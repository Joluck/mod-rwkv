# -*- coding: utf-8 -*-
"""
ModRWKV - Multimodal RWKV7 Model with Vision Encoder
Compatible with HuggingFace Transformers
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    PretrainedConfig,
    PreTrainedModel,
    Qwen3_5VisionModel,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from fla.models.utils import Cache, FLAGenerationMixin
from fla.models.rwkv7 import RWKV7Config, RWKV7Model
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss
from fla.modules.l2warp import l2_warp

logger = logging.get_logger(__name__)


@dataclass
class ModRWKVProjectorConfig:
    """Configuration for the vision-to-language projector."""
    encoder_dim: int = 768  # Qwen3.5 vision encoder output dim
    project_dim: int = 1024  # RWKV7 hidden size
    hidden_dim: Optional[int] = None  # MLP hidden dim (default: project_dim * 4)

    def to_dict(self):
        return {
            "encoder_dim": self.encoder_dim,
            "project_dim": self.project_dim,
            "hidden_dim": self.hidden_dim,
        }


class ModRWKVConfig(PretrainedConfig):
    """
    Configuration class for ModRWKV model.

    This config holds the configurations for all three components:
    1. Vision Encoder (Qwen3.5VisionModel)
    2. Projector (VisualAdapter)
    3. Language Model (RWKV7)
    """
    model_type = "modrwkv"
    is_composition = True

    @classmethod
    def from_text_vision_configs(
        cls,
        text_config: Union[RWKV7Config, Dict[str, Any]],
        vision_config: Union[Qwen3_5VisionConfig, Dict[str, Any]],
        projector_config: Optional[Union[ModRWKVProjectorConfig, Dict[str, Any]]] = None,
        **kwargs,
    ) -> "ModRWKVConfig":
        return cls(
            text_config=text_config,
            vision_config=vision_config,
            projector_config=projector_config,
            **kwargs,
        )

    def __init__(
        self,
        # Text/LLM config
        text_config: Optional[Union[RWKV7Config, Dict]] = None,
        # Vision encoder config
        vision_config: Optional[Union[Qwen3_5VisionConfig, Dict]] = None,
        # Projector config
        projector_config: Optional[Union[ModRWKVProjectorConfig, Dict]] = None,
        # Special tokens
        image_token_id: int = 65532,
        vision_start_token_id: int = 65530,
        vision_end_token_id: int = 65531,
        # Tie weights
        tie_word_embeddings: bool = False,
        # Model architecture flags
        use_conv_in_projector: bool = False,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        # Initialize text_config (RWKV7)
        if text_config is None:
            text_config = {}
        if isinstance(text_config, dict):
            text_config = RWKV7Config(**text_config)
        self.text_config = text_config

        # Initialize vision_config (Qwen3.5VisionModel)
        if vision_config is None:
            vision_config = {}
        if isinstance(vision_config, dict):
            vision_config = Qwen3_5VisionConfig(**vision_config)
        self.vision_config = vision_config

        # Initialize projector_config
        if projector_config is None:
            projector_config = ModRWKVProjectorConfig(
                encoder_dim=getattr(vision_config, "out_hidden_size", 768),
                project_dim=getattr(text_config, "hidden_size", 1024),
            )
        elif isinstance(projector_config, dict):
            projector_config = ModRWKVProjectorConfig(**projector_config)
        self.projector_config = projector_config

        # Special token IDs
        self.image_token_id = image_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

        # Architecture flags
        self.use_conv_in_projector = use_conv_in_projector

    def to_dict(self) -> Dict[str, Any]:
        """Convert this instance to a dictionary."""
        output = super().to_dict()
        output["text_config"] = self.text_config.to_dict() if hasattr(self.text_config, "to_dict") else self.text_config
        output["vision_config"] = self.vision_config.to_dict() if hasattr(self.vision_config, "to_dict") else self.vision_config
        output["projector_config"] = self.projector_config.to_dict() if hasattr(self.projector_config, "to_dict") else self.projector_config
        output["image_token_id"] = self.image_token_id
        output["vision_start_token_id"] = self.vision_start_token_id
        output["vision_end_token_id"] = self.vision_end_token_id
        output["use_conv_in_projector"] = self.use_conv_in_projector
        return output


class ModRWKVPreTrainedModel(PreTrainedModel):
    """Base class for ModRWKV models."""
    config_class = ModRWKVConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RWKV7Block"]
    _supports_cache_class = True
    _skip_keys_device_placement = ["past_key_values"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)


class VisualAdapter(nn.Module):
    """
    Projector that maps vision encoder outputs to LLM embedding space.
    Uses a 2-layer MLP with residual connection and pre-norm.
    """
    def __init__(self, encoder_dim: int, project_dim: int, hidden_dim: Optional[int] = None, use_conv: bool = False):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.project_dim = project_dim
        self.hidden_dim = hidden_dim or project_dim * 4
        self.use_conv = use_conv

        self.pre_norm = nn.LayerNorm(self.project_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.project_dim),
        )

        if use_conv:
            self.conv = nn.Conv1d(
                in_channels=encoder_dim,
                out_channels=encoder_dim,
                kernel_size=3,
                stride=2,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_conv:
            x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.mlp(x)
        return x + self.pre_norm(x)


class RWKV7VLModel(ModRWKVPreTrainedModel):
    """
    The bare ModRWKV model, consisting of:
    1. Vision Encoder (Qwen3.5VisionModel)
    2. Projector (VisualAdapter)
    3. Language Model (RWKV7Model)
    """

    def __init__(self, config: ModRWKVConfig):
        super().__init__(config)
        self.config = config

        # Vision encoder (Qwen3.5)
        self.encoder = Qwen3_5VisionModel(config.vision_config)

        # Projector (Vision -> LLM)
        proj_cfg = config.projector_config
        self.proj = VisualAdapter(
            encoder_dim=proj_cfg.encoder_dim,
            project_dim=proj_cfg.project_dim,
            hidden_dim=proj_cfg.hidden_dim,
            use_conv=config.use_conv_in_projector,
        )

        # Language model (RWKV7)
        self.llm = RWKV7Model(config.text_config)

        self.post_init()

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor,
    ) -> torch.Tensor:
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`,
        and checks that the placeholder token count is equal to the length of multimodal features.
        """
        if input_ids is None:
            raise ValueError("input_ids must be provided when injecting image features into text embeddings.")

        special_image_mask = input_ids == self.config.image_token_id

        n_image_tokens = special_image_mask.sum().item()
        n_image_features = image_features.shape[0]

        if n_image_tokens > n_image_features:
            # More image tokens than features — only fill the first n positions
            image_positions = special_image_mask.nonzero(as_tuple=False)
            excess = image_positions[n_image_features:]
            special_image_mask[excess[:, 0], excess[:, 1]] = False

        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

        if inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        return special_image_mask

    def _get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
    ) -> Tuple[List[torch.FloatTensor], torch.LongTensor]:
        """
        Get image features from the vision encoder and split them by image.

        Returns:
            image_embeds: List of tensors, one per image
            image_grid_thw: The image grid thw tensor
        """
        vision_output = self.encoder(pixel_values, image_grid_thw)

        if hasattr(vision_output, "pooler_output") and vision_output.pooler_output is not None:
            pooled_embeds = vision_output.pooler_output
            if pooled_embeds.shape[-1] == self.proj.encoder_dim:
                vision_embeds = pooled_embeds
            else:
                vision_embeds = vision_output.last_hidden_state
        elif hasattr(vision_output, "last_hidden_state"):
            vision_embeds = vision_output.last_hidden_state
        elif isinstance(vision_output, (tuple, list)):
            vision_embeds = vision_output[0]
        else:
            vision_embeds = vision_output

        # Split by image
        spatial_merge_size = getattr(self.encoder.config, 'spatial_merge_size', 2)
        split_sizes = (image_grid_thw.prod(-1) // (spatial_merge_size ** 2)).tolist()
        image_embeds = torch.split(vision_embeds, split_sizes)

        return list(image_embeds), image_grid_thw

    def _project_image_features(self, image_embeds: List[torch.FloatTensor]) -> torch.FloatTensor:
        """Project image features to LLM embedding space."""
        projected_features = []
        for embeds in image_embeds:
            if embeds.dim() == 2:
                embeds = embeds.unsqueeze(0)
            projected = self.proj(embeds)
            projected_features.append(projected.reshape(-1, projected.shape[-1]))

        if not projected_features:
            return torch.empty(0, self.config.text_config.hidden_size, device=self.proj.mlp[0].weight.device)

        return torch.cat(projected_features, dim=0)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        return_dict = return_dict if return_dict is not None else self.config.text_config.use_return_dict

        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must provide either input_ids or inputs_embeds.")
        if (pixel_values is None) != (image_grid_thw is None):
            raise ValueError("pixel_values and image_grid_thw must be provided together.")

        # Get text embeddings
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Process vision inputs if provided
        if pixel_values is not None and image_grid_thw is not None:
            # Handle both batched and unbatched inputs
            # pixel_values: [batch_size * num_patches, channels, height, width]
            # image_grid_thw: [batch_size * num_images, 3]

            # Get and project image features
            image_embeds_list, _ = self._get_image_features(pixel_values, image_grid_thw)
            projected_embeds = self._project_image_features(image_embeds_list)

            # Get placeholder mask and scatter image embeddings
            image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=projected_embeds
            )

            # Ensure device and dtype match
            projected_embeds = projected_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, projected_embeds)

        # Forward through LLM
        outputs = self.llm(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        return outputs


class RWKV7VLForConditionalGeneration(ModRWKVPreTrainedModel, FLAGenerationMixin):
    """
    ModRWKV Model for conditional generation with vision capabilities.
    """
    # Note: In newer transformers versions, _tied_weights_keys can be a dict {target: source}
    # or a list. We use an empty dict to avoid compatibility issues with the base class.
    _tied_weights_keys = {}

    def __init__(self, config: ModRWKVConfig):
        super().__init__(config)
        self.model = RWKV7VLModel(config)
        self.vocab_size = config.text_config.vocab_size
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False
        )
        self.criterion = None

        # Initialize weights
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def generate(self, *args, **kwargs):
        try:
            return super().generate(*args, **kwargs)
        except AttributeError as exception:
            if 'past_key_values' in str(exception):
                raise AttributeError(
                    f"You tried to call `generate` with a decoding strategy that manipulates `past_key_values`, "
                    f"which is not supported for {self.__class__.__name__}. "
                    f"Try another generation strategy instead. "
                    f"For the available generation strategies, check this doc: "
                    f"https://huggingface.co/docs/transformers/en/generation_strategies#decoding-strategies"
                )
            else:
                raise exception

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        labels: Optional[torch.LongTensor] = None,
        shift_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Optional[int] = 0,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.text_config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.text_config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.text_config.use_return_dict

        # Forward through the model
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        hidden_states = outputs.last_hidden_state

        # Compute logits
        loss, logits = None, None
        has_labels = (labels is not None) or (shift_labels is not None)

        if not (self.config.text_config.fuse_linear_cross_entropy and has_labels):
            logits = self.lm_head(hidden_states if logits_to_keep is None else hidden_states[:, -logits_to_keep:])

        # Compute loss if labels provided
        if has_labels:
            if getattr(self, 'criterion', None) is None:
                if self.config.text_config.fuse_linear_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss(use_l2warp=self.config.text_config.use_l2warp)
                elif self.config.text_config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion

            # Prepare shift_labels
            if shift_labels is None:
                shift_labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)
            shift_labels = shift_labels.to(hidden_states.device)

            # Compute loss
            if self.config.text_config.fuse_linear_cross_entropy:
                loss = criterion(hidden_states, shift_labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(shift_labels.numel(), -1), shift_labels.view(-1))
                loss = l2_warp(loss, logits) if self.config.text_config.use_l2warp else loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        pixel_values=None,
        image_grid_thw=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        """Prepare inputs for generation, handling multimodal inputs."""
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        is_first_step = past_key_values is None
        if cache_position is not None:
            first_cache_position = cache_position[0]
            if isinstance(first_cache_position, torch.Tensor):
                first_cache_position = first_cache_position.item()
            is_first_step = is_first_step or first_cache_position == 0

        if is_first_step:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_grid_thw"] = image_grid_thw

        return model_inputs


# Register models with AutoModel
AutoConfig.register(ModRWKVConfig.model_type, ModRWKVConfig, exist_ok=True)
AutoModel.register(ModRWKVConfig, RWKV7VLForConditionalGeneration, exist_ok=True)
AutoModelForCausalLM.register(ModRWKVConfig, RWKV7VLForConditionalGeneration, exist_ok=True)
AutoModelForImageTextToText.register(ModRWKVConfig, RWKV7VLForConditionalGeneration, exist_ok=True)

ModRWKVConfig.register_for_auto_class("AutoConfig")
RWKV7VLForConditionalGeneration.register_for_auto_class("AutoModel")
RWKV7VLForConditionalGeneration.register_for_auto_class("AutoModelForCausalLM")
