# -*- coding: utf-8 -*-

import warnings
from typing import Dict, List, Optional, Union

from transformers.configuration_utils import PretrainedConfig
from fla.models.rwkv7.configuration_rwkv7 import RWKV7Config
from transformers.models.auto import CONFIG_MAPPING, AutoConfig
from transformers import SiglipVisionConfig


# EncoderConfig: Dict[str, Type] = {
#     "clip": ClipConfig,
#     "whisper": WhisperConfig,
#     "speech": SpeechConfig,
#     "siglip2": SiglipConfig,
# }

class ProjConfig(PretrainedConfig):
    model_type = "siglp2"

    def __init__(
        self,
        encoder_type="siglp2",
        path=None,
        encoder_dim=None,
        project_dim=None,
        hidden_dim=None, 
        use_conv=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder_type = encoder_type
        self.path = path
        self.encoder_dim = encoder_dim
        self.project_dim = project_dim
        self.hidden_dim = hidden_dim
        self.use_conv = use_conv

class RWKV7VLConfig(PretrainedConfig):

    model_type = "rwkv7_vl"
    sub_configs = {"vision_config": AutoConfig, "proj_config":ProjConfig, "text_config": RWKV7Config}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        proj_config=None,
        image_token_id=65532,
        vision_start_token_id=65530,
        vision_end_token_id=65531,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"].for_model(**vision_config)
        # elif vision_config is None:
        #     self.vision_config = self.sub_configs["vision_config"]()
        if isinstance(proj_config, dict):
            self.proj_config = self.sub_configs["proj_config"](**proj_config)
        elif proj_config is None:
            self.proj_config = self.sub_configs["proj_config"]()
        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        self.image_token_id = image_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)


__all__ = ["RWKV7VLConfig"]