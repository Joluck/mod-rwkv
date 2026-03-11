# -*- coding: utf-8 -*-
"""Export a ModRWKV checkpoint to a self-contained Hugging Face bundle.

The exported directory contains:
- model weights and config for AutoModelForCausalLM
- processor/tokenizer assets for AutoProcessor
- remote-code Python files required by trust_remote_code=True

Example:
    /home/rwkv/vl/bin/python export_hf_model.py \
        --checkpoint rwkv7-0.4b-sft-qwen3_5/rwkv-step-1024.pth \
        --output-dir out/modrwkv-hf
"""

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

import fla  # noqa: F401
from fla.models.rwkv7 import RWKV7Config
from wrapper.modeling_modrwkv import (
    ModRWKVConfig,
    ModRWKVProjectorConfig,
    RWKV7VLForConditionalGeneration,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_VISION_BUNDLE = REPO_ROOT / "vision_bundle"
DEFAULT_PROCESSOR_BUNDLE = REPO_ROOT / "processor_bundle"

REMOTE_CODE_FILES = {
    REPO_ROOT / "wrapper" / "modeling_modrwkv.py": "modeling_modrwkv.py",
    REPO_ROOT / "wrapper" / "processor.py": "processor.py",
    REPO_ROOT / "wrapper" / "tokenizer.py": "tokenizer.py",
}

PROCESSOR_ASSET_FILES = [
    "processor_config.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "wr_vocab_v20230424.txt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ModRWKV model and processor to Hugging Face format")
    parser.add_argument("--checkpoint", required=True, help="Path to the training checkpoint (.pth file).")
    parser.add_argument("--output-dir", required=True, help="Directory to write the exported bundle into.")
    parser.add_argument(
        "--vision-bundle",
        default=str(DEFAULT_VISION_BUNDLE),
        help="Directory containing the exported Qwen3.5 vision config.",
    )
    parser.add_argument(
        "--processor-bundle",
        default=str(DEFAULT_PROCESSOR_BUNDLE),
        help="Directory containing processor/tokenizer assets to copy.",
    )
    parser.add_argument(
        "--precision",
        default="bf16",
        choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"],
        help="Weight dtype used for the exported model.",
    )
    parser.add_argument("--image-token-id", type=int, default=65532)
    parser.add_argument("--vision-start-token-id", type=int, default=65530)
    parser.add_argument("--vision-end-token-id", type=int, default=65531)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    parser.add_argument(
        "--skip-load-test",
        action="store_true",
        help="Skip the final AutoConfig/AutoProcessor load validation.",
    )
    return parser.parse_args()


def precision_to_dtype(precision: str) -> Tuple[torch.dtype, str]:
    mapping = {
        "float32": (torch.float32, "float32"),
        "fp32": (torch.float32, "float32"),
        "float16": (torch.float16, "float16"),
        "fp16": (torch.float16, "float16"),
        "bfloat16": (torch.bfloat16, "bfloat16"),
        "bf16": (torch.bfloat16, "bfloat16"),
    }
    return mapping[precision]


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory '{output_dir}' already exists and is not empty.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def load_checkpoint(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    return torch.load(checkpoint_path, map_location="cpu", weights_only=True)


def convert_llm_key_to_hf(name: str) -> Optional[str]:
    unused_names = {"blocks.0.att.v0", "blocks.0.att.v1", "blocks.0.att.v2"}
    if name in unused_names:
        return None

    emb_head_map = {
        "emb.weight": "embeddings.weight",
        "ln_out.weight": "norm.weight",
        "ln_out.bias": "norm.bias",
        "head.weight": "lm_head.weight",
    }
    if name in emb_head_map:
        return emb_head_map[name]

    if not name.startswith("blocks."):
        return None

    parts = name.split(".")
    block_type = parts[2]
    type_map = {
        "att": "attn",
        "ffn": "ffn",
        "ln0": "pre_norm",
        "ln1": "attn_norm",
        "ln2": "ffn_norm",
    }
    if block_type not in type_map:
        return None

    base = f"layers.{parts[1]}.{type_map[block_type]}"

    if block_type == "att" and len(parts) >= 4:
        component = parts[3]
        if len(component) == 2 and component[0] in "wvag" and component[1] in "012":
            suffix = {
                "0": "2.bias",
                "1": "0.weight",
                "2": "2.weight",
            }[component[1]]
            return f"{base}.{component[0]}_lora.lora.{suffix}"

        proj_map = {
            "receptance": "r_proj",
            "key": "k_proj",
            "value": "v_proj",
            "ln_x": "g_norm",
            "output": "o_proj",
        }
        if component in proj_map:
            remaining = ".".join(parts[4:]) if len(parts) > 4 else "weight"
            return f"{base}.{proj_map[component]}.{remaining}"

    remaining = ".".join(parts[3:])
    return f"{base}.{remaining}"


def split_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    components = {"encoder": {}, "proj": {}, "llm": {}, "other": {}}

    for name, tensor in state_dict.items():
        if name.startswith("encoder."):
            stripped = name.replace("encoder.", "", 1)
            if stripped.startswith("encoder."):
                stripped = stripped.replace("encoder.", "", 1)
            components["encoder"][stripped] = tensor
        elif name.startswith("proj."):
            components["proj"][name.replace("proj.", "", 1)] = tensor
        elif name.startswith("llm."):
            translated = convert_llm_key_to_hf(name.replace("llm.", "", 1))
            if translated:
                components["llm"][translated] = tensor
        else:
            components["other"][name] = tensor

    return components


def create_text_config(components: Dict[str, Dict[str, torch.Tensor]], original_state_dict: Dict[str, torch.Tensor]) -> RWKV7Config:
    llm_weights = components["llm"]
    emb_weight = llm_weights["embeddings.weight"]
    vocab_size, hidden_size = emb_weight.shape

    max_layer = -1
    for key in llm_weights:
        match = re.search(r"layers\.(\d+)\.", key)
        if match:
            max_layer = max(max_layer, int(match.group(1)))
    num_hidden_layers = max_layer + 1

    intermediate_size = llm_weights["layers.0.ffn.key.weight"].shape[0]
    hidden_ratio = intermediate_size / hidden_size

    decay_low_rank_dim = 64
    gate_low_rank_dim = 128
    a_low_rank_dim = 64
    v_low_rank_dim = 32
    head_dim = 64

    for key, tensor in original_state_dict.items():
        if key.endswith("att.r_k"):
            head_dim = tensor.shape[1]
        if not key.startswith("llm.blocks."):
            continue
        parts = key.split(".")
        if len(parts) >= 5 and parts[3] == "att":
            component = parts[4]
            if len(component) == 2 and component[1] == "1":
                if component[0] == "w":
                    decay_low_rank_dim = tensor.shape[1]
                elif component[0] == "g":
                    gate_low_rank_dim = tensor.shape[1]
                elif component[0] == "a":
                    a_low_rank_dim = tensor.shape[1]
                elif component[0] == "v":
                    v_low_rank_dim = tensor.shape[1]

    return RWKV7Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        hidden_ratio=hidden_ratio,
        head_dim=head_dim,
        decay_low_rank_dim=decay_low_rank_dim,
        gate_low_rank_dim=gate_low_rank_dim,
        a_low_rank_dim=a_low_rank_dim,
        v_low_rank_dim=v_low_rank_dim,
        attn_mode="chunk",
        fuse_cross_entropy=True,
        fuse_linear_cross_entropy=False,
        fuse_norm=True,
        use_cache=True,
        use_l2warp=True,
    )


def create_projector_config(components: Dict[str, Dict[str, torch.Tensor]], text_config: RWKV7Config) -> ModRWKVProjectorConfig:
    proj_weights = components["proj"]
    hidden_dim, encoder_dim = proj_weights["mlp.0.weight"].shape
    project_dim, hidden_dim_out = proj_weights["mlp.2.weight"].shape
    if hidden_dim_out != hidden_dim:
        raise ValueError("Projector checkpoint shapes are inconsistent.")
    if project_dim != text_config.hidden_size:
        raise ValueError(
            f"Projector output dim {project_dim} does not match text hidden size {text_config.hidden_size}."
        )
    return ModRWKVProjectorConfig(
        encoder_dim=encoder_dim,
        project_dim=project_dim,
        hidden_dim=hidden_dim,
    )


def reshape_weight(weight: torch.Tensor, target_shape: torch.Size, key: str) -> Optional[torch.Tensor]:
    if weight.shape == target_shape:
        return weight

    if len(weight.shape) == 3 and weight.shape[:2] == (1, 1) and len(target_shape) == 1:
        squeezed = weight.squeeze()
        if squeezed.shape == target_shape:
            return squeezed

    if "lora" in key and weight.ndim == 2 and weight.t().shape == target_shape:
        return weight.t()

    if weight.numel() == target_shape.numel():
        reshaped = weight.reshape(target_shape)
        if reshaped.shape == target_shape:
            return reshaped

    return None


def load_weights_into_model(
    model: RWKV7VLForConditionalGeneration,
    components: Dict[str, Dict[str, torch.Tensor]],
) -> Tuple[int, int]:
    encoder_missing, encoder_unexpected = model.model.encoder.load_state_dict(components["encoder"], strict=False)
    if encoder_unexpected:
        raise ValueError(f"Unexpected encoder keys: {encoder_unexpected[:5]}")

    proj_missing, proj_unexpected = model.model.proj.load_state_dict(components["proj"], strict=False)
    if proj_missing or proj_unexpected:
        raise ValueError(
            f"Projector load mismatch. Missing={proj_missing[:5]}, unexpected={proj_unexpected[:5]}"
        )

    llm_state = components["llm"]
    loaded = 0
    mismatched = 0

    for name, parameter in model.model.llm.named_parameters():
        if name not in llm_state:
            continue
        weight = reshape_weight(llm_state[name], parameter.shape, name)
        if weight is None:
            mismatched += 1
            continue
        parameter.data.copy_(weight.to(dtype=parameter.dtype))
        loaded += 1

    for name, buffer in model.model.llm.named_buffers():
        if name not in llm_state:
            continue
        weight = reshape_weight(llm_state[name], buffer.shape, name)
        if weight is None:
            mismatched += 1
            continue
        buffer.data.copy_(weight.to(dtype=buffer.dtype))
        loaded += 1

    if "lm_head.weight" in llm_state:
        weight = reshape_weight(llm_state["lm_head.weight"], model.lm_head.weight.shape, "lm_head.weight")
        if weight is None:
            mismatched += 1
        else:
            model.lm_head.weight.data.copy_(weight.to(dtype=model.lm_head.weight.dtype))
            loaded += 1

    return loaded, mismatched


def copy_remote_code(output_dir: Path) -> None:
    for src, dst_name in REMOTE_CODE_FILES.items():
        shutil.copy2(src, output_dir / dst_name)


def copy_processor_assets(processor_bundle: Path, output_dir: Path) -> None:
    for filename in PROCESSOR_ASSET_FILES:
        src = processor_bundle / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing processor asset: {src}")
        shutil.copy2(src, output_dir / filename)


def patch_config_auto_map(output_dir: Path) -> None:
    config_path = output_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config_data = json.load(handle)

    config_data["auto_map"] = {
        "AutoConfig": "modeling_modrwkv.ModRWKVConfig",
        "AutoModel": "modeling_modrwkv.RWKV7VLModel",
        "AutoModelForCausalLM": "modeling_modrwkv.RWKV7VLForConditionalGeneration",
        "AutoModelForImageTextToText": "modeling_modrwkv.RWKV7VLForConditionalGeneration",
    }

    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config_data, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def write_readme(output_dir: Path, checkpoint_path: Path) -> None:
    readme = output_dir / "README.md"
    readme.write_text(
        "# ModRWKV Hugging Face Export\n\n"
        f"Exported from checkpoint: `{checkpoint_path}`\n\n"
        "Load with `trust_remote_code=True`:\n\n"
        "```python\n"
        "from transformers import AutoModelForCausalLM, AutoProcessor\n\n"
        f"model = AutoModelForCausalLM.from_pretrained(\"{output_dir}\", trust_remote_code=True)\n"
        f"processor = AutoProcessor.from_pretrained(\"{output_dir}\", trust_remote_code=True)\n"
        "```\n",
        encoding="utf-8",
    )


def validate_export(output_dir: Path) -> None:
    from transformers import AutoConfig, AutoProcessor

    AutoConfig.from_pretrained(output_dir, trust_remote_code=True)
    AutoProcessor.from_pretrained(output_dir, trust_remote_code=True)


def export_bundle(args: argparse.Namespace) -> Path:
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    vision_bundle = Path(args.vision_bundle).resolve()
    processor_bundle = Path(args.processor_bundle).resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not vision_bundle.exists():
        raise FileNotFoundError(f"Vision bundle not found: {vision_bundle}")
    if not processor_bundle.exists():
        raise FileNotFoundError(f"Processor bundle not found: {processor_bundle}")

    dtype, dtype_name = precision_to_dtype(args.precision)
    prepare_output_dir(output_dir, args.overwrite)

    state_dict = load_checkpoint(checkpoint_path)
    components = split_state_dict(state_dict)

    text_config = create_text_config(components, state_dict)
    text_config.torch_dtype = dtype_name
    vision_config = Qwen3_5VisionConfig.from_pretrained(vision_bundle)
    projector_config = create_projector_config(components, text_config)

    config = ModRWKVConfig(
        text_config=text_config,
        vision_config=vision_config,
        projector_config=projector_config,
        image_token_id=args.image_token_id,
        vision_start_token_id=args.vision_start_token_id,
        vision_end_token_id=args.vision_end_token_id,
        use_conv_in_projector=False,
    )
    config.torch_dtype = dtype_name
    config.auto_map = {
        "AutoConfig": "modeling_modrwkv.ModRWKVConfig",
        "AutoModel": "modeling_modrwkv.RWKV7VLModel",
        "AutoModelForCausalLM": "modeling_modrwkv.RWKV7VLForConditionalGeneration",
        "AutoModelForImageTextToText": "modeling_modrwkv.RWKV7VLForConditionalGeneration",
    }

    model = RWKV7VLForConditionalGeneration(config).to(dtype=dtype)
    loaded, mismatched = load_weights_into_model(model, components)
    if mismatched:
        raise ValueError(f"Failed to map {mismatched} checkpoint tensors into the exported model state.")

    model.save_pretrained(output_dir, safe_serialization=True)
    copy_remote_code(output_dir)
    copy_processor_assets(processor_bundle, output_dir)
    patch_config_auto_map(output_dir)
    write_readme(output_dir, checkpoint_path)

    print(f"Loaded tensors into model: {loaded}")
    print(f"Exported bundle to: {output_dir}")
    return output_dir


def main() -> None:
    args = parse_args()
    output_dir = export_bundle(args)
    if not args.skip_load_test:
        validate_export(output_dir)
        print("Validation passed: AutoConfig and AutoProcessor can load the exported bundle.")


if __name__ == "__main__":
    main()