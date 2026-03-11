# -*- coding: utf-8 -*-
"""Smoke test an exported ModRWKV Hugging Face bundle.

This script verifies that an exported bundle can be loaded with:
- AutoProcessor
- AutoModelForImageTextToText
- AutoModelForCausalLM

It then runs:
- processor.apply_chat_template(...)
- an image-text model forward pass
- a causal LM forward pass
- generate()
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Smoke test an exported ModRWKV bundle")
    parser.add_argument(
        "--bundle-path",
        required=True,
        help="Path to the exported Hugging Face bundle.",
    )
    parser.add_argument(
        "--image-path",
        default=str(repo_root / "docs" / "03-Confusing-Pictures.jpg"),
        help="Image to use for the multimodal smoke test.",
    )
    parser.add_argument(
        "--prompt",
        default="Describe this image.",
        help="User prompt to pair with the image.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run the test on. CUDA is required for FLA RWKV forward/generation.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype for loading the exported bundle.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Max new tokens for generate().",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def move_batch_to_device(batch, device: str):
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def build_messages(image_path: Path, prompt: str):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": Image.open(image_path).convert("RGB"),
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def main() -> None:
    args = parse_args()
    bundle_path = Path(args.bundle_path).resolve()
    image_path = Path(args.image_path).resolve()

    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This exported RWKV stack requires CUDA for forward/generate.")

    dtype = resolve_dtype(args.dtype)
    messages = build_messages(image_path, args.prompt)

    processor = AutoProcessor.from_pretrained(bundle_path, trust_remote_code=True)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = move_batch_to_device(inputs, args.device)

    print("processor keys:", sorted(inputs.keys()))
    print("input_ids shape:", tuple(inputs["input_ids"].shape))
    if "pixel_values" in inputs:
        print("pixel_values shape:", tuple(inputs["pixel_values"].shape))
    if "image_grid_thw" in inputs:
        print("image_grid_thw:", inputs["image_grid_thw"].tolist())

    model = AutoModelForImageTextToText.from_pretrained(bundle_path, trust_remote_code=True, dtype=dtype).to(args.device)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    print(
        "AutoModelForImageTextToText output:",
        type(outputs).__name__,
        tuple(outputs.logits.shape),
        outputs.logits.dtype,
    )

    lm = AutoModelForCausalLM.from_pretrained(bundle_path, trust_remote_code=True, dtype=dtype).to(args.device)
    lm.eval()
    with torch.no_grad():
        lm_outputs = lm(**inputs)
    print("AutoModelForCausalLM output:", type(lm_outputs).__name__, tuple(lm_outputs.logits.shape), lm_outputs.logits.dtype)

    with torch.no_grad():
        generated = lm.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    prompt_length = inputs["input_ids"].shape[1]
    generated_suffix = generated[:, prompt_length:]
    print("generated shape:", tuple(generated.shape))
    print("prompt length:", prompt_length)
    print("new token ids:", generated[0, prompt_length:].tolist())
    print("decoded full:")
    print(processor.batch_decode(generated, skip_special_tokens=False)[0])
    print("decoded new text:")
    print(processor.batch_decode(generated_suffix, skip_special_tokens=False)[0])


if __name__ == "__main__":
    main()