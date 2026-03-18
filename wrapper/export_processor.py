import argparse
import os
from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoProcessor

from wrapper.processor import ModRWKVProcessor
from wrapper.tokenizer import RwkvTokenizer


RwkvTokenizer.register_for_auto_class("AutoTokenizer")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Mod-RWKV processor assets for AutoProcessor.from_pretrained(..., trust_remote_code=True)."
    )
    parser.add_argument("--output-dir", help="Directory to write the exported processor bundle into.")
    parser.add_argument(
        "--image-processor",
        dest="image_processor_name_or_path",
        default=None,
        help="Optional image processor source passed to AutoImageProcessor.from_pretrained.",
    )
    parser.add_argument(
        "--max-image-tokens",
        type=int,
        default=None,
        help="Max allowed image tokens to be fed to LLM.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output_dir if it already exists.",
    )
    return parser.parse_args()


def allowed_pixels_from_max_image_tokens(max_image_tokens: int, patch_size: int = 16, merge_size: int = 2) -> int:
    """
    Estimate the image pixel budget that corresponds to a max number of image tokens sent to the LLM.

    Qwen image token count at LLM side is approximately:
        llm_image_tokens = (H * W) / (patch_size^2 * merge_size^2)

    So the allowed pixel budget is:
        allowed_pixels = max_image_tokens * patch_size^2 * merge_size^2
    """
    if max_image_tokens <= 0:
        raise ValueError("max_image_tokens must be > 0")
    if patch_size <= 0 or merge_size <= 0:
        raise ValueError("patch_size and merge_size must be > 0")

    return max_image_tokens * (patch_size**2) * (merge_size**2)


def build_tokenizer():
    vocab_path = Path(__file__).with_name("wr_vocab_v20230424.txt")
    tokenizer = RwkvTokenizer(vocab_file=str(vocab_path))
    if tokenizer.pad_token_id != tokenizer.eos_token_id:
        raise ValueError(
            f"expected pad_token_id to match eos_token_id, got pad={tokenizer.pad_token_id} eos={tokenizer.eos_token_id}"
        )
    return tokenizer


def build_processor(image_processor_name_or_path=None, max_image_tokens:int=None):
    tokenizer = build_tokenizer()
    image_processor = None
    if image_processor_name_or_path:
        image_processor = AutoImageProcessor.from_pretrained(image_processor_name_or_path, trust_remote_code=True)
    processor = ModRWKVProcessor(tokenizer=tokenizer, image_processor=image_processor)

    if isinstance(max_image_tokens, int):
        allowed_pixels = allowed_pixels_from_max_image_tokens(
            max_image_tokens=max_image_tokens,
            patch_size=processor.image_processor.patch_size,
            merge_size=processor.image_processor.merge_size,
        )
        print(f"Configuring processor image_processor.size.longest_edge to {allowed_pixels} to correspond to max_image_tokens={max_image_tokens}")
        processor.image_processor.size["longest_edge"] = int(allowed_pixels)
    
    return processor


def prepare_output_dir(output_dir, force=False):
    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()) and not force:
        raise FileExistsError(
            f"Output directory '{output_path}' already exists and is not empty. Use --force to overwrite."
        )
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def assert_image_token_alignment(processor, inputs):
    if "image_grid_thw" not in inputs:
        return

    image_grid_thw = inputs["image_grid_thw"]
    if not isinstance(image_grid_thw, torch.Tensor):
        image_grid_thw = torch.as_tensor(image_grid_thw)

    merge_size = processor.image_processor.merge_size
    expected_image_pad = (image_grid_thw.prod(dim=-1) // (merge_size**2)).tolist()
    input_ids = inputs["input_ids"]
    actual_image_pad = (input_ids == processor.image_token_id).sum(dim=-1).tolist()
    actual_vision_start = (input_ids == processor.vision_start_token_id).sum(dim=-1).tolist()
    actual_vision_end = (input_ids == processor.vision_end_token_id).sum(dim=-1).tolist()

    expected_total_image_pad = sum(expected_image_pad)
    if actual_image_pad != [expected_total_image_pad]:
        raise ValueError(
            "image_pad token count does not match image_grid_thw: "
            f"expected {expected_total_image_pad}, got {actual_image_pad}."
        )

    expected_num_images = image_grid_thw.shape[0]
    if actual_vision_start != [expected_num_images] or actual_vision_end != [expected_num_images]:
        raise ValueError(
            "Vision boundary token count does not match image_grid_thw rows: "
            f"expected {expected_num_images}, got starts={actual_vision_start}, ends={actual_vision_end}."
        )


def test_saved_processor(processor_path):
    from PIL import Image

    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    print(processor)
    print("pad_token_id:", processor.tokenizer.pad_token_id, "eos_token_id:", processor.tokenizer.eos_token_id)
    if processor.tokenizer.pad_token_id != processor.tokenizer.eos_token_id:
        raise ValueError(
            f"reloaded tokenizer pad_token_id={processor.tokenizer.pad_token_id} does not match eos_token_id={processor.tokenizer.eos_token_id}"
        )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": Image.open("docs/03-Confusing-Pictures.jpg").convert("RGB"),
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    assert_image_token_alignment(processor, inputs)
    print(inputs.keys())
    outputs = processor.batch_decode(inputs["input_ids"], skip_special_tokens=False)
    print(outputs)


    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": Image.open("docs/03-Confusing-Pictures.jpg").convert("RGB"),
                },
                {"type": "text", "text": "Describe <image> and list out objects."},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    assert_image_token_alignment(processor, inputs)
    print(inputs.keys())
    outputs = processor.batch_decode(inputs["input_ids"], skip_special_tokens=False)
    print(outputs)

def main():
    args = parse_args()
    output_path = prepare_output_dir(args.output_dir, force=args.force)
    processor = build_processor(args.image_processor_name_or_path, args.max_image_tokens)
    processor.save_pretrained(output_path)
    print(f"Exported processor bundle to {os.fspath(output_path)}")
    test_saved_processor(output_path)


if __name__ == "__main__":
    main()
