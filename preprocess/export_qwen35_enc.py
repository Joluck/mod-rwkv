from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration, Qwen3_5VisionModel, Qwen3VLProcessor



def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3.5-0.8B", help="Path to the Qwen 3.5 model to export from")
    parser.add_argument("--output_path", type=str, required=True, help="Root folder to save the exported vision bundle (processor + vision encoder config and weights)")
    parser.add_argument("--max_image_tokens", type=int, default=4096, help="Max number of image tokens to configure for export. Corresponds to the max image token budget that the exported vision encoder + processor will be configured for.")
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



def export_vision_bundle(src_model_path: str, bundle_out: Path, max_image_tokens: int = 1024):
    bundle_out.mkdir(parents=True, exist_ok=False)
    processor :Qwen3VLProcessor= AutoProcessor.from_pretrained(src_model_path, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(src_model_path, trust_remote_code=True)

    vision_model = model.model.visual

    allowed_pixels = allowed_pixels_from_max_image_tokens(
        max_image_tokens=max_image_tokens,
        patch_size=processor.image_processor.patch_size,
        merge_size=processor.image_processor.merge_size,
    )
    print(f"Configuring processor image_processor.size.longest_edge to {allowed_pixels} to correspond to max_image_tokens={max_image_tokens}")
    processor.image_processor.size["longest_edge"] = int(allowed_pixels)

    


    vision_model.save_pretrained(bundle_out, safe_serialization=True)
    processor.save_pretrained(bundle_out)


    return vision_model, processor, allowed_pixels


if __name__ == "__main__":
    args = get_args()
    original_vision, processor, allowed_pixels = export_vision_bundle(
        src_model_path=args.model_path,
        bundle_out=Path(args.output_path),
        max_image_tokens=args.max_image_tokens,
    )


    print("Saved vision encoder + processor to:", args.output_path)
