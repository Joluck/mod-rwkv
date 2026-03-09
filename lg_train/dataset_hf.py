'''
hf compatible dataset
'''
import os

os.environ["HF_DATASETS_CACHE"]="/mnt/raid0_8t/huggingface/datasets"
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from transformers import AutoProcessor


ROLE_TABLE = {
    "user": "user",
    "assistant": "assistant",
    "system": "system",
    "human": "user",
    "gpt": "assistant",
}


_PROCESSOR_CACHE = {}


def _get_processor(processor_path):
    processor = _PROCESSOR_CACHE.get(processor_path)
    if processor is None:
        processor = AutoProcessor.from_pretrained(
            processor_path,
            trust_remote_code=True,
            use_fast=True,
        )
        _PROCESSOR_CACHE[processor_path] = processor
    return processor


def _normalize_chat(message):
    images = message["images"] if "images" in message else message["image"] if "image" in message else []
    texts = message["texts"] if "texts" in message else message["conversations"] if "conversations" in message else None

    if texts is None:
        raise ValueError("No valid text field found in the dataset item.")

    if not isinstance(images, list):
        images = [images]

    if not texts:
        raise ValueError("Empty chat data found in the dataset item.")

    first_text = texts[0]
    if not isinstance(first_text, dict):
        raise TypeError(f"Expected each chat turn to be a dict, but got {type(first_text).__name__}.")

    if "value" in first_text and "from" in first_text:
        normalized_texts = [{"from": msg["from"], "value": msg["value"]} for msg in texts]
    elif "content" in first_text and "role" in first_text:
        normalized_texts = [{"from": msg["role"], "value": msg["content"]} for msg in texts]
    elif "user" in first_text and "assistant" in first_text:
        normalized_texts = []
        for msg in texts:
            normalized_texts.append({"from": "user", "value": msg["user"]})
            normalized_texts.append({"from": "assistant", "value": msg["assistant"]})
    else:
        raise NotImplementedError("No valid chat format found in the dataset item.")

    return normalized_texts, images


def _build_messages(texts, images):
    messages = []
    for msg_index, msg in enumerate(texts):
        content = [{"type": "text", "text": msg["value"]}]
        if msg_index == 0:
            content.extend({"type": "image", "image": image} for image in images)
        messages.append(
            {
                "role": ROLE_TABLE.get(msg["from"], msg["from"]),
                "content": content,
            }
        )
    return messages


def _get_image_token_lengths(processor, image_grid_thw):
    if image_grid_thw is None:
        return []

    merge_size = processor.image_processor.merge_size
    return [int(grid[0] * grid[1] * grid[2]) // merge_size**2 for grid in image_grid_thw.tolist()]


def _build_inputs_and_labels(processor, texts, image_token_lengths, seqlen):
    input_ids = []
    label_ids = []
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = 0

    for msg_index, msg in enumerate(texts):
        role = ROLE_TABLE.get(msg["from"], msg["from"])
        role_name = "Assistant" if role == "assistant" else "System" if role == "system" else "User"
        content = msg["value"]

        if msg_index == 0 and image_token_lengths:
            for token_length in image_token_lengths:
                content += (
                    processor.vision_start_token
                    + processor.image_token * token_length
                    + processor.vision_end_token
                )

        tokenized = processor.tokenizer(
            f"\x16{role_name}: {content}\x17",
            add_special_tokens=False,
        )["input_ids"]
        input_ids.extend(tokenized)

        if role == "assistant":
            label_ids.extend(tokenized)
        else:
            label_ids.extend([-100] * len(tokenized))

    input_ids = input_ids[: seqlen + 1]
    label_ids = label_ids[: seqlen + 1]

    input_tensor = torch.tensor(input_ids, dtype=torch.long)
    label_tensor = torch.tensor(label_ids, dtype=torch.long)

    pad_length = max(seqlen - len(label_tensor) + 1, 0)
    if pad_length > 0:
        input_tensor = torch.nn.functional.pad(input_tensor, (0, pad_length), value=pad_token_id)
        label_tensor = torch.nn.functional.pad(label_tensor, (0, pad_length), value=-100)

    return input_tensor[:-1], label_tensor[1:]


def _tidyup_dataset_batch(item, processor_path, seqlen):
    processor = _get_processor(processor_path)
    batch_size = len(next(iter(item.values())))
    rendered_texts = []

    for index in range(batch_size):
        sample = {key: value[index] for key, value in item.items()}
        texts, images = _normalize_chat(sample)
        messages = _build_messages(texts, images)

        rendered_texts.append(
            processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        )

    return {"text": rendered_texts}


class HFDataset(Dataset):
    def __init__(self, dataset_path, processor_path, split='train', num_proc=24, seqlen=4096):
        dataset = load_dataset(dataset_path, split=split, num_proc=num_proc)
        self.processor_path = processor_path
        self.seqlen = seqlen
        self.raw_dataset = dataset
        remove_columns = list(dataset.column_names)

        self.dataset = dataset.map(
            function=_tidyup_dataset_batch,
            batched=True,
            batch_size=1024,
            num_proc=num_proc,
            fn_kwargs={"processor_path": processor_path, "seqlen": seqlen},
            remove_columns=remove_columns,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        '''
        Return processed input for the model. 
        - pixel_values: list of patched images from processor. 
        - input_ids: tokenized text inputs of shape [B, L]
        - label_ids: Labels for masking out the loss, of shape [B, L], where -100 indicates positions that should be ignored in the loss computation. Mask out user inputs in default.
        '''
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]

        processor = _get_processor(self.processor_path)
        raw_item = self.raw_dataset[idx]
        item = self.dataset[idx]
        texts, images = _normalize_chat(raw_item)
        rendered_text = item["text"]

        if images:
            model_inputs = processor(
                images=images,
                text=rendered_text,
                return_tensors="pt",
            )
            image_token_lengths = _get_image_token_lengths(processor, model_inputs.get("image_grid_thw"))
            pixel_values = {"pixel_values": model_inputs["pixel_values"]}
            if "image_grid_thw" in model_inputs:
                pixel_values["image_grid_thw"] = model_inputs["image_grid_thw"]
        else:
            model_inputs = processor(
                text=rendered_text,
                return_tensors="pt",
            )
            image_token_lengths = []
            pixel_values = None

        input_ids, label_ids = _build_inputs_and_labels(
            processor,
            texts,
            image_token_lengths,
            self.seqlen,
        )
        return pixel_values, input_ids, label_ids


def hf_collate_fn(batch):
    pixel_values, input_ids, label_ids = zip(*batch)
    batch_pixel_values = list(pixel_values)
    batch_input_ids = torch.stack(input_ids, dim=0)
    batch_label_ids = torch.stack(label_ids, dim=0)
    return batch_pixel_values, batch_input_ids, batch_label_ids


def create_hf_dataloader(
    dataset_path,
    processor_path,
    split='train',
    batch_size=1,
    shuffle=True,
    num_workers=0,
    num_proc=24,
    seqlen=4096,
    pin_memory=True,
    persistent_workers=False,
):
    dataset = HFDataset(
        dataset_path=dataset_path,
        processor_path=processor_path,
        split=split,
        num_proc=num_proc,
        seqlen=seqlen,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        collate_fn=hf_collate_fn,
    )
    

if __name__ == "__main__":
    
    dataset_path = "/mnt/sda1/HuggingFaceM4_FineVisionMax/"
    processor_name = "/home/rwkv/molin/mod-rwkv/processor_bundle"
    dataloader = create_hf_dataloader(
        dataset_path=dataset_path,
        processor_path=processor_name,
        batch_size=32,
        shuffle=True,
        num_workers=32,
        num_proc=24,
        seqlen=4096,
    )

    batch = next(iter(dataloader))
    print(batch[0][0].keys() if batch[0][0] is not None else None)
    print(batch[1].shape, batch[2].shape)