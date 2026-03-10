'''
hf compatible dataset
'''
import os

import lightning as L
import torch
from datasets import load_dataset, load_dataset_builder
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoProcessor


os.environ.setdefault("HF_DATASETS_CACHE", "/mnt/raid0_8t/huggingface/datasets")


ROLE_TABLE = {
    "user": "user",
    "assistant": "assistant",
    "system": "system",
    "human": "user",
    "gpt": "assistant",
}


_PROCESSOR_CACHE = {}


def _clear_processor_cache(_worker_id=None):
    _PROCESSOR_CACHE.clear()


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

    if images is None:
        images = []
    elif not isinstance(images, list):
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

    return normalized_texts, [image for image in images if image is not None]


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


def _normalize_images(images):
    normalized_images = []
    for image in images:
        if hasattr(image, "convert"):
            normalized_images.append(image.convert("RGB"))
        else:
            normalized_images.append(image)
    return normalized_images


def _render_chat(processor, texts, images):
    messages = _build_messages(texts, images)
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


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
            image_prefix = ""
            for token_length in image_token_lengths:
                image_prefix += (
                    processor.vision_start_token
                    + processor.image_token * token_length
                    + processor.vision_end_token
                )
            content = image_prefix + content

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


class HFDataset(Dataset):
    def __init__(self, dataset_path, processor_path, split='train', num_proc=None, seqlen=4096, cache_dir=None):
        self.processor_path = processor_path
        self.dataset_path = dataset_path
        self.split = split
        self.num_proc = num_proc
        self.seqlen = seqlen
        self.cache_dir = cache_dir or os.environ.get("HF_DATASETS_CACHE")
        self.dataset = None
        self.dataset_length = None

    def _get_dataset_length(self):
        if self.dataset_length is None:
            try:
                builder = load_dataset_builder(
                    self.dataset_path,
                    cache_dir=self.cache_dir,
                )
                split_info = builder.info.splits[self.split]
                self.dataset_length = split_info.num_examples
            except Exception:
                self.dataset_length = len(self._get_dataset())
        return self.dataset_length

    def _get_dataset(self):
        if self.dataset is None:
            self.dataset = load_dataset(
                self.dataset_path,
                split=self.split,
                cache_dir=self.cache_dir,
                keep_in_memory=False,
            )
        return self.dataset

    def release(self):
        self.dataset = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["dataset"] = None
        return state

    def __len__(self):
        return self._get_dataset_length()

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
        sample = self._get_dataset()[idx]
        texts, images = _normalize_chat(sample)
        images = _normalize_images(images)
        rendered_text = _render_chat(processor, texts, images)

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
            image_token_lengths = []
            pixel_values = None

        input_ids, label_ids = _build_inputs_and_labels(
            processor,
            texts,
            image_token_lengths,
            self.seqlen,
        )
        return pixel_values, input_ids, label_ids


class HFStreamingDataset(IterableDataset):
    """Streaming dataset that reads HF parquet files on-the-fly without caching."""

    def __init__(self, dataset_path, processor_path, split='train', seqlen=4096,
                 shuffle_buffer=1000, seed=42, cache_dir=None):
        self.dataset_path = dataset_path
        self.processor_path = processor_path
        self.split = split
        self.seqlen = seqlen
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.cache_dir = cache_dir or os.environ.get("HF_DATASETS_CACHE")
        self._epoch = 0
        self._length = None

    def _read_length(self):
        if self._length is None:
            try:
                builder = load_dataset_builder(
                    self.dataset_path,
                    cache_dir=self.cache_dir,
                )
                self._length = builder.info.splits[self.split].num_examples
            except Exception:
                self._length = 0
        return self._length

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __len__(self):
        return self._read_length()

    def __iter__(self):
        import torch.distributed as dist

        ds = load_dataset(self.dataset_path, split=self.split, streaming=True)

        rank, world_size = 0, 1
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        worker_info = torch.utils.data.get_worker_info()
        num_workers, worker_id = 1, 0
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        total_shards = world_size * num_workers
        shard_index = rank * num_workers + worker_id

        num_sources = ds.n_shards
        skip_mod = None
        if total_shards > 1:
            if num_sources >= total_shards:
                ds = ds.shard(num_shards=total_shards, index=shard_index)
            else:
                skip_mod = (total_shards, shard_index)

        if self.shuffle_buffer > 0:
            ds = ds.shuffle(
                buffer_size=self.shuffle_buffer,
                seed=self.seed + self._epoch,
            )

        processor = _get_processor(self.processor_path)
        for i, sample in enumerate(ds):
            if skip_mod is not None and i % skip_mod[0] != skip_mod[1]:
                continue
            try:
                yield _process_sample(sample, processor, self.seqlen)
            except Exception as e:
                import warnings
                warnings.warn(f"Skipping sample: {e}")
                continue


def _process_sample(sample, processor, seqlen):
    texts, images = _normalize_chat(sample)
    images = _normalize_images(images)
    rendered_text = _render_chat(processor, texts, images)

    if images:
        model_inputs = processor(
            images=images,
            text=rendered_text,
            return_tensors="pt",
        )
        image_token_lengths = _get_image_token_lengths(
            processor, model_inputs.get("image_grid_thw"),
        )
        pixel_values = {"pixel_values": model_inputs["pixel_values"]}
        if "image_grid_thw" in model_inputs:
            pixel_values["image_grid_thw"] = model_inputs["image_grid_thw"]
    else:
        image_token_lengths = []
        pixel_values = None

    input_ids, label_ids = _build_inputs_and_labels(
        processor, texts, image_token_lengths, seqlen,
    )
    return pixel_values, input_ids, label_ids


def hf_collate_fn(batch):
    pixel_values, input_ids, label_ids = zip(*batch)
    batch_pixel_values = list(pixel_values)
    batch_input_ids = torch.stack(input_ids, dim=0)
    batch_label_ids = torch.stack(label_ids, dim=0)
    return batch_pixel_values, batch_input_ids, batch_label_ids


def _resolve_dataloader_settings(num_workers, persistent_workers, prefetch_factor, max_inflight_batches):
    if num_workers <= 0:
        return 0, False, None, None

    effective_prefetch_factor = 1 if prefetch_factor is None else max(prefetch_factor, 1)
    effective_num_workers = min(
        num_workers,
        max(1, max(max_inflight_batches, 1) // effective_prefetch_factor),
    )
    return (
        effective_num_workers,
        persistent_workers and effective_num_workers > 0,
        effective_prefetch_factor,
        None,
    )


def create_hf_dataloader(
    dataset_path,
    processor_path,
    split='train',
    batch_size=1,
    shuffle=True,
    num_workers=0,
    num_proc=None,
    seqlen=4096,
    pin_memory=True,
    persistent_workers=False,
    prefetch_factor=None,
    max_inflight_batches=2,
):
    dataset = HFDataset(
        dataset_path=dataset_path,
        processor_path=processor_path,
        split=split,
        num_proc=num_proc,
        seqlen=seqlen,
    )
    effective_num_workers, effective_persistent_workers, effective_prefetch_factor, multiprocessing_context = _resolve_dataloader_settings(
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        max_inflight_batches=max_inflight_batches,
    )
    dataloader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": effective_num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": effective_persistent_workers,
        "collate_fn": hf_collate_fn,
        "worker_init_fn": _clear_processor_cache,
    }
    if effective_prefetch_factor is not None:
        dataloader_kwargs["prefetch_factor"] = effective_prefetch_factor
    if multiprocessing_context is not None:
        dataloader_kwargs["multiprocessing_context"] = multiprocessing_context
    return DataLoader(**dataloader_kwargs)


class HFDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_path,
        processor_path,
        train_split='train',
        val_split=None,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        num_proc=None,
        seqlen=4096,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=None,
        max_inflight_batches=2,
        streaming=False,
        shuffle_buffer=1000,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.processor_path = processor_path
        self.train_split = train_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.num_proc = num_proc
        self.seqlen = seqlen
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.max_inflight_batches = max_inflight_batches
        self.streaming = streaming
        self.shuffle_buffer = shuffle_buffer
        self.train_dataset = None
        self.val_dataset = None
        self.cache_dir = os.environ.get("HF_DATASETS_CACHE")

    @classmethod
    def from_args(cls, args):
        streaming = getattr(args, "data_type", "") == "chatimg"
        return cls(
            dataset_path=args.data_file,
            processor_path=args.processor_path,
            train_split=getattr(args, "sft_split", "train"),
            batch_size=args.micro_bsz,
            shuffle=bool(getattr(args, "data_shuffle", 1)),
            num_workers=args.num_workers,
            num_proc=getattr(args, "num_proc", None),
            seqlen=args.ctx_len,
            pin_memory=True,
            persistent_workers=getattr(args, "persistent_workers", False),
            prefetch_factor=getattr(args, "prefetch_factor", None),
            max_inflight_batches=getattr(args, "max_inflight_batches", 2),
            streaming=streaming,
        )

    def prepare_data(self):
        AutoProcessor.from_pretrained(
            self.processor_path,
            trust_remote_code=True,
            use_fast=True,
        )

    def setup(self, stage=None):
        if stage in (None, 'fit') and self.train_dataset is None:
            if self.streaming:
                self.train_dataset = HFStreamingDataset(
                    dataset_path=self.dataset_path,
                    processor_path=self.processor_path,
                    split=self.train_split,
                    seqlen=self.seqlen,
                    shuffle_buffer=self.shuffle_buffer,
                    cache_dir=self.cache_dir,
                )
            else:
                self.train_dataset = HFDataset(
                    dataset_path=self.dataset_path,
                    processor_path=self.processor_path,
                    split=self.train_split,
                    num_proc=self.num_proc,
                    seqlen=self.seqlen,
                    cache_dir=self.cache_dir,
                )

        if stage in (None, 'fit', 'validate') and self.val_split is not None and self.val_dataset is None:
            self.val_dataset = HFDataset(
                dataset_path=self.dataset_path,
                processor_path=self.processor_path,
                split=self.val_split,
                num_proc=self.num_proc,
                seqlen=self.seqlen,
                cache_dir=self.cache_dir,
            )

    def _build_dataloader(self, dataset, shuffle):
        effective_num_workers, effective_persistent_workers, effective_prefetch_factor, multiprocessing_context = _resolve_dataloader_settings(
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            max_inflight_batches=self.max_inflight_batches,
        )
        dataloader_kwargs = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": effective_num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": effective_persistent_workers,
            "collate_fn": hf_collate_fn,
            "worker_init_fn": _clear_processor_cache,
        }
        if effective_prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = effective_prefetch_factor
        if multiprocessing_context is not None:
            dataloader_kwargs["multiprocessing_context"] = multiprocessing_context
        return DataLoader(**dataloader_kwargs)

    def _build_streaming_dataloader(self, dataset):
        num_workers = self.num_workers
        kwargs = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "num_workers": num_workers,
            "pin_memory": self.pin_memory,
            "collate_fn": hf_collate_fn,
            "worker_init_fn": _clear_processor_cache,
        }
        if num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            kwargs["prefetch_factor"] = self.prefetch_factor or 2
        return DataLoader(**kwargs)

    def train_dataloader(self):
        if self.streaming:
            return self._build_streaming_dataloader(self.train_dataset)
        return self._build_dataloader(self.train_dataset, self.shuffle)

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return self._build_dataloader(self.val_dataset, False)

    def teardown(self, stage=None):
        if self.train_dataset is not None and hasattr(self.train_dataset, 'release'):
            self.train_dataset.release()
        if self.val_dataset is not None and hasattr(self.val_dataset, 'release'):
            self.val_dataset.release()

        if stage in (None, 'fit', 'validate', 'test', 'predict'):
            self.train_dataset = None
            self.val_dataset = None

        _PROCESSOR_CACHE.clear()


if __name__ == "__main__":
    dataset_path = "/mnt/sda1/xhs_caption_2/"
    processor_name = "/home/rwkv/molin/mod-rwkv/processor_bundle"
    # dataloader = create_hf_dataloader(
    #     dataset_path=dataset_path,
    #     processor_path=processor_name,
    #     batch_size=32,
    #     shuffle=True,
    #     num_workers=32,
    #     num_proc=24,
    #     seqlen=4096,
    # )

