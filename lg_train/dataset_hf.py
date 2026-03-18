'''
hf compatible dataset
'''
import os

import lightning as L
import torch
from datasets import load_dataset, load_dataset_builder
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoProcessor


os.environ.setdefault("HF_DATASETS_CACHE", "~/.cache/huggingface/datasets")


ROLE_TABLE = {
    "user": "user",
    "assistant": "assistant",
    "system": "system",
    "human": "user",
    "gpt": "assistant",
}

# Sources that contain <loc_*> grounding tags incompatible with our tokenizer.
_SKIP_SOURCES = frozenset({
    "SynthChartNet",
    "SynthFormulaNet",
    "SynthCodeNet",
    "DoclingMatix",
})


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


SYN_TOKEN_ID = 23   # \x16 — turn start
ETB_TOKEN_ID = 24   # \x17 — turn end
ASSISTANT_PREFIX_IDS = (23, 5585, 41693, 59)  # "\x16Assistant:"
_MAX_ASPECT_RATIO = 50


def _build_labels(input_ids, seqlen, pad_token_id=0):
    """Mask non-assistant turns using \x16/\x17 delimiters already in input_ids."""
    ids = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else list(input_ids)
    labels = [-100] * len(ids)
    prefix_len = len(ASSISTANT_PREFIX_IDS)

    start = None
    for i, tok in enumerate(ids):
        if tok == SYN_TOKEN_ID:
            start = i
        elif tok == ETB_TOKEN_ID and start is not None:
            if tuple(ids[start:start + prefix_len]) == ASSISTANT_PREFIX_IDS:
                for j in range(start, i + 1):
                    labels[j] = ids[j]
            start = None

    ids = ids[:seqlen + 1]
    labels = labels[:seqlen + 1]

    input_tensor = torch.tensor(ids, dtype=torch.long)
    label_tensor = torch.tensor(labels, dtype=torch.long)

    pad_length = max(seqlen + 1 - len(label_tensor), 0)
    if pad_length > 0:
        input_tensor = torch.nn.functional.pad(input_tensor, (0, pad_length), value=pad_token_id)
        label_tensor = torch.nn.functional.pad(label_tensor, (0, pad_length), value=-100)

    return input_tensor[:-1], label_tensor[1:]


class HFStreamingDataset(IterableDataset):
    """Streaming dataset that reads HF parquet files on-the-fly without caching."""

    def __init__(self, dataset_path, processor_path, split='train', seqlen=4096,
                 shuffle_buffer=512, seed=42, cache_dir=None):
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
            if sample.get("source", "") in _SKIP_SOURCES:
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
    for img in images:
        w, h = img.size
        if w == 0 or h == 0:
            raise ValueError("image has zero width or height")
        ratio = max(w / h, h / w)
        if ratio > _MAX_ASPECT_RATIO:
            raise ValueError(f"aspect ratio {ratio:.1f} exceeds limit {_MAX_ASPECT_RATIO}")
    messages = _build_messages(texts, images)
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    if images:
        model_inputs = processor(images=images, text=rendered, return_tensors="pt")
        pixel_values = {"pixel_values": model_inputs["pixel_values"]}
        if "image_grid_thw" in model_inputs:
            pixel_values["image_grid_thw"] = model_inputs["image_grid_thw"]
        input_ids_raw = model_inputs["input_ids"].squeeze(0)
    else:
        model_inputs = processor(text=rendered, return_tensors="pt")
        pixel_values = None
        input_ids_raw = model_inputs["input_ids"].squeeze(0)

    pad_token_id = processor.tokenizer.pad_token_id or 0
    input_ids, label_ids = _build_labels(input_ids_raw, seqlen, pad_token_id)
    return pixel_values, input_ids, label_ids


def hf_collate_fn(batch):
    pixel_values, input_ids, label_ids = zip(*batch)
    batch_pixel_values = list(pixel_values)
    batch_input_ids = torch.stack(input_ids, dim=0)
    batch_label_ids = torch.stack(label_ids, dim=0)
    return batch_pixel_values, batch_input_ids, batch_label_ids


class HFDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_path,
        processor_path,
        train_split='train',
        val_split=None,
        batch_size=1,
        num_workers=0,
        seqlen=4096,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=None,
        shuffle_buffer=1000,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.processor_path = processor_path
        self.train_split = train_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seqlen = seqlen
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.shuffle_buffer = shuffle_buffer
        self.train_dataset = None
        self.val_dataset = None
        self.cache_dir = os.environ.get("HF_DATASETS_CACHE")

    @classmethod
    def from_args(cls, args):
        return cls(
            dataset_path=args.data_file,
            processor_path=args.processor_path,
            train_split=getattr(args, "sft_split", "train"),
            batch_size=args.micro_bsz,
            num_workers=args.num_workers,
            seqlen=args.ctx_len,
            pin_memory=True,
            persistent_workers=getattr(args, "persistent_workers", False),
            prefetch_factor=getattr(args, "prefetch_factor", None),
        )

    def prepare_data(self):
        AutoProcessor.from_pretrained(
            self.processor_path,
            trust_remote_code=True,
            use_fast=True,
        )

    def setup(self, stage=None):
        if stage in (None, 'fit') and self.train_dataset is None:
            self.train_dataset = HFStreamingDataset(
                dataset_path=self.dataset_path,
                processor_path=self.processor_path,
                split=self.train_split,
                seqlen=self.seqlen,
                shuffle_buffer=self.shuffle_buffer,
                cache_dir=self.cache_dir,
            )
        if stage in (None, 'fit', 'validate') and self.val_split is not None and self.val_dataset is None:
            self.val_dataset = HFStreamingDataset(
                dataset_path=self.dataset_path,
                processor_path=self.processor_path,
                split=self.val_split,
                seqlen=self.seqlen,
                shuffle_buffer=0,
                cache_dir=self.cache_dir,
            )

    def _build_dataloader(self, dataset):
        kwargs = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "collate_fn": hf_collate_fn,
            "worker_init_fn": _clear_processor_cache,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            kwargs["prefetch_factor"] = max(self.prefetch_factor, 1) if self.prefetch_factor is not None else 2
        return DataLoader(**kwargs)

    def train_dataloader(self):
        return self._build_dataloader(self.train_dataset)

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return self._build_dataloader(self.val_dataset)

    def teardown(self, stage=None):
        self.train_dataset = None
        self.val_dataset = None
        _PROCESSOR_CACHE.clear()
