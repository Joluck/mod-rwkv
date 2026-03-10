import os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk, concatenate_datasets
from PIL import Image
import jsonlines
import librosa
from .utils import *


import PIL.PngImagePlugin
# 增加MAX_TEXT_CHUNK的大小，默认是1MB，可以设置为更大的值，例如10MB
PIL.PngImagePlugin.MAX_TEXT_CHUNK = 64 * 1024 * 1024
from .prepare.custom_transformers import get_image_processor
class WorldDataset(Dataset):
    def __init__(self, args, processor=None):
        """
        通用多模态数据集：
        支持 data_type = ['hf', 'img', 'arrow', 'jsonl', 'wav', 'state']
        """
        self.args = args
        self.processor = processor
        self.data_type = args.data_type
        self.hf_cache_dir = os.environ.get(
            "HF_DATASETS_CACHE",
            os.path.expanduser("~/.cache/huggingface/datasets"),
        )
        os.environ["HF_DATASETS_CACHE"] = self.hf_cache_dir

        # --- 1. 加载数据 ---
        if args.data_type == 'hf':
            self.data = self._load_hf_dataset(args.data_file)
        elif args.data_type == 'arrow':
            self.data = self._load_arrow_dataset(args.data_file)
        elif args.data_type in ['img', 'state']:
            self.data = self._load_vision_text(args.data_file)
            if hasattr(args, "copy"):
                self.data = self.data * args.copy
        elif args.data_type == 'jsonl':
            with jsonlines.open(args.data_file) as f:
                self.data = list(f)
        elif args.data_type == 'wav':
            with jsonlines.open(f'{args.data_file}/answer.jsonl') as f:
                self.data = list(f)
        elif args.data_type == 'autoimg':
            self.data = self._load_hf_dataset(args.data_file)
            self.image_processor = get_image_processor(768, 384, True)
        elif args.data_type == 'chatimg':
            self.data = self._load_hf_dataset(args.data_file)
            self.image_processor = get_image_processor(768, 384, True)
        else:
            raise ValueError(f"Unsupported data_type: {args.data_type}")
        data_nums = len(self.data)
        print(f"Loaded {len(self.data)} samples for {args.data_type} dataset.")
        if args.epoch_steps < data_nums and args.epoch_steps>0:
            if isinstance(self.data, list):
                self.data = self.data[:args.epoch_steps]
            else:
                self.data = self.data.select(range(args.epoch_steps))
            print(f"Trimmed to {len(self.data)} samples for epoch_steps {args.epoch_steps}.")
    # ------------------------------
    # 数据加载函数
    # ------------------------------

    def _load_hf_dataset(self, path):
        """加载 Hugging Face 格式数据"""
        subdirs = [
            os.path.join(path, d)
            for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))
        ]
        datasets = []
        for subdir in subdirs:
            try:
                ds = load_dataset(subdir, split="train", cache_dir=self.hf_cache_dir)
                datasets.append(ds)
            except Exception as e:
                print(f"⚠️ 跳过无效数据目录: {subdir}, 原因: {e}")

        if datasets:
            return concatenate_datasets(datasets)
        else:
            # 说明当前目录本身是dataset根目录
            return load_dataset(path, split="train", cache_dir=self.hf_cache_dir)

    def _load_arrow_dataset(self, path):
        """加载 Arrow 格式（支持多个子目录）"""
        subdirs = [
            os.path.join(path, d)
            for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))
        ]
        if subdirs:
            datasets = [load_from_disk(sd) for sd in subdirs]
            return concatenate_datasets(datasets)
        return load_from_disk(path)

    def _load_vision_text(self, path):
        """可根据项目自定义 load_vision_text"""
        # 假设格式 [{"image": "xxx.jpg", "conversations": [...]}, ...]
        return load_vision_text(path)

    def _normalize_images(self, sample, max_images=None):
        images = []

        if 'images' in sample and sample['images'] is not None:
            images = sample['images']
        elif 'image' in sample and sample['image'] is not None:
            single = sample['image']
            images = single if isinstance(single, list) else [single]

        if images is None:
            images = []
        if not isinstance(images, list):
            images = [images]

        images = [img for img in images if img is not None]

        if max_images is not None:
            images = images[:max_images]
        return images
 

    # ------------------------------
    # Dataset 必须方法
    # ------------------------------

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        while True:
            try:
                sample = self.data[idx]
                break
            except FileNotFoundError:
                idx = (idx + 1) % len(self.data)
        t = self.data_type

        if t == 'img':
            return self._process_img(sample)
        elif t == 'arrow':
            return self._process_arrow(sample)
        elif t == 'hf':
            return self._process_hf(sample)
        elif t == 'wav':
            return self._process_wav(sample)
        elif t == 'jsonl':
            return sample
        elif t == 'autoimg':
            return self._process_autoimg(sample)
        elif t == 'chatimg':
            return self._process_chatimg(sample)
        else:
            raise ValueError(f"Unsupported data_type in __getitem__: {t}")

    # ------------------------------
    # 各类型处理函数
    # ------------------------------

    def _process_img(self, sample):
        images = sample['image']
        if not isinstance(images, list):
            images = [images]
        images = [Image.open(os.path.join(self.args.data_file, "data", img)).convert("RGB") for img in images]

        texts = sample["conversations"]
        for i in range(len(images)):
                texts[0]["value"] = "<|placeholder|>" + texts[0]["value"]
        input_ids, label_ids = process_vision_text(texts, max_length=self.args.ctx_len, image_token_length=[576]*len(images))
        return  images, input_ids, label_ids

    def _process_arrow(self, sample):
        images = [img.convert("RGB") for img in sample["images"]][:3]  # ctx_len limit 3 image
        texts = convert_texts_to_conversations(sample["texts"])
        source = sample['source']
        while texts[0]["value"].startswith("<image>"):
            texts[0]["value"] = texts[0]["value"].replace("<image>", "", 1)
        for i in range(len(images)):
                texts[0]["value"] = "<|placeholder|>" + texts[0]["value"]
        input_ids, label_ids = process_vision_text(texts, max_length=self.args.ctx_len, image_token_length=[576]*len(images), source=source)
        return  images, input_ids, label_ids
    def _process_hf(self, sample):
        images = self._normalize_images(sample, max_images=3)
        images = [img.convert("RGB") for img in images]
        texts = convert_texts_to_conversations(sample["texts"])
        # texts = sample['conversations']
        while texts[0]["value"].startswith("<image>"):
            texts[0]["value"] = texts[0]["value"].replace("<image>", "", 1)
        for i in range(len(images)):
                texts[0]["value"] = "<|placeholder|>" + texts[0]["value"]
        input_ids, label_ids = process_vision_text(texts, max_length=self.args.ctx_len, image_token_length=[576]*len(images))
        images = images if images else None
        return  images, input_ids, label_ids
    def _process_autoimg(self, sample):
        images = self._normalize_images(sample, max_images=6)

        texts = convert_texts_to_conversations(sample["texts"])
        texts = placeholder_token(texts, len(images))
        image_token_length = []
        pixel_values = []

        for image in images:
            pixel_value,_ = self.image_processor(image.convert("RGB"))
            b,_,_,_ = pixel_value.shape
            image_token_length.append(b*115)
            pixel_values.append(pixel_value)
        pixel_values = torch.cat(pixel_values, dim=0) if pixel_values else None

        input_ids, label_ids = process_vision_text(texts, max_length=self.args.ctx_len, image_token_length=image_token_length)
        return pixel_values, input_ids, label_ids


    def _process_chatimg(self, sample):
        images = self._normalize_images(sample, max_images=6)

        # texts = convert_texts_to_conversations(sample["texts"])
        # 可能还在 conversations / messages ... 一类的地方
        texts = sample['conversations']
        texts = add_placeholder_token_to_chat(texts, len(images))
        image_token_length = []
        pixel_values = []

        for image in images:
            pixel_value,_ = self.image_processor(image.convert("RGB"))
            b,_,_,_ = pixel_value.shape
            image_token_length.append(b*115)
            pixel_values.append(pixel_value)
        pixel_values = torch.cat(pixel_values, dim=0) if pixel_values else None

        input_ids, label_ids = process_vision_text(texts, max_length=self.args.ctx_len, image_token_length=image_token_length)
        return pixel_values, input_ids, label_ids


    def _process_wav(self, sample):
        audio = librosa.load(sample["path"], sr=16000)[0]
        return {"audio": audio, "text": sample.get("text", "")}

def placeholder_token(texts, img_nums):
    while texts[0]["value"].startswith("<image>"):
        texts[0]["value"] = texts[0]["value"].replace("<image>", "", 1)
    for i in range(img_nums):
            texts[0]["value"] = "<|placeholder|>" + texts[0]["value"]
    return texts

def add_placeholder_token_to_chat(texts, img_nums):
    if "<image>" not in texts[0]["value"]:
        for i in range(img_nums):
            texts[0]["value"] = "<image>" + texts[0]["value"]
    texts[0]["value"] = texts[0]["value"].replace("<image>", "<|placeholder|>", img_nums)
    return texts

import lightning as L
from torch.utils.data import DataLoader

class WorldDataModule(L.LightningDataModule):
    def __init__(self, args, processor=None):
        super().__init__()
        self.args = args
        self.processor = processor

    def setup(self, stage=None):
        self.train_dataset = WorldDataset(self.args, self.processor)


    def train_dataloader(self):
        def custom_collate_fn(batch):
            signs, inputs_ids, labels = zip(*batch)
            all_images = list(signs)
            inputs_ids = torch.stack(inputs_ids, dim=0)
            labels = torch.stack(labels, dim=0)

            return all_images, inputs_ids, labels
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.micro_bsz,
            shuffle=True,    # Lightning 自动替换成 DistributedSampler
            collate_fn=custom_collate_fn,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
