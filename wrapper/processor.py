import copy

from transformers import BaseImageProcessor, PreTrainedTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack


CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '\x16' + ('Assistant' if message['role'] == 'assistant' else 'System' if message['role'] == 'system' else 'User') + ': ' }}"
    "{% if message['content'] is string %}"
    "{{ message['content'] }}"
    "{% else %}"
    "{% set ns = namespace(explicit_image_tags=0, image_items=0, text_parts=[]) %}"
    "{% for item in message['content'] %}"
    "{% if item['type'] == 'text' %}"
    "{% set ns.text_parts = ns.text_parts + [item['text']] %}"
    "{% set ns.explicit_image_tags = ns.explicit_image_tags + item['text'].count('<image>') %}"
    "{% elif item['type'] in ['image', 'image_url'] %}"
    "{% set ns.image_items = ns.image_items + 1 %}"
    "{% endif %}"
    "{% endfor %}"
    "{% for _ in range([ns.image_items - ns.explicit_image_tags, 0] | max) %}"
    "{{ '<image>' }}"
    "{% endfor %}"
    "{{ ns.text_parts | join('') }}"
    "{% endif %}"
    "{{ '\x17' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '\x16Assistant: ' }}{% endif %}"
)


class ModRWKVProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
        },
        "images_kwargs": {},
    }


class ModRWKVProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "RwkvTokenizer"
    user_image_tag = "<image>"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer = None,
        image_processor: BaseImageProcessor = None,
        chat_template=None,
    ):
        chat_template = CHAT_TEMPLATE if chat_template is None else chat_template
        super().__init__(tokenizer=tokenizer, image_processor=image_processor, chat_template=chat_template)

        self.image_token = "<|image_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.vision_start_token = (
            "<|vision_start|>" if not hasattr(tokenizer, "vision_start_token") else tokenizer.vision_start_token
        )
        self.vision_end_token = (
            "<|vision_end|>" if not hasattr(tokenizer, "vision_end_token") else tokenizer.vision_end_token
        )
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        self.vision_start_token_id = self.tokenizer.convert_tokens_to_ids(self.vision_start_token)
        self.vision_end_token_id = self.tokenizer.convert_tokens_to_ids(self.vision_end_token)
        self.vision_image_token = f"{self.vision_start_token}{self.image_token}{self.vision_end_token}"

    def to_dict(self):
        output = {}
        if self.image_processor is not None:
            output["image_processor"] = self.image_processor.to_dict()
        if getattr(self, "auto_map", None) is not None:
            output["auto_map"] = copy.deepcopy(self.auto_map)
        output["processor_class"] = self.__class__.__name__
        return output

    def _flatten_images(self, images):
        if images is None:
            return []
        if not isinstance(images, (list, tuple)):
            return [images]

        flat_images = []
        for item in images:
            if isinstance(item, (list, tuple)):
                flat_images.extend(self._flatten_images(item))
            else:
                flat_images.append(item)
        return flat_images

    def _get_num_images_per_text_sample(self, images, batch_size):
        if images is None:
            return [0] * batch_size
        if batch_size == 1:
            return [len(self._flatten_images(images))]
        if isinstance(images, (list, tuple)) and len(images) == batch_size:
            return [len(self._flatten_images(sample_images)) for sample_images in images]
        return None

    def _normalize_image_tags(self, text):
        return text.replace(self.user_image_tag, self.vision_image_token)

    def _append_missing_image_tags(self, text, num_missing_images):
        if num_missing_images <= 0:
            return text
        return text + self.vision_image_token * num_missing_images

    def _get_num_multimodal_tokens(self, image_grid_thw=None, **kwargs):
        vision_data = {}
        if image_grid_thw is not None:
            processor_defaults = getattr(self.image_processor, "_defaults", {})
            images_kwargs = dict(processor_defaults.get("images_kwargs", {}))
            images_kwargs.update(kwargs)
            merge_size = images_kwargs.get("merge_size", None) or self.image_processor.merge_size

            num_image_patches = [int(grid[0] * grid[1] * grid[2]) for grid in image_grid_thw]
            num_image_tokens = [num_patches // merge_size**2 for num_patches in num_image_patches]
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        return MultiModalData(**vision_data)

    def _count_token_occurrences(self, input_ids, token_id):
        counts = []
        for sample_ids in input_ids:
            counts.append(sum(1 for token in sample_ids if token == token_id))
        return counts

    def _validate_image_token_alignment(self, text_inputs, expected_image_tokens, expected_num_images):
        input_ids = text_inputs["input_ids"]
        actual_image_tokens = self._count_token_occurrences(input_ids, self.image_token_id)
        actual_vision_starts = self._count_token_occurrences(input_ids, self.vision_start_token_id)
        actual_vision_ends = self._count_token_occurrences(input_ids, self.vision_end_token_id)

        if actual_image_tokens != expected_image_tokens:
            raise ValueError(
                "Image token count does not match image_grid_thw-derived token count: "
                f"expected {expected_image_tokens}, got {actual_image_tokens}."
            )
        if actual_vision_starts != expected_num_images or actual_vision_ends != expected_num_images:
            raise ValueError(
                "Vision boundary token count does not match the number of image placeholders: "
                f"expected {expected_num_images}, got starts={actual_vision_starts}, ends={actual_vision_ends}."
            )



    def __call__(self, images=None, text=None, **kwargs: Unpack[ModRWKVProcessorKwargs]):
        output_kwargs = self._merge_kwargs(
            ModRWKVProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
            multimodal_tokens = self._get_num_multimodal_tokens(
                image_grid_thw=image_grid_thw,
                **output_kwargs["images_kwargs"],
            )
            num_image_tokens = multimodal_tokens.num_image_tokens
        else:
            image_inputs = {}
            image_grid_thw = None
            num_image_tokens = None

        if text is None:
            return BatchFeature(data=image_inputs)

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place
        expected_image_tokens = [0 for _ in text]
        expected_num_images = [0 for _ in text]
        if image_grid_thw is not None:
            num_images_per_sample = self._get_num_images_per_text_sample(images, len(text))
            index = 0
            for i in range(len(text)):
                text[i] = self._normalize_image_tags(text[i])
                if num_images_per_sample is not None:
                    missing_image_tags = num_images_per_sample[i] - text[i].count(self.image_token)
                    text[i] = self._append_missing_image_tags(text[i], missing_image_tags)
                while self.image_token in text[i]:
                    if index >= len(num_image_tokens):
                        # More <image> tags than actual images — strip the extras
                        text[i] = text[i].replace(self.vision_image_token, "")
                        break
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens[index], 1)
                    expected_image_tokens[i] += num_image_tokens[index]
                    expected_num_images[i] += 1
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

            if index != len(num_image_tokens):
                raise ValueError(
                    "Number of image placeholders in text does not match provided images: "
                    f"consumed {index}, available {len(num_image_tokens)}."
                )

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        if image_grid_thw is not None:
            self._validate_image_token_alignment(text_inputs, expected_image_tokens, expected_num_images)
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])
        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

ModRWKVProcessor.register_for_auto_class("AutoProcessor")


if __name__ == "__main__":
    from PIL import Image   
    from transformers import AutoProcessor
    from .tokenizer import RwkvTokenizer
    import os

    tokenizer = RwkvTokenizer(os.path.join(os.path.dirname(__file__), "wr_vocab_v20230424.txt"))
    img_processor = AutoProcessor.from_pretrained("/home/rwkv/models/Qwen3.5-0.8B").image_processor


    processor = ModRWKVProcessor(tokenizer=tokenizer, image_processor=img_processor)
    
    # Pure text test
    txt_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! How are you?"}, 
        {"role": "assistant", "content": "I'm good, thank you! How can I assist you today?"}
    ]

    
    # inputs = processor.apply_chat_template(txt_messages, tokenize=True, add_generation_prompt=True)
    # print(inputs)
    # outputs = processor.decode(inputs["input_ids"], skip_special_tokens = False)
    # print(outputs)

    # Image + text test without explicit image tag
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
    print(inputs.keys())
    outputs = processor.batch_decode(inputs["input_ids"], skip_special_tokens=False)
    print(outputs)


