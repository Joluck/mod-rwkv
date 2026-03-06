from transformers import BaseImageProcessor, PreTrainedTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack


CHAT_TEMPLATE = (
    "{{ '<|rwkv_tokenizer_end_of_text|>' }}"
    "{% for message in messages %}"
    "{{ '\x16' + message['role']|capitalize + ': ' }}"
    "{% if message['content'] is string %}"
    "{{ message['content'] }}"
    "{% else %}"
    "{% for item in message['content'] %}"
    "{% if item['type'] == 'text' %}"
    "{{ item['text'] }}"
    "{% elif item['type'] in ['image', 'image_url'] %}"
    "{{ '<|vision_start|><|image_pad|><|vision_end|>' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% endif %}"
    "{{ '\x17' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '\x16Assistant: <think></think>' }}{% endif %}"
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

    def _get_image_sizes(self, images):
        image_sizes = []
        for image in self._flatten_images(images):
            if hasattr(image, "size"):
                width, height = image.size
            elif hasattr(image, "shape") and len(image.shape) >= 2:
                height, width = image.shape[-2], image.shape[-1]
            else:
                raise TypeError(f"Unsupported image type for size extraction: {type(image)!r}")
            image_sizes.append([height, width])
        return image_sizes




    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.
        Args:
            image_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (height, width) per each image.
            video_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (num_frames, height, width) per each video.
        Returns:
            `MultiModalData`: A `MultiModalData` object holding number of tokens per each of the provided
            input modalities, along with other useful data.
        """

        vision_data = {}
        if image_sizes is not None:
            processor_defaults = getattr(self.image_processor, "_defaults", {})
            images_kwargs = processor_defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)
            merge_size = images_kwargs.get("merge_size", None) or self.image_processor.merge_size

            num_image_patches = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs)
                for image_size in image_sizes
            ]
            num_image_tokens = [(num_patches // merge_size**2) for num_patches in num_image_patches]
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})


        return MultiModalData(**vision_data)



    def __call__(self, images=None, text=None, **kwargs: Unpack[ModRWKVProcessorKwargs]):
        output_kwargs = self._merge_kwargs(
            ModRWKVProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            image_sizes = self._get_image_sizes(images)
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
            multimodal_tokens = self._get_num_multimodal_tokens(
                image_sizes=image_sizes,
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
        if image_grid_thw is not None:
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens[index], 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])
        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)




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

    # Image + text test

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": Image.open("/home/rwkv/molin/mod-rwkv/demo.jpeg").convert("RGB"),
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

    