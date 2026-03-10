import torch
from torch import nn
from transformers import Qwen3_5VisionModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.processing_utils import Unpack

class Qwen3_5VLEncoder(nn.Module):
    def __init__(
        self,
        encoder_path) -> None:
        super(Qwen3_5VLEncoder, self).__init__()

        
        self.encoder = Qwen3_5VisionModel.from_pretrained(encoder_path)
        self.encoder_dim = self.encoder.config.out_hidden_size

    def forward(self, pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor):
        # vision_output: BaseModelOutputWithPooling = self.encoder(
        #     pixel_values, image_grid_thw
        # )
        # return vision_output.pooler_output
        return self.get_image_features(pixel_values, image_grid_thw)



    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
        ) -> BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input images.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        """

        vision_output: BaseModelOutputWithPooling = self.encoder(
            pixel_values, image_grid_thw
        ).pooler_output  
        # image_embeds = vision_output.pooler_output
        split_sizes = (image_grid_thw.prod(-1) // self.encoder.spatial_merge_size**2).tolist()
        image_embeds = torch.split(vision_output, split_sizes)
        # vision_output.pooler_output = image_embeds

        return image_embeds


if __name__ == "__main__":
    from transformers import AutoProcessor
    from PIL import Image   
    encoder_path = "/home/rwkv/molin/mod-rwkv/vision_bundle"
    processor_path = "/home/rwkv/molin/mod-rwkv/processor_bundle"
    encoder = Qwen3_5VLEncoder(encoder_path)
    
    image = Image.open("/home/rwkv/molin/mod-rwkv/demo.jpeg").convert("RGB")

    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)

    inputs = processor(images=[image], text="What is in the image?", return_tensors="pt")
    
    # output = encoder(inputs["pixel_values"], inputs["image_grid_thw"])
    features = encoder(inputs["pixel_values"], inputs["image_grid_thw"])
    print(image.size)
    # print("Output shape:", output.shape)
    print("Features shape:", [im.shape for im in features])  # Assuming batch_size=1, so we take the first element of the list
    print(encoder.encoder_dim)