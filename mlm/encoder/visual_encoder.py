import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from diffusers import AutoencoderKL


class Patch(nn.Module):
    def __init__(self, Imgsize=64, Patchsize=16) -> None:
        super(Patch, self).__init__()
        self.Patchsize = Patchsize
        self.Imgsize = Imgsize
    def encoder(self, x):
        assert x.size(3)==self.Imgsize
        imgsize = self.Imgsize
        patchsize = self.Patchsize
        x = x.unfold(2, patchsize, patchsize).unfold(3, patchsize, patchsize)
        x = x.contiguous().reshape(x.size(0), x.size(1), int(pow(imgsize/patchsize, 2)), -1)
        x = x.transpose(1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
        return x

    def decoder(self, x):
        imgsize = self.Imgsize
        patchsize = self.Patchsize
        x = x.reshape(x.size(0), x.size(1), x.size(2)//patchsize, patchsize)
        x = x.transpose(1, 2)
        x = x.unfold(2,imgsize//patchsize,imgsize//patchsize).unfold(3, patchsize, patchsize)
        x = x.reshape(x.size(0), 4, imgsize, imgsize)
        return x

class SD_Auto():
    def __init__(self, path="sdxl", input_dtype=torch.float32) -> None:
        self.autoencoder = AutoencoderKL.from_pretrained(path, subfolder="vae")
        self.autoencoder = self.autoencoder.to('cuda', input_dtype)

    def encoder(self, x):
        with torch.no_grad():
            x = self.autoencoder.encode(x).latent_dist.sample()
        return x
    
    def decoder(self, x):

        with torch.no_grad():
            x = self.autoencoder.decode(x).sample
        #x = self.autoencoder.decode(x).sample
        return x
    
def kld_loss(mu, logvar):
    KLD = - 0.5 * torch.sum(1 + logvar - mu.pow(2) -
                            logvar.exp()) / mu.shape[0]
    return KLD


class VisualAdapter(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=512//8, patch_size=16, in_c=4, text_dim=2560, head_size=64):

        super().__init__()
        self.head_size = head_size
        # self.img_receptance = nn.Linear((patch_size*patch_size*in_c), text_dim, bias=False)
        # self.img_key = nn.Linear((patch_size*patch_size*in_c), text_dim, bias=False)
        # self.img_value = nn.Linear((patch_size*patch_size*in_c), text_dim, bias=False)
        self.linear = nn.Linear((patch_size*patch_size*in_c), text_dim, bias=False)
        self.patch = Patch(Imgsize=img_size, Patchsize=patch_size)

    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch.encoder(x)
        # r = self.img_receptance(x)
        # k = self.img_key(x)
        # v = self.img_value(x)
        # r = r.view(*x.shape[:2], -1, self.head_size).transpose(1, 2)
        # k = k.view(*x.shape[:2], -1, self.head_size).transpose(1, 2)
        # v = v.view(*x.shape[:2], -1, self.head_size).transpose(1, 2)
        # x_img = torch.nn.functional.scaled_dot_product_attention(
        #     r, k, v, is_causal=True, scale=1 / self.head_size
        # )
        # x = x_img.transpose(1, 2).reshape(*x.shape[:2], -1)
        return self.linear(x)



class VisualEncoder(nn.Module):
    def __init__(
        self,
        encoder_path,
        project_dim,
        train_mode="adapter",
        device="cuda",) -> None:
        super(VisualEncoder, self).__init__()

        self.model = AutoencoderKL.from_pretrained(path, subfolder="vae", allow_pickle=False).to('cuda', input_dtype)
        self.adapter = VisualAdapter(text_dim=llm_dim).to('cuda',dtype=torch.bfloat16)

    def forward(self, x):
        x = self.model.encode(x.unsqueeze(0)).latent_dist.sample()
        #print(x.view(-1))
        x = self.adapter(x)
        #print(x.view(-1))
        
        return x