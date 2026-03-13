import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, SiglipImageProcessor



class VisualAdapter(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, config):

        super().__init__()
        self.encoder_dim = config.encoder_dim
        self.project_dim = config.project_dim
        self.hidden_dim = config.hidden_dim
        self.use_conv = config.use_conv
        if self.hidden_dim==None:
            self.hidden_dim = config.project_dim*4

        self.pre_norm = nn.LayerNorm(self.project_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.project_dim),
        )
        if self.use_conv:
            self.conv = nn.Conv1d(
                    in_channels=self.encoder_dim,
                    out_channels=self.encoder_dim,
                    bias=False,
                    kernel_size=3,
                    stride=2
            )

    
    def forward(self, x):
        if self.use_conv:
            x = self.conv(x.permute(0,2,1)).permute(0,2,1)
        x = self.mlp(x)
        return x + self.pre_norm(x)

