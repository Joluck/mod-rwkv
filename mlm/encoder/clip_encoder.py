import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig



class VisualAdapter(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, encoder_dim, project_dim, hidden_dim=None):

        super().__init__()
        self.encoder_dim = encoder_dim
        self.project_dim = project_dim
        self.hidden_dim = hidden_dim

        if self.hidden_dim==None:
            self.hidden_dim = project_dim*2

        self.pre_norm = nn.LayerNorm(self.project_dim)
        self.proj = nn.Sequential(
            nn.Linear(self.encoder_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.project_dim),
        )
        # self.proj = nn.Sequential(
        #     nn.Linear(self.encoder_dim, self.hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(self.hidden_dim, self.hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(self.hidden_dim, self.project_dim),
        # )

    
    def forward(self, x):        
        x = self.proj(x)
        return x + self.pre_norm(x)



class ClipEncoder(nn.Module):
    
    def __init__(
        self,
        encoder_path,
        project_dim,
        train_mode="adapter",
        device="cuda",) -> None:
        super(ClipEncoder, self).__init__()

        self.device = device
        self.image_processor = CLIPImageProcessor.from_pretrained(encoder_path)
        self.model = CLIPVisionModel.from_pretrained(encoder_path)
        self.encoder_dim = self.model.config.hidden_size

        self.adapter = VisualAdapter(self.encoder_dim, project_dim)
    def forward(self, x):

        x= torch.from_numpy(self.image_processor(x)['pixel_values'][0]).to(self.device,dtype=torch.bfloat16)

        x = self.model(x.unsqueeze(0), output_hidden_states=True).last_hidden_state
        
        x = self.adapter(x)
        
        return x