import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import WhisperProcessor, WhisperForConditionalGeneration

class SpeechAdapter(nn.Module):
    def __init__(self, encoder_dim, project_dim, hidden_dim=None):
        super(SpeechAdapter, self).__init__()
        self.encoder_dim = encoder_dim
        self.project_dim = project_dim
        self.hidden_dim = hidden_dim

        if self.hidden_dim==None:
            self.hidden_dim = project_dim*2
        self.conv = nn.Conv1d(in_channels=self.encoder_dim , out_channels=self.hidden_dim, kernel_size=3, stride=2, padding=2)
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.project_dim),
        )
    def forward(self, x):
        # if x.size(1)<5 or x.size(1)>5000:
        #     return False
        
        # x shape: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)
        # x shape: (batch_size, input_dim, seq_len)
        x = self.conv(x).permute(0, 2, 1)
        # x shape after conv: (batch_size, input_dim, new_seq_len)
        x = self.proj(x)
        return x



class WhisperEncoder(nn.Module):
    def __init__(
        self,
        encoder_path,
        project_dim,
        train_mode="adapter",
        device="cuda",
    ):
        assert train_mode in ["adapter", "full"]
        super(WhisperEncoder, self).__init__()
        self.device = device
        self.processor = WhisperProcessor.from_pretrained(encoder_path)

        self.model = WhisperForConditionalGeneration.from_pretrained(encoder_path).model.encoder

        self.model_output_dim = self.model.config.d_model
            
        self.project_dim = project_dim
        self.adapter = SpeechAdapter(self.model_output_dim, self.project_dim)

    def forward(self, x):
        input_dict = self.processor(
            x, return_tensors="pt", sampling_rate=16000, return_attention_mask=True
        ).to(self.device,dtype=torch.bfloat16)

        chunk = torch.sum(input_dict['attention_mask'], dim=-1)//2+1
        
        x = self.model(**input_dict).last_hidden_state
        x = x[:,:chunk,:]
        x= self.adapter(x)#x:(B,T,hidden dim)
        
        return x
