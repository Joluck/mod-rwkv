import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np

from transformers import AutoProcessor, AutoModel



# class SpeechAdapter(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(SpeechAdapter, self).__init__()
#         self.conv = nn.Conv1d(in_channels=input_dim, out_channels=3072, kernel_size=3, stride=2)
#         self.transformer = nn.TransformerEncoderLayer(d_model=3072, nhead=8, dim_feedforward=4096)
#         self.linear = nn.Linear(3072, output_dim)
#     def forward(self, x):
#         # if x.size(1)<5 or x.size(1)>5000:
#         #     return False
#         # x shape: (batch_size, seq_len, input_dim)
#         x = x.permute(0, 2, 1)
#         # x shape: (batch_size, input_dim, seq_len)
#         x = self.conv(x)
#         # x shape after conv: (batch_size, input_dim, new_seq_len)
#         x = x.permute(2, 0, 1)  # Transformer expects (seq_len, batch_size, input_dim)
#         # x = self.transformer(x, src_key_padding_mask=mask.bool())
#         x = self.transformer(x)
#         x = x.permute(1, 0, 2)  # Back to (batch_size, seq_len, input_dim)
#         x = self.linear(x)
#         return x

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
        if x.size(1)>1023:
            return False
        return x

class SpeechEncoder(nn.Module):
    def __init__(
        self,
        encoder_path,
        project_dim,
        train_mode="adapter",
        device="cuda",
    ):
        assert train_mode in ["adapter", "full"]
        super(SpeechEncoder, self).__init__()

        self.device = device
        
        try:
            self.processor = AutoProcessor.from_pretrained(encoder_path)
        except:
            self.processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        
        self.time_reduction_factor = int(
            self.processor.feature_extractor.sampling_rate / 50
        )
        self.padding_length = 320
        
        self.model = AutoModel.from_pretrained(encoder_path)
        self.model.eval()
        self.model_output_dim = self.model.config.hidden_size
        self.project_dim = project_dim
            
        self.project_dim = project_dim
        self.adapter = SpeechAdapter(self.model_output_dim, self.project_dim).to(self.device,dtype=torch.bfloat16)
    #     self.set_gradient(train_mode)
        

   

    def forward(self, x):
        input_dict = self.processor(
            x, return_tensors="pt", padding=True, sampling_rate=16000
        ).to(self.device,dtype=torch.bfloat16)
        
        # encoder only
        x = self.model(**input_dict).last_hidden_state

        # stf encoder
        # x = self.model(**input_dict, output_hidden_states=True).hidden_states[-1]
        
        x= self.adapter(x)#x:(B,T,hidden dim)
        # mask = torch.ones(x.shape[0],x.shape[1]).to(self.device,dtype=torch.bfloat16)
        return x
