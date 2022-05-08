import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange


class Model(nn.Module):
    def __init__(self, dim, num_classes, max_tok, dropout=0.):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8, activation='gelu', batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.fc = nn.Linear(dim, num_classes)
        
        self.cls_token = nn.Parameter(torch.zeros(1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(max_tok, dim))

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

        

    def forward(self, x):
        print(x.shape)
        b, s, e = x.shape 
        print(x.shape)
        x = torch.cat([self.cls_token.repeat(b, 1, 1), x], dim=1)
        x[:, 1:, :] = self.pos_embed[:s, :].repeat(b, 1, 1)

        x = self.transformer_encoder(x)
        
        x = x[:, 0, :]
        x = self.fc(x)
        
        return x
