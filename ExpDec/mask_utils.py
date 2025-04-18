import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, n_head=4, dim_feedforward=128):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

    def forward(self, x):
        return self.transformer(x)

class MaskGenerator(nn.Module):
    def __init__(self, input_dim, seq_len, d_model=64):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.transformer = TransformerBlock(d_model)
        self.mask_fc = nn.Linear(d_model, input_dim)  # <- 변경: D차원 마스크 출력

    def forward(self, x):  # x: (B, T, D)
        h = self.proj(x)  # (B, T, d_model)
        z = self.transformer(h)
        mask = torch.sigmoid(self.mask_fc(z))  # (B, T, D), force (0,1)
        #mask = (mask > 0.8).float()
        return mask