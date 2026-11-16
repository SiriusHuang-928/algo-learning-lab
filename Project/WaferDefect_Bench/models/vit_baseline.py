import torch
import torch.nn as nn
import math

def sinusoidal_pos_embedding(num_patches, embed_dim):
    pe = torch.zeros(num_patches, embed_dim)
    position = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0) # [1, N, D]


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=4, embed_dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        B, C, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x


class ViT_Baseline(nn.Module):
    def __init__(self, img_size=24, patch_size=4, in_channels=1, num_classes=9, embed_dim=128, depth=4, num_heads=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        num_patches = (img_size // patch_size) ** 2
        pos_emb = sinusoidal_pos_embedding(num_patches, embed_dim)
        self.register_buffer("pos_embed", pos_emb)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=256,
            batch_first=True,
            dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        out = self.head(x)
        return out
