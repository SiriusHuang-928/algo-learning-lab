import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim)
        )

    def forward(self, x):
        # x: [B, T, D]
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = residual + attn_out

        residual = x
        x_norm = self.norm2(x)
        x = residual + self.mlp(x_norm)
        return x

class TemporalTransformer(nn.Module):
    def __init__(self, in_channels=38, window_len=96, embed_dim=64, num_heads=4, depth=2):
        super().__init__()
        self.in_channels = in_channels
        self.window_len = window_len
        self.embed_dim = embed_dim

        # 输入投影
        self.input_proj = nn.Linear(in_channels, embed_dim)
        # 可学习位置编码
        self.pos_emb = nn.Parameter(torch.randn(1, window_len, embed_dim))
        # transformer堆叠
        self.blocks = nn.Sequential(*[
            TemporalTransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        # 重建输出头
        self.recon_head = nn.Linear(embed_dim, in_channels)

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.shape
        x = self.input_proj(x)
        x = x + self.pos_emb
        x = self.blocks(x)
        recon = self.recon_head(x)
        return recon
