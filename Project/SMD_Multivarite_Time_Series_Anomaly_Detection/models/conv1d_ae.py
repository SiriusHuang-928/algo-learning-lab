import torch
import torch.nn as nn

class Conv1DAE(nn.Module):
    def __init__(self, in_channels=38, window_len=96):
        super().__init__()
        self.in_channels = in_channels
        self.window_len = window_len

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x: [B, T, C] -> Conv1d输入格式 [B, C, T]
        x = x.transpose(1, 2)
        z = self.encoder(x)
        recon = self.decoder(z)
        recon = recon.transpose(1, 2)
        return recon
