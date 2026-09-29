import torch
import torch.nn as nn
import math

SEQ_LEN = 20
EMBED_DIM = 32
NUM_HEADS = 4
FFN_HIDDEN = 64
NUM_LAYERS = 2
NUM_CLASSES = 3

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, seq_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, seq_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(pos * div_term)
        pe[0, :, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, embed_dim]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerTimeSeries(nn.Module):
    def __init__(self):
        super().__init__()
        # Input projection: raw signal -> embedding
        self.embedding = nn.Linear(1, EMBED_DIM)
        self.pos_encoder = PositionalEncoding(EMBED_DIM, SEQ_LEN)
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=FFN_HIDDEN,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        # Classification head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(EMBED_DIM, NUM_CLASSES)

    def forward(self, x):
        # x shape: [batch, seq_len]
        batch_size, seq_len = x.shape
        x = x.unsqueeze(-1)  # [batch, seq_len, 1]
        x = self.embedding(x) # [batch, seq_len, embed_dim]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x) # [batch, seq_len, embed_dim]
        # Global average pooling
        x = x.transpose(1, 2) # [batch, embed_dim, seq_len]
        x = self.pool(x).squeeze(-1) # [batch, embed_dim]
        logits = self.classifier(x)
        return logits
