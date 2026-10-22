import torch
import torch.nn as nn
from .layers import TransformerEncoderLayer

class CustomTransformerClassifier(nn.Module):
    """
    Custom Transformer classifier for 1D time series tasks.
    Uses learnable CLS token for classification, returns both logits and per-layer attention weights.
    Fully GPU-compatible, all parameters and buffers migrate with model.to(device).

    Args:
        seq_len: Length of input time series.
        d_model: Transformer embedding dimension.
        n_heads: Number of attention heads.
        num_layers: Number of stacked encoder layers.
        dim_feedforward: Hidden dimension of feed-forward network.
        num_classes: Number of output classes.
        dropout: Dropout probability.
    """
    def __init__(
        self,
        seq_len: int = 20,
        d_model: int = 32,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        num_classes: int = 3,
        dropout: float = 0.1,
        norm_mode: str = "pre",
        ffn_mode: str = "swiglu",
        window_size: int = None,
        num_experts: int = 4,
        expert_dim: int = 32,
        top_k: int = 2
    ):
        super().__init__()

        self.d_model = d_model

        # Input projection: map 1D scalar value to d_model dimension
        self.embedding = nn.Linear(1, d_model)

        # Learnable classification token, prepended to sequence (auto-migrate with model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))


        # Stacked Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, n_heads, dim_feedforward,
                dropout, norm_mode, ffn_mode, window_size,
                # ===== Added: pass MoE params to each encoder layer =====
                num_experts=num_experts,
                expert_dim=expert_dim,
                top_k=top_k
            )
            for _ in range(num_layers)
        ])

        # Final classification head
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Raw time series input, shape [batch_size, seq_len]
        Returns:
            logits: Classification scores, shape [batch_size, num_classes]
            attention_weights: List of attention weights from each encoder layer
        """
        batch_size = x.shape[0]

        # Step 1: Input embedding
        x = x.unsqueeze(-1)  # [batch, seq_len] → [batch, seq_len, 1]
        x = self.embedding(x)  # → [batch, seq_len, d_model]

        # Step 2: Prepend CLS token to sequence
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # → [batch, 1+seq_len, d_model]

        # Step 3: Add positional encoding
        # x = self.pos_encoding(x)

        # Step 4: Forward through all encoder layers
        attention_weights = []

        # ===== Added: initialize total auxiliary loss =====
        total_aux_loss = torch.tensor(0.0, device=x.device)

        for layer in self.encoder_layers:
            # ===== Modified: unpack 3 return values (added aux_loss) =====
            x, attn, aux_loss = layer(x)
            attention_weights.append(attn)
            total_aux_loss += aux_loss

        # Step 5: Use CLS token feature for classification
        cls_feature = x[:, 0, :]
        logits = self.classifier(cls_feature) #[batch_size,1,3]

        # ===== Modified: add total_aux_loss to return values =====
        return logits, attention_weights, total_aux_loss