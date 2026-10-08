import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        # 门控分支：升维 + Swish 激活
        self.W_gate = nn.Linear(d_model, dim_feedforward)
        # 数值分支：纯升维，不激活
        self.W_up = nn.Linear(d_model, dim_feedforward)
        # 降维投影
        self.W_down = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 双分支分别升维
        gate = F.silu(self.W_gate(x)) # 门控值：0~1之间
        up = self.W_up(x)             # 数值特征
        # 门控按元素相乘，动态抑制/增强特征
        x = gate * up
        # dropout + 降维
        x = self.dropout(x)
        x = self.W_down(x) 
        return x

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer.
    Fixed non-trainable buffer, injects sequence order information into input embeddings.

    Args:
        d_model: Dimension of input feature embeddings.
        dropout: Dropout probability applied after encoding addition.
        max_len: Maximum supported sequence length.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute sinusoidal encoding in advance (CPU init, auto-migrate with model via register_buffer)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # add batch dimension: [1, max_len, d_model]

        # Register as buffer: follows model device automatically, not counted as trainable parameter
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings, shape [batch_size, seq_len, d_model]
        Returns:
            x + positional encoding, same shape as input
        """
        seq_len = x.size(1)
        # Slice positional encoding to match actual sequence length
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Vanilla multi-head self-attention implemented from scratch.
    Splits embedding dimensions across multiple heads, computes scaled dot-product attention in parallel.
    Fully GPU-compatible, all constants adapt to input tensor device.

    Args:
        d_model: Total dimension of input features.
        n_heads: Number of parallel attention heads.
        dropout: Dropout probability on attention weights.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, window_size = None):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # dimension per head
        self.scale = math.sqrt(self.d_k)

        # Independent linear projections for Query, Key, Value
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection after head concatenation
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

        # ===== 新增：注意力门控线性层 =====
        self.W_gate_attn = nn.Linear(d_model, d_model)

        # ===== 新增：滑动窗口配置 =====
        # window_size=None 表示不启用，退化为全局全注意力
        self.window_size = window_size

    # generate RoPE
    def _get_rope(self, seq_len, device):
        theta = 1.0 / (10000 **(torch.arange(0, self.d_k, 2, device=device).float() / self.d_k))
        pos_idx = torch.arange(seq_len, device=device)
        angles = torch.outer(pos_idx, theta)  # shape: [seq_len, d_k/2]

        cos_emb = angles.cos()
        sin_emb = angles.sin()
        return cos_emb, sin_emb

    def _apply_rope(self, x, cos_emb, sin_emb):
        """对 Q/K 张量施加 RoPE 旋转
        x shape: [batch, num_heads, seq_len, d_k]
        """
        # 拆分奇偶维度
        x1 = x[..., 0::2]  # 偶维度
        x2 = x[..., 1::2]  # 奇维度

        # 旋转公式：(a+bi)(cosθ+isinθ) = (a·cosθ - b·sinθ) + (a·sinθ + b·cosθ)i
        rx1 = x1 * cos_emb - x2 * sin_emb
        rx2 = x1 * sin_emb + x2 * cos_emb

        # 拼回原来的形状
        return torch.stack([rx1, rx2], dim=-1).flatten(-2)

    def _get_window_mask(self, seq_len, device):
        """生成滑动窗口掩码：True = 需要屏蔽的位置"""
        if self.window_size is None:
            return None  # 不启用窗口

        half_window = self.window_size // 2
        # 生成位置索引 0~seq_len-1
        pos = torch.arange(seq_len, device=device)
        # 计算两两位置差的绝对值 → shape: [seq_len, seq_len]
        dist = torch.abs(pos.unsqueeze(0) - pos.unsqueeze(1))
        # 距离超过半窗的位置，标记为屏蔽
        mask = dist > half_window
        return mask

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        """
        Args:
            query, key, value: Input feature tensors, shape [batch_size, seq_len, d_model]
        Returns:
            output: Attention output, shape [batch_size, seq_len, d_model]
            attn_weights: Attention weight matrix, shape [batch_size, n_heads, seq_len, seq_len]
        """


        batch_size = query.shape[0]

        # Step 1: Linear projection
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Step 2: Reshape to [batch, n_heads, seq_len, d_k]
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # ========== 新增：施加 RoPE 旋转 ==========
        seq_len_q = Q.size(2)
        seq_len_k = K.size(2)
        cos_emb, sin_emb = self._get_rope(seq_len_k, Q.device)

        # Q 和 K 分别旋转（自注意力里 seq_len_q = seq_len_k）
        Q = self._apply_rope(Q, cos_emb[:seq_len_q], sin_emb[:seq_len_q])
        K = self._apply_rope(K, cos_emb, sin_emb)
        # ==========================================

        # Step 3: Scaled dot-product attention (scale factor created on input device to avoid mismatch)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # ===== 新增：滑动窗口屏蔽 =====
        if self.window_size is not None:
            mask = self._get_window_mask(scores.size(-1), scores.device)
            # 屏蔽位置设为负无穷，softmax 后权重趋近于 0
            scores = scores.masked_fill(mask, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Step 4: Weighted sum of values
        output = torch.matmul(attn_weights, V)

        # Step 5: Concatenate heads and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # ===== 新增：门控控制 =====
        # 用输入 q（自注意力中就是层的输入 x）生成门控权重
        gate = torch.sigmoid(self.W_gate_attn(query))
        # 按元素相乘，动态调节每个通道的注意力输出强度
        output = output * gate

        output = self.W_o(output)

        return output, attn_weights


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer encoder layer with post-layer normalization.
    Architecture: Self-Attention → Residual → LayerNorm → FFN → Residual → LayerNorm

    Args:
        d_model: Input/output feature dimension.
        n_heads: Number of attention heads.
        dim_feedforward: Hidden dimension of feed-forward network.
        dropout: Dropout probability.
    """
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 dim_feedforward: int,
                 dropout: float = 0.1,
                 norm_mode: str = "pre",
                 ffn_mode: str = "swiglu",
                 window_size: int = None,
                 ):
        super().__init__()
        self.norm_mode = norm_mode

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout,window_size=window_size)
        self.norm1 = nn.LayerNorm(d_model)

        # Two-layer feed-forward network with ReLU activation
        if ffn_mode == "relu":
            self.ffn = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Linear(dim_feedforward, d_model)
            )
        elif ffn_mode == "swiglu":
            self.ffn = SwiGLUFFN(d_model, dim_feedforward)

        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input features, shape [batch_size, seq_len, d_model]
        Returns:
            x: Output features, same shape as input
            attn_weights: Attention weights from this layer
        """
        # y = LN( x + F(x) )
        if self.norm_mode == "post":
            attn_out, attn_weights = self.self_attn(x, x, x)
            x = x + self.dropout(attn_out)
            x = self.norm1(x)

            ffn_out = self.ffn(x)
            x = x + self.dropout(ffn_out)
            x = self.norm2(x)

        # y = x + F( LN(x) )
        elif self.norm_mode == "pre":
            residual = x
            x = self.norm1(x)
            attn_out, attn_weights = self.self_attn(x, x, x)
            x = residual + self.dropout(attn_out)

            residual = x
            x = self.norm2(x)
            ffn_out = self.ffn(x)
            x = residual + self.dropout(ffn_out)

        return x, attn_weights


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
    ):
        super().__init__()

        self.d_model = d_model

        # Input projection: map 1D scalar value to d_model dimension
        self.embedding = nn.Linear(1, d_model)

        # Learnable classification token, prepended to sequence (auto-migrate with model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding module
        # self.pos_encoding = PositionalEncoding(d_model, dropout)



        # Stacked Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout, norm_mode,ffn_mode,window_size)
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
        for layer in self.encoder_layers:
            x, attn = layer(x)
            attention_weights.append(attn)

        # Step 5: Use CLS token feature for classification
        cls_feature = x[:, 0, :]
        logits = self.classifier(cls_feature)

        return logits, attention_weights
