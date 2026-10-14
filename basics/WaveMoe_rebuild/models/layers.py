import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

class MoELayer(nn.Module):
    """
    Mixture of Experts (MoE) Feed-Forward Layer.
    Routes each token to a subset of experts and aggregates weighted outputs.
    Includes load balancing auxiliary loss to prevent expert collapse.
    """

    def __init__(self, d_model, num_experts, expert_dim, top_k=2, dropout=0.1):
        """
        Args:
            d_model: Input feature dimension
            num_experts: Total number of expert networks
            expert_dim: Hidden dimension of each individual expert
            top_k: Number of experts activated per token
            dropout: Dropout rate applied inside each expert
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model

        # Router network: maps input to expert routing scores
        self.router = nn.Linear(d_model, num_experts)

        # Expert pool: each expert is an independent SwiGLU FFN
        self.experts = nn.ModuleList([
            SwiGLUFFN(d_model, expert_dim, dropout)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch, seq_len, d_model]
        Returns:
            output: MoE output of shape [batch, seq_len, d_model]
            aux_loss: Load balancing auxiliary loss (scalar)
        """
        batch_size, seq_len, _ = x.shape

        # Step 1: Compute routing probabilities for all experts
        router_logits = self.router(x)  # [batch, seq_len, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)

        # Save expert usage for debugging
        self.last_expert_usage = router_probs.mean(dim=(0, 1)).detach()

        # Step 2: Select top-k experts per token and their weights
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # Renormalize weights so that selected experts sum to 1
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # Step 3: Initialize output tensor
        output = torch.zeros_like(x)

        # Step 4: Dispatch tokens to corresponding experts and compute
        for expert_idx in range(self.num_experts):
            # Mask indicating which positions are assigned to this expert
            mask = (top_k_indices == expert_idx)
            if not mask.any():
                continue  # Skip expert if not activated in this batch

            # Gather input tokens assigned to this expert
            expert_input = x[mask.any(dim=-1)]  # [N, d_model], N = number of assigned tokens

            # Forward pass through the expert
            expert_output = self.experts[expert_idx](expert_input)

            # Get corresponding weights and apply weighting
            weight = top_k_weights[mask].unsqueeze(-1)  # [N, 1]
            # Write weighted output back
            output[mask.any(dim=-1)] += expert_output * weight

        # Step 5: Compute load balancing auxiliary loss
        # Average expert utilization across all tokens
        expert_usage = router_probs.mean(dim=(0, 1))  # [num_experts]
        # Entropy-based loss: encourages uniform expert utilization
        entropy = -torch.sum(expert_usage * torch.log(expert_usage + 1e-8))
        # Normalize: 0 = perfectly uniform, higher = more imbalanced
        max_entropy = torch.log(torch.tensor(self.num_experts, dtype=torch.float32, device=x.device))
        aux_loss = 1.0 - entropy / max_entropy

        return output, aux_loss

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
                 num_experts: int = 4,
                 expert_dim: int = 32,
                 top_k: int = 2
                 ):
        """
            Args:
                ffn_mode: "relu" / "swiglu" / "moe"
                num_experts: Number of experts (only used when ffn_mode="moe")
                expert_dim: Hidden dim per expert (only used when ffn_mode="moe")
                top_k: Activated experts per token (only used when ffn_mode="moe")
        """
        super().__init__()
        self.norm_mode = norm_mode
        self.ffn_mode = ffn_mode

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
        elif ffn_mode == "moe":
            self.ffn = MoELayer(
                d_model=d_model,
                num_experts=num_experts,
                expert_dim=expert_dim,
                top_k=top_k,
                dropout=dropout
            )

        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input features, shape [batch_size, seq_len, d_model]
        Returns:
            x: Output features, same shape as input
            attn_weights: Attention weights from this layer
            aux_loss: Load balancing auxiliary loss from MoE layer (0 if not in MoE mode)
        """

        # y = LN( x + F(x) )
        if self.norm_mode == "post":
            attn_out, attn_weights = self.self_attn(x, x, x)
            x = x + self.dropout(attn_out)
            x = self.norm1(x)

            # FFN branch: support both dense and MoE modes
            if self.ffn_mode == "moe":
                ffn_out, aux_loss = self.ffn(x)
            else:
                ffn_out = self.ffn(x)
                aux_loss = torch.tensor(0.0, device=x.device)

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

            # FFN branch: support both dense and MoE modes
            if self.ffn_mode == "moe":
                ffn_out, aux_loss = self.ffn(x)
            else:
                ffn_out = self.ffn(x)
                aux_loss = torch.tensor(0.0, device=x.device)

            x = residual + self.dropout(ffn_out)

        return x, attn_weights, aux_loss