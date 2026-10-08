# Time Series Waveform Classification Transformer Project

## Project Overview

This project implements a Transformer-based time series classification model from scratch using PyTorch. On top of the foundational Transformer architecture, it integrates multiple advanced optimization techniques widely adopted in industry and state-of-the-art large language models. Built around a waveform classification task, the project focuses on learning the design principles, engineering implementations, and trade-off logic of core Transformer components, rather than purely pursuing accuracy metrics.

## Core Advanced Technical Features

Four core optimization techniques are implemented on the baseline Transformer, with a pluggable implementation of sliding window sparse attention reserved. All techniques are standard industrial-grade Transformer solutions. While accuracy improvement is limited in this short-sequence simple task, they carry significant value for knowledge learning and engineering practice.

### 1. Pre-LayerNorm

- **Technical Description**: Places LayerNorm on the input side of each sub-layer (self-attention, feed-forward network), with no normalization after the residual connection. This differs from the post-layer normalization (Post-LN) in the original Transformer.
- **Core Advantages**
  - Improved training stability, effectively mitigating gradient vanishing in deep Transformers
  - Reduced model sensitivity to learning rate, enabling faster convergence
- **Learning Value**: Master the design logic of normalization placement in Transformers, understand the training differences and applicable scenarios of Pre-LN vs Post-LN, and build foundational knowledge of modern Transformer architectures.
- **Task Performance**: Accuracy improvement is not significant in this shallow, short-sequence scenario, but this architecture has become the standard design paradigm for modern Transformers.

### 2. Rotary Position Embedding (RoPE)

- **Technical Description**: Injects relative position information by applying rotary transformations to Query and Key feature vectors, instead of directly adding position embeddings to input features. It natively supports length extrapolation.
- **Core Advantages**
  - Relative position encoding property, better aligned with the distance-dependent nature of time series tasks
  - Supports inference extrapolation beyond the training sequence length after training
  - Less interference with attention distribution compared to traditional sinusoidal position encoding
- **Learning Value**: Understand the core role of position encoding, master the implementation principle and code of RoPE — the standard position encoding scheme used in mainstream large models like LLaMA. It is a core knowledge point for large model fundamentals.
- **Task Performance**: The extrapolation advantage of RoPE is not fully demonstrated in this short-sequence task, but its design philosophy is of high learning value as the standard position encoding solution for current large models.

### 3. SwiGLU Gated Feed-Forward Network

- **Technical Description**: Replaces the single-branch structure of the original FFN ("linear up-projection + ReLU activation + linear down-projection") with a dual-branch gated structure: the gate branch uses SiLU activation to generate weight coefficients between 0 and 1, which are element-wise multiplied with the value branch before down-projection.
- **Core Advantages**
  - Dynamic feature selection: the model can autonomously learn the importance of each feature channel, enhancing useful features and suppressing noise features
  - Smoother gradients: SiLU activation is differentiable everywhere, leading to more stable deep training compared to the hard truncation of ReLU
  - Stronger nonlinear expressiveness than the original ReLU-FFN under the same parameter count
- **Learning Value**: Master the application of gating mechanisms in MLP modules, understand the standard design paradigm of FFN layers in mainstream large models. It is an essential knowledge point for Transformer structure advancement.
- **Task Performance**: Accuracy improvement is moderate in this simple classification task, but gated FFN is the standard configuration of current industrial-grade Transformers.

### 4. Gated Multi-Head Attention

- **Technical Description**: Adds a gating branch to the attention output path, where the input features dynamically generate per-channel weight coefficients to adjust the attention-weighted output channel by channel.
- **Core Advantages**
  - Automatically focuses on valid attention information and suppresses attention noise from irrelevant positions
  - Forms a dual-gated structure ("attention gating + feature gating") with SwiGLU, improving feature selection capability across the entire pipeline
- **Learning Value**: Understand advanced optimization ideas for attention layers, master the implementation of gating mechanisms in attention modules, which can be transferred to time series tasks with complex noise scenarios.
- **Task Performance**: The gain from gating is not obvious in this task due to clean data and high feature distinguishability, but this technique is a fundamental idea for attention layer optimization.

### 5. Sliding Window Sparse Attention (Optional, Disabled in this project)

- **Technical Description**: Restricts each token to compute attention only with positions within a fixed window around itself. Positions outside the window are masked out, reducing attention computation complexity from O(n²) to O(n×w) where w is the window size.
- **Core Advantages**: Significantly reduces computation cost and memory usage in long-sequence scenarios, improving inference speed.
- **Learning Value**: Understand the design logic of sparse attention, master the engineering implementation of attention masking. It is one of the core techniques for long-sequence Transformer optimization.

## Trade-off Note on Sliding Window Attention

### Reason for Disabling Sliding Window in This Project

In this short-sequence waveform classification scenario (sequence length ≤ 30 steps), enabling sliding window (e.g., `window_size=7`) causes validation accuracy to plummet from ~99.6% to ~55%. Therefore, this feature is disabled by default. The core reasons are as follows:

1. **Severely insufficient receptive field**: The window size is far smaller than the total sequence length, directly cutting off global discriminative features such as waveform period, overall trend, and relative peak position. The model cannot obtain complete classification evidence.
2. **Low alignment with data characteristics**: This dataset is clean with no strong long-range noise; global information is all valid features. The core value of sliding window is filtering long-range noise, which does not exist in this task. Instead, it actively discards useful information.
3. **Amplified weakness in shallow models**: A 2-layer Transformer already has limited long-range modeling capability, and global attention is the core for integrating full-sequence information. Cutting off the window degrades the model into a local feature extractor, losing the sequence modeling advantage of Transformers.

### Applicable Scenarios for Sliding Window

- Long-sequence tasks (sequence length ≥ 100 steps) where the O(n²) computational cost of global attention becomes prohibitively high
- Tasks where core discriminative features are highly localized, such as pulse width detection and peak slope recognition
- Data with substantial long-range noise, requiring forced local focus

## Environment Dependencies

See `requirements.txt` for the complete dependency list. The core runtime environment is Python 3.8+ and PyTorch 2.0+.

## Usage Instructions

1. Install dependencies: `pip install -r requirements.txt`
2. Start training: run the training script to launch model training with the default optimal configuration (global attention, Pre-LN, RoPE, SwiGLU, gated attention).
3. Configuration modification: adjust parameters such as `window_size`, `ffn_mode`, and `norm_mode` during model initialization to verify the effects of different techniques.
