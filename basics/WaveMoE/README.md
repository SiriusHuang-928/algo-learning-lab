# WaveMoE: Mixture-of-Experts Transformer for Time Series Classification

## Overview
WaveMoE is an enhanced Transformer-based classifier optimized for 1D waveform / time series data. Built upon the previous custom dense Transformer architecture, this project introduces **Sparse Mixture-of-Experts (MoE)** layers to replace standard feed-forward networks, delivering higher representational capacity and better parameter efficiency through sparse activation.

## Key Features
- ✅ **Sparse Mixture-of-Experts (MoE)** feed-forward layers with top-k gating mechanism
- ✅ Entropy-based load balancing auxiliary loss to prevent expert collapse
- ✅ Seamless switching between 3 FFN modes: `relu`, `swiglu`, `moe`
- ✅ Pre-LN / Post-LN normalization mode selection
- ✅ Learnable CLS token for sequence-level classification
- ✅ Multi-head self-attention with optional sliding window support
- ✅ Complete training, validation and evaluation pipeline
- ✅ Built-in expert usage tracking for debugging and load analysis

## Architecture Details

### MoE Layer Core Design
The MoE module replaces the standard dense FFN layer with a dynamically routed expert pool:
1.  **Router Network**: A single linear layer maps each token's features to expert routing logits
2.  **Top-K Gating**: Each token is dispatched to only the top-k most relevant experts (default k=2)
3.  **Expert Pool**: A collection of independent SwiGLU feed-forward networks
4.  **Weighted Aggregation**: Outputs from selected experts are weighted by renormalized routing probabilities and summed
5.  **Load Balancing**: Entropy-based auxiliary loss encourages uniform expert utilization and avoids the expert collapse problem

### Full Model Pipeline
- Input projection layer for 1D scalar time series embedding
- Learnable classification token prepended to the input sequence
- Stacked Transformer encoder layers with configurable FFN mode
- Linear classification head built on the final CLS token output

## Requirements
See `requirements.txt` for the full dependency list. Core runtime requirements:
- Python 3.8+
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- scikit-learn >= 1.2.0
- tqdm >= 4.65.0
- Matplotlib >= 3.7.0

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Place your time series dataset in the `data/` directory. The model accepts raw input of shape `(batch_size, seq_len)`.

### 3. Run Training

```
python train.py
```

You can tune all hyperparameters (d_model, num_experts, top_k, ffn_mode, etc.) directly in the training script.

### 4. Run Evaluation

```
python eval.py
```

## Project Structure

```
WaveMoE/

├── model.py          # Core implementation: MoELayer, TransformerEncoderLayer, CustomTransformerClassifier

├── train.py          # Training and validation pipeline with loss tracking

├── eval.py           # Evaluation and inference script

├── dataset.py        # Data loading, preprocessing and augmentation

├── requirements.txt  # Python dependency list

└── README.md
```

## Key Improvements from Previous Version

This release introduces major architecture upgrades based on the previous dense Transformer classifier:

### 1. Full MoE Architecture Implementation

- Replaced dense feed-forward networks with sparse Mixture-of-Experts layers
- Top-k sparse activation: only a subset of experts runs per token, reducing per-token compute overhead
- Fully configurable expert count and top-k ratio to trade off performance and efficiency

### 2. Load Balancing Mechanism

- Added entropy-based auxiliary loss to enforce uniform expert workload distribution
- Built-in `last_expert_usage` buffer via `register_buffer` for real-time expert load monitoring
- Effectively mitigates the expert collapse problem common in MoE training

### 3. Full Backward Compatibility

- One-click mode switching via the `ffn_mode` parameter: `relu` / `swiglu` / `moe`
- All original features are fully preserved: Pre/Post-LN, sliding window attention, CLS token classification
- Auxiliary loss automatically returns 0 in dense modes, no changes needed to the existing training pipeline

### 4. Enhanced Training Pipeline

- Auxiliary loss is automatically aggregated across all encoder layers
- Configurable auxiliary loss weight to balance main task performance and load balancing
- Device-agnostic implementation: all tensors automatically align with the model device (CPU/CUDA)

## Benchmark Results

On standard time series classification tasks:

- Expert utilization stabilizes around ~1/num_experts (e.g. ~0.25 for 4 experts), verifying effective load balancing
- MoE mode achieves comparable or better classification accuracy than dense SwiGLU mode with higher total parameter capacity
- Sparse activation delivers better compute efficiency per parameter than equivalent dense architectures

## License

MIT License
