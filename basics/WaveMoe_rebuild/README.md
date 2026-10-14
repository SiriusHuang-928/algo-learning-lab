```
# WaveMoE: Mixture-of-Experts Transformer for Time Series Classification

## Overview
WaveMoE is a modular Transformer-based classifier built for 1D waveform / time series classification tasks. Built upon a standard dense Transformer architecture, this project replaces vanilla feed-forward networks with **sparse Mixture-of-Experts (MoE)** layers, achieving higher representational capacity with sparse activation and better parameter efficiency.

The project implements a complete end-to-end pipeline including data generation, model training, validation, evaluation, and visualization, with fully configurable hyperparameters and seamless mode switching.

## Key Features
- ✅ **Sparse Mixture-of-Experts (MoE)** feed-forward layers with top-k gating mechanism
- ✅ Entropy-based load balancing auxiliary loss to prevent expert collapse
- ✅ Seamless switching between 3 FFN modes: `relu`, `swiglu`, `moe`
- ✅ Support for both Pre-LN and Post-LN normalization strategies
- ✅ Learnable  token for sequence-level classification
- ✅ Multi-head self-attention with optional sliding window attention
- ✅ Full training & validation pipeline with standard classification metrics
- ✅ Built-in expert utilization tracking for load balance debugging
- ✅ Automatic visualization output: learning curves, confusion matrix, attention heatmap

## Architecture

### MoE Layer Design
The core MoE module replaces dense feed-forward networks with a dynamically routed expert pool:
1.  **Router Network**: A single linear layer maps each token's features to expert routing logits
2.  **Top-K Gating**: Each token is dispatched to only the top-k most relevant experts (default k=2) for sparse computation
3.  **Expert Pool**: A collection of independent SwiGLU feed-forward networks, each specializing in different feature patterns
4.  **Weighted Aggregation**: Outputs from selected experts are weighted by renormalized routing probabilities and summed
5.  **Load Balancing**: Entropy-based auxiliary loss encourages uniform expert utilization and mitigates the expert collapse problem

### Model Pipeline
- Input projection layer for 1D scalar time series embedding
- Learnable classification token prepended to the input sequence
- Stacked Transformer encoder layers with configurable FFN mode
- Linear classification head built on the final  token output

## Requirements
Core runtime dependencies:
- Python 3.8+
- PyTorch >= 2.0.0
- NumPy >= 1.21.0
- scikit-learn >= 1.0.0
- Matplotlib >= 3.5.0

See `requirements.txt` for the full dependency list.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training

Default configuration (MoE mode, 4 experts, top-2 gating):

```
python train.py
```

Run with custom configurations:

```
# Train with dense SwiGLU FFN (baseline)
python train.py --ffn_mode swiglu

# Train with 8 experts and top-3 gating
python train.py --ffn_mode moe --num_experts 8 --top_k 3

# Use Post-LN normalization
python train.py --norm_mode post
```

### 3. Run Batch Inference

```
python infer.py
```

## Project Structure

```
WaveMoE_rebuild/

├── models/

│   ├── __init__.py

│   ├── layers.py          # Core components: Attention, FFN, MoE, Encoder layer

│   └── classifier.py     # Full Transformer classifier model

├── utils/

│   ├── __init__.py

│   ├── metrics.py        # Classification metrics calculation

│   └── visualize.py      # Training curves, confusion matrix, attention visualization

├── dataset.py            # Synthetic waveform dataset generation

├── train.py              # Training and validation pipeline

├── infer.py              # Batch inference demo

├── requirements.txt      # Python dependencies

└── README.md
```

## Key Improvements Over Dense Transformer

This project introduces major architecture upgrades compared to a standard dense Transformer classifier:

1. **Sparse MoE Architecture**
   - Replaced monolithic dense feed-forward networks with dynamically routed sparse expert pools
   - Top-k activation reduces per-token compute overhead while maintaining total model capacity
   - Fully configurable expert count and top-k ratio to trade off performance and efficiency
2. **Load Balancing Mechanism**
   - Entropy-based auxiliary loss to enforce uniform expert workload distribution
   - Built-in `last_expert_usage` buffer via `register_buffer` for real-time expert load monitoring
   - Effectively mitigates the expert collapse problem common in MoE training
3. **Full Backward Compatibility**
   - One-click mode switching via `ffn_mode` parameter
   - All original Transformer features are fully preserved
   - Auxiliary loss automatically returns 0 in dense modes with no changes to the training pipeline
4. **Modular Engineering**
   - Clean layered architecture with separated model components, utility functions and training logic
   - Full command-line configuration via argparse, no hard-coded hyperparameters
   - Device-agnostic implementation: all tensors automatically align with the model device (CPU/CUDA)

## Outputs

After training completes, the following outputs will be generated automatically:

- Best model checkpoint saved to `saved_weights/`
- Training & validation loss/accuracy curves
- Confusion matrix on validation set
- Attention weight heatmap from the first encoder layer

All visual outputs are saved to `output_figures/` by default.

## License

MIT License
