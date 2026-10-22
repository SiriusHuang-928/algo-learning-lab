# WaveMoE-Bench: Mixture-of-Experts Transformer Benchmark for 1D Waveform Classification

## Overview

WaveMoE-Bench is a full-pipeline, reproducible benchmark project for 1D time series / waveform classification, built upon Transformer architecture with sparse Mixture-of-Experts (MoE) feed-forward layers. It provides end-to-end support from synthetic data generation, multi-architecture comparative training, quantitative evaluation, model visualization, to industrial-grade ONNX export and deployment.

The project implements three FFN modes with unified interface: vanilla ReLU, modern SwiGLU, and sparse MoE with load balancing mechanism, enabling fair horizontal comparison of accuracy, parameter efficiency, and inference performance.

## Key Features

- ✅ **3 FFN modes with unified interface**: `relu`, `swiglu`, `moe` one-click switch
- ✅ **Sparse MoE layer** with top-k gating and entropy-based load balancing auxiliary loss
- ✅ Support both Pre-LN and Post-LN normalization strategies
- ✅ Learnable  token for sequence-level classification
- ✅ Multi-head self-attention with optional RoPE position embedding
- ✅ Full training pipeline with standard classification metrics and automatic visualization
- ✅ Real-time expert utilization monitoring for load balance analysis
- ✅ **ONNX export & deployment support** for dense models, with ONNX Runtime benchmark
- ✅ Fixed random seed for fully reproducible experiments
- ✅ Automatic CSV training log archiving

## Architecture

### MoE Layer Design

The core MoE module replaces monolithic dense feed-forward networks with a dynamically routed expert pool:

1. **Router Network**: Linear layer mapping token features to expert routing logits
2. **Top-K Gating**: Each token is dispatched to only the top-k most relevant experts (default k=2) for sparse computation
3. **Expert Pool**: Collection of independent SwiGLU feed-forward networks, each specializing in different feature patterns
4. **Weighted Aggregation**: Outputs from selected experts are weighted by renormalized routing probabilities
5. **Load Balancing**: Entropy-based auxiliary loss to prevent expert collapse and ensure uniform expert utilization

### Model Pipeline

1. Input projection layer for 1D scalar waveform embedding
2. Learnable classification token prepended to the input sequence
3. Stacked Transformer encoder layers with configurable FFN mode
4. Linear classification head built on the final  token output

## Requirements

- Python 3.8+
- PyTorch >= 2.0.0
- NumPy >= 1.21.0
- scikit-learn >= 1.0.0
- Matplotlib >= 3.5.0
- ONNX Runtime >= 1.14.0 (for deployment inference)

Install all dependencies:

```
pip install -r requirements.txt
```

## Quick Start

### 1. Train Default MoE Model

Default configuration: 4 experts, top-2 gating, Pre-LN, 20 epochs

```
python train.py
```

### 2. Run Baseline Experiments

Fair comparison with all other hyperparameters fixed:

```
# ReLU dense FFN baseline
python train.py --ffn_mode relu --weight_filename baseline_relu.pth

# SwiGLU dense FFN baseline
python train.py --ffn_mode swiglu --weight_filename baseline_swiglu.pth
```

### 3. Export Model to ONNX

Dense models support standard ONNX export for industrial deployment:

```
# Export SwiGLU baseline
python export_onnx.py --ffn_mode swiglu --weight_path ./saved_weights/baseline_swiglu.pth --output_path ./saved_weights/baseline_swiglu.onnx
```

### 4. ONNX Runtime Inference Benchmark

```
python onnx_infer.py --onnx_path ./saved_weights/baseline_swiglu.onnx
```

## Experimental Results

All experiments are conducted on unified configuration: seq_len=20, d_model=32, 2 encoder layers, 20 epochs, fixed random seed=42.

表格

| Model Mode | Total Parameters | Val Accuracy | Macro F1 | Total Training Time | PyTorch GPU Inference (per sample) | ONNX CPU Inference (per sample) | Expert Usage Std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ReLU FFN | 19395 (0.0194 M) | 99.60% | 0.9960 | 63.19 s | ~0.117 ms | - | - |
| SwiGLU FFN | 23619 (0.0236 M) | 99.70% | 0.9969 | 64.76 s | ~0.118 ms | 0.224 ms | - |
| MoE (4 experts, Top-2) | 36619 (0.0366 M) | 99.65% | 0.9961 | 201.74 s | ~0.372 ms | Not supported natively | ~0.006 |

### Key Conclusions

1. SwiGLU activation achieves higher accuracy with slightly more parameters than vanilla ReLU, confirming the advantage of gated activation functions.
2. MoE model expands total parameter capacity by ~55% compared to SwiGLU baseline, with uniform expert utilization (std ~0.006). The load balancing mechanism effectively prevents expert collapse.
3. On this simple waveform classification task with high class separability, all three models achieve over 99.6% accuracy. The capacity advantage of MoE will be more significant on more complex tasks with strong noise, multiple categories, or small samples.
4. Dense models support standard ONNX export and can be directly deployed in industrial environments; the sparse MoE implementation in this project is designed for algorithm research and principle verification.

## Project Structure

```
WaveMoE_Bench/
├── models/
│   ├── __init__.py
│   ├── layers.py          # Core components: Attention, FFN, MoE, Encoder layer, RoPE
│   └── classifier.py     # Complete Transformer classifier model
├── utils/
│   ├── __init__.py
│   ├── metrics.py        # Classification metrics calculation & training log saving
│   └── visualize.py      # Learning curves, confusion matrix, attention visualization
├── dataset.py            # Synthetic waveform dataset generation
├── train.py              # Training & validation pipeline with CLI config
├── infer.py              # PyTorch native batch inference demo
├── export_onnx.py        # ONNX model export script
├── onnx_infer.py         # ONNX Runtime inference benchmark
├── saved_weights/        # Model checkpoints & exported ONNX files
├── output_figures/       # Visualization results & training logs
├── requirements.txt
└── README.md
```

## Deployment Notes

1. **Dense FFN modes (ReLU / SwiGLU)**: Support standard ONNX format export, compatible with all mainstream inference engines (ONNX Runtime, TensorRT, OpenVINO), and can be directly used for production deployment.
2. **Sparse MoE mode**: This project adopts dynamic routing sparse computation implementation for algorithm research. Native PyTorch ONNX export does not support dynamic sparse routing static graph.
3. **Industrial MoE deployment**: Production-level sparse MoE inference usually relies on dedicated large model inference frameworks such as vLLM, TensorRT-LLM, and Megatron-LM, which natively support expert routing, expert parallelism and sparse computation scheduling.

## License

MIT License
