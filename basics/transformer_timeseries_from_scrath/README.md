# Transformer Time Series Classification from Scratch

## Introduction

This project implements a complete Transformer encoder architecture from scratch using PyTorch, designed for 1D time series waveform classification tasks. All core components are built manually without relying on the built-in `nn.Transformer` module. The repository includes full pipelines for synthetic data generation, training, validation, and metric calculation, making it ideal for deep learning beginners to understand the underlying principles and engineering implementation of Transformers.

## Key Features

- **Pure manual implementation**: Multi-head self-attention, encoder layers, sinusoidal positional encoding, and CLS classification token are all implemented from scratch with clear, traceable logic.
- **Complete training pipeline**: Covers synthetic dataset generation, batch-wise training loop, validation evaluation, and multi-dimensional metric calculation.
- **Interpretability support**: Outputs attention weights from each layer, enabling visualization of the feature focus regions of the CLS token.
- **Lightweight and extensible**: Modular code structure with configurable parameters, can be quickly adapted to other time series classification tasks.

## Environment Requirements


| Dependency | Minimum Version | Description |
| --- | --- | --- |
| Python | 3.7+ | Recommended 3.8+ for development |
| PyTorch | 1.10.0+ | Core deep learning framework |
| NumPy | 1.19.0+ | Numerical computation and data generation |
| scikit-learn | 1.0.0+ | Optional, for confusion matrix and detailed classification metrics |

## Quick Start

### 1. Clone the repository

```
git clone <your-repository-url>
cd transformer-timeseries-from-scratch
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run basic tests

Verify the forward propagation and backpropagation functionality of the model:

```
python test.py
```

A successful run will confirm that the forward dimension test passes and parameter updates via backpropagation work properly.

### 4. Run training demo

Execute the complete training pipeline on the synthetic 3-class waveform classification task:

```
python train.py
```

The script will automatically generate three classes of waveform datasets, train for 30 epochs, and output training/validation loss and accuracy for each epoch.

## Project Structure

```
transformer-timeseries-from-scratch/
├── model.py          # Core model definitions: positional encoding, multi-head attention, encoder layer, classifier
├── dataset.py        # Synthetic waveform dataset generation (no peak / single peak / double peak)
├── train.py          # Full training and validation pipeline with metric calculation
├── test.py           # Unit tests for forward and backward propagation
├── requirements.txt  # Dependency list
└── README.md         # Project documentation
```

## Core Modules

All modules are defined in `model.py`:

1. **PositionalEncoding**
Sinusoidal positional encoding that pre-computes position features up to the maximum sequence length. It injects temporal position information into sequences to solve the permutation invariance limitation of self-attention.
2. **MultiHeadAttention**
Core implementation of multi-head self-attention, including linear projection, head splitting, scaled dot-product attention, head concatenation, and output projection.
3. **TransformerEncoderLayer**
Standard Post-LN (Post-LayerNorm) encoder layer, composed of self-attention sublayer + residual connection + layer norm + feed-forward network sublayer + residual connection + layer norm. Input and output dimensions are fully aligned for arbitrary stacking.
4. **CustomTransformerClassifier**
End-to-end classification model with a complete data flow: input embedding, CLS token concatenation, positional encoding, stacked encoder layers, and a CLS-based classification head.

## Training & Evaluation

### Task Description

Three classes of 1D waveforms with 20 time steps are synthesized:

- **Class 0**: Gaussian noise with no significant peak
- **Class 1**: Noise with one distinct peak superimposed
- **Class 2**: Noise with two distinct peaks superimposed

### Training Output

Output per epoch includes:

- Training set average loss and training accuracy
- Validation set average loss and validation accuracy

After training, a confusion matrix and classification report can be generated to evaluate per-class precision, recall, and F1-score in detail.

## Example Results

- After 30 training epochs, validation accuracy can reach 95% and above.
- The CLS token attention accurately aligns with waveform peak positions, and multiple attention heads exhibit clear division of labor.
- Two encoder layers achieve hierarchical feature abstraction: local peak detection at the first layer, and global count judgment at the second layer.

## Parameter Tuning

Core parameters can be modified at model initialization in `train.py`:

```
model = CustomTransformerClassifier(
    d_model=32,        # Feature dimension
    num_heads=4,       # Number of attention heads
    num_layers=2,      # Number of stacked encoder layers
    dim_feedforward=64 # Hidden dimension of the feed-forward network
)
```

- For simple tasks, 1~3 encoder layers are recommended; for complex tasks, depth can be increased to 6~12 layers.
- `dim_feedforward` is typically set to 2~4 times the value of `d_model`.
- The number of attention heads must evenly divide the feature dimension.

## License

MIT License
