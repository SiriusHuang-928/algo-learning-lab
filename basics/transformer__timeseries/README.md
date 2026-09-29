# Transformer Time Series Classification
1D time series waveform classification based on Transformer encoder for 3-class pulse peak signal recognition. Includes full implementation of sinusoidal positional encoding, multi-head self-attention, and standard training & evaluation pipeline.

## Project Overview
This project implements a lightweight Transformer model for time series classification, which identifies three types of 1D waveforms: **no peak, single peak, and double peak**. Built on standard Transformer Encoder architecture, it injects sequential order information via sinusoidal positional encoding and captures global temporal dependencies with multi-head self-attention. The repository includes complete training, evaluation and inference scripts, ready to use out of the box.

## Core Features
- Pure PyTorch implementation with native standard Transformer Encoder architecture
- Sinusoidal positional encoding that assigns unique position representation to each time step
- Multi-head self-attention mechanism that automatically focuses on key peak positions in waveforms
- Automatic train/test split, with best-performing weights saved based on test accuracy
- Auto-generated loss curve and confusion matrix, saved locally without popup windows
- Standalone offline inference script that runs batch prediction with pretrained weights

## Environment Requirements
See `requirements.txt` for full dependency list. Core requirements:
- Python 3.8+
- PyTorch >= 2.0.0
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

## Quick Start
### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_eval.py
```

The script will automatically:

- Generate 10,000 synthetic waveform samples, split into train/test sets with 8:2 ratio
- Train for 20 epochs, save the best checkpoint (by test accuracy) to `saved_weights/transformer_best.pth`
- Generate loss-accuracy curve and confusion matrix after training, saved to `output_figures/`

### 3. Offline inference

```bash
python infer_demo.py
```

Load pretrained weights, generate 10 random waveforms and print ground truth / predicted labels.

## Project Structure

```
transformer-timeseries-classification/

├── dataset.py          # Waveform generation and dataset construction

├── model.py            # Transformer classification model (with positional encoding)

├── train_eval.py       # Main script for training, evaluation and visualization

├── infer_demo.py       # Offline batch inference demo

├── requirements.txt    # Dependency list

├── README.md           # Project documentation

├── saved_weights/      # Checkpoint save directory (auto-generated)

└── output_figures/     # Visualization output directory (auto-generated)

```

## Key Learning Points

### 1. Positional Encoding

- Transformer has no built-in sequential order awareness, unlike CNN or RNN
- Sinusoidal positional encoding generates a unique vector for each time step, added to feature embeddings to inject position information
- Fixed constant buffer, not trainable, generalizable to input sequences of arbitrary length

### 2. Multi-Head Self-Attention

- Computes attention weights via three linear projections (Q, K, V), automatically learns correlations between positions
- Splits embedding dimensions across multiple heads, learning different patterns of temporal dependency (local / long-range) in parallel
- Global parallel computation, better at capturing long-range temporal dependencies compared to RNN

### 3. Transformer Encoder Architecture

- Standard encoder layer: multi-head attention + layer normalization + feed-forward network + residual connections
- Stacked layers extract higher-level temporal features hierarchically
- Input and output dimensions remain consistent, enabling residual connections and deep stacking

### 4. Adaptation for Time Series Classification

- Raw 1D time series is projected to embedding dimension via a linear layer
- Encoder outputs per-time-step features, aggregated into a global feature vector via global average pooling
- Final linear classification head outputs class logits

## Performance

On 10,000 synthetic samples (8:2 train/test split), the model achieves over 99% overall test accuracy. Confusion mainly occurs between single-peak and double-peak samples due to feature similarity, while pure noise samples are classified with 100% accuracy.
