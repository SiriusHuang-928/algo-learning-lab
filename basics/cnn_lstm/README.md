# CNN-LSTM Time Series Classification
This project implements a hybrid CNN-LSTM model for 1D time-series classification.
CNN extracts local waveform features, then LSTM learns sequential temporal dependencies.

## Project Overview
- Task: Synthetic 1D waveform classification (3 types of signals)
- Model: Conv1d layers → MaxPool1d → LSTM → Linear classifier
- Dataset: Auto-generated synthetic time series
- Train/Test split: 80% / 20%

## Requirements
torch>=2.0
numpy>=1.24

## Run Command
```bash
python cnn_lstm.py

## Core Features

1. Conv1d extracts local waveform features
2. MaxPool reduces sequence length
3. LSTM captures temporal sequential information
4. Complete training loop, test accuracy evaluation, single sample inference
