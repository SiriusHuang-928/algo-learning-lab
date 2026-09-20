# CNN-LSTM Time Series Classification
This project implements a hybrid CNN-LSTM model for 1D time-series classification.
CNN extracts local waveform features, then LSTM learns sequential temporal dependencies.

## Project Overview
- Task: Synthetic 1D waveform classification (3 types of signals)
- Model: Conv1d layers → MaxPool1d → LSTM → Linear classifier
- Dataset: Automatically generated synthetic time series samples
- Train/Test split: 80% / 20%
