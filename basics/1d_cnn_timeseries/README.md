# 1D-CNN Time Series Classification Demo
This demo implements a 1D convolutional neural network for time-series signal classification.
The dataset is synthetically generated, supports multi-class waveform recognition.

## Model Architecture
- Conv1d -> ReLU -> MaxPool1d
- Conv1d -> ReLU -> MaxPool1d
- Flatten -> Linear classifier

## Input shape
Input tensor: `[batch_size, in_channels, seq_len]`
For this task: `(N, 1, 20)`

## How to run
```bash
python 1d_cnn_timeseries.py

## Features

- Generate synthetic time-series samples
- Train 1D-CNN model
- Evaluate classification accuracy
- Single sample inference for new waveform
