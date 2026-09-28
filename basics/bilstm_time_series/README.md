# BiLSTM Time Series Classifier
Bidirectional LSTM for synthetic 1D time-series multi-class classification.

## Project Overview
This project uses BiLSTM to classify three types of simulated sensor waveforms:
- Class 0: Low noise normal signal
- Class 1: Signal with sharp peak
- Class 2: Slow drifting signal

## Model Architecture
- BiLSTM (bidirectional, batch_first=True)
- Concatenate forward & backward final hidden state
- Fully connected layer for 3-class output

## How to Run
```bash
python bilstm.py

