# CNN-BiLSTM Hybrid Time Series Classification
Hybrid model combining CNN local waveform feature extractor and Bidirectional LSTM temporal modeling, designed for simulated micro-electrode sensor signal multi-class classification.
Auto generate lightweight temporary training loss & accuracy curve after training, support single & batch waveform inference accuracy test.

## Core Functions
1. Conv1d layers extract local spike/peak features from raw 1D sensor signal
2. BiLSTM captures forward & backward long-range temporal dependencies of waveform
3. Record training history, lightweight temporary curve visualization (no heavy high-DPI image saving)
4. Evaluate global test accuracy + independent per-class recognition accuracy
5. Batch inference demo: randomly generate 10 waveforms and calculate inference accuracy
6. Complete PyTorch standard training loop with gradient control, train/eval mode & gradient-free inference

## Dataset Introduction
3-class synthetic simulated sensor time series (random peak position distribution):
- Class 0: Pure low-amplitude Gaussian noise (normal baseline signal)
- Class 1: Single random-position sharp peak abnormal waveform
- Class 2: Two separated random double peaks abnormal waveform

## Model Input & Dimension Flow
Input tensor shape: `[batch, channel=1, sequence_length=20]`
1. Two Conv1d + MaxPool blocks compress sequence length from 20 → 5, output feature channel=32
2. Permute dimension to match LSTM input format: `[batch, seq_len=5, feature_dim=32]`
3. BiLSTM extracts bidirectional final hidden state, concatenate forward & backward feature vector
4. Fully connected layer maps combined feature to 3-class classification logits

## Output Artifacts After Running
1. Console real-time training log: epoch total loss, global test accuracy, per-class separate accuracy
2. Pop-up temporary loss & accuracy curve figure (no local file saved, fast rendering)
3. Batch inference log: true label vs predicted class of 10 random waveforms + final inference accuracy

## How to Run
```bash
python cnn_bilstm.py
