# Time Series MLP Classifier for Microelectrode QC
## Task
Binary classification for simulated microelectrode sensor waveform.
- Label 0: Normal baseline signal with small Gaussian noise
- Label 1: Abnormal signal with sharp spike artifact

## Model Architecture

Input(20) → Linear(20,32) → ReLU → Linear(32,8) → ReLU → Linear(8,1)

- Output raw logits, use `BCEWithLogitsLoss` for binary classification
- Optimizer: Adam

## Pipeline
1. Generate synthetic microelectrode waveform dataset
2. MLP forward propagation
3. Backward propagation, compute gradients and update weights
4. Evaluate classification accuracy on training set
5. Single-sample inference demo for new waveform

## How to Run
```bash
pip install torch numpy matplotlib
python timeseries_mlp_classifier.py

### Limitation

This is a simple MLP baseline.
MLP treats every sampling point as independent feature, cannot capture local waveform patterns.
If spike moves to another position, prediction performance may drop.
Next plan: implement 1D CNN for time-series signal.
