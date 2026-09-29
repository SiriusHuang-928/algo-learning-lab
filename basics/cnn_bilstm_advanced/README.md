# Advanced CNN-BiLSTM Time Series Classification
Modular engineering project for simulated micro-electrode sensor waveform multi-class classification.
All folders are automatically created by code, clone & run directly without manual folder creation.

## Dataset Introduction
3 types synthetic sensor signal:
0: Normal noise baseline
1: Single sharp peak anomaly
2: Double separated peaks anomaly

## File Structure
cnn_bilstm_advanced/
├── dataset.py         Waveform data generation & train/test split

├── model.py           CNN-BiLSTM network + forward hook for conv feature capture

├── train_eval.py      Main training script, auto save best weight & figures

├── infer_demo.py      Offline inference script, load pretrained weight only

├── requirements.txt   All python dependencies

├── README.md

## Auto-generated folders after running train_eval.py

├── saved_weights/     Store best model weight cnn_bilstm_best.pth

└── output_figures/    Auto save 3 plotting results (no pop-up window)

## Environment Install
```bash
pip install -r requirements.txt

## Run Command
1. Train model, auto generate all figures & save optimal weight
python train_eval.py

After training finished:
- saved_weights/cnn_bilstm_best.pth : Best model parameter
- output_figures/ contains 3 pictures: loss curve, confusion matrix, conv1 feature map

2. Offline batch inference (NO retrain, only load saved weight)
python infer_demo.py

## Key Feature

1. Full modular split: dataset / model / train / inference separated
2. Auto create output folders, no manual creation needed
3. No pop-up plot window, all figures saved to local directory
4. Dynamic forward hook, only capture conv feature when visualize
5. Auto save best model based on test accuracy
6. Complete evaluation: overall accuracy, per-class accuracy, confusion matrix

## Running Result Preview

1. Confusion Matrix: Class 0 fully recognized, tiny confusion between class1 & class2
2. Conv1 Feature Map: 32 convolution channels capture waveform spike features
3. Training curve: Accuracy converge over 99% at epoch19
4. Inference demo: Random generated waveform prediction, 10 samples all correct


