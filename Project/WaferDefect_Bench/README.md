# Wafer Defect Classification
PyTorch implementation for wafer defect pattern classification, supporting multiple backbone models for comparison.

## Project Overview
This project is built for semiconductor wafer map defect classification. We implement and compare 4 representative vision models:
- `cnn_baseline`: Hand-crafted simple CNN baseline
- `resnet18`: ResNet-18
- `efficientnet_b0`: EfficientNet-B0
- `vit`: Vision Transformer (ViT)

All models accept single-channel gray-scale wafer map input. Training log, metric CSV and model weights will be saved automatically. ONNX export & inference speed test scripts are provided for deployment validation.

## Directory Structure

wafer-defect-classification/
├── dataset.py
├── train.py
├── export_onnx.py
├── onnx_infer.py
├── models/
│   ├── **init**.py
│   ├── cnn_baseline.py
│   ├── resnet.py
│   ├── efficientnet_b0.py
│   └── vit_baseline.py
├── utils/
│   ├── **init**.py
│   └── metrics.py
├── saved_weights/       # model checkpoint output
├── output_figures/      # curve/plot output
├── requirements.txt
└── README.md

## Environment
Python >=3.9, CUDA environment recommended for GPU acceleration.

## Install Dependencies
```
pip install -r requirements.txt

## Training Commands

```
# Train CNN baseline
python train.py --model cnn_baseline

# Train ResNet18
python train.py --model resnet18

# Train EfficientNet-B0
python train.py --model efficientnet_b0

# Train ViT
python train.py --model vit
```

Available arguments:

- `--model`: model selection, choices: `cnn_baseline`, `resnet18`, `efficientnet_b0`, `vit`
- `--epochs`: training epoch count
- `--batch_size`: batch size
- `--lr`: learning rate
- `--save_dir`: path to save `.pth` weights
- `--figure_dir`: path to save training figures

## ONNX Export & Inference

# Export trained pytorch weight to ONNX
python export_onnx.py

# Run ONNX Runtime inference benchmark
python onnx_infer.py

## Experiment Notes

- Training log and metrics will be saved as CSV for later comparison.
- ViT is sensitive to dataset scale; CNN-style models perform better on this small wafer dataset.
- Sparse MoE logic is not fully compatible with ONNX static graph, only dense models are exported.

## License

MIT

