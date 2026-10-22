import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, save_dir="./output_figures"):
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Val Accuracy')
    ax2.set_title('Accuracy Curve')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "learning_curves.png"), dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, label_names, save_dir="./output_figures"):
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


def plot_attention_visualization(model, save_dir="./output_figures", seq_len=20):
    os.makedirs(save_dir, exist_ok=True)
    device = next(model.parameters()).device

    # Generate a sample input
    x_sample = torch.randn(1, seq_len).to(device)
    model.eval()

    with torch.no_grad():
        _, attn_weights, _ = model(x_sample)

    # Plot attention from first layer, first head
    fig, axes = plt.subplots(1, len(attn_weights), figsize=(5 * len(attn_weights), 4))
    if len(attn_weights) == 1:
        axes = [axes]

    for i, attn in enumerate(attn_weights):
        attn_np = attn.mean(dim=1)[0].cpu().numpy()
        im = axes[i].imshow(attn_np, cmap='viridis')
        axes[i].set_title(f'Encoder Layer {i+1} Attention')
        axes[i].set_xlabel('Key Position')
        axes[i].set_ylabel('Query Position')
        plt.colorbar(im, ax=axes[i])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "attention_map.png"), dpi=150)
    plt.close()
