import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from dataset import generate_waveform, SEQ_LEN
from model import CustomTransformerClassifier

# ==================== Configuration ====================
# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3

# Model architecture
D_MODEL = 32
N_HEADS = 4
NUM_LAYERS = 2
FFN_DIM = 64
NUM_CLASSES = 3
DROPOUT = 0.1

# Dataset
TOTAL_SAMPLES = 10000
TRAIN_RATIO = 0.8

# Paths
SAVE_DIR = "./saved_weights"
FIGURE_DIR = "./output_figures"
WEIGHT_FILENAME = "custom_transformer_best.pth"

# Runtime device & GPU optimization
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Speed up fixed-size input computation
LABEL_NAMES = ["No Peak", "Single Peak", "Double Peak"]


# ==================== Dataset Construction ====================
def build_dataset(num_samples: int):
    """
    Generate synthetic waveform dataset and convert to PyTorch tensors.

    Args:
        num_samples: Total number of samples to generate.
    Returns:
        signals: Tensor of shape [num_samples, seq_len]
        labels: Tensor of shape [num_samples]
    """
    signals = []
    labels = []
    for _ in range(num_samples):
        sig, label = generate_waveform(SEQ_LEN)
        signals.append(sig)
        labels.append(label)

    signals = torch.tensor(np.array(signals), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    return signals, labels


# ==================== Training & Evaluation ====================
def train_one_epoch(model, dataloader, criterion, optimizer):
    """
    Train model for one epoch.

    Returns:
        avg_loss: Average loss per sample
        accuracy: Training accuracy
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        logits, _ = model(x_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y_batch).sum().item()
        total += x_batch.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, criterion):
    """
    Evaluate model on validation set.

    Returns:
        avg_loss: Average loss per sample
        accuracy: Validation accuracy
        all_labels: Ground truth labels
        all_preds: Predicted labels
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.to(DEVICE, non_blocking=True)

        logits, _ = model(x_batch)
        loss = criterion(logits, y_batch)

        total_loss += loss.item() * x_batch.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y_batch).sum().item()
        total += x_batch.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_labels), np.array(all_preds)


# ==================== Visualization ====================
def plot_learning_curves(train_losses, val_losses, train_accs, val_accs):
    """Plot training & validation loss and accuracy curves."""
    os.makedirs(FIGURE_DIR, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label="Train Loss", linewidth=1.5)
    ax1.plot(val_losses, label="Val Loss", linewidth=1.5)
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(train_accs, label="Train Accuracy", linewidth=1.5)
    ax2.plot(val_accs, label="Val Accuracy", linewidth=1.5)
    ax2.set_title("Accuracy Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "learning_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(labels, preds):
    """Plot confusion matrix heatmap."""
    os.makedirs(FIGURE_DIR, exist_ok=True)

    # Compute confusion matrix
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_attention_visualization(model):
    """
    Visualize CLS token attention weights overlaid on original waveform.
    Generates one sample per class, shows attention distribution of each head.
    """
    os.makedirs(FIGURE_DIR, exist_ok=True)
    model.eval()

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    for class_idx in range(NUM_CLASSES):
        # Generate a sample of target class
        while True:
            sig, label = generate_waveform(SEQ_LEN)
            if label == class_idx:
                break

        x_tensor = torch.tensor(sig, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            _, attn_weights = model(x_tensor)

        # Extract CLS attention from last encoder layer
        # Shape: [1, n_heads, 1+seq_len, 1+seq_len] → [n_heads, seq_len]
        last_layer_attn = attn_weights[-1].squeeze(0).cpu().numpy()
        cls_attn = last_layer_attn[:, 0, 1:]  # CLS position 0 attends to all time steps

        ax = axes[class_idx]
        ax.plot(sig, color="#1f77b4", linewidth=1.5, label="Waveform")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Class: {LABEL_NAMES[class_idx]} | CLS Attention per Head")

        # Plot attention weight for each head
        for h in range(N_HEADS):
            ax.plot(cls_attn[h], label=f"Head {h+1}", alpha=0.7, linestyle="--")

        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time Step")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_DIR, "cls_attention_visualization.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("Attention visualization saved to output_figures/cls_attention_visualization.png")


# ==================== Main Pipeline ====================
def main():
    # Create output directories
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    # Build dataset
    print("Building dataset...")
    signals, labels = build_dataset(TOTAL_SAMPLES)
    split_idx = int(TOTAL_SAMPLES * TRAIN_RATIO)

    train_set = TensorDataset(signals[:split_idx], labels[:split_idx])
    val_set = TensorDataset(signals[split_idx:], labels[split_idx:])

    # pin_memory=True speeds up CPU→GPU data transfer
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=torch.cuda.is_available(), num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        pin_memory=torch.cuda.is_available(), num_workers=0
    )

    # Initialize model, loss function and optimizer
    model = CustomTransformerClassifier(
        seq_len=SEQ_LEN,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        num_layers=NUM_LAYERS,
        dim_feedforward=FFN_DIM,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
        norm_mode="pre"
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print(f"\nStart training on {DEVICE}...")
    best_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_labels, val_preds = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch+1:2d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2%}"
        )

        # Save best checkpoint by validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, WEIGHT_FILENAME))
            print(f"  → New best model saved, accuracy: {best_acc:.2%}")

    print(f"\nTraining finished. Best validation accuracy: {best_acc:.2%}")

    # Generate all visualizations
    print("\nGenerating visualizations...")
    plot_learning_curves(train_losses, val_losses, train_accs, val_accs)
    plot_confusion_matrix(val_labels, val_preds)

    # Load best model for attention visualization
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, WEIGHT_FILENAME), map_location=DEVICE))
    plot_attention_visualization(model)

    print("All results saved to output_figures/")


if __name__ == "__main__":
    main()
