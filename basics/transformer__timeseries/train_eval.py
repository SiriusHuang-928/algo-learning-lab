import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from dataset import build_dataset, SEQ_LEN, NUM_CLASSES
from model import TransformerTimeSeries

# Create auto folders
os.makedirs("./saved_weights", exist_ok=True)
os.makedirs("./output_figures", exist_ok=True)

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3

# Build dataset
X_tensor, y_tensor = build_dataset(total_samples=10000)
split_idx = int(0.8 * len(X_tensor))
X_train, X_test = X_tensor[:split_idx], X_tensor[split_idx:]
y_train, y_test = y_tensor[:split_idx], y_tensor[split_idx:]

# Model, loss, optimizer
model = TransformerTimeSeries()
loss_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Record history
train_loss_list = []
train_acc_list = []
best_acc = 0.0

def plot_loss_acc_curve(loss_rec, acc_rec):
    plt.rcParams["font.size"] = 10
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    ep = list(range(1, len(loss_rec)+1))
    ax1.plot(ep, loss_rec, c="r")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.grid(alpha=0.2)
    ax2.plot(ep, acc_rec, c="g")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test Accuracy")
    ax2.grid(alpha=0.2)
    plt.savefig("./output_figures/loss_acc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_confusion_mat(all_true, all_pred):
    cm = confusion_matrix(all_true, all_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Confusion Matrix")
    plt.savefig("./output_figures/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

# Train loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    shuffle_idx = torch.randperm(len(X_train))
    x_shuf, y_shuf = X_train[shuffle_idx], y_train[shuffle_idx]
    # Batch train
    for start in range(0, len(x_shuf), BATCH_SIZE):
        end = start + BATCH_SIZE
        x_batch = x_shuf[start:end]
        y_batch = y_shuf[start:end]

        optimizer.zero_grad()
        pred_logits = model(x_batch)
        loss = loss_criterion(pred_logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / (len(x_shuf) / BATCH_SIZE)
    train_loss_list.append(avg_loss)

    # Evaluate
    model.eval()
    all_pred = []
    all_true = []
    total_correct = 0
    with torch.no_grad():
        for start in range(0, len(X_test), BATCH_SIZE):
            end = start + BATCH_SIZE
            x_batch = X_test[start:end]
            y_batch = y_test[start:end]
            logits = model(x_batch)
            pred_idx = torch.argmax(logits, dim=1)
            total_correct += (pred_idx == y_batch).sum().item()
            all_pred.extend(pred_idx.numpy())
            all_true.extend(y_batch.numpy())
    test_acc = total_correct / len(X_test)
    train_acc_list.append(test_acc)

    # Per class accuracy
    class_acc = []
    for cls in range(NUM_CLASSES):
        cls_mask = np.array(all_true) == cls
        cls_correct = (np.array(all_pred)[cls_mask] == cls).sum()
        cls_total = cls_mask.sum()
        class_acc.append(cls_correct / cls_total if cls_total > 0 else 0)

    print(f"Epoch {epoch+1} | Loss:{avg_loss:.4f} | Acc:{test_acc:.4f}")
    print(f"Class Acc: {class_acc}")

    # Save best model
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "./saved_weights/transformer_best.pth")
        print(f"Best model saved, acc={best_acc:.4f}\n")

# Post training draw all figures
plot_loss_acc_curve(train_loss_list, train_acc_list)
plot_confusion_mat(all_true, all_pred)
print("Training complete, figures saved to output_figures/")
