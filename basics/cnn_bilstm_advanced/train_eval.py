import os
os.makedirs("./output_figures", exist_ok=True)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from dataset import build_dataset, SEQ_LEN, NUM_CLASSES
from model import CNNBiLSTM

# 超参
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
SAVE_WEIGHT_PATH = "./saved_weights/cnn_bilstm_best.pth"

def train_epoch(model, opt, loss_fn, x_train, y_train):
    model.train()
    total_loss = 0.0
    shuffle_idx = torch.randperm(len(x_train))
    x_shuf, y_shuf = x_train[shuffle_idx], y_train[shuffle_idx]
    for start in range(0, len(x_shuf), BATCH_SIZE):
        bx = x_shuf[start:start+BATCH_SIZE]
        by = y_shuf[start:start+BATCH_SIZE]
        opt.zero_grad()
        pred = model(bx)
        by = by.long()
        loss = loss_fn(pred, by)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss

def evaluate(model, x_test, y_test):
    model.eval()
    all_pred = []
    all_true = []
    with torch.no_grad():
        y_test = y_test.long()
        logits = model(x_test)
        preds = torch.argmax(logits, dim=1)
        all_pred.extend(preds.numpy())
        all_true.extend(y_test.numpy())
        total_correct = (preds == y_test).sum().item()
        overall_acc = total_correct / len(y_test)
        class_acc = []
        for cls in range(NUM_CLASSES):
            mask = (y_test == cls)
            cnt = torch.sum(mask).item()
            if cnt == 0:
                class_acc.append(0.0)
                continue
            correct = torch.sum(preds[mask]==cls).item()
            class_acc.append(correct/cnt)
    cm = confusion_matrix(all_true, all_pred)
    return overall_acc, class_acc, cm

def plot_temp_curve(loss_rec, acc_rec):
    plt.rcParams["font.size"] = 10
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,6))
    ep = list(range(1, len(loss_rec)+1))
    ax1.plot(ep, loss_rec, c="r")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.grid(alpha=0.2)
    ax2.plot(ep, acc_rec, c="g")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test Acc")
    ax2.grid(alpha=0.2)
    # 保存文件，不弹窗
    plt.savefig("./output_figures/loss_acc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_confusion_matrix(cm):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Confusion Matrix")
    # 保存文件
    plt.savefig("./output_figures/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_conv_feature(model, sample_x):
    model.eval()
    # 临时注册钩子
    handle = model.cnn[0].register_forward_hook(model.hook_conv1)
    with torch.no_grad():
        model(sample_x)
    feat = model.conv1_feature[0].numpy()
    fig, axes = plt.subplots(4,8, figsize=(14,6))
    axes = axes.flatten()
    for i in range(32):
        axes[i].plot(feat[i])
        axes[i].set_xticks([])
    plt.suptitle("Conv1 Layer 32 Channels Feature Map")
    # 保存文件
    plt.savefig("./output_figures/conv1_feature_map.png", dpi=150, bbox_inches="tight")
    plt.close()
    # 解绑钩子
    handle.remove()
    model.conv1_feature = None

if __name__ == "__main__":
    os.makedirs("./saved_weights", exist_ok=True)
    X_train, X_test, y_train, y_test = build_dataset()
    model = CNNBiLSTM()
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=LR)
    loss_history, acc_history = [], []
    best_acc = 0.0

    print("===== Start Advanced CNN-BiLSTM Training =====")
    for ep in range(EPOCHS):
        loss_val = train_epoch(model, opt, loss_fn, X_train, y_train)
        test_acc, per_cls_acc, cm = evaluate(model, X_test, y_test)
        loss_history.append(loss_val)
        acc_history.append(test_acc)
        print(f"Epoch {ep+1:2d} | Loss:{loss_val:.4f} | Acc:{test_acc:.4f}")
        print(f"Class Acc: {per_cls_acc}\n")
        # 保存最优权重
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), SAVE_WEIGHT_PATH)
            print(f"Best model saved, acc={best_acc:.4f}\n")

    # 绘图
    plot_temp_curve(loss_history, acc_history)
    plot_confusion_matrix(cm)
    # 卷积特征可视化
    single_sample = X_test[0:1]
    plot_conv_feature(model, single_sample)
