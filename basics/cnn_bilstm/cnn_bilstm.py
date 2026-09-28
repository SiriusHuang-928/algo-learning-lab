import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ===================== Hyper Parameters =====================
SEQ_LEN = 20
NUM_CLASSES = 3
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
CNN_OUT_CHANNEL = 32
LSTM_HIDDEN = 16

# ===================== Dataset Generator =====================
def generate_waveform(seq_len: int):
    label = np.random.randint(0, NUM_CLASSES)
    signal = np.random.normal(0, 0.1, seq_len).astype(np.float32)
    if label == 1:
        pos = np.random.randint(3, 17)
        signal[pos] += 0.7
    elif label == 2:
        pos1 = np.random.randint(2, 8)
        pos2 = np.random.randint(12, 18)
        signal[pos1] += 0.6
        signal[pos2] += 0.6
    return signal, label

# Build full dataset
total_samples = 10000
X_raw, y_raw = [], []
for _ in range(total_samples):
    sig, lab = generate_waveform(SEQ_LEN)
    X_raw.append(sig)
    y_raw.append(lab)

X_np = np.array(X_raw, dtype=np.float32)
y_np = np.array(y_raw, dtype=np.int64)

# CNN input shape: [N, channel=1, seq_len]
X_tensor = torch.from_numpy(X_np).unsqueeze(1)
y_tensor = torch.from_numpy(y_np)

# Train / Test split 8:2
split_idx = int(0.8 * total_samples)
X_train, X_test = X_tensor[:split_idx], X_tensor[split_idx:]
y_train, y_test = y_tensor[:split_idx], y_tensor[split_idx:]

# ===================== CNN-BiLSTM Hybrid Model =====================
class CNNBiLSTM(nn.Module):
    def __init__(self, cnn_out_ch, lstm_hidden, num_classes):
        super().__init__()
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=cnn_out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(cnn_out_ch, cnn_out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        # BiLSTM layer
        self.bilstm = nn.LSTM(
            input_size=cnn_out_ch,
            hidden_size=lstm_hidden,
            bidirectional=True,
            batch_first=True
        )
        # Classifier head
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x):
        # x: [batch, 1, seq_len=20]
        cnn_feat = self.cnn(x)          # [batch, 32, 5]
        cnn_feat = cnn_feat.permute(0, 2, 1) # [batch, seq=5, feature=32]
        lstm_out, (h_n, _) = self.bilstm(cnn_feat)
        concat_h = torch.cat((h_n[0], h_n[1]), dim=-1)
        logits = self.fc(concat_h)
        return logits

# ===================== Train & Evaluation Function =====================
def train_epoch(model, opt, loss_fn, x_train, y_train, batch_size):
    model.train()
    total_loss = 0.0
    shuffle_idx = torch.randperm(len(x_train))
    x_shuf, y_shuf = x_train[shuffle_idx], y_train[shuffle_idx]
    for start in range(0, len(x_shuf), batch_size):
        bx = x_shuf[start:start+batch_size]
        by = y_shuf[start:start+batch_size]
        opt.zero_grad()
        pred = model(bx)
        loss = loss_fn(pred, by)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss

def evaluate(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(x_test)
        preds = torch.argmax(logits, dim=1)
        total_correct = (preds == y_test).sum().item()
        overall_acc = total_correct / len(y_test)

        # Calculate per-class accuracy
        class_acc = []
        for cls in range(NUM_CLASSES):
            cls_mask = (y_test == cls)
            cls_total = torch.sum(cls_mask).item()
            if cls_total == 0:
                class_acc.append(0.0)
                continue
            cls_correct = torch.sum(preds[cls_mask] == cls).item()
            class_acc.append(cls_correct / cls_total)
    return overall_acc, class_acc

# ===================== Plot Training Curve =====================
def plot_curve_temp(loss_record, acc_record):
    import matplotlib.pyplot as plt
    # 极简配置，减少渲染耗时
    plt.rcParams["font.size"] = 10
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    epochs = list(range(1, len(loss_record)+1))

    # 损失曲线
    ax1.plot(epochs, loss_record, c="r")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.grid(alpha=0.2)

    # 准确率曲线
    ax2.plot(epochs, acc_record, c="g")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test Acc")
    ax2.grid(alpha=0.2)

    # 弹出临时窗口展示，不存文件
    plt.show()
    # 立刻销毁画布释放内存
    plt.close(fig)

# ===================== Main Training Process =====================
if __name__ == "__main__":
    model = CNNBiLSTM(CNN_OUT_CHANNEL, LSTM_HIDDEN, NUM_CLASSES)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Record history for plotting
    loss_history = []
    acc_history = []

    print("===== Start CNN-BiLSTM Hybrid Training =====")
    for epoch in range(EPOCHS):
        epoch_loss = train_epoch(model, optimizer, loss_func, X_train, y_train, BATCH_SIZE)
        test_acc, per_class_acc = evaluate(model, X_test, y_test)
        loss_history.append(epoch_loss)
        acc_history.append(test_acc)

        # Print log
        print(f"Epoch {epoch+1:2d} | Total Loss: {epoch_loss:.4f} | Overall Acc: {test_acc:.4f}")
        print(f"  Per-class Acc | Class0:{per_class_acc[0]:.4f} Class1:{per_class_acc[1]:.4f} Class2:{per_class_acc[2]:.4f}\n")

    # Save loss & acc curve figure
    plot_curve_temp(loss_history, acc_history)

    # test 10 random signals and calculate accuracy
    print("===== Inference Demo: Test 10 Random Waveforms =====")
    model.eval()
    with torch.no_grad():
        correct_count = 0
        total_test = 10
        for _ in range(total_test):
            test_sig, true_label = generate_waveform(SEQ_LEN)
            test_tensor = torch.from_numpy(test_sig).unsqueeze(0).unsqueeze(1)
            logit = model(test_tensor)
            pred_cls = torch.argmax(logit, dim=1).item()

            if pred_cls == true_label:
                correct_count += 1
            print(f"True label: {true_label}, Predicted class: {pred_cls}")

        acc = correct_count / total_test
        print(f"\n10 samples inference accuracy: {acc:.2f}")
