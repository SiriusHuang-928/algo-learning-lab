import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------- Global Config --------------------------
SEQ_LEN = 20
NUM_CLASSES = 3
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
HIDDEN_DIM = 16  # LSTM hidden state dimension

# -------------------------- Synthetic Waveform Dataset Generator --------------------------
def generate_signal_sample(seq_len: int):
    """
    Generate 3 types of synthetic time-series waveform
    Class 0: Pure Gaussian noise (baseline signal)
    Class 1: Noise + single sharp peak
    Class 2: Noise + two separate peaks (contains sequential order feature)
    """
    label = np.random.randint(0, NUM_CLASSES)
    signal = np.random.normal(loc=0, scale=0.1, size=seq_len).astype(np.float32)
    if label == 1:
        pos = np.random.randint(3, 17)  # random position
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
    sig, lab = generate_signal_sample(SEQ_LEN)
    X_raw.append(sig)
    y_raw.append(lab)

# Convert to numpy array
X_np = np.array(X_raw, dtype=np.float32)
y_np = np.array(y_raw, dtype=np.int64)

# Convert to tensor & add channel dim for Conv1d: [N, seq_len] -> [N, channel, seq_len]
X_tensor = torch.from_numpy(X_np).unsqueeze(1)
y_tensor = torch.from_numpy(y_np)

# Train / Test split (8:2)
split_idx = int(0.8 * total_samples)
X_train, X_test = X_tensor[:split_idx], X_tensor[split_idx:]
y_train, y_test = y_tensor[:split_idx], y_tensor[split_idx:]

# -------------------------- Hybrid CNN-LSTM Model Definition --------------------------
class CNNLSTMClassifier(nn.Module):
    def __init__(self, num_classes, lstm_hidden):
        super().__init__()
        # 1D CNN backbone for local waveform feature extraction
        self.cnn_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # seq_len 20 -> 10
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)   # seq_len 10 -> 5
        )
        # LSTM layer to capture temporal dependency
        # batch_first=True: input shape [batch, seq_len, feature_dim]
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )
        # Final classification head
        self.classifier = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        # Step 1: CNN forward, output shape [batch, channel, seq_len] = [B,32,5]
        cnn_feat = self.cnn_extractor(x)
        # Step 2: Permute dimension to fit LSTM input: [B, channel, seq_len] -> [B, seq_len, channel]
        lstm_input = cnn_feat.permute(0, 2, 1)
        # Step 3: LSTM forward propagation
        lstm_out, (h_n, _) = self.lstm(lstm_input)
        # Only take last time step hidden state for classification, remove layer dim
        final_feat = h_n.squeeze(0)
        # Step 4: Linear classifier output logits
        logits = self.classifier(final_feat)
        return logits

# -------------------------- Pure CNN Baseline Model (for comparison) --------------------------
class PureCNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(32 * 5, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------- Initialize Model, Loss & Optimizer --------------------------
hybrid_model = CNNLSTMClassifier(NUM_CLASSES, HIDDEN_DIM)
cnn_baseline = PureCNNClassifier(NUM_CLASSES)
loss_func = nn.CrossEntropyLoss()
opt_hybrid = optim.Adam(hybrid_model.parameters(), lr=LR)
opt_cnn = optim.Adam(cnn_baseline.parameters(), lr=LR)

# -------------------------- Training & Evaluation Helper Function --------------------------
def train_one_epoch(model, optimizer, x_train, y_train, batch_size):
    model.train()
    total_loss = 0.0
    shuffle_idx = torch.randperm(len(x_train))
    x_shuf, y_shuf = x_train[shuffle_idx], y_train[shuffle_idx]
    for start in range(0, len(x_train), batch_size):
        batch_x = x_shuf[start:start+batch_size]
        batch_y = y_shuf[start:start+batch_size]
        optimizer.zero_grad()
        pred_logits = model(batch_x)
        loss = loss_func(pred_logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss

def evaluate_acc(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(x_test)
        pred = torch.argmax(logits, dim=1)
        correct = (pred == y_test).sum().item()
        acc = correct / len(y_test)
    return acc

# -------------------------- Main Training Loop --------------------------
print("===== Start Training: CNN-LSTM Hybrid vs Pure CNN Baseline =====")
for epoch in range(EPOCHS):
    # Train two models separately
    loss_hybrid = train_one_epoch(hybrid_model, opt_hybrid, X_train, y_train, BATCH_SIZE)
    loss_cnn = train_one_epoch(cnn_baseline, opt_cnn, X_train, y_train, BATCH_SIZE)

    # Evaluate test accuracy
    acc_hybrid = evaluate_acc(hybrid_model, X_test, y_test)
    acc_cnn = evaluate_acc(cnn_baseline, X_test, y_test)

    print(f"Epoch {epoch+1:2d} | Hybrid Loss: {loss_hybrid:.4f} | Hybrid Acc: {acc_hybrid:.4f} | CNN Baseline Acc: {acc_cnn:.4f}")

# -------------------------- Single Sample Inference Demo --------------------------
print("\n===== Single Waveform Inference Test =====")
hybrid_model.eval()
with torch.no_grad():
    # Create a double peak signal (class 2)
    test_wave = np.random.normal(0, 0.1, SEQ_LEN).astype(np.float32)
    test_wave[4] += 1.8
    test_wave[14] += 1.8
    # Reshape to match model input [1, channel, seq_len]
    test_tensor = torch.from_numpy(test_wave).unsqueeze(0).unsqueeze(1)
    pred_logit = hybrid_model(test_tensor)
    pred_class = torch.argmax(pred_logit, dim=1).item()
    print(f"Single test waveform predicted class: {pred_class}")
    print("Class definition: 0=noise, 1=single peak, 2=double peaks")
