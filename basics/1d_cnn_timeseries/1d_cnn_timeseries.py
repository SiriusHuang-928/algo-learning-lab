import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ---------------------- 1. Generate synthetic multi-class time series dataset ----------------------
def generate_signal_sample(seq_len=20):
    cls = np.random.randint(0,3)
    t = np.arange(seq_len)
    base_noise = np.random.normal(0, 0.1, seq_len)
    if cls == 0:
        # Class 0: pure gaussian noise
        sig = base_noise
    elif cls == 1:
        # Class 1: single spike pulse
        sig = base_noise
        spike_idx = np.random.randint(3, seq_len-3)
        sig[spike_idx] += 2.2
    else:
        # Class 2: periodic sine oscillation
        sig = base_noise + 0.4 * np.sin(2 * np.pi * t / 6)
    return sig, cls

def build_dataset(n_samples=10000, seq_len=20):
    X_list, y_list = [], []
    for _ in range(n_samples):
        sig, label = generate_signal_sample(seq_len)
        X_list.append(sig)
        y_list.append(label)
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    # Conv1d input shape: [batch, channel, sequence_length]
    X_tensor = torch.from_numpy(X).unsqueeze(1)
    y_tensor = torch.from_numpy(y)
    return X_tensor, y_tensor

# ---------------------- 2. 1D CNN Model Definition ----------------------
class TimeSeries1DCNN(nn.Module):
    def __init__(self, seq_len=20, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(32 * 5, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# ---------------------- 3. Training Pipeline ----------------------
if __name__ == "__main__":
    # Hyperparameters
    SEQ_LEN = 20
    N_SAMPLES = 10000
    EPOCHS = 40
    LR = 1e-3

    # Prepare data
    X_data, y_label = build_dataset(N_SAMPLES, SEQ_LEN)
    model = TimeSeries1DCNN(seq_len=SEQ_LEN, num_classes=3)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Train loop
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        pred_logits = model(X_data)
        loss = loss_func(pred_logits, y_label)

        loss.backward()
        optimizer.step()

        # Calculate accuracy
        with torch.no_grad():
            pred_idx = torch.argmax(pred_logits, dim=1)
            acc = (pred_idx == y_label).sum() / len(y_label)
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}, Acc: {acc.item():.4f}")

    # ---------------------- Inference Demo ----------------------
    print("\n==== Inference Demo ====")
    model.eval()
    with torch.no_grad():
        # Test case 0: pure noise
        test0 = np.random.normal(0,0.1,SEQ_LEN).astype(np.float32)
        test0_tensor = torch.from_numpy(test0).unsqueeze(0).unsqueeze(1)
        logit0 = model(test0_tensor)
        pred0 = torch.argmax(logit0, dim=1)
        print(f"Test 0 (pure noise), predicted class: {pred0.item()}")

        # Test case1: spike signal
        test1 = np.random.normal(0,0.1,SEQ_LEN).astype(np.float32)
        test1[10] +=2.2
        test1_tensor = torch.from_numpy(test1).unsqueeze(0).unsqueeze(1)
        logit1 = model(test1_tensor)
        pred1 = torch.argmax(logit1, dim=1)
        print(f"Test 1 (spike signal), predicted class: {pred1.item()}")

        # Test case2: sine oscillation
        t = np.arange(SEQ_LEN)
        test2 = (0.1*np.random.randn(SEQ_LEN) + 0.4 * np.sin(2 * np.pi * t /6)).astype(np.float32)
        test2_tensor = torch.from_numpy(test2).unsqueeze(0).unsqueeze(1)
        logit2 = model(test2_tensor)
        pred2 = torch.argmax(logit2, dim=1)
        print(f"Test 2 (sine wave), predicted class: {pred2.item()}")
