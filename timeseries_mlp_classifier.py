# Suppress Windows OpenMP duplicate lib warning
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# ---------------------- 1. MLP Model Definition ----------------------
class TimeSeriesMLP(nn.Module):
    def __init__(self, input_length: int = 20):
        super().__init__()
        self.mlp_stack = nn.Sequential(
            nn.Linear(input_length, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1)  # Output raw logits without sigmoid
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation
        :param x: input tensor shape [batch_size, input_length]
        :return: logits tensor shape [batch_size, 1]
        """
        logits = self.mlp_stack(x)
        return logits


# ---------------------- 2. Synthetic Microelectrode Waveform Data Generator ----------------------
def generate_electrode_dataset(sample_num: int = 10000):
    """
    Generate simulated 1D time-series signal for microelectrode binary QC task
    Label 0: Normal baseline with tiny noise
    Label 1: Abnormal waveform with sharp spike
    :param sample_num: total generated samples
    :return: X(tensor [N, 20]), y(tensor [N])
    """
    signal_list = []
    label_list = []

    for _ in range(sample_num):
        baseline = np.random.normal(loc=0.0, scale=0.1, size=20)
        if np.random.random() > 0.5:
            waveform = baseline
            label = 0.0
        else:
            spike_pos = np.random.randint(low=5, high=15)
            waveform = baseline
            waveform[spike_pos] += 2.2
            label = 1.0

        signal_list.append(waveform)
        label_list.append(label)

    X_np = np.array(signal_list, dtype=np.float32)
    y_np = np.array(label_list, dtype=np.float32)
    X_tensor = torch.from_numpy(X_np)
    y_tensor = torch.from_numpy(y_np)

    return X_tensor, y_tensor


# ---------------------- 3. Main Training Entry ----------------------
if __name__ == "__main__":
    # Hyper Parameters
    INPUT_LEN = 20
    TOTAL_SAMPLES = 10000
    TRAIN_EPOCHS = 80
    LEARNING_RATE = 1e-3

    # Initialize core modules
    model = TimeSeriesMLP(input_length=INPUT_LEN)
    loss_criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Load synthetic dataset
    X_data, y_label = generate_electrode_dataset(sample_num=TOTAL_SAMPLES)

    # ---------------------- 4. Training Loop ----------------------
    train_loss_record = []
    print("===== Start MLP Training for Microelectrode Signal Classification =====")

    for epoch in range(TRAIN_EPOCHS):
        optimizer.zero_grad()
        pred_logits = model(X_data).squeeze(dim=-1)
        train_loss = loss_criterion(pred_logits, y_label)

        train_loss.backward()
        optimizer.step()

        train_loss_record.append(train_loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1:3d}/{TRAIN_EPOCHS}] | Train Loss: {train_loss.item():.4f}")

    # ---------------------- 5. Accuracy Evaluation ----------------------
    print("\n===== Training Set Evaluation =====")
    with torch.no_grad():
        pred_prob = torch.sigmoid(pred_logits)
        pred_class = (pred_prob > 0.5).float()
        total_correct = (pred_class == y_label).sum()
        train_accuracy = total_correct / len(y_label)
    print(f"Train Set Accuracy: {train_accuracy:.4f}")

    # ---------------------- 6. Plot Loss Curve ----------------------
    plt.figure(figsize=(8, 4))
    plt.plot(train_loss_record, color="#1f77b4")
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.grid(alpha=0.3)
    plt.show()

    # ---------------------- 7. Single Sample Inference Demo ----------------------
    print("\n===== Single Waveform Inference Demo =====")
    with torch.no_grad():
        test_waveform = torch.normal(mean=0.0, std=0.1, size=(1, INPUT_LEN))
        test_waveform[0, 10] += 2.2

        test_logit = model(test_waveform)
        test_prob = torch.sigmoid(test_logit)
        print(f"Predicted abnormal signal probability: {test_prob.item():.3f}")
