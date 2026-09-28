import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Global Config
SEQ_LEN = 20
NUM_CLASSES = 3
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
HIDDEN_DIM = 16

# Generate random waveform dataset (random peak position version)
def generate_signal_sample(seq_len: int):
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

# Build dataset
total_samples = 10000
X_raw, y_raw = [], []
for _ in range(total_samples):
    sig, lab = generate_signal_sample(SEQ_LEN)
    X_raw.append(sig)
    y_raw.append(lab)

X_np = np.array(X_raw, dtype=np.float32)
y_np = np.array(y_raw, dtype=np.int64)

# Original shape [N, seq_len], convert to LSTM input [batch, seq_len, input_size=1]
X_tensor = torch.from_numpy(X_np).unsqueeze(-1)
y_tensor = torch.from_numpy(y_np)

# Train test split 8:2
split_idx = int(0.8 * total_samples)
X_train, X_test = X_tensor[:split_idx], X_tensor[split_idx:]
y_train, y_test = y_tensor[:split_idx], y_tensor[split_idx:]

# Pure BiLSTM Model
class BiLSTMClassifier(nn.Module):
    def __init__(self, num_classes, hidden_dim):
        super().__init__()
        # input_size=1: each time step only 1 signal value
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        # bidirectional output hidden_dim * 2
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x shape: [batch, seq_len, 1]
        lstm_out, (h_n, _) = self.lstm(x)
        # h_n shape: [2 layers(bidir), batch, hidden_dim]
        concat_h = torch.cat((h_n[0], h_n[1]), dim=-1)
        logits = self.classifier(concat_h)
        return logits

# Train & Eval Utils
def train_one_epoch(model, optimizer, x_train, y_train, batch_size, loss_fn):
    model.train()
    total_loss = 0.0
    shuffle_idx = torch.randperm(len(x_train))
    x_shuf, y_shuf = x_train[shuffle_idx], y_train[shuffle_idx]
    for start in range(0, len(x_train), batch_size):
        batch_x = x_shuf[start:start+batch_size]
        batch_y = y_shuf[start:start+batch_size]
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = loss_fn(pred, batch_y)
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
        return correct / len(y_test)

# Main run
if __name__ == "__main__":
    model = BiLSTMClassifier(NUM_CLASSES, HIDDEN_DIM)
    loss_func = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=LR)
    print("===== Start Training Pure BiLSTM Model =====")
    for epoch in range(EPOCHS):
        loss_total = train_one_epoch(model, opt, X_train, y_train, BATCH_SIZE, loss_func)
        test_acc = evaluate_acc(model, X_test, y_test)
        print(f"Epoch {epoch+1:2d} | Train Loss: {loss_total:.4f} | Test Acc: {test_acc:.4f}")
