import numpy as np
import torch

SEQ_LEN = 20
NUM_CLASSES = 3

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

def build_dataset(total_samples=10000):
    X_raw, y_raw = [], []
    for _ in range(total_samples):
        sig, lab = generate_waveform(SEQ_LEN)
        X_raw.append(sig)
        y_raw.append(lab)
    X_np = np.array(X_raw, dtype=np.float32)
    y_np = np.array(y_raw, dtype=np.int64)
    X_tensor = torch.from_numpy(X_np).unsqueeze(1)
    y_tensor = torch.from_numpy(y_np)
    split_idx = int(0.8 * total_samples)
    X_train, X_test = X_tensor[:split_idx], X_tensor[split_idx:]
    y_train, y_test = y_tensor[:split_idx], y_tensor[split_idx:]
    return X_train, X_test, y_train, y_test
