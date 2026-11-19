import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class SMDDataset(Dataset):
    def __init__(self, data, label, window_len=96, step=1):
        self.window_len = window_len
        self.step = step
        self.data = data
        self.label = label
        self.samples = []
        self.sample_labels = []

        for i in range(0, len(self.data) - self.window_len, self.step):
            x_win = self.data[i:i + self.window_len]
            y_last = self.label[i + self.window_len - 1]
            self.samples.append(x_win)
            self.sample_labels.append(y_last)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32)
        y = torch.tensor(self.sample_labels[idx], dtype=torch.float32)
        return x, y


def build_dataloaders(root_dir, window_size=96, batch_size=32):
    # 适配你当前的目录结构
    train_file = os.path.join(root_dir, "train", "machine-1-1.txt")
    test_file = os.path.join(root_dir, "test", "machine-1-1.txt")
    test_label_file = os.path.join(root_dir, "test_label", "machine-1-1.txt")

    print(f"train path: {train_file}")
    print(f"test path: {test_file}")
    print(f"test label path: {test_label_file}")

    train_data = np.loadtxt(train_file, delimiter=",")
    test_data = np.loadtxt(test_file, delimiter=",")
    test_label = np.loadtxt(test_label_file, delimiter=",")

    # 训练集没有标签，无监督训练，训练标签全部置0
    train_label = np.zeros(len(train_data))

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    train_set = SMDDataset(train_data, train_label, window_len=window_size, step=1)
    test_set = SMDDataset(test_data, test_label, window_len=window_size, step=1)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, test_loader, scaler


if __name__ == "__main__":
    train_loader, test_loader, _ = build_dataloaders("ServerMachineDataset", window_size=96, batch_size=8)
    x, y = next(iter(train_loader))
    print("x shape [B, T, C]:", x.shape)
    print("y shape [B]:", y.shape)
