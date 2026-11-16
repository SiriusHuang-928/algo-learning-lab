import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def load_wafer_data(data_dir="./data", sample_num=None):
    """
    Load preprocessed wafer dataset from numpy files.
    """
    x = np.load(os.path.join(data_dir, "wafer_x.npy"))
    y = np.load(os.path.join(data_dir, "wafer_y.npy"))
    raw_labels = np.load(os.path.join(data_dir, "label_names.npy"), allow_pickle=True)
    # 修复标签格式：numpy数组转普通字符串
    label_names = [str(item[0]) if isinstance(item, np.ndarray) else str(item) for item in raw_labels]

    if sample_num is not None:
        indices = np.random.choice(len(x), sample_num, replace=False)
        x = x[indices]
        y = y[indices]

    print(f"Total samples: {len(x)}")
    print(f"Number of classes: {len(label_names)}")
    print(f"Class names: {label_names}")
    return x, y, label_names


class WaferDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def build_dataloaders(data_dir="./data", batch_size=128, sample_num=50000):
    """Build train/val dataloaders with stratified split."""
    x, y, label_names = load_wafer_data(data_dir, sample_num)

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    train_dataset = WaferDataset(x_train, y_train)
    val_dataset = WaferDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, label_names
