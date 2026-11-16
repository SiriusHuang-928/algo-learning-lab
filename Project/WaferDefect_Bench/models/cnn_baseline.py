import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Lightweight CNN baseline for wafer defect classification.
    Input: [batch, 1, 26, 26] grayscale wafer map
    Output: [batch, num_classes] logits
    """

    def __init__(self, num_classes=9):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        # 26x26 -> 13x13 -> 6x6 -> 3x3 after 3 pooling
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 26->13
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 13->6
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 6->3

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        logits = self.fc2(x)

        return logits