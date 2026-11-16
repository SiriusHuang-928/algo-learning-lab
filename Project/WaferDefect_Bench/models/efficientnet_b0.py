import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 输入单通道灰度图
        self.base = models.efficientnet_b0(weights=None)
        # 替换第一层卷积，适配单通道
        self.base.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # 替换分类头
        in_feat = self.base.classifier[1].in_features
        self.base.classifier[1] = nn.Linear(in_feat, num_classes)

    def forward(self, x):
        return self.base(x)
