import torch
import torch.nn as nn

CNN_OUT_CHANNEL = 32
LSTM_HIDDEN = 16
NUM_CLASSES = 3

class CNNBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, CNN_OUT_CHANNEL, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(CNN_OUT_CHANNEL, CNN_OUT_CHANNEL, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.bilstm = nn.LSTM(
            input_size=CNN_OUT_CHANNEL,
            hidden_size=LSTM_HIDDEN,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(LSTM_HIDDEN * 2, NUM_CLASSES)
        self.conv1_feature = None
        # 钩子捕获第一层卷积输出
        self.cnn[0].register_forward_hook(self.hook_conv1)

    def hook_conv1(self, module, input, output):
        self.conv1_feature = output.detach().cpu()

    def forward(self, x):
        cnn_feat = self.cnn(x)
        cnn_feat = cnn_feat.permute(0, 2, 1)
        _, (h_n, _) = self.bilstm(cnn_feat)
        concat_h = torch.cat([h_n[0], h_n[1]], dim=-1)
        logits = self.fc(concat_h)
        return logits
