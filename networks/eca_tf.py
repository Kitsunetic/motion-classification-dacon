import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class ECALayer(nn.Module):
    def __init__(self, k_size=3):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, l = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        # y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class ECABasicBlock(nn.Module):
    def __init__(self, inchannels, channels, stride=1, downsample=None, k_size=3):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(inchannels, channels, 1),
            nn.BatchNorm1d(channels),
            nn.ELU(inplace=True),
            nn.Conv1d(channels, channels, 3, stride=stride, padding=1, padding_mode="circular"),
            nn.BatchNorm1d(channels),
            nn.ELU(inplace=True),
            nn.Conv1d(channels, channels, 1),
            nn.BatchNorm1d(channels),
            ECALayer(k_size),
        )

        self.downsample = downsample
        if self.downsample is None and (stride != 1 or inchannels != channels):
            self.downsample = nn.Sequential(
                nn.Conv1d(inchannels, channels, 1, stride=stride),
                nn.BatchNorm1d(channels),
            )

        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        return self.elu(x + residual)


class ECABottleneck(nn.Module):
    def __init__(self, inchannels, channels, stride=1, downsample=None, k_size=3):
        super().__init__()

        width = channels // 2

        self.conv = nn.Sequential(
            nn.Conv1d(inchannels, width, 1),
            nn.BatchNorm1d(width),
            nn.ELU(inplace=True),
            nn.Conv1d(width, width, 3, stride=1, padding=1, padding_mode="cirulcar"),
            nn.BatchNorm1d(width),
            nn.ELU(inplace=True),
            nn.Conv1d(width, channels, 1),
            nn.BatchNorm1d(channels),
            ECALayer(k_size),
        )

        self.downsample = downsample
        if self.downsample is None and (stride != 1 or inchannels != channels):
            self.downsample = nn.Sequential(
                nn.Conv1d(inchannels, channels, 1, stride=stride),
                nn.BatchNorm1d(channels),
            )

        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        return self.elu(x + residual)


class ECATF(nn.Module):
    def __init__(self, block, n_layers):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(8, 64, 7, stride=2, padding=3, padding_mode="circular"),
            nn.BatchNorm1d(64),
            nn.ELU(inplace=True),
        )
        self.layer1 = self._make_layer(block, 64, 256, stride=1, num_layers=n_layers[0])
        self.layer2 = self._make_layer(block, 256, 512, stride=1, num_layers=n_layers[1])
        self.layer3 = self._make_layer(block, 512, 1024, stride=2, num_layers=n_layers[2])

        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=1024,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                activation="relu",
            ),
            num_layers=8,
            norm=nn.LayerNorm(1024),
        )

        self.layer4 = self._make_layer(block, 1024, 2048, stride=2, num_layers=n_layers[3])

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(inplace=True),
            nn.Linear(1024, 61),
        )

    def forward(self, x):
        x = self.conv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = x.transpose(1, 2)

        x = self.layer4(x)

        x = self.fc(x)

        return x

    def _make_layer(self, block, inchannels, channels, stride, num_layers):
        m = []
        m.append(ECABasicBlock(inchannels, channels, stride))
        for _ in range(1, num_layers):
            m.append(ECABasicBlock(channels, channels))
        return nn.Sequential(*m)


def ecatf18():
    return ECATF(ECABasicBlock, [2, 2, 2, 2])


def ecatf34():
    return ECATF(ECABasicBlock, [3, 4, 6, 3])


def ecatf50():
    return ECATF(ECABottleneck, [3, 4, 6, 3])


def ecatf101():
    return ECATF(ECABottleneck, [3, 4, 23, 3])


def ecatf152():
    return ECATF(ECABottleneck, [3, 4, 36, 3])
