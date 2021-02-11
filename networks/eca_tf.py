import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.modules.conv import Conv1d

from .common import Activation, conv3x3, cba3x3


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

        self.conv1 = conv3x3(inchannels, channels, stride)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act = Activation()
        self.conv2 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.eca = ECALayer(k_size)
        self.downsample = downsample

        if self.downsample is None and (stride != 1 or inchannels != channels):
            self.downsample = nn.Sequential(
                nn.Conv1d(inchannels, channels, 1, stride=stride),
                nn.BatchNorm1d(channels),
            )

    def forward(self, x):
        h = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.eca(x)

        if self.downsample is not None:
            h = self.downsample(h)

        x += h
        x = self.act(x)

        return x


class ECATF(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1s = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(1, 12, 7, 2, padding=3),
                    nn.InstanceNorm1d(12),
                )
                for _ in range(6)
            ]
        )
        self.elayer1 = nn.Sequential(ECABasicBlock(12 * 6, 128), ECABasicBlock(128, 128))
        self.elayer2 = nn.Sequential(ECABasicBlock(128, 256), ECABasicBlock(256, 256))
        self.elayer3 = nn.Sequential(ECABasicBlock(256, 512, stride=2), nn.AvgPool1d(2))
        self.elayer4 = nn.Sequential(ECABasicBlock(512, 1024, stride=2), nn.AvgPool1d(2))

        encoder_layer = TransformerEncoderLayer(
            d_model=1024,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
        )
        self.encoder = TransformerEncoder(encoder_layer, 8)

        self.decoder = ECABasicBlock(1024, 2048)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p=0.05),
            nn.Linear(2048, 1024),
            Activation(),
            nn.Dropout(p=0.05),
            nn.Linear(1024, 61),
        )

    def forward(self, x):
        xs = [self.conv1s[i](x[:, i : i + 1]) for i in range(6)]
        x = torch.cat(xs, dim=1)
        x = self.elayer1(x)
        x = self.elayer2(x)
        x = self.elayer3(x)
        x = self.elayer4(x)

        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = x.transpose(1, 2)

        x = self.decoder(x)
        x = self.fc(x)

        return x
