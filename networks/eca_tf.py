import math

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

        self.conv1 = nn.Conv1d(inchannels, channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act = nn.ELU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
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


class PosEncoder(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=600):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        x = self.dropout(x)

        return x


class ECATF(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(24, 120, 7, stride=1, padding=3, groups=3),
            nn.InstanceNorm1d(120),
            nn.ELU(inplace=True),
            nn.Conv1d(120, 128, 3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ELU(inplace=True),
        )
        self.elayer1 = nn.Sequential(ECABasicBlock(128, 128), ECABasicBlock(128, 128))
        self.elayer2 = nn.Sequential(ECABasicBlock(128, 256), ECABasicBlock(256, 256))
        self.elayer3 = nn.Sequential(ECABasicBlock(256, 512, stride=2))
        self.elayer4 = nn.Sequential(ECABasicBlock(512, 1024, stride=2), nn.AvgPool1d(2))

        encoder_layer = TransformerEncoderLayer(
            d_model=1024,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
        )
        self.encoder = TransformerEncoder(encoder_layer, 8)

        self.decoder1 = ECABasicBlock(1024, 1536, stride=2)
        self.decoder2 = ECABasicBlock(1536, 2048)
        self.decoder3 = ECABasicBlock(2048, 3070, stride=2)
        self.decoder4 = ECABasicBlock(3070, 4094)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p=0.05),
            nn.Linear(4094, 1024),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.05),
            nn.Linear(1024, 61),
        )

    def forward(self, x):
        x = self.conv(x)

        x = self.elayer1(x)
        x = self.elayer2(x)
        x = self.elayer3(x)
        x = self.elayer4(x)

        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = x.transpose(1, 2)

        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.decoder4(x)

        x = self.fc(x)

        return x
