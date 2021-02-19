import math

import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules import padding

from .common import PADDING_MODE, Activation, cba3x3, conv3x3


class TFEncoderBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_head,
        n_layers,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_len=600,
        transpose=True,
    ):
        super().__init__()

        self.pe = PositionalEncoder(d_model=d_model, max_seq_len=max_seq_len)
        self.encoder = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="relu",
            ),
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.transpose = transpose

    def forward(self, x):
        if self.transpose:
            x = x.transpose_(1, 2)

        x = self.pe(x)
        x = self.encoder(x)

        if self.transpose:
            x = x.transpose_(1, 2)

        return x


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


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
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


def ECABasicGroup(inchannels, channels, stride=1, downsample=None, k_size=3, n_layers=1):
    m = []
    m.append(ECABasicBlock(inchannels, channels, stride=stride, downsample=downsample, k_size=k_size))

    for _ in range(1, n_layers):
        m.append(ECABasicBlock(channels, channels, stride=1, downsample=downsample, k_size=k_size))

    return nn.Sequential(*m)


class SEBasicBlock(nn.Module):
    def __init__(self, inchannels, channels, stride=1, downsample=None, k_size=3):
        super().__init__()

        self.conv1 = conv3x3(inchannels, channels, stride)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act = Activation()
        self.conv2 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.se = SELayer(channels, reduction=16)
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
        x = self.se(x)

        if self.downsample is not None:
            h = self.downsample(h)

        x += h
        x = self.act(x)

        return x


def SEBasicGroup(inchannels, channels, stride=1, downsample=None, k_size=3, n_layers=1):
    m = []
    m.append(SEBasicBlock(inchannels, channels, stride=stride, downsample=downsample, k_size=k_size))

    for _ in range(1, n_layers):
        m.append(SEBasicBlock(channels, channels, stride=1, downsample=downsample, k_size=k_size))

    return nn.Sequential(*m)


class PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=600):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        with torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.size(1)
            pe = self.pe[:, :seq_len].view(1, seq_len, self.d_model)
            x = x + pe
            return x


class ECATF(nn.Module):
    def __init__(self, num_classes=61):
        super().__init__()

        self.conv1_list = nn.ModuleList([nn.Conv1d(1, 6, 7, stride=2, padding=3, padding_mode=PADDING_MODE) for _ in range(6)])
        self.norm1_list = nn.ModuleList([nn.InstanceNorm1d(12) for _ in range(6)])
        self.act = Activation()
        self.pool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv1d(36, 72, 3, stride=1, padding=1, groups=2, padding_mode=PADDING_MODE)
        self.norm2 = nn.InstanceNorm1d(72)

        C = 128
        G = ECABasicGroup

        self.elayer1 = G(72, 1 * C, stride=1, n_layers=2)
        self.elayer2 = G(1 * C, 2 * C, stride=1, n_layers=2)
        self.elayer3 = G(2 * C, 4 * C, stride=2, n_layers=2)
        self.elayer4 = G(4 * C, 8 * C, stride=2, n_layers=2)

        self.tf_encoder = TFEncoderBlock(
            d_model=8 * C,
            n_head=8,
            n_layers=8,
            dim_feedforward=16 * C,
            dropout=0.1,
            max_seq_len=math.ceil(600 / (2 ** 3)),
            transpose=True,
        )

        self.zembedding = nn.Sequential(
            nn.Conv1d(12, 1 * C, 7, stride=2, padding=3, groups=2),
            nn.InstanceNorm1d(1 * C),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            Activation(),
            nn.Conv1d(1 * C, 2 * C, 3, stride=1, padding=1),
            nn.InstanceNorm1d(2 * C),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            Activation(),
        )

        self.decoder1 = G((8 + 2) * C, 12 * C, stride=2, n_layers=1)
        self.decoder2 = G(12 * C, 16 * C, stride=2, n_layers=1)
        self.decoder3 = G(16 * C, 20 * C, stride=2, n_layers=1)
        self.decoder4 = G(20 * C, 24 * C, stride=2, n_layers=1)

        fc_modules = [
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p=0.05),
            nn.Linear(24 * C, 4 * C),
            Activation(),
            nn.Dropout(p=0.05),
            nn.Linear(4 * C, num_classes),
        ]
        if num_classes == 1:
            fc_modules.append(nn.Sigmoid())
        self.fc = nn.Sequential(*fc_modules)

    def forward(self, input):
        x = input[:, :6, :]
        z = input[:, 6:, :]

        xs = []
        for i in range(6):
            h = self.conv1_list[i](x[:, i : i + 1])
            h = self.norm1_list[i](h)
            xs.append(h)
        x = torch.cat(xs, dim=1)
        x = self.act(x)
        # x = self.pool(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        # x = self.pool(x)

        x = self.elayer1(x)
        x = self.elayer2(x)
        x = self.elayer3(x)
        x = self.elayer4(x)

        x = self.tf_encoder(x)

        z = self.zembedding(z)
        x = torch.cat([x, z], dim=1)

        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.decoder4(x)

        x = self.fc(x)

        return x
