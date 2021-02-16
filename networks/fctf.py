import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.activation import MultiheadAttention

from .common import PADDING_MODE, Activation, CircularHalfPooling, cba3x3, conv3x3
from .resnest import ResNeStBottleneck


class ConvBatchNorm(nn.Sequential):
    def __init__(self, inchannels, channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__(
            nn.Conv1d(inchannels, channels, kernel_size, stride, padding, groups=groups, padding_mode="circular"),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )


class ChannelSpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()

        psize = kernel_size // 2
        self.conv = nn.Conv2d(1, 1, kernel_size, padding=psize, padding_mode="circular")
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        h = x.unsqueeze(1)
        h = self.conv(h)
        h = h.squeeze(1)
        h = h.sigmoid()

        x = x + self.gamma * x * h

        return x


class ChannelSpatialAttentionBlock(nn.Module):
    def __init__(self, inchannels, channels, kernel_size, stride=1):
        super().__init__()

        self.conv1 = ConvBatchNorm(inchannels, channels, 1)
        self.attn = ChannelSpatialAttention(kernel_size)
        self.conv2 = ConvBatchNorm(channels, channels, 3, stride=stride, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.attn(x)
        x = self.conv2(x)

        return x


class ChannelSpatialAttentionGroup(nn.Module):
    def __init__(self, inchannels, channels, kernel_size, n_layers, pool=False):
        super().__init__()

        psize = kernel_size // 2
        self.conv = ConvBatchNorm(inchannels, channels, kernel_size, padding=psize)

        self.m = nn.ModuleList()
        for _ in range(n_layers):
            self.m.append(ChannelSpatialAttentionBlock(channels, channels, kernel_size))

        self.pool = CircularHalfPooling() if pool else None

    def forward(self, x):
        x = self.conv(x)
        h = x

        for m in self.m:
            x = m(h)

        x = h + x
        if self.pool is not None:
            x = self.pool(x)

        return x


class ConvTransformerModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.shallow_extraction = nn.Sequential(
            nn.Conv1d(6, 18, 3, padding=1, groups=6, padding_mode="circular"),
            nn.InstanceNorm1d(30),
            nn.ReLU(inplace=True),
            nn.Conv1d(18, 36, 3, padding=1, groups=2, padding_mode="circular"),
            nn.InstanceNorm1d(36),
            nn.ReLU(inplace=True),
            nn.Conv1d(36, 64, 3, padding=1, padding_mode="circular"),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = CircularHalfPooling()

        self.feature_extraction1 = self._fe_layer(64, 64, stride=1)
        self.feature_extraction2 = self._fe_layer(128, 128, stride=1)

        self.self_attn1 = ChannelSpatialAttentionGroup(256, 384, 3, 2)
        self.self_attn2 = ChannelSpatialAttentionGroup(384, 512, 3, 2)
        self.self_attn3 = ChannelSpatialAttentionGroup(512, 768, 3, 2, pool=False)
        self.self_attn4 = ChannelSpatialAttentionGroup(768, 1024, 3, 2, pool=True)

        # TODO 푸리에 변환한 값 추가

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.decision = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 61),
        )

    def forward(self, x):
        x = self.shallow_extraction(x)
        x = self.pool(x)  # 300

        x = self.feature_extraction1(x)
        x = self.feature_extraction2(x)
        x = self.pool(x)  # 150

        x = self.self_attn1(x)
        x = self.self_attn2(x)
        x = self.self_attn3(x)  # 75
        x = self.self_attn4(x)  # 75

        x = self.global_pool(x)
        x = self.decision(x)

        return x

    @staticmethod
    def _fe_layer(inchannels, channels, stride):
        return ResNeStBottleneck(
            inchannels,
            channels,
            stride=stride,
            radix=2,
            bottleneck_width=64,
            avd=True,
            avd_first=False,
            expansion=2,
        )
