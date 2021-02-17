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
            nn.Conv1d(6, 60, 3, padding=1, groups=6, padding_mode="circular"),
            nn.InstanceNorm1d(60),
            nn.ReLU(inplace=True),
            nn.Conv1d(60, 64, 3, padding=1, groups=2, padding_mode="circular"),
            nn.InstanceNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 3, padding=1, padding_mode="circular"),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = CircularHalfPooling()  # 300

        self.body = nn.Sequential(
            # self._fe_layer(128, 128, stride=1),
            # CircularHalfPooling(),  # 150
            ChannelSpatialAttentionGroup(128, 256, 5, 2, pool=False),
            # self._fe_layer(256, 256, stride=1),
            # CircularHalfPooling(),  # 75
            ChannelSpatialAttentionGroup(256, 512, 5, 4, pool=True),
            # self._fe_layer(512, 512, stride=1),
            # CircularHalfPooling(),  # 37
            ChannelSpatialAttentionGroup(512, 1024, 5, 8, pool=True),
            # self._fe_layer(1024, 1024, stride=1),
            # CircularHalfPooling(),  # 18
            ChannelSpatialAttentionGroup(1024, 2048, 5, 4, pool=True),
        )

        # TODO layer 수 늘리기
        # TODO 채널 늘리기
        # TODO GAP 전에 conv
        # TODO 푸리에 변환한 값 추가

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.decision = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 61),
        )

    def forward(self, x):
        x = self.shallow_extraction(x)
        x = self.pool(x)  # 300

        x = self.body(x)

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
