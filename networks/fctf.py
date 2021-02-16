import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.activation import MultiheadAttention

from .common import CircularHalfPooling, PADDING_MODE, Activation, cba3x3, conv3x3
from .resnest import ResNeStBottleneck


class ConvBatchNorm(nn.Sequential):
    def __init__(self, inchannels, channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__(
            nn.Conv1d(inchannels, channels, kernel_size, stride, padding, groups=groups, padding_mode="circular"),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )


class ChannelSpatialAttention(nn.Module):
    def __init__(self, inchannels, channels, kernel_size):
        super().__init__()

        psize = kernel_size // 2
        self.conv = nn.Conv2d(
            inchannels,
            channels,
            kernel_size,
            padding=psize,
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        h = x.unsqueeze(1)
        h = self.conv(h)
        h = h.squeeze(1)
        h = h.sigmoid()

        x = torch.addcmul(x, self.gamma, x, h)

        return x


class MultiHeadConvolutionalAttention(nn.Module):
    def __init__(self, inchannels, channels, kernel_size):
        super().__init__()

        self.conv_q = ConvBatchNorm(inchannels, channels, 1)
        self.conv_k = ConvBatchNorm(inchannels, channels, 1)
        self.conv_v = ConvBatchNorm(inchannels, channels, 1)

        self.attn = ChannelSpatialAttention(channels, channels, kernel_size)
        self.conv = ConvBatchNorm(channels, channels, 3)

    def forward(self, q, k, v):
        q = self.conv_q(q)
        k = self.conv_k(k)
        v = self.conv_v(v)

        x = torch.cat([q, k, v], dim=1)
        x = self.conv(x)

        return x


class MultiHeadConvolutionalAttentionGroup(nn.Module):
    def __init__(self, inchannels, channels, kernel_size, n_layers, pool=False):
        super().__init__()

        m = [MultiHeadConvolutionalAttention(inchannels, channels, kernel_size)]
        for _ in range(1, n_layers):
            m.append(MultiHeadConvolutionalAttention(channels, channels, kernel_size))

        if pool:
            m.append(CircularHalfPooling())

        self.body = nn.Sequential(*m)

    def forward(self, x):
        return x + self.body(x)


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

        self.feature_extraction1 = self._fe_layer(64, 128, stride=1)
        self.feature_extraction2 = self._fe_layer(128, 256, stride=1)

        self.self_attn1 = MultiHeadConvolutionalAttentionGroup(256, 384, 3, 2)
        self.self_attn2 = MultiHeadConvolutionalAttentionGroup(384, 512, 3, 2)
        self.self_attn3 = MultiHeadConvolutionalAttentionGroup(512, 768, 3, 2, pool=True)
        self.self_attn4 = MultiHeadConvolutionalAttentionGroup(768, 1024, 3, 2, pool=True)

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
        x = self.self_attn4(x)  # 34

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
