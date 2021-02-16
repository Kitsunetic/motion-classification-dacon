import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .common import CircularHalfPooling, PADDING_MODE, Activation, cba3x3, conv3x3
from .resnest import ResNeStBottleneck


class ChannelSpatialAttentionLayer(nn.Module):
    def __init__(self, inchannels, channels, kernel_size):
        super().__init__()

        psize = kernel_size // 2
        self.conv = nn.Conv2d(
            inchannels,
            channels,
            kernel_size,
            padding=psize,
        )

    def forward(self, x):
        pass


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
            CircularHalfPooling(),
        )

        self.feature_extraction1 = self._fe_layer(64, 128, stride=1)
        self.feature_extraction2 = self._fe_layer(128, 256, stride=1)

    def forward(self, x):
        x = self.shallow_extraction(x)

        pass

    def _fe_layer(self, inchannels, channels, stride):
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
