import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialDropout1d(nn.Module):
    def __init__(self, p=0.5, min_size=1, max_size=10):
        super().__init__()

        self.p = p
        self.min_size = min_size
        self.max_size = max_size

    def forward(self, x):
        if not self.training or random.random() > self.p:
            return x

        pos = torch.randint(0, x.size(2) - self.max_size, (x.size(0),))
        length = torch.randint(self.min_size, self.max_size, (x.size(0),))
        print(pos)
        print(length)
        x[..., pos : pos + length] *= 0

        return x


class ConvBlock3(nn.Module):
    def __init__(self, inchannels, channels, kernel_size):
        super().__init__()

        self.act = nn.ELU(inplace=True)

        psize = kernel_size // 2
        self.conv1 = nn.Conv1d(inchannels, channels, kernel_size, padding=psize)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=psize)
        self.bn2 = nn.BatchNorm1d(channels)
        self.conv3 = nn.Conv1d(channels, channels, kernel_size, padding=psize)
        self.bn3 = nn.BatchNorm1d(channels)

    def forward(self, x):
        x = self.conv1(x)
        res = x
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.conv3(x)
        x += res
        x = self.bn3(x)
        x = self.act(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # self.dropout = SpatialDropout1d(0.1)
        self.dropout = nn.Dropout(0.1)

        self.conv1 = ConvBlock3(18, 128, 9)
        self.conv2 = ConvBlock3(128, 128, 9)

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, 61),
        )

    def forward(self, x):
        # x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.fc(x)

        return x
