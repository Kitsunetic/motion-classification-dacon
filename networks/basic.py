import torch
from torch import nn
from torch.nn import functional as F

from .nl import NLBlockND


class BasicConv(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(18, 64, 7, stride=2),
            nn.BatchNorm1d(64),
            nn.ELU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=0),
            nn.Conv1d(64, 128, 3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=0),
            NLBlockND(128, 32, dimension=1),
            # SELayer(128),
            nn.Conv1d(128, 256, 3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 61),
        )

    def forward(self, x):
        x = self.conv1(x)

        return x
