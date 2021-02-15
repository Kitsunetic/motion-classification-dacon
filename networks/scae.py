"""
SCAE: Stacked CNN Auto-Encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SCAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(6, 32, 5, padding=2, padding_mode="circular")
        self.conv2 = nn.Conv1d(32, 64, 5, padding=2, padding_mode="circular")
        self.conv3 = nn.Conv1d(64, 32, 5, padding=2, padding_mode="circular")
        self.conv4 = nn.Conv1d(32, 6, 5, padding=2, padding_mode="circular")

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x
