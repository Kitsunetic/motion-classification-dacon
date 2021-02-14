"""
SCAE: Stacked CNN Auto-Encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SCAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(6, 3, 5, padding=2, padding_mode="circular")
        self.conv2 = nn.Conv1d(3, 6, 5, padding=2, padding_mode="circular")

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x
