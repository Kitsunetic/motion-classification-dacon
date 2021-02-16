import torch.nn as nn
import torch.nn.functional as F

PADDING_MODE = "circular"
Activation = nn.ELU


def conv3x3(ca, cb, stride=1):
    return nn.Conv1d(ca, cb, 3, stride, 1, padding_mode=PADDING_MODE)


def cba3x3(ca, cb, stride=1):
    return nn.Sequential(
        conv3x3(ca, cb, stride),
        nn.BatchNorm1d(cb),
        Activation(),
    )


class CircularHalfPooling(nn.AvgPool1d):
    def __init__(self):
        super().__init__(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x = F.pad(x, (1, 1), "circular")
        super().__call__(x)

        return x
