import torch.nn as nn

Activation = nn.ELU


def conv3x3(ca, cb, stride=1):
    return nn.Conv1d(ca, cb, 3, stride, 1)


def cba3x3(ca, cb, stride=1):
    return nn.Sequential(
        conv3x3(ca, cb, stride),
        nn.BatchNorm1d(cb),
        Activation(),
    )
