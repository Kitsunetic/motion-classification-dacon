import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import BasicBlock, BottleNeck

from .common import Activation


class LegacyResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()

        self.inchannels = 32

        self.conv = nn.Sequential(
            nn.Conv1d(18, self.inchannels, 3, padding=1, bias=False),
            nn.BatchNorm1d(self.inchannels),
            Activation(),
        )

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.layer4 = self._make_layer(block, 512, layers[3])

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.inchannels, 61),
        )

    def forward(self, x):
        x = self.conv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.fc(x)

        return x

    def _make_layer(self, block, channels, blocks, stride=1):
        layers = []
        layers.append(block(self.inchannels, channels, stride=stride))
        self.inchannels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inchannels, channels))

        return nn.Sequential(*layers)


class LegacyResNet18(LegacyResNet):
    def __init__(self):
        super().__init__(BasicBlock, [2, 2, 2, 2])


class LegacyResNet34(LegacyResNet):
    def __init__(self):
        super().__init__(BasicBlock, [3, 4, 6, 3])


class LegacyResNet50(LegacyResNet):
    def __init__(self):
        super().__init__(BottleNeck, [3, 4, 6, 3])


class LegacyResNet101(LegacyResNet):
    def __init__(self):
        super().__init__(BottleNeck, [3, 4, 23, 3])


class LegacyResNet152(LegacyResNet):
    def __init__(self):
        super().__init__(BottleNeck, [3, 8, 36, 3])
