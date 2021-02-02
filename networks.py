import torch
import torch.nn as nn

Activation = nn.ELU

# TODO resnet으로 함 해보고
# TODO repVGG 한번 해보자


def cba(inchannels, channels, kernel_size, stride=1, padding=0):
    conv = []
    conv.append(nn.Conv1d(inchannels, channels, kernel_size, stride, padding))
    conv.append(nn.BatchNorm1d(channels))
    conv.append(Activation(inplace=True))
    return nn.Sequential(*conv)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inchannels, channels, stride=1, groups=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(inchannels, channels, 3, stride=stride, padding=1, groups=groups),
            nn.BatchNorm1d(channels),
            Activation(),
            nn.Conv1d(channels, channels, 3, padding=1, groups=groups),
            nn.BatchNorm1d(channels),
        )
        self.act = Activation()

        self.conv2 = None
        if inchannels != channels or stride != 1:
            self.conv2 = nn.Sequential(
                nn.Conv1d(inchannels, channels, 1, stride=stride, groups=groups), nn.BatchNorm1d(channels)
            )

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        if self.conv2 is not None:
            identity = self.conv2(identity)
        x += identity
        x = self.act(x)

        return x


class BottleNeck(nn.Module):
    expansion = 2

    def __init__(self, inchannels, channels, stride=1, groups=1):
        super(BottleNeck, self).__init__()

        width = int(channels * (64 / 64.0)) * groups
        self.conv1 = nn.Sequential(nn.Conv1d(inchannels, width, 1), nn.BatchNorm1d(width))
        self.conv2 = nn.Sequential(nn.Conv1d(width, width, 3, stride=stride, groups=groups, padding=1), nn.BatchNorm1d(width))
        self.conv3 = nn.Sequential(nn.Conv1d(width, channels * self.expansion, 1), nn.BatchNorm1d(channels * self.expansion))
        self.act = Activation()

        self.downsample = None
        if stride != 1 or channels * self.expansion != inchannels:
            self.downsample = nn.Conv1d(inchannels, channels * self.expansion, 1)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity
        x = self.act(x)

        return x


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()

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
            nn.Linear(self.inchannels, 2048),
            nn.Dropout(0.1),
            nn.Linear(2048, 61),
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


class ResNet18(ResNet):
    def __init__(self):
        super(ResNet18, self).__init__(BasicBlock, [2, 2, 2, 2])


class ResNet34(ResNet):
    def __init__(self):
        super(ResNet34, self).__init__(BasicBlock, [3, 4, 6, 3])


class ResNet50(ResNet):
    def __init__(self):
        super(ResNet50, self).__init__(BottleNeck, [3, 4, 6, 3])
