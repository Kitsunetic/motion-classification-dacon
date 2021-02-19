import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import Activation
import math


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inchannels, channels, stride=1, groups=1, last_gamma=False, act=nn.ReLU, norm=nn.BatchNorm1d):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(inchannels, channels, 3, stride=stride, padding=1, groups=groups),
            norm(channels),
            act(inplace=True),
            nn.Conv1d(channels, channels, 3, padding=1, groups=groups),
        )
        self.bn = norm(channels)
        self.act = act(inplace=True)

        self.downsample = None
        if inchannels != channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(inchannels, channels, 1, stride=stride, groups=groups),
                norm(channels),
            )

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if last_gamma:
            nn.init.constant_(self.bn.weight, 0)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity
        x = self.act(x)

        return x


class BottleNeck(nn.Module):
    expansion = 4  # TODO 2??

    def __init__(self, inchannels, channels, stride=1, groups=1, last_gamma=False):
        super(BottleNeck, self).__init__()

        width = int(channels * (64 / 64.0)) * groups
        self.conv1 = nn.Sequential(
            nn.Conv1d(inchannels, width, 1),
            nn.BatchNorm1d(width),
            Activation(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(width, width, 3, stride=stride, groups=groups, padding=1),
            nn.BatchNorm1d(width),
            Activation(),
        )
        self.conv3 = nn.Conv1d(width, channels * self.expansion, 1)
        self.bn = nn.BatchNorm1d(channels * self.expansion)
        self.act = Activation()

        self.downsample = None
        if stride != 1 or channels * self.expansion != inchannels:
            self.downsample = nn.Sequential(
                nn.Conv1d(inchannels, channels * self.expansion, 1, stride=stride),
                nn.BatchNorm1d(channels * self.expansion),
            )

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if last_gamma:
            nn.init.constant_(self.bn.weight, 0)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity
        x = self.act(x)

        return x


class PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=600):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        with torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.size(1)
            pe = self.pe[:, :seq_len].view(1, seq_len, self.d_model)
            x = x + pe
            return x


class TFEncoderBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_head,
        n_layers,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_len=600,
        transpose=True,
    ):
        super().__init__()

        self.pe = PositionalEncoder(d_model=d_model, max_seq_len=max_seq_len)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="relu",
            ),
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.transpose = transpose

    def forward(self, x):
        if self.transpose:
            x = x.transpose_(1, 2)

        x = self.pe(x)
        x = self.encoder(x)

        if self.transpose:
            x = x.transpose_(1, 2)

        return x


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()

        self.inchannels = 64
        self.conv = nn.Sequential(
            nn.Conv1d(18, 18, 1, groups=6),
            nn.Conv1d(18, 64, 7, 2, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            # nn.AvgPool1d(2),
        )

        self.tf = TFEncoderBlock(
            d_model=64,
            n_head=8,
            n_layers=2,
            dim_feedforward=2048,
            dropout=0.1,
            max_seq_len=600,
        )

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1])  # , stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2])  # , stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.inchannels, 2048),
            nn.Dropout(0.1),
            nn.Linear(2048, 61),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.tf(x)

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
        super().__init__(BasicBlock, [2, 2, 2, 2])


class ResNet34(ResNet):
    def __init__(self):
        super().__init__(BasicBlock, [3, 4, 6, 3])


class ResNet50(ResNet):
    def __init__(self):
        super().__init__(BottleNeck, [3, 4, 6, 3])


class ResNet101(ResNet):
    def __init__(self):
        super().__init__(BottleNeck, [3, 4, 23, 3])


class ResNet152(ResNet):
    def __init__(self):
        super().__init__(BottleNeck, [3, 8, 36, 3])
