import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .resnet import BasicBlock, BottleNeck

from .common import Activation


class SGCNN(nn.Module):
    def __init__(self, in_channels, channels, activation, norm):
        assert in_channels % 3 == 0
        super().__init__()

        self.convs = [
            nn.Sequential(
                nn.Conv1d(in_channels // 3, in_channels, 9),
                norm(in_channels),
            )
            for _ in range(3)
        ]
        self.convs = nn.ModuleList(self.convs)
        self.conv2 = nn.Sequential(
            activation(inplace=True),
            nn.Conv1d(in_channels * 3, channels, 9),
            norm(channels),
            activation(inplace=True),
        )

    def forward(self, x):
        xs = torch.split(x, split_size_or_sections=6, dim=1)
        xt = []
        for i in range(3):
            xt.append(self.convs[i](xs[i]))
        x = torch.cat(xt, dim=1)
        x = self.conv2(x)

        return x


"""====================================================================
                       [2018] Transformer

- https://tutorials.pytorch.kr/beginner/transformer_tutorial.html
===================================================================="""


class TransformerModel_v1(nn.Module):
    def __init__(self, activation=nn.ReLU, norm=nn.InstanceNorm1d):
        super().__init__()

        self.conv1 = SGCNN(18, 128, activation, norm)
        encoder_layer = TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dropout=0.2,
            activation="relu",
        )
        self.encoder = TransformerEncoder(encoder_layer, 2)
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(584 * 128, 2048),
            nn.Dropout(p=0.2),
            nn.Linear(2048, 61),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = self.decoder(x)

        return x


class TransformerModel_v2(nn.Module):
    def __init__(self, activation=nn.ReLU, norm=nn.InstanceNorm1d):
        super().__init__()

        self.conv1 = SGCNN(18, 128, activation, norm)
        encoder_layer = TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dropout=0.2,
            activation="relu",
        )
        self.encoder = TransformerEncoder(encoder_layer, 2)
        self.decoder = nn.Sequential(
            nn.Conv1d(128, 512, 5, 2),
            nn.BatchNorm1d(512),
            activation(inplace=True),
            nn.AvgPool1d(2),
            nn.Conv1d(512, 1024, 5, 2),
            nn.BatchNorm1d(1024),
            activation(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(1024, 61),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = x.transpose(1, 2)
        x = self.decoder(x)

        return x


class TransformerModel_v3(nn.Module):
    def __init__(self, activation=nn.ReLU, norm=nn.InstanceNorm1d):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(6, 60, 3, groups=6),
            norm(60),
            activation(inplace=True),
            nn.Conv1d(60, 128, 3),
            nn.BatchNorm1d(128),
            activation(inplace=True),
            # SGCNN(18, 128, activation, norm),
            nn.AvgPool1d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, 3),
            nn.BatchNorm1d(256),
            activation(inplace=True),
            nn.Conv1d(256, 512, 3),
            nn.BatchNorm1d(512),
            activation(inplace=True),
            nn.AvgPool1d(2),
        )
        encoder_layer = TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dropout=0.2,
            activation="relu",
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=6)
        self.decoder = nn.Sequential(
            nn.Conv1d(512, 1024, 3),
            nn.BatchNorm1d(1024),
            activation(inplace=True),
            nn.AvgPool1d(2),
            nn.Conv1d(1024, 2048, 3),
            nn.BatchNorm1d(2048),
            activation(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            nn.Dropout(0.1),
            nn.Linear(1024, 61),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = x.transpose(1, 2)
        x = self.decoder(x)
        x = self.fc(x)

        return x


class TransformerModel_v4(nn.Module):
    def __init__(self, activation=nn.ReLU, norm=nn.InstanceNorm1d):
        super().__init__()

        self.conv1 = nn.Sequential(
            SGCNN(18, 128, activation, norm),
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512, stride=2),
        )
        encoder_layer = TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dropout=0.2,
            activation="relu",
        )
        self.encoder = TransformerEncoder(encoder_layer, 6)
        self.decoder = nn.Sequential(
            BasicBlock(512, 1024, stride=2),
            BasicBlock(1024, 2048, stride=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.Linear(1024, 61),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = x.transpose(1, 2)
        x = self.decoder(x)

        return x


class TransformerModel_v4(nn.Module):
    def __init__(self, act=nn.ELU, norm=nn.InstanceNorm1d):
        super().__init__()

        self.conv1s = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(1, 12, 7, 2, padding=3),
                    nn.InstanceNorm1d(12),
                )
                for _ in range(6)
            ]
        )
        self.elayer1 = nn.Sequential(BasicBlock(12 * 6, 128, act=act), BasicBlock(128, 128, act=act))
        self.elayer2 = nn.Sequential(BasicBlock(128, 256, act=act), BasicBlock(256, 256, act=act))
        self.elayer3 = nn.Sequential(BasicBlock(256, 512, stride=2, act=act), nn.AvgPool1d(2))
        self.elayer4 = nn.Sequential(BasicBlock(512, 1024, stride=2, act=act), nn.AvgPool1d(2))

        encoder_layer = TransformerEncoderLayer(
            d_model=1024,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
        )
        self.encoder = TransformerEncoder(encoder_layer, 8)

        self.decoder = BasicBlock(1024, 2048, act=act)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p=0.05),
            nn.Linear(2048, 1024),
            act(inplace=True),
            nn.Dropout(p=0.05),
            nn.Linear(1024, 61),
        )

    def forward(self, x):
        xs = [self.conv1s[i](x[:, i : i + 1]) for i in range(6)]
        x = torch.cat(xs, dim=1)
        x = self.elayer1(x)
        x = self.elayer2(x)
        x = self.elayer3(x)
        x = self.elayer4(x)

        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = x.transpose(1, 2)

        x = self.decoder(x)
        x = self.fc(x)

        return x
