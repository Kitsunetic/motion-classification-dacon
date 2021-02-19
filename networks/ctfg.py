import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.transformer import TransformerDecoder, TransformerDecoderLayer

from .common import Activation, PADDING_MODE, cba3x3, conv3x3
from .eca_tf import ECABasicBlock, ECALayer, PositionalEncoder


class CALayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv_du = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1),
            Activation(inplace=True),
            nn.Conv1d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(x)
        x = x * y

        return x


class RCAB(nn.Module):
    def __init__(self, channels, kernel_size, reduction=16):
        super().__init__()

        m = []
        psize = kernel_size // 2
        for i in range(2):
            m.append(nn.Conv1d(channels, channels, kernel_size, padding=psize, padding_mode=PADDING_MODE))
            m.append(nn.BatchNorm1d(channels))
            if i == 0:
                m.append(Activation(inplace=True))

        m.append(CALayer(channels, reduction))

        self.body = nn.Sequential(*m)

    def forward(self, x):
        x = x + self.body(x)

        return x


class RG(nn.Module):
    def __init__(self, channels, kernel_size, n_resblocks, reduction=16):
        super().__init__()

        self.rcab = nn.Sequential(*[RCAB(channels, kernel_size, reduction) for _ in range(n_resblocks)])

        psize = kernel_size // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=psize, padding_mode=PADDING_MODE)

    def forward(self, x):
        res = self.rcab(x)
        res = self.conv(res)
        x = x + res

        return x


class RTFG(nn.Module):
    """Residual TransFormer Group"""

    expansion = 2

    def __init__(
        self,
        channels,
        kernel_size,
        n_resblocks,
        n_head,
        n_layers,
        reduction=16,
        pos_encoder=True,
    ):
        self.pos_encoder = pos_encoder
        super().__init__()

        self.rg = RG(channels, kernel_size, n_resblocks, reduction)

        if self.pos_encoder:
            self.pe = PositionalEncoder(channels, addition="cat")
        elayer = TransformerEncoderLayer(
            d_model=channels,
            nhead=n_head,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
        )
        self.encoder = TransformerEncoder(elayer, n_layers)

        psize = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels * self.expansion, 3, padding=psize, padding_mode=PADDING_MODE),
            nn.BatchNorm1d(channels * self.expansion),
            Activation(inplace=True),
        )

        # self.transpose = transpose

    def forward(self, x):
        x = self.rg(x)

        x.transpose_(1, 2)
        if self.pos_encoder:
            x = self.pe(x)
        x = self.encoder(x)
        x.transpose_(1, 2)

        x = self.conv(x)

        return x


class TFEncoderBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_head,
        n_layers,
        pos_encoder=True,
        addition_mode="sum",
        transpose=True,
    ):
        super().__init__()

        self.pe = None
        if pos_encoder:
            self.pe = PositionalEncoder(d_model, addition=addition_mode)
        self.encoder = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(d_model=d_model, nhead=n_head),
            num_layers=n_layers,
        )

        self.transpose = transpose

    def forward(self, x):
        if self.transpose:
            x = x.transpose_(1, 2)

        if self.pe is not None:
            x = self.pe(x)
        x = self.encoder(x)

        if self.transpose:
            x = x.transpose_(1, 2)

        return x


class TFDecoderBlock(nn.Module):
    def __init__(self, d_model, n_head, n_layers, transpose=True):
        super().__init__()

        self.decoder = TransformerDecoder(
            decoder_layer=TransformerDecoderLayer(d_model, n_head),
            num_layers=n_layers,
        )

    def forward(self, x):
        if self.transpose:
            x = x.transpose_(1, 2)

        x = self.decoder(x)

        if self.transpose:
            x = x.transpose_(1, 2)

        return x


class RTFModel(nn.Module):
    def __init__(self, n_resblocks, n_layers, n_head, reduction=16, pos_encoder=True):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(6, 120, 7, stride=2, padding=1, groups=2, padding_mode=PADDING_MODE),
            nn.InstanceNorm1d(120),
            Activation(inplace=True),
            nn.Conv1d(120, 256, 3, padding=1, padding_mode=PADDING_MODE),
            nn.BatchNorm1d(256),
            Activation(inplace=True),
        )
        self.avg_pool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = ECABasicBlock(256, 512, stride=2)
        self.layer2 = ECABasicBlock(512, 1024, stride=2)

        self.group1 = RTFG(
            channels=1024,
            kernel_size=3,
            n_resblocks=n_resblocks[0],
            n_head=n_head[0],
            n_layers=n_layers[0],
            reduction=reduction,
            pos_encoder=pos_encoder,
        )
        self.group2 = RTFG(
            channels=2048,
            kernel_size=3,
            n_resblocks=n_resblocks[1],
            n_head=n_head[1],
            n_layers=n_layers[1],
            reduction=reduction,
            pos_encoder=pos_encoder,
        )

        self.layer3 = ECABasicBlock(4096, 4096, stride=2)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(4096, 61),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.avg_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.group1(x)
        x = self.avg_pool(x)
        x = self.group2(x)
        x = self.avg_pool(x)

        x = self.layer3(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
