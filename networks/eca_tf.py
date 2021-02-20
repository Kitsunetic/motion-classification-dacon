import math

import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules import padding
from torch.nn.modules.activation import Tanh
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear

from .common import PADDING_MODE, Activation, cba3x3, conv3x3


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

        # self.pe = PositionalEncoder(d_model=d_model, max_seq_len=max_seq_len)
        self.encoder = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
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

        # x = self.pe(x)
        x = self.encoder(x)

        if self.transpose:
            x = x.transpose_(1, 2)

        return x


class ECALayer(nn.Module):
    def __init__(self, k_size=3):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, l = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        # y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ECABasicBlock(nn.Module):
    def __init__(self, inchannels, channels, stride=1, downsample=None, k_size=3):
        super().__init__()

        self.conv1 = conv3x3(inchannels, channels, stride)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act = Activation()
        self.conv2 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.eca = ECALayer(k_size)
        self.downsample = downsample

        if self.downsample is None and (stride != 1 or inchannels != channels):
            self.downsample = nn.Sequential(
                nn.Conv1d(inchannels, channels, 1, stride=stride),
                nn.BatchNorm1d(channels),
            )

    def forward(self, x):
        h = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.eca(x)

        if self.downsample is not None:
            h = self.downsample(h)

        x += h
        x = self.act(x)

        return x


def ECABasicGroup(inchannels, channels, stride=1, downsample=None, k_size=3, n_layers=1):
    m = []
    m.append(ECABasicBlock(inchannels, channels, stride=stride, downsample=downsample, k_size=k_size))

    for _ in range(1, n_layers - 1):
        m.append(ECABasicBlock(channels, channels, stride=1, downsample=downsample, k_size=k_size))
    m.append(NLBlockND(channels, inter_channels=1024, dimension=1))
    m.append(ECABasicBlock(channels, channels, stride=1, downsample=downsample, k_size=k_size))
    return nn.Sequential(*m)


class SEBasicBlock(nn.Module):
    def __init__(self, inchannels, channels, stride=1, downsample=None, k_size=3):
        super().__init__()

        self.conv1 = conv3x3(inchannels, channels, stride)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act = Activation()
        self.conv2 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.se = SELayer(channels, reduction=16)
        self.downsample = downsample

        if self.downsample is None and (stride != 1 or inchannels != channels):
            self.downsample = nn.Sequential(
                nn.Conv1d(inchannels, channels, 1, stride=stride),
                nn.BatchNorm1d(channels),
            )

    def forward(self, x):
        h = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.se(x)

        if self.downsample is not None:
            h = self.downsample(h)

        x += h
        x = self.act(x)

        return x


class SEBottleneck(nn.Module):
    def __init__(self, inchannels, channels, stride=1, downsample=None, k_size=3):
        super().__init__()

        C = channels // 4

        self.conv1 = nn.Sequential(
            nn.Conv1d(inchannels, C, 1, bias=False),
            nn.BatchNorm1d(C),
            nn.Hardswish(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(C, C, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(C),
            nn.Hardswish(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(C, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.se = SELayer(channels, reduction=16)

        self.downsample = downsample
        if self.downsample is None and (stride != 1 or inchannels != channels):
            self.downsample = nn.Sequential(
                nn.Conv1d(inchannels, channels, 1, stride=stride),
                nn.BatchNorm1d(channels),
            )

        self.relu = nn.Hardswish(inplace=True)

    def forward(self, x):
        h = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.se(x)

        if self.downsample is not None:
            h = self.downsample(h)

        return self.relu(x + h)


def SEBasicGroup(inchannels, channels, stride=1, downsample=None, k_size=3, n_layers=1):
    m = []
    m.append(SEBasicBlock(inchannels, channels, stride=stride, downsample=downsample, k_size=k_size))

    for _ in range(1, n_layers):
        m.append(SEBasicBlock(channels, channels, stride=1, downsample=downsample, k_size=k_size))

    return nn.Sequential(*m)


def SEBottleneckGroup(inchannels, channels, stride=1, downsample=None, k_size=3, n_layers=1):
    m = []
    m.append(SEBottleneck(inchannels, channels, stride, downsample, k_size))
    for _ in range(1, n_layers - 1):
        m.append(SEBottleneck(channels, channels, stride, downsample, k_size))
    # m.append(NLBlockND(channels, dimension=1))
    m.append(SEBottleneck(channels, channels, stride, downsample, k_size))

    return nn.Sequential(*m)


class NLBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=512, mode="embedded", dimension=3, bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        if mode not in ["gaussian", "embedded", "dot", "concatenate"]:
            raise ValueError("`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`")

        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        conv_nd = nn.Conv1d
        bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1), bn(self.in_channels)
            )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1), nn.ReLU())

    def forward(self, x):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """

        batch_size = x.size(0)

        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)  # number of position in x
            f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)

        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        W_y = self.W_z(y)
        # residual connection
        z = W_y + x

        return z


class NNNLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


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
            # pe = self.pe[:, :seq_len].view(1, seq_len, self.d_model)
            pe = self.pe[:, :seq_len].expand_as(x)
            # x = x + pe
            x = torch.cat([x, pe], dim=1)
            return x


class ECATF(nn.Module):
    def __init__(self, num_classes=61):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Conv1d(6, 60, 7, stride=2, padding=1, groups=6, padding_mode="circular"),
            nn.InstanceNorm1d(60),
            nn.Hardswish(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        C = 128
        G = SEBasicGroup  # ECABasicGroup  # SEBasicGroup  # SEBottleneckGroup

        self.elayer1 = G(60, 1 * C, stride=1, n_layers=2)
        self.elayer2 = G(1 * C, 2 * C, stride=1, n_layers=3)
        self.elayer3 = G(2 * C, 4 * C, stride=1, n_layers=4)
        self.elayer4 = G(4 * C, 8 * C, stride=2, n_layers=2)
        self.elayer5 = nn.Conv1d(8 * C, 8 * C, 1)

        self.tf_encoder = TFEncoderBlock(
            d_model=8 * C,
            n_head=8,
            n_layers=8,
            dim_feedforward=24 * C,
            dropout=0.1,
            max_seq_len=math.ceil(600 / (2 ** 3)),
            transpose=True,
        )

        self.decoder1 = G(8 * C, 12 * C, stride=1, n_layers=2)
        self.decoder2 = G(12 * C, 16 * C, stride=1, n_layers=2)
        self.decoder3 = G(16 * C, 20 * C, stride=2, n_layers=3)
        self.decoder4 = G(20 * C, 24 * C, stride=2, n_layers=4)

        self.zembedding = nn.Sequential(
            nn.Conv1d(12, 4, 3, padding=0, bias=False),
            nn.Flatten(),
            nn.Linear(4 * 598, 2 * C),
        )

        self.fc1 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(24 * C, 8 * C),
        )
        fc = [
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(10 * C, 4 * C),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(4 * C, num_classes),
        ]
        """if num_classes == 1:
            fc.append(nn.Sigmoid())"""
        self.fc2 = nn.Sequential(*fc)

    def forward(self, input):
        x = input[:, :6, :]
        z = input[:, 6:, :]

        x = self.embedding(x)

        x = self.elayer1(x)
        x = self.elayer2(x)
        x = self.elayer3(x)
        x = self.elayer4(x)
        x = self.elayer5(x)

        x = self.tf_encoder(x)

        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.decoder4(x)

        x = self.fc1(x)
        z = self.zembedding(z)
        x = torch.cat([x, z], dim=1)
        x = self.fc2(x)

        return x
