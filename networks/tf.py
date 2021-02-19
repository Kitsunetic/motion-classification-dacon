import math

import torch
from torch import functional as F
from torch import nn


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


class QKVConv1x1(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)

        return q, k, v


def SpatialGate(q, k, v):
    cm = torch.bmm(q.transpose(1, 2), k)  # B, L, L
    x = torch.bmm(v, cm)  # B, C, L
    return x


def ChannelGate(q, k, v):
    cm = torch.bmm(q, k.transpose(1, 2))  # B, C, C
    x = torch.bmm(cm, v)  # B, C, L
    return x


class GatedResidualConv(nn.Module):
    def __init__(self, channels, gating="spatial"):
        super().__init__()

        self.conv = nn.Conv1d(channels, channels, 3, padding=1, padding_mode="circular")
        self.relu = nn.ELU()
        self.embedding = QKVConv1x1(channels)
        self.gate = SpatialGate if gating == "spatial" else ChannelGate
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x

        x = self.relu(x)
        x = self.conv(x)
        q, k, v = self.embedding(x)
        x = self.gate(q, k, v) + residual
        x = self.norm(x)
        return x


class TSequential(nn.Sequential):
    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


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


class TF_v1(nn.Module):
    def __init__(self):
        super().__init__()

        d_model = 64

        self.embedding = nn.Sequential(
            nn.Conv1d(6, d_model, 7, stride=2, padding=1, padding_mode="circular"),
            nn.BatchNorm1d(d_model),
            nn.ELU(inplace=True),
            nn.AvgPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(d_model, 2 * d_model, 3, padding=1, padding_mode="circular"),
            nn.BatchNorm1d(2 * d_model),
            nn.ELU(inplace=True),
            nn.AvgPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(2 * d_model, 4 * d_model, 3, padding=1, padding_mode="circular"),
            nn.BatchNorm1d(4 * d_model),
            nn.ELU(inplace=True),
            nn.AvgPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(4 * d_model, 8 * d_model, 3, padding=1, padding_mode="circular"),
            nn.BatchNorm1d(8 * d_model),
            nn.ELU(inplace=True),
        )
        self.grn = GatedResidualConv(8 * d_model, gating="spatial")

        self.tf = TSequential(
            PositionalEncoder(d_model=8 * d_model, max_seq_len=600),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=8 * d_model,
                    nhead=8,
                    dim_feedforward=2048,
                    dropout=0.1,
                    activation="relu",
                ),
                num_layers=2,
                norm=nn.LayerNorm(8 * d_model),
            ),
        )

        self.repr = nn.Sequential(
            nn.Conv1d(8 * d_model, 16 * d_model, 3, padding=1, padding_mode="circular"),
            nn.BatchNorm1d(16 * d_model),
            nn.ELU(inplace=True),
            ECALayer(),
            nn.AvgPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(16 * d_model, 32 * d_model, 3, padding=1, padding_mode="circular"),
            nn.BatchNorm1d(32 * d_model),
        )
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32 * d_model, 16 * d_model),
            nn.Linear(16 * d_model, 61),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.grn(x)
        x = self.tf(x)
        x = self.repr(x)
        x = self.fc(x)
        return x

    def tfgrn(self, d_model, seq_len):
        grn = GatedResidualConv(d_model, gating="spatial")
        tf = TSequential(
            PositionalEncoder(d_model=d_model, max_seq_len=seq_len),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=8,
                    dim_feedforward=2048,
                    dropout=0.1,
                    activation="relu",
                ),
                num_layers=2,
                norm=nn.LayerNorm(d_model),
            ),
        )
        return nn.Sequential(grn, tf)
