import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

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


class TransformerModel_v3(nn.Module):
    def __init__(self, activation=nn.ReLU, norm=nn.InstanceNorm1d):
        super().__init__()

        self.conv1 = nn.Sequential(
            SGCNN(18, 128, activation, norm),
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


class FocalLoss(nn.Module):
    """
    Referenced to
    - https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html
    - https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/4
    - https://github.com/DingKe/pytorch_workplace/blob/master/focalloss/loss.py
    - https://github.com/Kitsunetic/focal_loss_pytorch
    """

    def __init__(self, gamma=2.0, eps=1e-6, reduction="mean"):
        assert reduction in ["mean", "sum"], f"reduction should be mean or sum not {reduction}."
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        plog = F.log_softmax(input, dim=-1)  # adjust log directly could occurs gradient exploding???
        p = torch.exp(plog)
        focal_weight = (1 - p) ** self.gamma
        loss = F.nll_loss(focal_weight * plog, target)

        return loss


class ClassBalancedLoss(nn.Module):
    """
    Referenced to
    - "Class-Balanced Loss Based on Effective Number of Samples"
    - https://github.com/vandit15/Class-balanced-loss-pytorch/blob/master/class_balanced_loss.py
    """

    def __init__(self, samples_per_cls, no_of_classes, loss_type="focal", beta=0.9999, gamma=2.0):
        loss_types = ["focal", "sidmoid", "softmax"]
        assert loss_type.lower() in loss_types, f"loss_type must be one of {loss_types} not {loss_type}"

        if not isinstance(samples_per_cls, torch.Tensor):
            samples_per_cls = torch.tensor(samples_per_cls, dtype=torch.float32)

        super().__init__()

        self.effective_num = 1.0 - beta ** samples_per_cls
        self.weights = (1.0 - beta) / self.effective_num
        # TODO no_of_classes를 왜 곱하는지??
        self.weights = self.weights / torch.sum(self.weights) * no_of_classes
        self.weights.unsqueeze_(0)

        self.no_of_classes = no_of_classes
        self.loss_type = loss_type.lower()
        self.beta = beta
        self.gamma = gamma

    def forward(self, input, target):
        one_hot = F.one_hot(target, self.no_of_classes).float()

        weights = self.weights.repeat(one_hot.size(0), 1) * one_hot
        weights = weights.sum(1).unsqueeze(1)
        weights = weights.repeat(1, self.no_of_classes)

        if self.loss_type == "focal":
            cb_loss = self.focal_loss(one_hot, input, weights, self.gamma)
        elif self.loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input, one_hot, weight=self.weights)
        elif self.loss_type == "softmax":
            pred = torch.softmax(input, dim=1)
            cb_loss = F.binary_cross_entropy(pred, one_hot, weight=weights)
        else:
            raise f"Unknown loss_type {self.loss_type}"

        return cb_loss

    @staticmethod
    def focal_loss(labels, logits, alpha, gamma):
        """Compute the focal loss between `logits` and the ground truth `labels`.
        Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
        where pt is the probability of being classified to the true class.
        pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
        Args:
        labels: A float tensor of size [batch, num_classes].
        logits: A float tensor of size [batch, num_classes].
        alpha: A float tensor of size [batch_size]
            specifying per-example weight for balanced cross entropy.
        gamma: A float scalar modulating loss from hard and easy examples.
        Returns:
        focal_loss: A float32 scalar representing normalized total loss.
        """
        BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

        if gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

        loss = modulator * BCLoss

        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)

        focal_loss /= torch.sum(labels)
        return focal_loss
