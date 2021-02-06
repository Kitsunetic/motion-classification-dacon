import torch
import torch.nn as nn
import torch.nn.functional as F

Activation = nn.ELU


def cba(inchannels, channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv1d(inchannels, channels, kernel_size, stride, padding),
        nn.BatchNorm1d(channels),
        Activation(inplace=True),
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inchannels, channels, stride=1, groups=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(inchannels, channels, 3, stride=stride, padding=1, groups=groups),
            nn.BatchNorm1d(channels),
            Activation(),
            nn.Conv1d(channels, channels, 3, padding=1, groups=groups),
        )
        self.bn = nn.BatchNorm1d(channels)
        self.act = Activation()

        self.downsample = None
        if inchannels != channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(inchannels, channels, 1, stride=stride, groups=groups),
                nn.BatchNorm1d(channels),
            )

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
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
    expansion = 4

    def __init__(self, inchannels, channels, stride=1, groups=1):
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


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()

        # 18개의 채널은 각각 3개씩 {A가속, G가속, A속, G속, A위치, G위치} 6개로 나눠짐.
        # 그러므로 embedding conv도 6그룹으로 나눈다.
        # 이후 각각 나눠서 embedding된걸 합치기 위해서 CNN 층 하나 더 추가
        self.inchannels = 64  # 6의 배수여야 함
        """self.conv = nn.Sequential(
            cba(18, 36, 7, 1, 2, groups=6),
            cba(36, 36, 1, 1, 0),
            cba(36, 64, 3, 2, 1),
            nn.AvgPool1d(2),
        )"""
        self.conv = nn.Sequential(
            cba(18, 64, 7, 2, 3),
            nn.AvgPool1d(2),
        )

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.inchannels, 2048),
            nn.Dropout(0.1),
            nn.Linear(2048, 61),  # for total classification
            # nn.Linear(2048, 1),
            # nn.Sigmoid(),  # for ss only
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


"""
====================================================================
                       [2020] ResNeSt

https://github.com/zhanghang1989/ResNeSt.git
====================================================================
"""


class rSoftMax(nn.Module):
    # https://github.com/zhanghang1989/ResNeSt/blob/master/resnest/torch/splat.py
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        if self.radix > 1:
            batch = x.size(0)
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplAtConv1d(nn.Module):
    # https://github.com/zhanghang1989/ResNeSt/blob/master/resnest/torch/splat.py
    def __init__(
        self,
        in_channels,
        channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        radix=2,
        reduction_factor=4,
        norm_layer=None,
        **kwargs,
    ):
        super().__init__()

        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels

        self.conv = nn.Conv1d(
            in_channels,
            channels * radix,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=groups * radix,
            bias=bias,
            **kwargs,
        )
        self.bn0 = norm_layer(self.channels * radix)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv1d(self.channels, inter_channels, 1, groups=self.cardinality)
        self.bn1 = norm_layer(inter_channels)
        self.fc2 = nn.Conv1d(inter_channels, self.channels * radix, 1, groups=self.cardinality)
        self.rsoftmax = rSoftMax(radix, self.cardinality)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn0(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, int(rchannel // self.radix), dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool1d(gap, 1)
        gap = self.fc1(gap)
        gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1)

        if self.radix > 1:
            attens = torch.split(atten, int(rchannel // self.radix), dim=1)
            outs = []
            for att, split in zip(attens, splited):
                outs.append(att * split)
            out = sum(outs)
        else:
            out = atten * x

        return out.contiguous()


class ResNeStBottleneck(nn.Module):
    # https://github.com/zhanghang1989/ResNeSt/blob/master/resnest/torch/resnet.py
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        radix=1,
        cardinality=1,
        bottleneck_width=64,
        avd=False,
        avd_first=False,
        dilation=1,
        is_first=False,
        norm_layer=None,
        last_gamma=False,
    ):
        super().__init__()
        group_width = int(planes * (bottleneck_width / 64.0)) * cardinality

        self.conv1 = nn.Conv1d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool1d(3, stride, padding=1)
            stride = 1

        if radix >= 1:
            self.conv2 = SplAtConv1d(
                group_width,
                group_width,
                kernel_size=3,
                stride=stride,
                padding=1 * dilation,
                dilation=dilation,
                groups=cardinality,
                bias=False,
                radix=radix,
                norm_layer=norm_layer,
            )
        else:
            self.conv2 = nn.Conv1d(
                group_width,
                group_width,
                kernel_size=3,
                stride=stride,
                padding=1 * dilation,
                dilation=dilation,
                groups=cardinality,
                bias=False,
            )
            self.bn2 = norm_layer(group_width)

        self.conv3 = nn.Conv1d(group_width, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)

        if last_gamma:
            from torch.nn.init import zeros_

            zeros_(self.bn3.weight)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)
        if self.radix == 0:
            out = self.bn2(out)
            out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out


class ResNeSt1d(nn.Module):
    # https://github.com/zhanghang1989/ResNeSt/blob/master/resnest/torch/resnet.py
    def __init__(
        self,
        inchannels,
        block,
        layers,
        radix=1,
        groups=1,
        bottleneck_width=64,
        num_classes=1000,
        dilated=False,
        dilation=1,
        deep_stem=False,
        stem_width=64,
        avg_down=False,
        avd=False,
        avd_first=False,
        final_drop=0.0,
        last_gamma=False,
        norm_layer=nn.BatchNorm1d,
    ):
        super().__init__()

        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        # ResNeSt params
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first

        act = nn.ReLU

        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv1d(inchannels, stem_width, 3, 2, 1, bias=False),
                norm_layer(stem_width),
                act(inplace=True),
                nn.Conv1d(stem_width, stem_width, 3, 1, 1, bias=False),
                norm_layer(stem_width),
                act(inplace=True),
                nn.Conv1d(stem_width, self.inplanes, 3, 1, 1, bias=False),
            )
        else:
            self.conv1 = nn.Conv1d(inchannels, self.inplanes, 7, 2, 3, bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = act(inplace=True)
        self.maxpool = nn.MaxPool1d(3, 2, 1)

        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[2], stride=1, dilation=2, norm_layer=norm_layer)
        elif dilation == 2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=1, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[2], stride=1, dilation=2, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=1, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[2], stride=2, dilation=1, norm_layer=norm_layer)

        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None, is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool1d(stride, stride, ceil_mode=True, count_include_pad=False))
                else:
                    down_layers.append(nn.AvgPool1d(1, 1, ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv1d(self.inplanes, planes * block.expansion, 1, 1, 0, bias=False))
            else:
                down_layers.append(nn.Conv1d(self.inplanes, planes * block.expansion, 1, stride, 0, bias=False))

            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride,
                    downsample=downsample,
                    radix=self.radix,
                    cardinality=self.cardinality,
                    bottleneck_width=self.bottleneck_width,
                    avd=self.avd,
                    avd_first=self.avd_first,
                    dilation=1,
                    is_first=is_first,
                    norm_layer=norm_layer,
                    last_gamma=self.last_gamma,
                )
            )
        elif dilation == 4:
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride,
                    downsample=downsample,
                    radix=self.radix,
                    cardinality=self.cardinality,
                    bottleneck_width=self.bottleneck_width,
                    avd=self.avd,
                    avd_first=self.avd_first,
                    dilation=2,
                    is_first=is_first,
                    norm_layer=norm_layer,
                    last_gamma=self.last_gamma,
                )
            )
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    radix=self.radix,
                    cardinality=self.cardinality,
                    bottleneck_width=self.bottleneck_width,
                    avd=self.avd,
                    avd_first=self.avd_first,
                    dilation=dilation,
                    norm_layer=norm_layer,
                    last_gamma=self.last_gamma,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if self.drop:
            x = self.drop(x)
        x = self.fc(x)

        return x


def resnest50(inchannels, **kwargs):
    model = ResNeSt1d(
        inchannels,
        ResNeStBottleneck,
        [3, 4, 6, 3],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=32,
        avg_down=True,
        avd=True,
        avd_first=False,
        **kwargs,
    )
    return model


def resnest101(inchannels, **kwargs):
    model = ResNeSt1d(
        inchannels,
        ResNeStBottleneck,
        [3, 4, 23, 3],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=64,
        avg_down=True,
        avd=True,
        avd_first=False,
        **kwargs,
    )
    return model


def resnest200(inchannels, **kwargs):
    model = ResNeSt1d(
        inchannels,
        ResNeStBottleneck,
        [3, 24, 36, 3],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=64,
        avg_down=True,
        avd=True,
        avd_first=False,
        **kwargs,
    )
    return model


def resnest269(inchannels, **kwargs):
    model = ResNeSt1d(
        inchannels,
        ResNeStBottleneck,
        [3, 30, 48, 8],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=64,
        avg_down=True,
        avd=True,
        avd_first=False,
        **kwargs,
    )
    return model
