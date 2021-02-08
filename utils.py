import os

import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed, submission=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = submission
        torch.backends.cudnn.benchmark = not submission


def pprint_args(args, title="", printing=True):
    d = args.__dict__
    keylens = tuple(map(lambda k: len(k), d.keys()))
    keylen = max(keylens) + 1

    result = "Arguments: " + title
    for k, v in d.items():
        k += " " * (keylen - len(k))
        result += f"\r\n - {k}: {v}"

    if printing:
        print(result)

    return result


def generate_experiment_directory(base_dir, comment=None):
    base_dir = Path(base_dir)
    experiment_id = 0
    if base_dir.exists():
        for d in sorted(list(base_dir.iterdir()), reverse=True):
            if d.is_dir() and d.name[:4].isdigit():  # and len(list(d.iterdir())) > 0:
                experiment_id = int(d.name[:4]) + 1
                break

    dirname = f"{experiment_id:04d}"
    if comment:
        dirname += f"-{comment}"
    dirpath = base_dir / dirname
    dirpath.mkdir(parents=True, exist_ok=True)

    return dirpath, dirpath.name


def convert_markdown(text):
    return text.replace("\n", "<br>").replace(" ", "&nbsp;")


class AverageMeter(object):
    """
    AverageMeter, referenced to https://dacon.io/competitions/official/235626/codeshare/1684
    """

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.losses = []

    def update(self, val, n=1):
        self.losses.append(val)
        if n > 0:
            self.sum += val * n
            self.cnt += n
            self.avg = self.sum / self.cnt

    def get(self):
        return self.avg

    def __call__(self):
        return self.avg


class AccuracyMeter:
    def __init__(self):
        self.correct = 0
        self.numel = 0
        self.acc = 0

    def update(self, y, p):
        pp = torch.argmax(p, dim=1)
        self.correct += (y == pp).sum().item()
        self.numel += y.shape[0]

        if self.numel > 0:
            self.acc = self.correct / self.numel

    def __call__(self):
        return self.acc


class BinaryAccuracyMeter:
    def __init__(self):
        self.correct = 0
        self.numel = 0
        self.acc = 0

    def update(self, y, p):
        pp = torch.round(p)
        self.correct += (y == pp).sum().item()
        self.numel += y.shape[0]

        if self.numel > 0:
            self.acc = self.correct / self.numel

    def __call__(self):
        return self.acc


class FocalLoss(nn.Module):
    """
    Referenced to
    - https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html
    - https://github.com/DingKe/pytorch_workplace/blob/master/focalloss/loss.py
    - https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/4
    - https://github.com/Kitsunetic/focal_loss_pytorch
    """

    def __init__(self, gamma=0, eps=1e-6, reduction="mean"):
        assert reduction in ["mean", "sum"], f"reduction should be mean or sum not {reduction}."
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        p = F.softmax(input, dim=-1)
        focal_weight = (1 - p) ** self.gamma
        loss = F.nll_loss(focal_weight * torch.log(p), target)

        return loss
