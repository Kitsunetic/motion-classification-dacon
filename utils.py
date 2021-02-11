import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
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


def combine_submissions(dics, expath=None):
    tdic = {"id": dics[0]["id"].to_list()}

    vdic = np.zeros((len(dics[0]), 61), dtype=np.float32)
    for dic in dics:
        vdic += dic.to_numpy()[:, 1:]
    vdic /= len(dics)
    for i in range(61):
        tdic[str(i)] = vdic[:, i].tolist()

    tdic = pd.DataFrame(tdic)
    if expath is not None:
        dic.to_csv(Path(expath) / "submission-{self.exname}.csv", index=False)

    return tdic


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
