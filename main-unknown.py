import math
import random
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.model_selection import StratifiedKFold, KFold
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

warnings.filterwarnings("ignore")

import networks as ww
from utils import (
    AccuracyMeter,
    AverageMeter,
    FocalLoss,
    combine_submissions,
    generate_experiment_directory,
    strtime,
)

LOGDIR = Path("log-unknown")
RESULT_DIR = Path("results-unknown")
DATA_DIR = Path("data")
COMMENT = "ECATF3"

EXPATH, EXNAME = generate_experiment_directory(RESULT_DIR, COMMENT)

BATCH_SIZE = 256
NUM_CPUS = 8
EPOCHS = 300

DO_KFOLD = True
VAL_N_TTA = 10
TEST_N_TTA = 50

EARLYSTOP_PATIENCE = 10

NUM_FOLDS = 4
FOLD = 1

CORRECT_THRESHOLD = 0.6


class UniKLDiv(nn.Module):
    def __init__(self):
        super().__init__()

        uniform = torch.softmax(torch.zeros(1, 60, dtype=torch.float32), dim=1)
        self.register_buffer("uniform", uniform)

    def forward(self, pred):
        return torch.mean(-torch.log(torch.softmax(pred, dim=1) / self.uniform))


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        criterion_uni: nn.Module,
        optimizer,
        writer: SummaryWriter,
        exname: str,
        expath: Path,
        fold: int,
    ):
        self.model = model
        self.criterion = criterion
        self.criterion_uni = criterion_uni
        self.optimizer = optimizer
        self.writer = writer
        self.exname = exname
        self.expath = expath
        self.fold = fold

    def fit(self, dl_train1, dl_valid1, dl_train2, dl_valid2, num_epochs, start_epoch=1, checkpoint=None):
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            factor=0.5,
            patience=3,
            verbose=True,
            threshold=1e-8,
            cooldown=0,
        )
        self.earlystop_cnt = 0
        self.best_loss = math.inf
        self.epoch = start_epoch
        self.num_epochs = num_epochs
        self.dl_train1 = dl_train1
        self.dl_valid1 = dl_valid1
        self.dl_train2 = dl_train2
        self.dl_valid2 = dl_valid2

        if checkpoint is not None and Path(checkpoint).exists():
            print("Load state dict", checkpoint)
            ckpt = torch.load(checkpoint)
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.epoch = ckpt["epoch"]
            self.num_epochs = ckpt["num_epochs"]
            self.best_loss = ckpt["best_loss"]
            self.earlystop_cnt = ckpt["earlystop_cnt"]

        for self.epoch in range(start_epoch, num_epochs + 1):
            self.cepoch = f"{self.epoch:03d}/{self.num_epochs:03d}:{self.fold}"
            self.fepoch = self.fold * 1000 + self.epoch
            self.fsepoch = f"{self.fepoch:04d}"

            self.train_loop1()
            self.valid_loop1()
            self.train_loop2()
            self.valid_loop2()
            self.callback()

            if self.earlystop_cnt > EARLYSTOP_PATIENCE:
                print("[Early Stop] fold", self.fold)
                break

    def train_loop1(self):
        self.model.train()

        ys, ps = [], []
        _loss, _acc = AverageMeter(), AccuracyMeter()
        with tqdm(total=len(self.dl_train1.dataset), ncols=100, leave=False, desc=f"{self.cepoch} train1") as t:
            for x, y in self.dl_train1:
                x_, y_ = x.cuda(), y.cuda()
                p_ = self.model(x_)
                loss = self.criterion(p_, y_)

                # SAM
                loss.backward()
                self.optimizer.first_step(zero_grad=True)
                self.criterion(self.model(x_), y_).backward()
                self.optimizer.second_step(zero_grad=True)

                _loss.update(loss.item())
                _acc.update(y_, p_)
                ys.append(y)
                ps.append(p_.detach().cpu())

                t.set_postfix_str(f"loss:{loss.item():.6f} acc:{_acc():5.2f}%", refresh=False)
                t.update(len(y))

        self.tys = torch.cat(ys)
        self.tps = torch.cat(ps).softmax(dim=1)
        self.tloss1 = _loss()

        self.tacc1 = (self.tys == torch.argmax(self.tps, dim=1)).sum().item() / len(self.tys) * 100

    def train_loop2(self):
        self.model.train()

        ps = []
        _loss = AverageMeter()
        correct, numel = 0, 0
        with tqdm(total=len(self.dl_train2.dataset), ncols=100, leave=False, desc=f"{self.cepoch} train2") as t:
            for (x,) in self.dl_train2:
                x_ = x.cuda()
                p_ = self.model(x_)
                loss = self.criterion_uni(p_)

                # SAM
                loss.backward()
                self.optimizer.first_step(zero_grad=True)
                self.criterion_uni(self.model(x_)).backward()
                self.optimizer.second_step(zero_grad=True)

                _loss.update(loss.item())
                ps.append(p_.detach().cpu())

                correct += (torch.argmax(torch.softmax(p_, dim=1), dim=1) < CORRECT_THRESHOLD).sum().item()
                numel += len(x)

                t.set_postfix_str(f"loss:{loss.item():.6f}, acc:{correct/numel*100:5.2f}%", refresh=False)
                t.update(len(x))

        self.tus = torch.cat(ps).softmax(dim=1)
        self.tloss2 = _loss()
        self.tacc2 = correct / numel * 100

    @torch.no_grad()
    def valid_loop1(self):
        self.model.eval()

        pss = []
        _loss = AverageMeter()
        with tqdm(total=len(self.dl_valid1.dataset) * VAL_N_TTA, ncols=100, leave=False, desc=f"{self.cepoch} valid1") as t:
            for _ in range(VAL_N_TTA):
                ys, ps = [], []
                correct, numel = 0, 0
                for x, y in self.dl_valid1:
                    x_, y_ = x.cuda(), y.cuda()
                    p_ = self.model(x_)

                    loss = self.criterion(p_, y_)
                    _loss.update(loss.item())

                    ys.append(y)
                    ps.append(p_.cpu())

                    correct += (torch.argmax(p_, dim=1) == y_).sum().item()
                    numel += len(x)

                    t.set_postfix_str(f"loss:{loss.item():.6f}, acc:{correct/numel*100:5.2f}%", refresh=False)
                    t.update(len(y))

                pss.append(torch.cat(ps))

        self.vys = torch.cat(ys)
        self.vps = torch.stack(pss).softmax(dim=2).mean(dim=0)
        self.vloss1 = _loss()
        self.vacc1 = (self.vys == torch.argmax(self.vps, dim=1)).sum().item() / len(self.vys) * 100

    @torch.no_grad()
    def valid_loop2(self):
        self.model.eval()

        ps = []
        _loss = AverageMeter()
        correct, numel = 0, 0
        with tqdm(total=len(self.dl_valid2.dataset), ncols=100, leave=False, desc=f"{self.cepoch} valid2") as t:
            for (x,) in self.dl_valid2:
                x_ = x.cuda()
                p_ = self.model(x_)

                loss = self.criterion_uni(p_)
                _loss.update(loss.item())

                ps.append(p_.detach().cpu())

                correct += (torch.argmax(torch.softmax(p_, dim=1), dim=1) < CORRECT_THRESHOLD).sum().item()
                numel += len(x)

                t.set_postfix_str(f"loss: {loss.item():.6f}, acc:{correct/numel*100:5.2f}%", refresh=False)
                t.update(len(x))

        self.vus = torch.cat(ps).softmax(dim=1)
        self.vloss2 = _loss()
        self.vacc2 = correct / numel * 100

    @torch.no_grad()
    def callback(self):
        self.scheduler.step(self.vloss1)

        tas = torch.argmax(self.tps, dim=1)
        vas = torch.argmax(self.vps, dim=1)

        print(
            f"[{strtime()} {self.cepoch}:{self.fold}]",
            f"loss:{self.tloss1:.6f}:{self.vloss1:.6f}",
            f"acc1:{self.tacc1:5.2f}:{self.vacc1:5.2f}%",
            f"loss2:{self.tloss2:.6f}:{self.vloss2:.6f}",
            f"acc2:{self.tacc2:5.2f}:{self.vacc2:5.2f}%",
        )

        # Tensorboard
        loss1_scalars = {"tloss1": self.tloss1, "vloss1": self.vloss2}
        acc1_scalars = {"tacc1": self.tacc1, "vacc1": self.vacc1}
        loss2_scalars = {"tloss2": self.tloss1, "vloss2": self.vloss2}
        acc2_scalars = {"tacc2": self.tacc2, "vacc2": self.vacc2}
        self.writer.add_scalars(self.exname + "/loss1", loss1_scalars, self.fepoch)
        self.writer.add_scalars(self.exname + "/loss2", loss2_scalars, self.fepoch)
        self.writer.add_scalars(self.exname + "/acc1", acc1_scalars, self.fepoch)
        self.writer.add_scalars(self.exname + "/acc2", acc2_scalars, self.fepoch)

        # Classification Report
        self.classification_report(self.tys, tas, self.vys, vas)

        if self.best_loss - self.vloss1 > 1e-8:
            # Early Stop
            self.best_loss = self.vloss1
            self.earlystop_cnt = 0

            # Save Checkpoint
            ckpt = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": self.epoch,
                "num_epochs": self.num_epochs,
                "best_loss": self.best_loss,
                "earlystop_cnt": self.earlystop_cnt,
            }
            torch.save(ckpt, self.expath / f"best-ckpt-{self.fold}.pth")

            # Confusion Matrix
            if self.epoch > 5:
                self.confusion_matrix(self.tys, tas, self.vys, vas)
        else:
            self.earlystop_cnt += 1

    def classification_report(self, tys, tps, vys, vps):
        treport = classification_report(tys, tps, zero_division=0)
        vreport = classification_report(vys, vps, zero_division=0)
        with open(self.expath / f"classification_report-{self.fsepoch}.txt", "w") as f:
            f.write(treport)
            f.write("\r\n\r\n")
            f.write(vreport)

    def confusion_matrix(self, tys, tps, vys, vps):
        tcm, vcm = confusion_matrix(tys, tps), confusion_matrix(vys, vps)
        idx = [k for k in range(60)]
        tcm = pd.DataFrame(np.ma.masked_greater(tcm, 100), index=idx, columns=idx)
        vcm = pd.DataFrame(np.ma.masked_greater(vcm, 100), index=idx, columns=idx)
        plt.figure(figsize=(24, 20))
        sns.heatmap(tcm, annot=True, cbar=False)
        plt.tight_layout()
        plt.savefig(self.expath / f"cm-train-{self.fsepoch}.png")
        plt.close()
        plt.figure(figsize=(24, 20))
        sns.heatmap(vcm, annot=True, cbar=False)
        plt.tight_layout()
        plt.savefig(self.expath / f"cm-valid-{self.fsepoch}.png")
        plt.close()

    def load_best_checkpoint(self):
        ckpt_path = self.expath / f"best-ckpt-{self.fold}.pth"
        print("Load best checkpoint", ckpt_path)
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt["model"])

    @torch.no_grad()
    def evaluate(self, dl):
        self.model.eval()

        pss = []
        with tqdm(total=len(dl.dataset) * TEST_N_TTA, ncols=100, leave=False, desc="Submission") as t:
            for _ in range(TEST_N_TTA):
                ps = []
                for (x,) in dl:
                    ps.append(self.model(x.cuda()).cpu())
                    t.update(len(x))
                pss.append(torch.cat(ps))

        return torch.stack(pss).softmax(dim=2).mean(dim=0)

    @torch.no_grad()
    def submission(self, dl):
        self.load_best_checkpoint()
        ps = self.evaluate(dl)

        dic = defaultdict(list)
        for i, p in enumerate(ps, 3125):
            dic["id"].append(i)
            for j, v in enumerate(p):
                dic[str(j)].append(v.item())
        dic = pd.DataFrame(dic)

        submission_path = self.expath / f"submission-{self.exname}-{self.fold}.csv"
        print("Write submission to", submission_path)
        dic.to_csv(submission_path, index=False)

        return ps


@torch.no_grad()
def random_shift(x, p=0.5):
    if random.random() > p:
        return x

    shift = random.randint(0, 600)
    return torch.roll(x, shift, dims=1)


@torch.no_grad()
def random_sin(x, power=0.3):
    freqs = [100, 150, 200, 300, 600]
    wave = torch.sin(torch.tensor(list(range(600))) / random.sample(freqs, 1)[0] * math.pi)
    amplitude = random.random() * power
    signal = 1 + wave * amplitude
    return x * signal.reshape(1, -1)


@torch.no_grad()
def random_cos(x, power=0.3):
    freqs = [100, 150, 200, 300, 600]
    wave = torch.cos(torch.tensor(list(range(600))) / random.sample(freqs, 1)[0] * math.pi)
    amplitude = random.random() * power
    signal = 1 + wave * amplitude
    return x * signal.reshape(1, -1)


@torch.no_grad()
def random_gaussian(x, p=0.5, ksize=5, sigma=(0.0, 1.0)):
    if random.random() > p:
        return x

    sigma = random.random() * (sigma[1] - sigma[0]) + sigma[0]

    k = cv2.getGaussianKernel(ksize, sigma)
    k = torch.tensor(k, dtype=torch.float32)
    k = torch.repeat_interleave(k, 6, dim=1)
    k.transpose_(0, 1)
    k.unsqueeze_(1)

    psize = ksize // 2
    x.unsqueeze_(0)
    x = F.pad(x, [psize, psize], "circular")
    x = F.conv1d(x, k, groups=6)
    x.squeeze_(0)

    return x


class MyDataset(TensorDataset):
    def __init__(self, *tensors):
        super().__init__(*tensors)

    def __getitem__(self, index):
        items = super().__getitem__(index)

        x_total = items[0]
        x_total = self._augmentation(x_total)
        if len(items) == 2:
            y = items[1]

            return x_total, y
        else:
            return (x_total,)

    def _augmentation(self, x_total):
        x = x_total[:6]
        x_deriv = x_total[6:]
        x = random_shift(x)
        x = random_sin(x, power=0.5)
        x = random_cos(x, power=0.5)
        x = random_gaussian(x, ksize=3, sigma=(0.01, 0.2))
        x_total = torch.cat([x, x_deriv], dim=0)
        return x_total


def load_dataset():
    data = np.load(DATA_DIR / "unknown.npz")
    X_train1 = data["X_train1"]
    Y_train1 = data["Y_train1"]
    X_train2 = data["X_train2"]
    X_test = data["X_test"]

    X_train1 = torch.tensor(X_train1, dtype=torch.float32)
    Y_train1 = torch.tensor(Y_train1, dtype=torch.long)
    X_train2 = torch.tensor(X_train2, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    print(X_train1.shape, Y_train1.shape, X_train2.shape, X_test.shape)

    ds1 = MyDataset(X_train1, Y_train1)
    ds2 = MyDataset(X_train2)
    ds_test = MyDataset(X_test)
    dl_kwargs = dict(num_workers=6, pin_memory=True)
    dl_test = DataLoader(ds_test, **dl_kwargs, shuffle=False, batch_size=2 * BATCH_SIZE)

    kf1 = StratifiedKFold(n_splits=6, shuffle=True, random_state=261342)
    kf2 = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=261342)
    dl_list = []
    for (train_idx1, valid_idx1), (train_idx2, valid_idx2) in zip(kf1.split(X_train1, Y_train1), kf2.split(X_train2)):
        ds_train1 = Subset(ds1, train_idx1)
        ds_valid1 = Subset(ds1, valid_idx1)
        dl_train1 = DataLoader(ds_train1, **dl_kwargs, shuffle=True, batch_size=BATCH_SIZE)
        dl_valid1 = DataLoader(ds_valid1, **dl_kwargs, shuffle=False, batch_size=2 * BATCH_SIZE)
        ds_train2 = Subset(ds2, train_idx2)
        ds_valid2 = Subset(ds2, valid_idx2)
        dl_train2 = DataLoader(ds_train2, **dl_kwargs, shuffle=True, batch_size=BATCH_SIZE)
        dl_valid2 = DataLoader(ds_valid2, **dl_kwargs, shuffle=True, batch_size=BATCH_SIZE)
        dl_list.append((dl_train1, dl_valid1, dl_train2, dl_valid2))

    return dl_list, dl_test


def main():
    print(EXPATH)
    writer = SummaryWriter(LOGDIR)

    pss = []
    dl_list, dl_test = load_dataset()
    dl_train1, dl_valid1, dl_train2, dl_valid2 = dl_list[FOLD - 1]

    model = ww.ECATF().cuda()
    criterion = FocalLoss(gamma=2.0).cuda()
    criterion_uni = UniKLDiv().cuda()
    optimizer = ww.SAM(model.parameters(), AdamW, lr=0.0001)

    trainer = Trainer(model, criterion, criterion_uni, optimizer, writer, EXNAME, EXPATH, FOLD)
    trainer.fit(dl_train1, dl_valid1, dl_train2, dl_valid2, EPOCHS)
    pss.append(trainer.submission(dl_test))


if __name__ == "__main__":
    main()
