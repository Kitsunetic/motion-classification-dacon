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
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
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
    ClassBalancedLoss,
    FocalLoss,
    combine_submissions,
    generate_experiment_directory,
    strtime,
)

LOGDIR = Path("log3/jamm")
RESULT_DIR = Path("results3/jamm")
DATA_DIR = Path("data")
COMMENT = ""

EXPATH, EXNAME = generate_experiment_directory(RESULT_DIR, COMMENT)

BATCH_SIZE = 512
NUM_CPUS = 8
EPOCHS = 300

VAL_N_TTA = 1
TEST_N_TTA = 1

EARLYSTOP_PATIENCE = 50


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer,
        writer: SummaryWriter,
        exname: str,
        expath: Path,
        fold: int,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.writer = writer
        self.exname = exname
        self.expath = expath
        self.fold = fold

    def fit(self, dl_train, dl_valid, num_epochs, start_epoch=1, checkpoint=None):
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            factor=0.5,
            patience=5,
            verbose=True,
            threshold=1e-8,
            cooldown=0,
        )
        self.earlystop_cnt = 0
        self.best_loss = math.inf
        self.epoch = start_epoch
        self.num_epochs = num_epochs
        self.dl_train = dl_train
        self.dl_valid = dl_valid

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

            self.train_loop()
            self.valid_loop()
            self.callback()

            if self.earlystop_cnt > EARLYSTOP_PATIENCE:
                print("[Early Stop] fold", self.fold)
                break

    def train_loop(self):
        self.model.train()

        ys, ps = [], []
        _loss, _acc = AverageMeter(), AccuracyMeter()
        with tqdm(total=len(self.dl_train.dataset), ncols=100, leave=False, desc=f"{self.cepoch} train") as t:
            for x, y in self.dl_train:
                x_, y_ = x.cuda(), y.cuda()
                p_ = self.model(x_)
                loss = self.criterion(p_, y_)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # SAM
                """loss.backward()
                clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.first_step(zero_grad=True)
                self.criterion(self.model(x_), y_).backward()
                clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.second_step(zero_grad=True)"""

                _loss.update(loss.item())
                _acc.update(y_, p_)
                ys.append(y)
                ps.append(p_.detach().cpu())

                t.set_postfix_str(f"loss:{loss.item():.6f} acc:{_acc():.2f}%", refresh=False)
                t.update(len(y))

        self.tys = torch.cat(ys)
        self.tps = torch.cat(ps).softmax(dim=1)
        self.tloss = _loss()

        self.tacc = (self.tys == torch.argmax(self.tps, dim=1)).sum().item() / len(self.tys) * 100

    @torch.no_grad()
    def valid_loop(self):
        self.model.eval()

        pss = []
        _loss = AverageMeter()
        with tqdm(total=len(self.dl_valid.dataset) * VAL_N_TTA, ncols=100, leave=False, desc=f"{self.cepoch} valid") as t:
            for _ in range(VAL_N_TTA):
                ys, ps = [], []
                correct, numel = 0, 0
                for x, y in self.dl_valid:
                    x_, y_ = x.cuda(), y.cuda()
                    p_ = self.model(x_)

                    loss = self.criterion(p_, y_)
                    _loss.update(loss.item())

                    ys.append(y)
                    ps.append(p_.cpu())

                    correct += (torch.argmax(p_, dim=1) == y_).sum().item()
                    numel += len(x)

                    t.set_postfix_str(f"loss:{loss.item():.6f} acc:{correct/numel*100:.2f}", refresh=False)
                    t.update(len(y))

                pss.append(torch.cat(ps))

        self.vys = torch.cat(ys)
        self.vps = torch.stack(pss).softmax(dim=2).mean(dim=0)
        self.vloss = _loss()
        self.vacc = (self.vys == torch.argmax(self.vps, dim=1)).sum().item() / len(self.vys) * 100

    @torch.no_grad()
    def callback(self):
        tas = torch.argmax(self.tps, dim=1)
        vas = torch.argmax(self.vps, dim=1)

        np.savez_compressed("test.npz", tas=tas.numpy(), vas=vas.numpy(), tys=self.tys.numpy(), vys=self.vys.numpy())

        # 26번에 대해 accuracy 따로 구하기
        len_tys26 = (self.tys == 26).sum().item()
        len_vys26 = (self.vys == 26).sum().item()
        tacc26 = ((tas == 26) * (self.tys == 26)).sum().item() / len_tys26 * 100
        taccoo = ((tas != 26) * (tas == self.tys)).sum().item() / (len(self.tys) - len_tys26) * 100
        vacc26 = ((vas == 26) * (self.vys == 26)).sum().item() / len_vys26 * 100
        vaccoo = ((vas != 26) * (vas == self.vys)).sum().item() / (len(self.vys) - len_vys26) * 100

        # LogLoss
        tll = log_loss(self.tys, self.tps)
        vll = log_loss(self.vys, self.vps)
        print(
            f"[{strtime()} {self.cepoch}:{self.fold}]",
            f"loss:{self.tloss:.6f}:{self.vloss:.6f}",
            f"acc:{self.tacc:.2f}:{self.vacc:.2f}%",
            f"acc26:{tacc26:.2f}:{vacc26:.2f}%",
            f"accoo:{taccoo:.2f}:{vaccoo:.2f}%",
            f"ll:{tll:.6f}:{vll:.6f}",
        )

        # Tensorboard
        loss_scalars = {"tloss": self.tloss, "vloss": self.vloss}
        acc_scalars = {"tacc": self.tacc, "vacc": self.vacc}
        ll_scalars = {"tll": tll, "vll": vll}
        self.writer.add_scalars(self.exname + "/loss", loss_scalars, self.fepoch)
        self.writer.add_scalars(self.exname + "/acc", acc_scalars, self.fepoch)
        self.writer.add_scalars(self.exname + "/ll", ll_scalars, self.fepoch)

        # Classification Report
        # self.classification_report(self.tys, tas, self.vys, vas)

        self.scheduler.step(vll)

        if self.best_loss - vll > 1e-8:
            # Early Stop
            self.best_loss = vll
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
            """if self.epoch > 5:
                self.confusion_matrix(self.tys, tas, self.vys, vas)"""
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
        idx = [k for k in range(61)]
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
    #if random.random() > p:
    #    return x

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

        x = items[0]
        y = items[1]
        if y.item() != 26:
            x = self._augmentation(x)

        return x, y

    def _augmentation(self, x):
        x = random_shift(x)
        # x = random_sin(x, power=0.7)
        # x = random_cos(x, power=0.7)
        # x = random_gaussian(x, ksize=3, sigma=(0.01, 1))
        return x


def load_dataset():
    data = np.load(DATA_DIR / "jamm.npz")
    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_test = data["X_test"]

    X_train = torch.tensor(X_train, dtype=torch.float32).transpose(1, 2)
    Y_train = torch.tensor(Y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32).transpose(1, 2)
    print(X_train.shape, Y_train.shape, X_test.shape)

    # samples_per_cls
    samples_per_cls = [(Y_train == i).sum().item() for i in range(61)]
    print(samples_per_cls)

    ds = MyDataset(X_train, Y_train)
    ds_test = TensorDataset(X_test)
    dl_kwargs = dict(num_workers=6, pin_memory=True)
    dl_test = DataLoader(ds_test, **dl_kwargs, shuffle=False, batch_size=2 * BATCH_SIZE)

    skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=261342)
    dl_list = []
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, Y_train), 1):
        ds_train = Subset(ds, train_idx)
        ds_valid = Subset(ds, valid_idx)
        dl_train = DataLoader(ds_train, **dl_kwargs, shuffle=True, batch_size=BATCH_SIZE)
        dl_valid = DataLoader(ds_valid, **dl_kwargs, shuffle=False, batch_size=2 * BATCH_SIZE)
        dl_list.append((fold, dl_train, dl_valid))

    return dl_list, dl_test, samples_per_cls


def main():
    print(EXPATH)
    writer = SummaryWriter(LOGDIR)

    pss = []
    dl_list, dl_test, samples_per_cls = load_dataset()
    for fold, dl_train, dl_valid in dl_list:
        # model = ww.ECATF().cuda()
        model = ww.SimpleCNN().cuda()
        # criterion = FocalLoss(gamma=2.4).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = AdamW(model.parameters(), lr=0.001)
        # optimizer = ww.SAM(model.parameters(), AdamW, lr=0.001)

        trainer = Trainer(model, criterion, optimizer, writer, EXNAME, EXPATH, fold)
        trainer.fit(dl_train, dl_valid, EPOCHS)
        pss.append(trainer.submission(dl_test))

    # submission 파일들 합치기
    combine_submissions(pss, EXPATH)


if __name__ == "__main__":
    main()
