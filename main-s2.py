import matplotlib.pyplot as plt
import seaborn as sns
import math
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch_optimizer
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch_poly_lr_decay import PolynomialLRDecay

import networks as ww
from datasets import D0219
from utils import (
    AccuracyMeter,
    AverageMeter,
    ClassBalancedLoss,
    FocalLoss,
    combine_submissions,
    convert_markdown,
    generate_experiment_directory,
)

LOGDIR = Path("log")
RESULT_DIR = Path("results")
DATA_DIR = Path("data")
COMMENT = "ECATF-CB-D0219-B64-KFold4-TTA20-S2"

EXPATH, EXNAME = generate_experiment_directory(RESULT_DIR, COMMENT)

BATCH_SIZE = 64
NUM_CPUS = 8
EPOCHS = 100

DO_KFOLD = True
N_TTA = 20

EARLYSTOP_PATIENCE = 10


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

    def fit(self, dl_train, dl_valid, num_epochs):
        self.num_epochs = num_epochs
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            factor=0.5,
            patience=3,
            verbose=True,
            threshold=1e-8,
            cooldown=0,
        )
        # self.scheduler = CosineAnnealingLR(self.optimizer, T_max=20, eta_min=0, last_epoch=-1)
        """self.scheduler = PolynomialLRDecay(
            self.optimizer,
            max_decay_steps=len(dl_train) * num_epochs // 2,
            end_learning_rate=1e-6,
            power=0.9,
        )"""
        self.best_loss = math.inf

        self.earlystop_cnt = 0

        for epoch in range(1, num_epochs + 1):
            result_train = self.train_loop(dl_train)
            result_valid = self.valid_loop(dl_valid)
            self.callback(epoch, result_train, result_valid)

            if self.earlystop_cnt > EARLYSTOP_PATIENCE:
                print("[Early Stop]")
                break

    def train_loop(self, dl):
        self.model.train()

        ys, ps = [], []
        self.loss, self.aeloss, self.acc = AverageMeter(), AverageMeter(), AccuracyMeter()
        with tqdm(total=len(dl), ncols=100, leave=False) as t:
            for x, y in dl:
                x_, y_ = x.cuda(), y.cuda()
                p_ = self.model(x_)
                loss = self.criterion(p_, y_)

                # SAM
                loss.backward()
                self.optimizer.first_step(zero_grad=True)
                self.criterion(self.model(x_), y_).backward()
                self.optimizer.second_step(zero_grad=True)
                # self.scheduler.step()  # polynomial lr decay

                self.loss.update(loss.item())
                self.acc.update(y_, p_)
                ys.append(y)
                ps.append(p_.detach().cpu())

                t.set_postfix_str(f"loss: {loss.item():.6f} acc: {self.acc()*100:.2f}%", refresh=False)
                t.update()

        ys = torch.cat(ys)
        ps = torch.cat(ps)
        self.loss = self.loss()
        self.aeloss = self.aeloss()
        self.acc = self.acc()

        return ys, ps  # classification_report 때문에 순서 바뀌면 안됨

    @torch.no_grad()
    def valid_loop(self, dl):
        self.model.eval()

        pss = []
        self.val_loss, self.val_aeloss, self.val_acc = AverageMeter(), AverageMeter(), AccuracyMeter()
        with tqdm(total=len(dl) * N_TTA, ncols=100, leave=False) as t:
            for _ in range(N_TTA):
                ys, ps = [], []
                for x, y in dl:
                    x_, y_ = x.cuda(), y.cuda()
                    p_ = self.model(x_)

                    loss = self.criterion(p_, y_)
                    self.val_loss.update(loss.item())
                    self.val_acc.update(y_, p_)

                    ys.append(y)
                    ps.append(p_.cpu())
                    t.update()

                ps = torch.cat(ps)
                pss.append(ps)

        ys = torch.cat(ys)
        ps = torch.stack(pss).mean(dim=0)

        self.val_loss = self.val_loss()
        self.val_acc = self.val_acc()

        return ys, ps

    @torch.no_grad()
    def callback(self, epoch, result_train, result_valid):
        foldded_epoch = self.fold * 1000 + epoch
        ys_train, ps_train = result_train
        ys_valid, ps_valid = result_valid

        # LogLoss
        ll_train = log_loss(ys_train, torch.softmax(ps_train, dim=1))
        ll_valid = log_loss(ys_valid, torch.softmax(ps_valid, dim=1))

        now = datetime.now()
        print(
            f"[{now.month:02d}:{now.day:02d}-{now.hour:02d}:{now.minute:02d} {epoch:03d}/{self.num_epochs:03d}:{self.fold}]",
            f"loss: {self.loss:.6f}:{self.val_loss:.6f}",
            f"acc: {self.acc*100:.2f}:{self.val_acc*100:.2f}%",
            f"ll: {ll_train:.6f}:{ll_valid:.6f}",
        )

        # Tensorboard
        loss_scalars = {"loss": self.loss, "val_loss": self.val_loss}
        acc_scalars = {"acc": self.acc, "val_acc": self.val_acc}
        ll_scalars = {"ll": ll_train, "val_ll": ll_valid}
        self.writer.add_scalars(self.exname + "/loss", loss_scalars, foldded_epoch)
        self.writer.add_scalars(self.exname + "/acc", acc_scalars, foldded_epoch)
        self.writer.add_scalars(self.exname + "/ll", ll_scalars, foldded_epoch)

        # Classification Report
        qs_train = torch.argmax(ps_train, dim=1)
        qs_valid = torch.argmax(ps_valid, dim=1)
        report_train = classification_report(ys_train, qs_train, zero_division=0)
        report_valid = classification_report(ys_valid, qs_valid, zero_division=0)
        self.writer.add_text(self.exname + "/CR_train", convert_markdown(report_train), foldded_epoch)
        self.writer.add_text(self.exname + "/CR_valid", convert_markdown(report_valid), foldded_epoch)

        # LR scheduler
        # self.scheduler.step()
        self.scheduler.step(self.val_loss)

        # Early Stop
        if self.best_loss - self.val_loss > 1e-8:
            self.best_loss = self.val_loss
            self.earlystop_cnt = 0

            # Save Checkpoint
            ckpt = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "epoch": epoch}
            torch.save(ckpt, self.expath / f"best-ckpt-{self.fold}.pth")

            cm_train = pd.DataFrame(
                confusion_matrix(ys_train, qs_train),
                index=[k for k in range(61)],
                columns=[k for k in range(61)],
            )
            cm_valid = pd.DataFrame(
                confusion_matrix(ys_valid, qs_valid),
                index=[k for k in range(61)],
                columns=[k for k in range(61)],
            )
            plt.figure(figsize=(20, 20))
            sns.heatmap(cm_train, annot=True, cbar=False)
            plt.tight_layout()
            plt.savefig(self.expath / f"cm-train-epoch{epoch:03d}.png")
            plt.close()
            plt.figure(figsize=(20, 20))
            sns.heatmap(cm_valid, annot=True, cbar=False)
            plt.tight_layout()
            plt.savefig(self.expath / f"cm-valid-epoch{epoch:03d}.png")
            plt.close()
        else:
            self.earlystop_cnt += 1

        # Last Cross Entropy
        """if epoch == 20:
            self.criterion = FocalLoss(gamma=2.0)
        elif epoch == 40:
            self.criterion = nn.CrossEntropyLoss().cuda()"""

    @torch.no_grad()
    def evaluate(self, dl):
        self.model.eval()

        ps = []
        for (x,) in dl:
            p = self.model(x.cuda()).cpu()
            ps.append(p)

        return torch.cat(ps)

    @torch.no_grad()
    def submission(self, dl):
        # Load best checkpoint
        ckpt = torch.load(self.expath / f"best-ckpt-{self.fold}.pth")
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        pss = []
        for _ in range(N_TTA):
            pss.append(self.evaluate(dl))
        ps = torch.stack(pss)
        ps = torch.softmax(ps, dim=2).mean(dim=0)

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


def main():
    print(EXPATH)
    writer = SummaryWriter(LOGDIR)

    pss = []
    dl_ss_list, dl_ss_test, samples_per_cls = D0219(DATA_DIR, BATCH_SIZE, ssonly=True)
    dl_list, dl_test, samples_per_cls = D0219(DATA_DIR, BATCH_SIZE, ssonly=False)
    for fold, dl_train, dl_valid in dl_list:
        model = ww.ECATF().cuda()
        criterion = ClassBalancedLoss(samples_per_cls, 61, beta=0.9999, gamma=2.0).cuda()
        # criterion = FocalLoss(gamma=3.2).cuda()
        optimizer = ww.SAM(model.parameters(), AdamW, lr=0.0001)

        trainer = Trainer(model, criterion, optimizer, writer, EXNAME, EXPATH, fold)
        trainer.fit(dl_train, dl_valid, EPOCHS)
        pss += [trainer.submission(dl_test)]

        if not DO_KFOLD:
            break

    # submission 파일들 합치기
    if DO_KFOLD:
        combine_submissions(pss, EXPATH)


if __name__ == "__main__":
    main()