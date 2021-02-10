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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import networks
from datasets import D0201_v1, D0206_org_v4_4, D0206_org_v4_5
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
COMMENT = "TransformerModel_v4-AdamW-FocalLoss_gamma2.0-D0201_v1-B256"

EXPATH, EXNAME = generate_experiment_directory(RESULT_DIR, COMMENT)

BATCH_SIZE = 256
NUM_CPUS = 8
EPOCHS = 200

DO_KFOLD = True


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
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.25, patience=5, verbose=True, threshold=1e-8, cooldown=3)
        self.best_loss = math.inf
        self.earlystop_cnt = 0
        self.earlystop = False

        for epoch in range(1, num_epochs + 1):
            result_train = self.train_loop(dl_train)
            result_valid = self.valid_loop(dl_valid)
            self.callback(epoch, result_train, result_valid)

            if self.earlystop:
                break

    def train_loop(self, dl):
        self.model.train()

        ys, ps = [], []
        self.loss, self.acc = AverageMeter(), AccuracyMeter()
        with tqdm(total=len(dl), ncols=100, leave=False) as t:
            for x, y in dl:
                y_ = y.cuda()
                p = self.model(x.cuda())
                # print(y_.shape, p.shape)
                loss = self.criterion(p, y_)
                self.optimizer.zero_grad()
                loss.backward()
                # clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                self.loss.update(loss.item())
                self.acc.update(y_, p)
                ys.append(y)
                ps.append(p.detach().cpu())

                t.set_postfix_str(f"loss: {loss.item():.6f} acc: {self.acc()*100:.2f}%", refresh=False)
                t.update()

        ys = torch.cat(ys)
        ps = torch.cat(ps)
        ps_ = torch.argmax(ps, dim=1)

        return ys, ps_, ps  # classification_report 때문에 순서 바뀌면 안됨

    def valid_loop(self, dl):
        self.model.eval()

        ys, ps = [], []
        self.val_loss, self.val_acc = AverageMeter(), AccuracyMeter()
        with torch.no_grad():
            with tqdm(total=len(dl), ncols=100, leave=False) as t:
                for x, y in dl:
                    y_ = y.cuda()
                    p = self.model(x.cuda())
                    loss = self.criterion(p, y_)

                    self.val_loss.update(loss.item())
                    self.val_acc.update(y_, p)
                    ys.append(y)
                    ps.append(p.cpu())

                    t.set_postfix_str(f"val_loss: {loss.item():.6f} val_acc: {self.val_acc()*100:.2f}%", refresh=False)
                    t.update()

        ys = torch.cat(ys)
        ps = torch.cat(ps)
        ps_ = torch.argmax(ps, dim=1)

        return ys, ps_, ps  # classification_report 때문에 순서 바뀌면 안됨

    def callback(self, epoch, result_train, result_valid):
        foldded_epoch = self.fold * 1000 + epoch

        # LogLoss
        with torch.no_grad():
            ll_train = log_loss(result_train[0], result_train[2])
            ll_valid = log_loss(result_valid[0], result_valid[2])

        now = datetime.now()
        print(
            f"[{now.month:02d}:{now.day:02d}-{now.hour:02d}:{now.minute:02d} {epoch:03d}/{self.num_epochs:03d}:{self.fold}]",
            f"loss: {self.loss():.6f} acc: {self.acc()*100:.2f}% ll: {ll_train:.6f}",
            f"val_loss: {self.val_loss():.6f} val_acc: {self.val_acc()*100:.2f}%  val_ll: {ll_valid:.6f}",
        )

        # Tensorboard
        loss_scalars = {"loss": self.loss(), "val_loss": self.val_loss()}
        acc_scalars = {"acc": self.acc(), "val_acc": self.val_acc()}
        ll_scalars = {"ll": ll_train, "val_ll": ll_valid}
        self.writer.add_scalars(self.exname + "/loss", loss_scalars, foldded_epoch)
        self.writer.add_scalars(self.exname + "/acc", acc_scalars, foldded_epoch)
        self.writer.add_scalars(self.exname + "/ll", ll_scalars, foldded_epoch)

        # Classification Report
        report_train = classification_report(result_train[0], result_train[1], zero_division=0)
        report_valid = classification_report(result_valid[0], result_valid[1], zero_division=0)
        self.writer.add_text(self.exname + "/CR_train", convert_markdown(report_train), foldded_epoch)
        self.writer.add_text(self.exname + "/CR_valid", convert_markdown(report_valid), foldded_epoch)

        # LR scheduler
        self.scheduler.step(self.val_loss())

        # Early Stop
        if self.best_loss - self.val_loss() > 1e-8:
            self.best_loss = self.val_loss()
            self.earlystop_cnt = 0

            # Save Checkpoint
            torch.save(self.model.state_dict(), self.expath / f"best-ckpt-{self.fold}.pth")
        else:
            self.earlystop_cnt += 1

        if self.earlystop_cnt > 20:
            print(f"[Early Stop:{self.fold}] Stop training")
            self.earlystop = True

    def evaluate(self, dl):
        self.model.eval()

        with torch.no_grad():
            ps = []
            for (x,) in dl:
                p = self.model(x.cuda()).cpu()
                ps.append(p)

            return torch.cat(ps)

    def submission(self, dl):
        ps = self.evaluate(dl)
        ps = torch.softmax(ps, dim=1)
        dic = defaultdict(list)
        for i, p in enumerate(ps, 3125):
            dic["id"].append(i)
            for j, v in enumerate(p):
                dic[str(j)].append(v.item())
        dic = pd.DataFrame(dic)

        submission_path = self.expath / f"submission{self.fold}.csv"
        print("Write submission to", submission_path)
        dic.to_csv(submission_path, index=False)

        return dic


def main():
    print(EXPATH)
    writer = SummaryWriter(LOGDIR)

    dics = []
    dl_list, dl_test = D0206_org_v4_4(DATA_DIR, BATCH_SIZE)
    # dl_list, dl_test = D0201_v1(DATA_DIR, BATCH_SIZE)
    for fold, dl_train, dl_valid in dl_list:
        model = networks.TransformerModel_v3().cuda()
        # criterion = ClassBalancedLoss(, 61, beta=0.9999, gamma=2.0)
        criterion = FocalLoss()  # TODO gamma 키워서?
        optimizer = AdamW(model.parameters(), lr=1e-4)

        trainer = Trainer(model, criterion, optimizer, writer, EXNAME, EXPATH, fold)
        trainer.fit(dl_train, dl_valid, EPOCHS)
        dics += [trainer.submission(dl_test)]

        if not DO_KFOLD:
            break

    # submission 파일들 합치기
    if DO_KFOLD:
        combine_submissions(dics, EXPATH)


if __name__ == "__main__":
    main()
