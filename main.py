import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch_optimizer
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import networks
from datasets import D0206_org_v4_4
from utils import AccuracyMeter, AverageMeter, convert_markdown, generate_experiment_directory

LOGDIR = Path("log")
RESULT_DIR = Path("results")
DATA_DIR = Path("data")
COMMENT = "ResNeSt50_fc-0206_org_v4_4"

EXPATH, EXNAME = generate_experiment_directory(RESULT_DIR, COMMENT)

BATCH_SIZE = 256
NUM_CPUS = 8
EPOCHS = 200


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
        for x, y in dl:
            y_ = y.cuda()
            p = self.model(x.cuda())
            # print(y_.shape, p.shape)
            loss = self.criterion(p, y_)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.loss.update(loss.item())
            self.acc.update(y_, p)
            ys.append(y)
            ps.append(p.detach().cpu())
        ys = torch.cat(ys)
        ps = torch.argmax(torch.cat(ps), dim=1)

        return ys, ps  # classification_report 때문에 순서 바뀌면 안됨

    def valid_loop(self, dl):
        self.model.eval()

        ys, ps = [], []
        self.val_loss, self.val_acc = AverageMeter(), AccuracyMeter()
        with torch.no_grad():
            for x, y in dl:
                y_ = y.cuda()
                p = self.model(x.cuda())
                loss = self.criterion(p, y_)

                self.val_loss.update(loss.item())
                self.val_acc.update(y_, p)
                ys.append(y)
                ps.append(p.cpu())
        ys = torch.cat(ys)
        ps = torch.argmax(torch.cat(ps), dim=1)

        return ys, ps  # classification_report 때문에 순서 바뀌면 안됨

    def callback(self, epoch, result_train, result_valid):
        foldded_epoch = self.fold * 1000 + epoch

        now = datetime.now()
        print(
            f"[{now.month:02d}:{now.day:02d}-{now.hour:02d}:{now.minute:02d} {epoch:03d}/{self.num_epochs:03d}:{self.fold}]",
            f"loss: {self.loss():.6f} acc: {self.acc()*100:.2f}%",
            f"val_loss: {self.val_loss():.6f} val_acc: {self.val_acc()*100:.2f}%",
        )

        # Tensorboard
        loss_scalars = {"loss": self.loss(), "val_loss": self.val_loss()}
        acc_scalars = {"acc": self.acc(), "val_acc": self.val_acc()}
        self.writer.add_scalars(self.exname + "/loss", loss_scalars, foldded_epoch)
        self.writer.add_scalars(self.exname + "/acc", acc_scalars, foldded_epoch)

        # Classification Report
        report_train = classification_report(*result_train, zero_division=0)
        report_valid = classification_report(*result_valid, zero_division=0)
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
            for x in dl:
                p = self.model(x.cuda()).cpu()
                ps.append(p)

            return torch.cat(ps)


def main():
    print(EXPATH)
    writer = SummaryWriter(LOGDIR)

    dl_list, dl_test = D0206_org_v4_4(DATA_DIR, BATCH_SIZE)
    for fold, dl_train, dl_valid in dl_list:
        # model = networks.LegacyResNet152().cuda()
        model = nn.Sequential(
            networks.resnest50(18, num_classes=1000),
            # nn.Dropout(0.2),  # dropout은 안하는게 더 좋다는 결론
            nn.Linear(1000, 61),
        ).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch_optimizer.RAdam(model.parameters())

        trainer = Trainer(model, criterion, optimizer, writer, EXNAME, EXPATH, fold)
        trainer.fit(dl_train, dl_valid, EPOCHS)

        # TODO submission 만들기
        break  # TODO 아직 KFold 안함
    # TODO submission 파일들 합치기


if __name__ == "__main__":
    main()
