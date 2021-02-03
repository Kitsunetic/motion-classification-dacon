import math
import random
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import Subset
import torch_optimizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
import matplotlib.pyplot as plt

import networks
from utils import AccuracyMeter, AverageMeter, generate_experiment_directory

LOGDIR = Path("log")
RESULT_DIR = Path("results")
DATA_PATH = Path("data/0203")
COMMENT = "resnet152-os2"

EXPATH, EXNAME = generate_experiment_directory(RESULT_DIR, COMMENT)

BATCH_SIZE = 256
NUM_CPUS = 8
EPOCHS = 200


class QDataset(Dataset):
    def __init__(self, files: List[Path]):
        super().__init__()

        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        x = data["x"]
        y = data["y"]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int64)

        return x, y


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
            self.train_loop(dl_train)
            self.valid_loop(dl_valid)
            self.callback(epoch)

            if self.earlystop:
                self.report(dl_valid)
                break

    def train_loop(self, dl):
        self.model.train()

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

    def valid_loop(self, dl):
        self.model.eval()

        self.val_loss, self.val_acc = AverageMeter(), AccuracyMeter()
        with torch.no_grad():
            for x, y in dl:
                y_ = y.cuda()
                p = self.model(x.cuda())
                loss = self.criterion(p, y_)

                self.val_loss.update(loss.item())
                self.val_acc.update(y_, p)

    def callback(self, epoch):
        now = datetime.now()
        print(
            f"[{now.month:02d}:{now.day:02d}-{now.hour:02d}:{now.minute:02d} {epoch:03d}/{self.num_epochs:03d}:{self.fold}]",
            f"loss: {self.loss():.6f} acc: {self.acc()*100:.2f}%",
            f"val_loss: {self.val_loss():.6f} val_acc: {self.val_acc()*100:.2f}%",
        )

        # Tensorboard
        loss_scalars = {"loss": self.loss(), "val_loss": self.val_loss()}
        acc_scalars = {"acc": self.acc(), "val_acc": self.val_acc()}
        self.writer.add_scalars(self.exname + "/loss", loss_scalars, epoch)
        self.writer.add_scalars(self.exname + "/acc", acc_scalars, epoch)

        # LR scheduler
        self.scheduler.step(self.val_loss())

        # Early Stop
        if self.best_loss - self.val_loss() > 1e-8:
            self.best_loss = self.val_loss()
            self.earlystop_cnt = 0

            # Save Checkpoint
            torch.save(self.model.state_dict(), self.expath / "best-ckpt.pth")
        else:
            self.earlystop_cnt += 1

        if self.earlystop_cnt > 20:
            print("[Early Stop] Stop training")
            self.earlystop = True

    def report(self, dl):
        # 가장 좋은 checkpoint 로딩
        ckpt = torch.load(self.expath / "best-ckpt.pth")
        self.model.load_state_dict(ckpt)
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

        Y = torch.cat(ys).numpy()
        P = torch.cat(ps)
        P = torch.argmax(P, dim=1).numpy()

        # DEBUG: Y, P txt파일로 저장
        with open(self.expath / "YP.txt", "w") as f:
            for y, p in zip(Y, P):
                f.write(f"{y} {p}\r\n")

        # loss, acc
        self.writer.add_text(
            self.exname + "/best",
            f"loss: {self.val_loss():.6f}, acc: {self.val_acc()*100:.2f}%",
        )

        # classification report
        cr = classification_report(Y, P, labels=list(range(61)))
        cr = cr.replace("\n", "<br>").replace(" ", "&nbsp;")
        self.writer.add_text(self.exname + "/classification_report", cr)

        # confusion matrix
        cm = confusion_matrix(Y, P, labels=list(range(61)))
        fig = plt.figure(figsize=(24, 20))
        sns.heatmap(cm, annot=True, fmt="g", cmap="Blues", cbar=False)
        plt.xlabel("pred")
        plt.ylabel("real")
        plt.title(EXNAME)
        self.writer.add_figure(self.exname + "/confusion_matrix", fig)

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

    legacy_files, legacy_labels = [], []
    for i in range(61):
        for f in (DATA_PATH / "train" / f"{i:02d}").glob("*.npz"):
            legacy_files.append(f)
            legacy_labels.append(i)
    test_files = list((DATA_PATH / "test").glob("*.npz"))
    ds_test = QDataset(test_files)

    skf = StratifiedKFold(shuffle=True, random_state=143151)
    for fold, (train_idx, valid_idx) in enumerate(skf.split(legacy_files, legacy_labels), 1):
        # 0203 train은 aug된거까지 로드. valid는 원본 데이터(legacy)만 로드
        train_files = []
        for i in train_idx:
            train_files.append(legacy_files[i])

            file_id = legacy_files[i].name[:-4]
            aug_dir = DATA_PATH / "train_aug" / f"{legacy_labels[i]:02d}"
            train_files.extend(aug_dir.glob(f"{file_id}_*.npz"))
        valid_files = [legacy_files[i] for i in valid_idx]

        ds_train = QDataset(train_files)
        ds_valid = QDataset(valid_files)
        dl_kwargs = dict(batch_size=BATCH_SIZE, num_workers=NUM_CPUS, pin_memory=True)
        dl_train = DataLoader(ds_train, **dl_kwargs, shuffle=True)
        dl_valid = DataLoader(ds_valid, **dl_kwargs, shuffle=False)

        model = networks.ResNet152().cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch_optimizer.RAdam(model.parameters(), lr=1e-3)  # TODO 1e-3??

        trainer = Trainer(model, criterion, optimizer, writer, EXNAME, EXPATH, fold)
        trainer.fit(dl_train, dl_valid, EPOCHS)

        # TODO submission 만들기

        break  # TODO 아직 KFold 안함


if __name__ == "__main__":
    main()
