import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch_optimizer
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter

import networks
from utils import AccuracyMeter, AverageMeter, generate_experiment_directory

LOGDIR = Path("log")
RESULT_DIR = Path("results")
DATA_PATH = Path("data/0201.npz")
COMMENT = "resnet50-aug_shift240"

EXPATH, EXNAME = generate_experiment_directory(RESULT_DIR, COMMENT)

BATCH_SIZE = 256
NUM_CPUS = 8
EPOCHS = 200


class QDataset(TensorDataset):
    def __getitem__(self, idx):
        data = super(QDataset, self).__getitem__(idx)

        if len(data) != 1:
            x, y = data
            # TODO augmentation
            x = self.augmentation(x)
            return x, y
        else:
            # test
            return data

    def augmentation(self, x):
        # random shift
        x = self._aug_random_shift(x)

        return x

    @staticmethod
    def _aug_random_shift(x):
        """
        80% 확률로 랜덤하게 왼쪽/오른쪽으로 1~240만큼 shift한다.
        잘리는 부분은 버리고, 새로운 부분은 0으로 채움.
        """
        if random.random() >= 0.8:
            return x

        dist = random.randint(-240, 240)
        x = torch.roll(x, dist, dims=1)

        return x


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
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.25, patience=3, verbose=True, threshold=1e-8, cooldown=2)
        self.best_loss = math.inf
        self.earlystop_cnt = 0
        self.earlystop = False

        for epoch in range(1, num_epochs + 1):
            self.train_loop(dl_train)
            self.valid_loop(dl_valid)
            self.callback(epoch)

            if self.earlystop:
                break

    def train_loop(self, dl):
        self.model.train()

        self.loss, self.acc = AverageMeter(), AccuracyMeter()
        for x, y in dl:
            y_ = y.cuda()
            p = self.model(x.cuda())
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

        if self.earlystop_cnt > 10:
            print("[Early Stop] Stop training")
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

    data = np.load(DATA_PATH)
    X_train, Y_train, X_test = data["X_train"], data["Y_train"], data["X_test"]
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.int64)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    skf = StratifiedKFold(shuffle=True, random_state=143151)
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, Y_train), 1):
        ds_train = QDataset(X_train[train_idx], Y_train[train_idx])
        ds_valid = QDataset(X_train[valid_idx], Y_train[valid_idx])
        dl_kwargs = dict(batch_size=BATCH_SIZE, num_workers=NUM_CPUS, pin_memory=True)
        dl_train = DataLoader(ds_train, **dl_kwargs, shuffle=True)
        dl_valid = DataLoader(ds_valid, **dl_kwargs, shuffle=False)

        model = networks.ResNet50().cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch_optimizer.RAdam(model.parameters(), lr=1e-4)

        trainer = Trainer(model, criterion, optimizer, writer, EXNAME, EXPATH, fold)
        trainer.fit(dl_train, dl_valid, EPOCHS)

        # TODO submission 만들기

        break  # TODO 아직 KFold 안함


if __name__ == "__main__":
    main()
