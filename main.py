import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch_optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from utils import AccuracyMeter, AverageMeter, generate_experiment_directory

LOGDIR = Path("../log2")
RESULT_DIR = Path("../results")
DATA_DIR = Path("../data/M128_50000")
COMMENT = "M128-50000"

EXPATH, EXNAME = generate_experiment_directory(RESULT_DIR, COMMENT)

BATCH_SIZE = 256
NUM_CPUS = 8
EPOCHS = 200


class NPZDataset(Dataset):
    def __init__(self, npz_file, is_train=True):
        super(NPZDataset, self).__init__()

        self.is_train = is_train

        data = np.load(npz_file)
        if self.is_train:
            self.X = data["X_train"]
            self.Y = data["Y_train"]
        else:
            self.X = data["X_test"]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)

        # TODO augmentation

        if self.is_train:
            y = torch.tensor([self.Y[idx]], dtype=torch.float32)
            return x, y
        else:
            return x


class Trainer:
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer, writer: SummaryWriter, exname: str, expath: Path):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.writer = writer
        self.exname = exname
        self.expath = expath

    def fit(self, dl_train, dl_valid, num_epochs):
        self.num_epochs = num_epochs
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.25, patience=3, verbose=True, threshold=1e-8, cooldown=10)
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

        self.loss, self.acc, self.acc1 = AverageMeter(), AccuracyMeter(), AccuracyMeter(generosity=1)
        for x, y, k in dl:
            y_ = y.cuda()
            p = self.model(x.cuda())
            loss = self.criterion(p, y_)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.loss.update(loss.item())
            self.acc.update(y_, p)
            self.acc1.update(y_, p)

    def valid_loop(self, dl):
        self.model.eval()

        self.val_loss, self.val_acc, self.val_acc1 = AverageMeter(), AccuracyMeter(), AccuracyMeter(generosity=1)
        with torch.no_grad():
            for x, y, k in dl:
                y_ = y.cuda()
                p = self.model(x.cuda())
                loss = self.criterion(p, y_)

                self.val_loss.update(loss.item())
                self.val_acc.update(y_, p)
                self.val_acc1.update(y_, p)

    def callback(self, epoch):
        now = datetime.now()
        print(
            f"[{now.month:02d}:{now.day:02d}-{now.hour:02d}:{now.minute:02d} {epoch:03d}/{self.num_epochs:03d}]",
            f"loss: {self.loss():.6f} acc: {self.acc()*100:.1f}% acc1: {self.acc1()*100:.1f}",
            f"val_loss: {self.val_loss():.6f} val_acc: {self.val_acc()*100:.1f}% val_acc1: {self.val_acc1()*100:.1f}",
        )

        # Tensorboard
        loss_scalars = {"loss": self.loss(), "val_loss": self.val_loss()}
        acc_scalars = {"acc": self.acc(), "acc1": self.acc1(), "val_acc": self.val_acc(), "val_acc1": self.val_acc1()}
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
            ps, ys = [], []
            for x, y, k in dl:
                p = self.model(x.cuda()).cpu()
                ps.append(p)
                ys.append(y)

            return torch.cat(ps), torch.cat(ys)


def main():
    pass


if __name__ == "__main__":
    main()
