import math
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.dataset import Subset


def D0206_org(data_dir, batch_size) -> Tuple[List[Tuple[int, DataLoader, DataLoader]], DataLoader]:
    data = np.load(data_dir / "0206_org.npz")
    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_test = data["X_test"]

    X_train = tensor(X_train, dtype=torch.float32)
    Y_train = tensor(Y_train, dtype=torch.long)
    X_test = tensor(X_test, dtype=torch.float32)
    print(X_train.shape, Y_train.shape, X_test.shape)

    ds = TensorDataset(X_train, Y_train)
    ds_test = TensorDataset(X_test)
    dl_kwargs = dict(batch_size=batch_size, num_workers=6, pin_memory=True)
    dl_test = DataLoader(ds_test, **dl_kwargs, shuffle=False)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=261342)
    dl_list = []
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, Y_train), 1):
        ds_train = Subset(ds, train_idx)
        ds_valid = Subset(ds, valid_idx)
        dl_train = DataLoader(ds_train, **dl_kwargs, shuffle=True)
        dl_valid = DataLoader(ds_valid, **dl_kwargs, shuffle=False)
        dl_list.append((fold, dl_train, dl_valid))

    return dl_list, dl_test


def random_shift(x):
    shift = random.randint(0, 600)
    return torch.roll(x, shift, dims=1)


def random_sin(x, power=0.3):
    freqs = [100, 150, 200, 300, 600]
    wave = torch.sin(torch.tensor(list(range(600))) / random.sample(freqs, 1)[0] * math.pi)
    amplitude = random.random() * power
    signal = 1 + wave * amplitude
    return x * signal.reshape(1, -1)


def random_cos(x, power=0.3):
    freqs = [100, 150, 200, 300, 600]
    wave = torch.cos(torch.tensor(list(range(600))) / random.sample(freqs, 1)[0] * math.pi)
    amplitude = random.random() * power
    signal = 1 + wave * amplitude
    return x * signal.reshape(1, -1)


def random_gnaw(x):
    gnaw_l = int(random.random() * 50)
    gnaw_r = int(random.random() * 50)

    x[:, 0:gnaw_l] = 0
    x[:, -gnaw_r:] = 0
    return x


def omnirandom(x):
    x = random_shift(x)
    x = random_sin(x)
    x = random_cos(x)
    return x


def D0206_org_base(data_dir, batch_size, augc) -> Tuple[List[Tuple[int, DataLoader, DataLoader]], DataLoader]:
    data = np.load(data_dir / "0206_org.npz")
    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_test = data["X_test"]

    X_train = tensor(X_train, dtype=torch.float32)
    Y_train = tensor(Y_train, dtype=torch.long)
    X_test = tensor(X_test, dtype=torch.float32)
    print(X_train.shape, Y_train.shape, X_test.shape)

    ds = augc(X_train, Y_train)
    ds_test = TensorDataset(X_test)
    dl_kwargs = dict(batch_size=batch_size, num_workers=6, pin_memory=True)
    dl_test = DataLoader(ds_test, **dl_kwargs, shuffle=False)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=261342)
    dl_list = []
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, Y_train), 1):
        ds_train = Subset(ds, train_idx)
        ds_valid = Subset(ds, valid_idx)
        dl_train = DataLoader(ds_train, **dl_kwargs, shuffle=True)
        dl_valid = DataLoader(ds_valid, **dl_kwargs, shuffle=False)
        dl_list.append((fold, dl_train, dl_valid))

    return dl_list, dl_test


class C0206_org_v2(TensorDataset):
    # 2.480509 60.96% --> 2.398791 63.84% 까지 증가함
    def __getitem__(self, index):
        x, y = super().__getitem__(index)

        x = omnirandom(x)
        return x, y


def D0206_org_v2(data_dir, batch_size) -> Tuple[List[Tuple[int, DataLoader, DataLoader]], DataLoader]:
    return D0206_org_base(data_dir, batch_size, C0206_org_v2)


class C0206_org_v3(TensorDataset):
    # 2.480509 60.96% --> 2.592381 59.84% 감소함
    def __getitem__(self, index):
        x, y = super().__getitem__(index)

        x = random_shift(x)
        # x = random_sin(x)
        # x = random_cos(x)
        return x, y


def D0206_org_v3(data_dir, batch_size) -> Tuple[List[Tuple[int, DataLoader, DataLoader]], DataLoader]:
    return D0206_org_base(data_dir, batch_size, C0206_org_v3)


class C0206_org_v4(TensorDataset):
    # v2 2.398791 63.84% --> 2.163285 64.00%, 1.822508 61.60% 증가
    def __getitem__(self, index):
        x, y = super().__getitem__(index)

        # x = random_shift(x)
        x = random_sin(x)
        x = random_cos(x)
        return x, y


def D0206_org_v4(data_dir, batch_size) -> Tuple[List[Tuple[int, DataLoader, DataLoader]], DataLoader]:
    return D0206_org_base(data_dir, batch_size, C0206_org_v4)


class C0206_org_v4_2(TensorDataset):
    # v4에서 p=0.5를 둠
    # v4 2.163285 64.00%, 1.822508 61.60% --> 0.64%정도 나빠짐
    def __getitem__(self, index):
        x, y = super().__getitem__(index)

        # x = random_shift(x)
        x = random_sin(x)
        x = random_cos(x)
        return x, y


def D0206_org_v4_2(data_dir, batch_size) -> Tuple[List[Tuple[int, DataLoader, DataLoader]], DataLoader]:
    return D0206_org_base(data_dir, batch_size, C0206_org_v4_2)


class C0206_org_v4_3(TensorDataset):
    # power를 0.3에서 0.5로 증가
    # v4 2.163285 64.00%, 1.822508 61.60% --> v4_4 1.882854 63.04% 2.317307 65.28% 증가
    def __getitem__(self, index):
        x, y = super().__getitem__(index)

        # x = random_shift(x)
        x = random_sin(x, power=0.5)
        x = random_cos(x, power=0.5)
        return x, y


def D0206_org_v4_3(data_dir, batch_size) -> Tuple[List[Tuple[int, DataLoader, DataLoader]], DataLoader]:
    return D0206_org_base(data_dir, batch_size, C0206_org_v4_3)


class C0206_org_v4_4(TensorDataset):
    # power를 0.7에서 0.7로 증가
    # v4_4 1.882854 63.04% 2.317307 65.28% --> 1.608719 60.64% 1.930776 64.16%. loss는 감소
    # 0073-ResNeSt101_fc-0206_org_v4_4          - 1.811438 val_acc: 57.60% 나빠짐...
    # 0074-ResNeSt200_fc-0206_org_v4_4-B256     - 1.656662 val_acc: 57.92%
    # 0075-ResNeSt269_fc-0206_org_v4_4-B192     - 2.553638 val_acc: 48.64%
    # 0076-LegacyResNet50_fc-0206_org_v4_4-B128 - 1.265038 val_acc: 69.28%
    # 왜지...?? receptive field가 작을텐데... 왜 압도적으로 더 잘되지??
    def __getitem__(self, index):
        x, y = super().__getitem__(index)

        # x = random_shift(x)
        x = random_sin(x, power=0.7)
        x = random_cos(x, power=0.7)
        return x, y


def D0206_org_v4_4(data_dir, batch_size) -> Tuple[List[Tuple[int, DataLoader, DataLoader]], DataLoader]:
    return D0206_org_base(data_dir, batch_size, C0206_org_v4_4)


class C0206_org_v4_5(TensorDataset):
    # power를 0.7에서 1.0로 증가
    # v4_4 1.608719 60.64% 1.930776 64.16%. loss는 감소 --> 1.672721 63.04% 1.989787 63.84%
    def __getitem__(self, index):
        x, y = super().__getitem__(index)

        # x = random_shift(x)
        x = random_sin(x, power=1)
        x = random_cos(x, power=1)
        return x, y


def D0206_org_v4_5(data_dir, batch_size) -> Tuple[List[Tuple[int, DataLoader, DataLoader]], DataLoader]:
    return D0206_org_base(data_dir, batch_size, C0206_org_v4_5)


class C0206_org_v5(TensorDataset):
    # random_gnaw
    # 많이 나빠짐
    def __getitem__(self, index):
        x, y = super().__getitem__(index)

        # x = random_shift(x)
        x = random_sin(x)
        x = random_cos(x)
        x = random_gnaw(x)
        return x, y


def D0206_org_v5(data_dir, batch_size) -> Tuple[List[Tuple[int, DataLoader, DataLoader]], DataLoader]:
    return D0206_org_base(data_dir, batch_size, C0206_org_v5)


class C0206_org_v6(TensorDataset):
    # 6개의 채널만 사용
    def __getitem__(self, index):
        x, y = super().__getitem__(index)

        # x = random_shift(x)
        x = random_sin(x, power=0.7)
        x = random_cos(x, power=0.7)
        y = y[:6, :]
        return x, y


def D0206_org_v6(data_dir, batch_size) -> Tuple[List[Tuple[int, DataLoader, DataLoader]], DataLoader]:
    return D0206_org_base(data_dir, batch_size, C0206_org_v6)


class C0206_org_v7(TensorDataset):
    def __getitem__(self, index):
        x, y = super().__getitem__(index)

        # x = random_shift(x)
        x = random_sin(x, power=0.7)
        x = random_cos(x, power=0.7)
        return x, y


def D0201_base(data_dir, batch_size, augc) -> Tuple[List[Tuple[int, DataLoader, DataLoader]], DataLoader]:
    data = np.load(data_dir / "0201.npz")
    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_test = data["X_test"]

    X_train = tensor(X_train, dtype=torch.float32)
    Y_train = tensor(Y_train, dtype=torch.long)
    X_test = tensor(X_test, dtype=torch.float32)
    print(X_train.shape, Y_train.shape, X_test.shape)

    ds = augc(X_train, Y_train)
    ds_test = TensorDataset(X_test)
    dl_kwargs = dict(batch_size=batch_size, num_workers=6, pin_memory=True)
    dl_test = DataLoader(ds_test, **dl_kwargs, shuffle=False)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=261342)
    dl_list = []
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, Y_train), 1):
        ds_train = Subset(ds, train_idx)
        ds_valid = Subset(ds, valid_idx)
        dl_train = DataLoader(ds_train, **dl_kwargs, shuffle=True)
        dl_valid = DataLoader(ds_valid, **dl_kwargs, shuffle=False)
        dl_list.append((fold, dl_train, dl_valid))

    return dl_list, dl_test


class C0201_v1(TensorDataset):
    def __getitem__(self, index):
        x, y = super().__getitem__(index)

        # x = random_shift(x)
        x = random_sin(x, power=0.7)
        x = random_cos(x, power=0.7)
        return x, y


def D0201_v1(data_dir, batch_size) -> Tuple[List[Tuple[int, DataLoader, DataLoader]], DataLoader]:
    return D0201_base(data_dir, batch_size, C0201_v1)
