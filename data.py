"""
Data loading and processing utilities.
"""

import typing as tp

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


def get_data(
    data_path: str,  # eg data/full/
    num_sites: int,  # eg 95
    num_instances: int,  # eg 100
    file_ext: str = "",
    max_length: int = 2000,
) -> np.ndarray:
    data = []
    for site in range(0, num_sites):
        for instance in range(0, num_instances):
            file_name = str(site) + "-" + str(instance)
            with open(data_path + file_name + file_ext, "r", encoding="utf-8") as file_pt:
                directions = []
                for line in file_pt:
                    x = line.strip().split("\t")
                    directions.append(1 if float(x[1]) > 0 else -1)
                if len(directions) < max_length:
                    zend = max_length - len(directions)
                    directions.extend([0] * zend)
                elif len(directions) > max_length:
                    directions = directions[:max_length]
                data.append(directions + [site])
    return np.array(data)


class WFDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, condense: bool) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int64)
        self.condense = condense

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, index: int) -> tp.Tuple[Tensor, Tensor]:
        x = self.condense_tensor(self.X[index]) if self.condense else self.X[index]
        y = self.y[index]
        return x, y

    @staticmethod
    def condense_tensor(x: Tensor) -> Tensor:
        x_ = []

        previous = None
        count = 1

        for i in x:
            if previous is None:
                previous = i.item()
                continue

            current = i.item()
            if previous == current:
                count += 1
            else:
                x_.append(count * previous)
                count = 1

            previous = current

        return torch.tensor(x_, dtype=torch.float32)


def collate_fn(
    batch: tp.Iterable[tp.Tuple[Tensor, Tensor]],
    max_length: int = None,
) -> tp.Tuple[Tensor, Tensor]:
    X, y = [], []
    for sample, label in batch:
        if max_length:
            sample = F.pad(sample, (0, max_length - sample.shape[0]), mode="constant", value=0)
        X.append(sample)
        y.append(label)
    X = pad_sequence(X, batch_first=True)
    y = torch.tensor(y, dtype=torch.int64)
    return X, y


def get_collate_fn(*args) -> tp.Callable:
    return lambda batch: collate_fn(batch, *args)
