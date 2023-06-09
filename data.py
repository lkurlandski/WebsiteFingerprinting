"""
Data loading and processing utilities.
"""

from pathlib import Path
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


def read_wf_file(path: Path, max_length: int = 2000) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as file_pt:
        directions = [0] * max_length
        for i, line in enumerate(file_pt):
            if i == max_length - 1:
                break
            x = line.strip().split("\t")
            directions[i] = 1 if float(x[1]) > 0 else -1
        return np.array(directions)


def get_all_data(path: tp.Union[Path, str]) -> tp.Tuple[np.ndarray, np.array]:
    y = []
    X = []
    for p in Path(path).iterdir():
        y.append(int(p.name.split("-")[0]))
        X.append(read_wf_file(p))
    return np.array(X), np.array(y)


class WFDataset(Dataset):
    X: tp.List[tp.List[int]]
    y: Tensor

    def __init__(self, X: np.ndarray, y: np.ndarray, condense: bool) -> None:
        self.X = [self.condense_tensor(x) for x in X] if condense else X.tolist()
        self.y = torch.tensor(y, dtype=torch.int64)
        self.condense = condense

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, index: int) -> tp.Tuple[Tensor, Tensor]:
        return torch.tensor(self.X[index], dtype=torch.float32), self.y[index]

    @staticmethod
    def condense_tensor(x: tp.Union[tp.List[int], np.ndarray, Tensor]) -> tp.List[int]:
        x_ = []

        previous = None
        count = 1

        for i in x:
            if previous is None:
                previous = int(i)
                continue

            current = int(i)
            if previous == current:
                count += 1
            else:
                x_.append(count * previous)
                count = 1

            previous = current

        return x_


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
