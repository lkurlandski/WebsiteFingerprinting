"""
Various helper functions.
"""

from pathlib import Path
import typing as tp

import numpy as np


def get_highest_file(directory: tp.Union[str, Path]) -> tp.Optional[Path]:
    directory = Path(directory)
    if not directory.exists():
        return None
    files = list(directory.iterdir())
    if not files:
        return None
    ext = files[0].suffix
    latest = max((int(p.stem) for p in files))
    file = directory / (str(latest) + ext)
    return file


class EarlyStopper:
    def __init__(self, patience: int = 1, min_delta: float = 0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss: float) -> bool:
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
