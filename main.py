"""
Train and evaluate models.
"""

from itertools import product
import json
from pathlib import Path
from pprint import pformat
import shutil
import sys
import typing as tp

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import optimizers as tf_optim
from tensorflow.keras.utils import to_categorical
import torch
from torch import nn
from torch import optim
from torch.utils.data import random_split, DataLoader

from cnn import CNN
from data import get_collate_fn, get_data, WFDataset
from rnn import evaluate_rnn_classifier, train_rnn_classifier, RNNClassifier
from utils import get_highest_file, EarlyStopper


def get_torch_optimizer(optimizer: str, model: nn.Module, **kwargs) -> optim.Optimizer:
    if optimizer == "Adam":
        return optim.Adam(model.parameters(), **kwargs)
    if optimizer == "Adamax":
        return optim.Adamax(model.parameters(), **kwargs)
    if optimizer == "SGD":
        return optim.SGD(model.parameters(), **kwargs)
    if optimizer == "RMSprop":
        return optim.RMSprop(model.parameters(), **kwargs)
    raise ValueError(f"Invalid {optimizer=}")


def get_tensorflow_optimizer(optimizer: str, **kwargs) -> tf_optim.Optimizer:
    if optimizer == "Adam":
        return tf_optim.Adam(**kwargs)
    if optimizer == "Adamax":
        return tf_optim.Adamax(**kwargs)
    if optimizer == "SGD":
        return tf_optim.SGD(**kwargs)
    if optimizer == "RMSprop":
        return tf_optim.RMSprop(**kwargs)
    raise ValueError(f"Invalid {optimizer=}")


def rnns(
    tr_X: np.ndarray,
    tr_y: np.ndarray,
    vl_X: np.ndarray,
    vl_y: np.ndarray,
    ts_X: np.ndarray,
    ts_y: np.ndarray,
) -> None:
    # Caution: nested functions share state with parent function
    def single_run_test() -> None:
        # Load the best model if early stopping was engaged
        if PATIENCE is not None:
            epoch_f = int(get_highest_file(path_models).stem)
            if epoch_f != EPOCHS - 1:
                checkpoint = path_models / f"{epoch_f-PATIENCE}.pt"
                model.load_state_dict(torch.load(checkpoint), strict=False)
        # Evaluate the best model
        report = evaluate_rnn_classifier(model, ts_loader, DEVICE_PT)
        with open(path / "report.json", "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=4)

    # Caution: nested functions share state with parent function
    def single_run_train() -> None:
        # Resume previous experiment if checkpoint exists
        if (checkpoint := get_highest_file(path_models)) is not None:
            epoch_s = int(checkpoint.stem)
            model.load_state_dict(torch.load(checkpoint), strict=False)
        else:
            epoch_s = 0
            path.mkdir(parents=True)
            path_models.mkdir()
            path_reports.mkdir()
        # Train the model for a number of epochs
        optimizer_ = get_torch_optimizer(optimizer, model)
        early_stopper = EarlyStopper(patience=PATIENCE) if PATIENCE is not None else None
        train_rnn_classifier(
            model,
            tr_loader,
            vl_loader,
            optimizer_,
            path_models,
            path_reports,
            epoch_e=EPOCHS,
            epoch_s=epoch_s,
            early_stopper=early_stopper,
            device=DEVICE_PT,
        )

    print(f"{tr_X.shape=}")
    print(f"{tr_y.shape=}")
    print(f"{vl_X.shape=}")
    print(f"{vl_y.shape=}")
    print(f"{ts_X.shape=}")
    print(f"{ts_y.shape=}")

    tr_dataset = WFDataset(tr_X, tr_y, condense=True)
    vl_dataset = WFDataset(vl_X, vl_y, condense=True)
    ts_dataset = WFDataset(ts_X, ts_y, condense=True)
    tr_loader = DataLoader(tr_dataset, BATCH_SIZE, collate_fn=get_collate_fn(False))
    vl_loader = DataLoader(vl_dataset, BATCH_SIZE, collate_fn=get_collate_fn(False))
    ts_loader = DataLoader(ts_dataset, BATCH_SIZE, collate_fn=get_collate_fn(False))

    order = (
        "optimizer",
        "dropout",
        "architecture",
        "hidden_size",
        "num_layers",
        "bidirectional",
    )
    order = {key: idx for idx, key in enumerate(order)}
    for combo in product(*[PARAMS_CNN[k] for k in order]):
        try:
            kwargs = {
                "architecture": combo[order["architecture"]],
                "hidden_size": combo[order["hidden_size"]],
                "num_layers": combo[order["num_layers"]],
                "bidirectional": combo[order["bidirectional"]],
                "dropout": combo[order["dropout"]],
                "num_classes": NUM_SITES,
            }
            print(f"{'-' * 78}\n{pformat(kwargs)}")
            model = RNNClassifier(**kwargs)
            path = OUTPUT.joinpath(*[str(combo[idx]) for idx in order.values()])
            if path.exists() and CLEAN_RNN:
                shutil.rmtree(path)
            path_models = path / "models"
            path_reports = path / "reports"
            if TRAIN_RNN:
                single_run_train()
            if TEST_RNN:
                single_run_test()
        except Exception as e:
            if ERRORS == "raise":
                raise e
            if ERRORS == "warn":
                print(f"{'-' * 78}\n{e}{'-' * 78}")


# TODO: save models to faciliate reevaluating them without needing to train again
def cnns(
    tr_X: np.ndarray,
    tr_y: np.ndarray,
    vl_X: np.ndarray,
    vl_y: np.ndarray,
    ts_X: np.ndarray,
    ts_y: np.ndarray,
) -> None:
    tr_X = np.expand_dims(tr_X, axis=2) if tr_X.ndim < 3 else tr_X
    vl_X = np.expand_dims(vl_X, axis=2) if vl_X.ndim < 3 else vl_X
    ts_X = np.expand_dims(ts_X, axis=2) if ts_X.ndim < 3 else ts_X
    tr_y = to_categorical(tr_y) if tr_y.ndim < 2 else tr_y
    vl_y = to_categorical(vl_y) if vl_y.ndim < 2 else vl_y
    ts_y = to_categorical(ts_y) if ts_y.ndim < 2 else ts_y

    print(f"{tr_X.shape=}")
    print(f"{tr_y.shape=}")
    print(f"{vl_X.shape=}")
    print(f"{vl_y.shape=}")
    print(f"{ts_X.shape=}")
    print(f"{ts_y.shape=}")

    num_features = tr_X.shape[1] if tr_X is not None else ts_X.shape[1]
    order = (
        "optimizer",
        "dropout_rate",
        "architecture",
        "filter_size",
        "act_func",
    )
    order = {key: idx for idx, key in enumerate(order)}
    for combo in product(*[PARAMS_CNN[k] for k in order]):
        try:
            kwargs = {
                "num_features": num_features,
                "num_classes": NUM_SITES,
                "filter_size": combo[order["filter_size"]],
                "act_func": combo[order["act_func"]],
                "dropout_rate": combo[order["dropout_rate"]],
                "optimizer": get_tensorflow_optimizer(combo[order["optimizer"]]),
            }
            print(f"{'-' * 78}\n{pformat(kwargs)}")
            model = CNN(**kwargs)

            path = OUTPUT.joinpath(*[str(combo[idx]) for idx in order.values()])
            if path.exists() and CLEAN_CNN:
                shutil.rmtree(path)

            path.mkdir(parents=True)
            if TRAIN_CNN:
                vl = (vl_X, vl_y) if (vl_X is not None and vl_y is not None) else None
                with tf.device(DEVICE_TF):
                    history = model.fit(tr_X, tr_y, BATCH_SIZE, EPOCHS, validation_data=vl, patience=PATIENCE)
                with open(path / "history.json", "w", encoding="utf-8") as handle:
                    json.dump(history.history, handle, indent=4)
            if EVAL_CNN:
                with tf.device(DEVICE_TF):
                    report = model.model.evaluate(ts_X, ts_y, BATCH_SIZE, return_dict=True)
                with open(path / "report.json", "w", encoding="utf-8") as handle:
                    json.dump(report, handle, indent=4)
        except Exception as e:
            if ERRORS == "raise":
                raise e
            if ERRORS == "warn":
                print(f"{'-' * 78}\n{e}{'-' * 78}")


def main() -> None:
    torch.manual_seed(SEED)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    raw = get_data(DATA_PATH, NUM_SITES, NUM_INSTANCES)
    X = raw[:, : raw.shape[1] - 1]
    y = raw[:, -1]

    tr_vl_X, ts_X, tr_vl_y, ts_y = train_test_split(X, y, test_size=int(TS * len(y)), random_state=SEED)
    tr_X, vl_X, tr_y, vl_y = train_test_split(tr_vl_X, tr_vl_y, test_size=int(VL * len(y)), random_state=SEED)

    print(f"{torch.cuda.device_count()=}")
    print(f"{tf.config.list_physical_devices('GPU')=}")
    print(f"n_classes={np.unique(y).shape[0]}")
    print(f"{X.shape=}")
    print(f"{y.shape=}")

    print(f"{tr_X.shape=}")
    print(f"{tr_y.shape=}")
    print(f"{vl_X.shape=}")
    print(f"{vl_y.shape=}")
    print(f"{ts_X.shape=}")
    print(f"{ts_y.shape=}")

    if TRAIN_RNN or EVAL_RNN:
        rnns(tr_X, tr_y, vl_X, vl_y, ts_X, ts_y)

    if TRAIN_CNN or EVAL_CNN:
        cnns(tr_X, tr_y, vl_X, vl_y, ts_X, ts_y)


if __name__ == "__main__":
    # Hyperparameters
    PARAMS_RNN = {
        "architecture": ["LSTM"],
        "hidden_size": [64],
        "num_layers": [3],
        "bidirectional": [False],
        "dropout": [0.0],
        "optimizer": ["Adam"],
    }
    PARAMS_CNN = {
        "architecture": ["CNN"],  # lazy
        "filter_size": [4],
        "act_func": ["relu", "elu"],
        "dropout_rate": [0.2],
        "optimizer": ["Adam"],
    }
    TR, VL, TS = 0.8, 0.1, 0.1
    assert sum((TR, VL, TS)) == 1, f"Train/Validation/Test split should sum to 1, not {sum((TR, VL, TS))=}"
    BATCH_SIZE = 128
    EPOCHS = 3
    PATIENCE = 3

    # Experiment
    DATA_PATH = "./data/full/"
    NUM_SITES = 95
    NUM_INSTANCES = 100
    SEED = 0
    DEVICE_PT = "cuda:1"  # "cpu" or "cuda:0"
    DEVICE_TF = "/GPU:1"  # "/device:CPU:0" or "/GPU:0"
    OUTPUT = Path("./outputs")

    # Delete existing data
    CLEAN_RNN = True
    CLEAN_CNN = True

    # Flags
    ERRORS = "raise"
    TRAIN_RNN = False
    EVAL_RNN = False
    TRAIN_CNN = True
    EVAL_CNN = True

    main()
