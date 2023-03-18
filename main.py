"""
Train and evaluate models.
"""

from itertools import chain, product
import json
from pathlib import Path
from pprint import pformat
import shutil

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import optimizers as tf_optim
from tensorflow.keras.utils import to_categorical
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from cnn import CNN
from data import get_collate_fn, get_all_data, WFDataset
from rnn import evaluate_rnn_classifier, train_rnn_classifier, RNNClassifier
from utils import get_highest_file, EarlyStopper


def get_torch_optimizer(model: nn.Module, optimizer: str, **kwargs) -> optim.Optimizer:
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
        early_stopper = EarlyStopper(patience=PATIENCE) if PATIENCE is not None else None
        train_rnn_classifier(
            model,
            tr_loader,
            vl_loader,
            optimizer,
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

    num_sites = np.unique(tr_y).shape[0]

    tr_dataset = WFDataset(tr_X, tr_y, condense=True)
    vl_dataset = WFDataset(vl_X, vl_y, condense=True)
    ts_dataset = WFDataset(ts_X, ts_y, condense=True)
    tr_loader = DataLoader(tr_dataset, BATCH_SIZE, collate_fn=get_collate_fn(False))
    vl_loader = DataLoader(vl_dataset, BATCH_SIZE, collate_fn=get_collate_fn(False))
    ts_loader = DataLoader(ts_dataset, BATCH_SIZE, collate_fn=get_collate_fn(False))

    order = (
        "optimizer",
        "learning_rate",
        "dropout",
        "architecture",
        "hidden_size",
        "num_layers",
        "bidirectional",
    )
    order = {key: idx for idx, key in enumerate(order)}
    for combo in product(*[PARAMS_RNN[k] for k in order]):
        try:
            kwargs = {
                "architecture": combo[order["architecture"]],
                "hidden_size": combo[order["hidden_size"]],
                "num_layers": combo[order["num_layers"]],
                "bidirectional": combo[order["bidirectional"]],
                "dropout": combo[order["dropout"]],
                "num_classes": num_sites,
            }
            print(f"{'-' * 78}\n{pformat(kwargs)}")
            model = RNNClassifier(**kwargs)
            path = OUTPUT.joinpath(*[str(combo[idx]) for idx in order.values()])
            if path.exists() and CLEAN_RNN:
                shutil.rmtree(path)
            elif (path / "report.json").exists():
                print("SKIPPING. Use CLEAN_RNN=True to clean out old data.")
                continue
            path_models = path / "models"
            path_reports = path / "reports"
            if TRAIN_RNN:
                optimizer = get_torch_optimizer(model, combo[order["optimizer"]], lr=combo[order["learning_rate"]])
                single_run_train()
            if EVAL_RNN:
                single_run_test()
        except Exception as e:
            if ERRORS == "raise":
                raise e
            if ERRORS == "warn":
                print(f"{'-' * 78}\n{e}{'-' * 78}")


def cnns(
    tr_X: np.ndarray,
    tr_y: np.ndarray,
    vl_X: np.ndarray,
    vl_y: np.ndarray,
    ts_X: np.ndarray,
    ts_y: np.ndarray,
) -> None:
    num_sites = np.unique(tr_y).shape[0]
    num_features = tr_X.shape[1]

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

    order = (
        "optimizer",
        "learning_rate",
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
                "num_classes": num_sites,
                "filter_size": combo[order["filter_size"]],
                "act_func": combo[order["act_func"]],
                "dropout_rate": combo[order["dropout_rate"]],
                "optimizer": get_tensorflow_optimizer(
                    combo[order["optimizer"]], learning_rate=combo[order["learning_rate"]]
                ),
            }
            print(f"{'-' * 78}\n{pformat(kwargs)}")
            model = CNN(**kwargs)

            path = OUTPUT.joinpath(*[str(combo[idx]) for idx in order.values()])
            if path.exists() and CLEAN_CNN:
                shutil.rmtree(path)
            elif (path / "report.json").exists():
                print("SKIPPING. Use CLEAN_CNN=True to clean out old data.")
            path.mkdir(parents=True)
            if TRAIN_CNN:
                with tf.device(DEVICE_TF):
                    history = model.fit(tr_X, tr_y, BATCH_SIZE, EPOCHS, (vl_X, vl_y), PATIENCE)
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

    X, y = get_all_data(DATA_PATH)
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
    # Hyperparameters for the RNNs (see below for an idea of valid values)
    PARAMS_RNN = {
        "architecture": ["LSTM", "GRU"],
        "hidden_size": [128],
        "num_layers": [4],
        "bidirectional": [False],
        "dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "optimizer": ["Adam", "Adamax", "SGD", "RMSprop"],
        "learning_rate": [0.001, 0.01, 0.1],
    }

    # Hyperparameters for the CNNs (see below for an idea of valid values)
    PARAMS_CNN = {
        "architecture": ["CNN"],
        "filter_size": [2, 4, 8, 16, 32],
        "act_func": ["elu"],
        "dropout_rate": [0.0],
        "optimizer": ["Adam", "Adamax", "SGD", "RMSprop"],
        "learning_rate": [0.001, 0.01, 0.1],
    }

    # Shared hyperparameters
    TR, VL, TS = 0.90, 0.05, 0.05  # training, validation, test split
    BATCH_SIZE = 128
    EPOCHS = 50
    PATIENCE = 10  # halt training if validation loss does not decrease for this many epochs

    # Experiment configurations
    DATA_PATH = "./data/full/"  # location of the unzipped data files
    SEED = 0  # random seed for controlling random randomization
    DEVICE_PT = "cuda:1"  # device for the Pytorch training/evaluation, "cpu" or "cuda:0"
    DEVICE_TF = "/GPU:1"  # device for the TensorFlow training/evaluation, "/device:CPU:0" or "/GPU:0"
    OUTPUT = Path("./outputs")  # location where experimental data is stored

    # If True, the program will delete any existing output data it encounters
    CLEAN_RNN = False
    CLEAN_CNN = False

    # Flags controlling what occurs in experiment
    ERRORS = "warn"  # how to handle errors, "warn" or "raise"
    TRAIN_RNN = False  # if True, will train the RNN models
    EVAL_RNN = False  # if True, will evaluate the best RNN model
    TRAIN_CNN = True  # if True, will train the CNN models
    EVAL_CNN = True  # if True, will evaluate the best CNN model

    # Verify parameters for the RNNs are valid
    assert all(p in ("RNN", "LSTM", "GRU") for p in PARAMS_RNN["architecture"])
    assert all(isinstance(p, int) for p in chain(PARAMS_RNN["hidden_size"], PARAMS_RNN["num_layers"]))
    assert all(isinstance(p, float) for p in chain(PARAMS_RNN["learning_rate"], PARAMS_RNN["dropout"]))
    assert all(isinstance(p, bool) for p in PARAMS_RNN["bidirectional"])
    assert all(p in ("Adam", "Adamax", "SGD", "RMSprop") for p in PARAMS_RNN["optimizer"])

    # Verify parameters for the CNNs are valid
    assert all(p in ("CNN",) for p in PARAMS_CNN["architecture"])
    assert all(isinstance(p, int) for p in PARAMS_CNN["filter_size"])
    assert all(isinstance(p, float) for p in chain(PARAMS_CNN["learning_rate"], PARAMS_CNN["dropout_rate"]))
    assert all(p in ("relu", "elu", "tanh") for p in PARAMS_CNN["act_func"])
    assert all(p in ("Adam", "Adamax", "SGD", "RMSprop") for p in PARAMS_CNN["optimizer"])

    main()
