"""
Recurrent Neural Network classifiers and their associated training loops.
"""

import json
from pathlib import Path
import typing as tp

from sklearn.metrics import classification_report
import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import EarlyStopper


class RNNClassifier(nn.Module):
    def __init__(
        self,
        architecture: str,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        dropout: float,
        num_classes: int,
        input_size: int = 1,
        batch_first: bool = True,
    ) -> None:
        super().__init__()

        self.architecture = architecture
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        if architecture == "RNN":
            self.rnn = nn.RNN(
                input_size,
                hidden_size,
                num_layers,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        elif architecture == "LSTM":
            self.rnn = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        elif architecture == "GRU":
            self.rnn = nn.GRU(
                input_size,
                hidden_size,
                num_layers,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError(f"Invalid {architecture=}")

        self.d = 2 if self.rnn.bidirectional else 1
        self.mlp = nn.Linear(hidden_size * self.d, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        initial = self.get_initial_state(x)
        out, _ = self.rnn(x, initial)
        out = self.mlp(out[:, -1, :])
        return out

    def get_initial_state(self, x: Tensor) -> tp.Union[Tensor, tp.Tuple[Tensor, Tensor]]:
        h0 = torch.zeros(self.rnn.num_layers * self.d, x.size(0), self.rnn.hidden_size).to(x.device)
        if isinstance(self.rnn, (nn.RNN, nn.GRU)):
            return h0
        if isinstance(self.rnn, nn.LSTM):
            c0 = torch.zeros(self.rnn.num_layers * self.d, x.size(0), self.rnn.hidden_size).to(x.device)
            return (h0, c0)
        raise TypeError(f"Unexpected type of RNN: {type(self.rnn)}")

    def output_path(self) -> Path:
        components = (
            self.dropout,
            self.architecture,
            self.hidden_size,
            self.num_layers,
            self.bidirectional,
        )
        return Path(".").joinpath(*[str(c) for c in components])


def train_rnn_classifier(
    model: nn.Module,
    tr_loader: DataLoader,
    vl_loader: DataLoader,
    optimizer: optim.Optimizer,
    path_models: Path,
    path_reports: Path,
    epoch_e: int,
    epoch_s: int = 0,
    early_stopper: EarlyStopper = None,
    device: tp.Optional[str] = "cpu",
) -> None:
    device = torch.device(device)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(list(range(epoch_s, epoch_e))):
        model.train()
        tr_loss = 0
        for X, y in tqdm(tr_loader, leave=False):
            X = X.unsqueeze(2)
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()

        tr_loss /= len(tr_loader)
        report = evaluate_rnn_classifier(model, vl_loader, device)
        report["tr_loss"] = tr_loss

        torch.save(model.state_dict(), path_models / f"{epoch}.pt")
        with open(path_reports / f"{epoch}.json", "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=4)

        if early_stopper is not None and early_stopper.early_stop(report["loss"]):
            break


def evaluate_rnn_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: tp.Optional[str] = "cpu",
) -> tp.Dict[str, tp.Any]:
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    loss = 0
    y_true = []
    y_pred = []
    for X, y in loader:
        X = X.unsqueeze(2)
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            outputs = model(X)
        loss += criterion(outputs, y).item()
        y_true.extend(y.detach().tolist())
        y_pred.extend(torch.argmax(outputs, dim=1).detach().tolist())
    loss /= len(loader)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report["loss"] = loss
    return report
