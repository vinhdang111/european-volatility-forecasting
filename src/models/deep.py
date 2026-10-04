"""Deep learning: an LSTM and a Transformer on sequences of daily data (PyTorch).

Architecture ("wide and deep")
------------------------------
                 tabular features of day t ──────────────┬──> linear layer ───────────┐
                                                         │                            (+)──> log variance
    last 22 days ──> sequence encoder ──> summary ──> [concatenate] ──> small network ─┘      for 1, 5, 22 days
                     (LSTM or Transformer)

* The **linear path** is a HAR-X regression inside the network. It is initialised
  with the least-squares solution, so training starts from HAR-X, and it lets
  the forecast follow volatility to levels never seen in training.
* The **deep path** reads the raw daily sequence and adds what a linear model
  cannot capture. Its last layer starts at zero: at the first step the network
  *is* the linear model.
* One network forecasts the three horizons at once (multi-task learning).

lstm         LSTM (Long Short-Term Memory): reads the sequence day by day and
             keeps a memory of what it has seen.
transformer  Transformer encoder: every day of the sequence can look directly at
             every other day (self-attention).

Training
--------
The loss is QLIKE, the project's primary loss. The number of epochs is chosen by
early stopping on the last year of the training window; the network is then
retrained on the whole window for that number of epochs. A new network is trained
for every test year, like the other models (`refit_every` can space the
trainings out to save time; the years in between then reuse the latest network).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.features.build_features import HORIZONS
from src.models.base import Forecaster, add_is_index
from src.models.sequences import (SEQUENCE_CHANNELS, TABULAR_FEATURES, TARGET_SHIFT, WINDOW, SequenceData,
                                  Standardiser, ols_start)

REFIT_EVERY = 1              # a new network for every test year (3 = one training every three years)
MAX_EPOCHS = 10              # upper limit; early stopping chooses the actual number
PATIENCE = 2                 # epochs without improvement on the validation year before stopping
VALIDATION_DAYS = 252        # one year of trading days
EVAL_BATCH = 8192

logger = logging.getLogger(__name__)


class VolNet(nn.Module):
    """Linear path on the tabular features + sequence encoder with a small head."""

    def __init__(self, n_tabular: int, n_channels: int, n_outputs: int, encoder: str,
                 hidden: int = 32, window: int = WINDOW, dropout: float = 0.1) -> None:
        super().__init__()
        self.kind = encoder
        self.linear = nn.Linear(n_tabular, n_outputs)
        if encoder == "lstm":
            self.encoder = nn.LSTM(n_channels, hidden, batch_first=True)
        elif encoder == "transformer":
            self.embed = nn.Linear(n_channels, hidden)
            self.position = nn.Parameter(torch.zeros(1, window, hidden))      # learned position of each day
            nn.init.normal_(self.position, std=0.02)
            layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=4, dim_feedforward=2 * hidden,
                                               dropout=dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        else:
            raise ValueError(f"Unknown encoder: {encoder!r}")
        self.head = nn.Sequential(nn.Linear(hidden + n_tabular, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, n_outputs))
        nn.init.zeros_(self.head[-1].weight)          # the deep path starts at zero
        nn.init.zeros_(self.head[-1].bias)

    def start_from_linear(self, weights: np.ndarray, bias: np.ndarray) -> None:
        """Initialise the linear path with a least-squares solution (weights: features x outputs)."""
        with torch.no_grad():
            self.linear.weight.copy_(torch.as_tensor(weights.T, dtype=torch.float32))
            self.linear.bias.copy_(torch.as_tensor(bias, dtype=torch.float32))

    def forward(self, tabular: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
        """tabular: (batch, features); sequence: (batch, days, channels). Returns (batch, horizons)."""
        if self.kind == "lstm":
            _, (hidden, _) = self.encoder(sequence)
            summary = hidden[-1]                                   # memory after the last day
        else:
            encoded = self.encoder(self.embed(sequence) + self.position)
            summary = encoded[:, -1]                               # representation of the last day
        return self.linear(tabular) + self.head(torch.cat([summary, tabular], dim=1))


def qlike_terms(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """QLIKE of each forecast; prediction and target are log variances."""
    difference = (target - prediction).clamp(-12.0, 12.0)         # log(actual / forecast)
    return torch.exp(difference) - difference - 1.0


def qlike_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average QLIKE over the targets selected by `mask` (1 = use, 0 = ignore)."""
    return (qlike_terms(prediction, target) * mask).sum() / mask.sum().clamp(min=1.0)


@dataclass
class FittedNetwork:
    net: VolNet
    tabular_scaler: Standardiser
    channel_scaler: Standardiser


class DeepForecaster(Forecaster):
    """Base class of the neural forecasters; subclasses choose the sequence encoder."""

    name = "deep"
    encoder = "lstm"

    def __init__(self, refit_every: int = REFIT_EVERY, max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE,
                 hidden: int = 32, batch_size: int = 1024, learning_rate: float = 1e-3,
                 validation_days: int = VALIDATION_DAYS, seed: int = 0, device: str | None = None) -> None:
        self.refit_every = refit_every
        self.max_epochs = max_epochs
        self.patience = patience
        self.hidden = hidden
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_days = validation_days
        self.seed = seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._seq: SequenceData | None = None
        self._networks: dict[int, FittedNetwork] = {}      # fold anchor -> network used for that fold
        self._latest: FittedNetwork | None = None           # most recently trained network
        self._active: FittedNetwork | None = None           # network whose scaled inputs are loaded
        self.history_: list[dict] = []                      # one record per training and horizon

    # -- data -------------------------------------------------------------------------------
    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        data = add_is_index(data).sort_values(["ticker", "date"]).reset_index(drop=True)
        self._seq = SequenceData(data)
        self._positions = torch.from_numpy(self._seq.positions).to(self.device)
        self._valid = torch.from_numpy(self._seq.valid).to(self.device)
        self._targets = torch.from_numpy(np.nan_to_num(self._seq.targets).astype(np.float32)).to(self.device)
        return data

    def _scaled_inputs(self, fitted: FittedNetwork) -> tuple[torch.Tensor, torch.Tensor]:
        tabular = torch.from_numpy(fitted.tabular_scaler.transform(self._seq.tabular)).to(self.device)
        channels = torch.from_numpy(fitted.channel_scaler.transform(self._seq.channels)).to(self.device)
        return tabular, channels

    def _forward(self, net: VolNet, tabular: torch.Tensor, channels: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        sequence = channels[self._positions[rows]] * self._valid[rows].unsqueeze(-1)   # days before the series start: 0
        return net(tabular[rows], sequence)

    # -- training -----------------------------------------------------------------------------
    def _train(self, mask: np.ndarray, epochs: int, validation: np.ndarray | None = None):
        """Train a new network on the targets selected by `mask` (rows x horizons).

        With a `validation` mask, the validation QLIKE is measured after every
        epoch and training stops early. Returns (network, best number of epochs,
        validation QLIKE per horizon at that epoch).
        """
        seq = self._seq
        torch.manual_seed(self.seed)
        rows = np.flatnonzero(mask.any(axis=1))
        fitted = FittedNetwork(
            net=VolNet(len(TABULAR_FEATURES), len(SEQUENCE_CHANNELS), len(HORIZONS), self.encoder, self.hidden).to(self.device),
            tabular_scaler=Standardiser.fit(seq.tabular[rows]),
            channel_scaler=Standardiser.fit(seq.channels[rows]),
        )
        net = fitted.net
        net.start_from_linear(*ols_start(fitted.tabular_scaler.transform(seq.tabular[rows]), seq.targets[rows], mask[rows]))
        tabular, channels = self._scaled_inputs(fitted)
        rows_t = torch.from_numpy(rows).to(self.device)
        mask_t = torch.from_numpy(mask.astype(np.float32)).to(self.device)
        optimiser = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        shuffle = torch.Generator().manual_seed(self.seed)

        best_epoch, best_scores, best_value = epochs, None, float("inf")
        for epoch in range(1, epochs + 1):
            net.train()
            order = rows_t[torch.randperm(len(rows), generator=shuffle).to(self.device)]
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                loss = qlike_loss(self._forward(net, tabular, channels, batch), self._targets[batch], mask_t[batch])
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimiser.step()
            if validation is not None:
                scores = self._evaluate(net, tabular, channels, validation)
                if scores.mean() < best_value:
                    best_epoch, best_scores, best_value = epoch, scores, float(scores.mean())
                elif epoch - best_epoch >= self.patience:
                    break
        return fitted, best_epoch, best_scores

    def _evaluate(self, net: VolNet, tabular: torch.Tensor, channels: torch.Tensor, mask: np.ndarray) -> np.ndarray:
        """Average QLIKE per horizon on the targets selected by `mask`."""
        rows_t = torch.from_numpy(np.flatnonzero(mask.any(axis=1))).to(self.device)
        mask_t = torch.from_numpy(mask.astype(np.float32)).to(self.device)
        total = torch.zeros(len(HORIZONS), device=self.device)
        count = torch.zeros(len(HORIZONS), device=self.device)
        net.eval()
        with torch.no_grad():
            for start in range(0, len(rows_t), EVAL_BATCH):
                batch = rows_t[start:start + EVAL_BATCH]
                terms = qlike_terms(self._forward(net, tabular, channels, batch), self._targets[batch])
                total += (terms * mask_t[batch]).sum(dim=0)
                count += mask_t[batch].sum(dim=0)
        return (total / count.clamp(min=1.0)).cpu().numpy()

    def _fit_network(self, anchor: int) -> FittedNetwork:
        started = time.time()
        seq = self._seq
        cutoffs = seq.cutoffs(anchor)
        mask = seq.training_mask(cutoffs)
        split = seq.validation_split(mask, cutoffs, self.validation_days)
        if split is None:                                    # history too short for a validation year
            epochs, scores = min(self.max_epochs, 5), np.full(len(HORIZONS), np.nan)
        else:
            inner, validation = split
            _, epochs, scores = self._train(inner, self.max_epochs, validation)
            if scores is None:                               # the validation loss was never finite
                scores = np.full(len(HORIZONS), np.nan)
        fitted, _, _ = self._train(mask, epochs)             # final network: whole training window

        seconds = time.time() - started
        for j, horizon in enumerate(HORIZONS):
            record = {"model": self.name, "horizon": horizon, "train_end": cutoffs[horizon],
                      "n_train": int(mask[:, j].sum()), "epochs": epochs, "training_seconds": seconds}
            if not np.isnan(scores[j]):
                record["validation_qlike"] = float(scores[j])
            self.history_.append(record)
        logger.info("%-12s trained to %s: %d rows, %d epochs, validation QLIKE %s, %.0f s on %s", self.name,
                    pd.Timestamp(cutoffs[min(HORIZONS)]).date(), int(mask.any(axis=1).sum()), epochs,
                    " / ".join(f"{s:.3f}" for s in scores), seconds, self.device)
        return fitted

    # -- walk-forward interface ----------------------------------------------------------------
    def fit(self, train: pd.DataFrame, horizon: int) -> "DeepForecaster":
        """Use the network of this fold: train a new one every `refit_every` folds, else reuse the latest.

        The network covers the three horizons, so it is trained once per fold,
        the first time the fold is seen.
        """
        anchor = self._seq.fold_anchor(train["date"].max(), horizon)
        if anchor not in self._networks:
            if len(self._networks) % self.refit_every == 0:
                self._latest = self._fit_network(anchor)
            self._networks[anchor] = self._latest
        self._activate(self._networks[anchor])
        return self

    def _activate(self, fitted: FittedNetwork) -> None:
        if self._active is not fitted:
            self._active = fitted
            self._tabular, self._channels = self._scaled_inputs(fitted)

    def predict(self, test: pd.DataFrame, horizon: int) -> pd.Series:
        seq = self._seq
        rows = seq.rows_of(test)
        usable = rows >= 0
        usable[usable] = seq.complete[rows[usable]]
        forecast = pd.Series(np.nan, index=test.index)
        if usable.any():
            rows_t = torch.from_numpy(rows[usable]).to(self.device)
            net = self._active.net
            net.eval()
            outputs = []
            with torch.no_grad():
                for start in range(0, len(rows_t), EVAL_BATCH):
                    outputs.append(self._forward(net, self._tabular, self._channels, rows_t[start:start + EVAL_BATCH]))
            log_variance = torch.cat(outputs)[:, HORIZONS.index(horizon)].cpu().numpy().astype(float)
            forecast[usable] = np.exp(log_variance - TARGET_SHIFT)
        return forecast


class Lstm(DeepForecaster):
    """LSTM on the last 22 days, with a linear path on the HAR-X features."""

    name = "lstm"
    encoder = "lstm"


class Transformer(DeepForecaster):
    """Transformer encoder on the last 22 days, with a linear path on the HAR-X features."""

    name = "transformer"
    encoder = "transformer"
