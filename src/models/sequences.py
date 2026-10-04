"""Data preparation for the neural networks (NumPy only, no deep-learning library needed).

A network receives, for each series and day t:

* a **sequence**: the last 22 daily observations of a few raw variables
  (variance, return, volume, market variance, VIX), so that it can learn by
  itself how to summarise the recent past;
* the **tabular features** of Step 4 on day t (the same ones as HAR-X).

Everything a network sees on day t is dated t or earlier. The rules that prevent
any look-ahead (which rows may be used for training, how the features are
scaled) live in this module and are covered by tests/test_sequences.py.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.build_features import HORIZONS
from src.models.har import EXTRA_FEATURES, HAR_FEATURES

TABULAR_FEATURES = HAR_FEATURES + EXTRA_FEATURES + ["is_index"]
SEQUENCE_CHANNELS = ["log_rv_d", "ret_d", "log_volume_ratio", "mkt_log_rv_d", "log_vix"]
REQUIRED = HAR_FEATURES              # a row needs these to be used for training or to get a forecast
WINDOW = 22                          # days of history in the sequence
TARGET_SHIFT = float(np.log(1e4))    # targets are handled as log variance in percent squared (values around 0 to 2)


@dataclass
class Standardiser:
    """Subtract the mean and divide by the standard deviation measured on the training rows.

    Missing values become 0, i.e. the training average.
    """

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardiser":
        with warnings.catch_warnings():                      # a column that is entirely missing gets mean 0, std 1
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nan_to_num(np.nanmean(values, axis=0))
            std = np.nan_to_num(np.nanstd(values, axis=0))
        std[std < 1e-8] = 1.0
        return cls(mean, std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.nan_to_num((values - self.mean) / self.std).astype(np.float32)


class SequenceData:
    """Arrays for the networks, built once from the full dataset sorted by ticker and date."""

    def __init__(self, data: pd.DataFrame, window: int = WINDOW) -> None:
        if not data.reset_index(drop=True).equals(data.sort_values(["ticker", "date"]).reset_index(drop=True)):
            raise ValueError("data must be sorted by ticker and date")
        n = len(data)
        self.window = window
        self.dates = data["date"].to_numpy()
        self.calendar = np.sort(data["date"].unique())
        self.tabular = data[TABULAR_FEATURES].to_numpy(dtype=float)
        self.channels = data[SEQUENCE_CHANNELS].to_numpy(dtype=float)
        self.targets = data[[f"target_{h}d" for h in HORIZONS]].to_numpy(dtype=float) + TARGET_SHIFT
        self.complete = data[REQUIRED].notna().all(axis=1).to_numpy()

        # Sequence of each row: the `window` rows of the same ticker ending at that row.
        codes = pd.factorize(data["ticker"])[0]
        is_first = np.r_[True, codes[1:] != codes[:-1]]
        first_row = np.maximum.accumulate(np.where(is_first, np.arange(n), 0))
        positions = np.arange(n)[:, None] + np.arange(-window + 1, 1)[None, :]
        self.valid = positions >= first_row[:, None]             # False = before the start of the series
        self.positions = np.where(self.valid, positions, 0)

        self._row = pd.Series(np.arange(n), index=pd.MultiIndex.from_arrays([data["ticker"], data["date"]]))

    def rows_of(self, frame: pd.DataFrame) -> np.ndarray:
        """Row number of each (ticker, date) of `frame`; -1 if unknown."""
        keys = pd.MultiIndex.from_arrays([frame["ticker"], frame["date"]])
        return self._row.reindex(keys).fillna(-1).to_numpy(dtype=np.int64)

    def fold_anchor(self, train_end, horizon: int) -> int:
        """Calendar position that identifies a walk-forward fold, whatever the horizon.

        The backtest ends the training window `horizon` days before the test
        period (embargo), so position(train_end) + horizon is the same for the
        three horizons of one fold.
        """
        return int(np.searchsorted(self.calendar, np.datetime64(train_end))) + horizon

    def cutoffs(self, anchor: int) -> dict[int, np.datetime64]:
        """Last training date allowed for each horizon: its target must end before the test period."""
        return {h: self.calendar[anchor - h] for h in HORIZONS}

    def training_mask(self, cutoffs: dict[int, np.datetime64]) -> np.ndarray:
        """Boolean array (rows x horizons): True where a target may be used for training."""
        allowed = np.column_stack([self.dates <= np.datetime64(cutoffs[h]) for h in HORIZONS])
        return allowed & ~np.isnan(self.targets) & self.complete[:, None]

    def validation_split(self, mask: np.ndarray, cutoffs: dict[int, np.datetime64], validation_days: int,
                         min_history: int = 250) -> tuple[np.ndarray, np.ndarray] | None:
        """Split the training targets into an inner training set and the last `validation_days` days.

        For each horizon h, the h days before the validation period are left out
        of the inner set (embargo). Returns None when the history is too short.
        """
        last = int(np.searchsorted(self.calendar, np.datetime64(cutoffs[min(HORIZONS)])))
        start = last - validation_days + 1
        if start - max(HORIZONS) - 1 < min_history:
            return None
        validation = mask & (self.dates >= self.calendar[start])[:, None]
        inner = mask & np.column_stack([self.dates <= self.calendar[start - 1 - h] for h in HORIZONS])
        return inner, validation


def ols_start(tabular: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares regression of each target on the standardised tabular features.

    Used to initialise the linear path of the network: training starts from the
    HAR-X regression instead of from random weights. Returns (weights with one
    column per horizon, intercepts).
    """
    weights = np.zeros((tabular.shape[1], targets.shape[1]))
    bias = np.zeros(targets.shape[1])
    for j in range(targets.shape[1]):
        rows = mask[:, j]
        if not rows.any():
            continue
        design = np.column_stack([np.ones(rows.sum()), tabular[rows]])
        solution, *_ = np.linalg.lstsq(design, targets[rows, j], rcond=None)
        bias[j], weights[:, j] = solution[0], solution[1:]
    return weights, bias
