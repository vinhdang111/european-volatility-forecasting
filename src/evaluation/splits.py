"""Walk-forward (expanding window) splits for time-ordered panel data.

For each test year Y:

    train = all rows dated before Y, minus the last `embargo` trading days
    test  = all rows dated in Y

The embargo matters because a row dated t carries a target computed from days
t+1 ... t+h. Without it, the last training rows of year Y-1 would have targets
that reach into the test year, and the model would be trained on information
from the period it is evaluated on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FIRST_TEST_YEAR = 2006      # five years of data before the first forecast


@dataclass(frozen=True)
class Fold:
    test_year: int
    train_end: pd.Timestamp      # last date allowed in training (inclusive)
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def train_mask(self, dates: pd.Series) -> pd.Series:
        return dates <= self.train_end

    def test_mask(self, dates: pd.Series) -> pd.Series:
        return (dates >= self.test_start) & (dates <= self.test_end)


def walk_forward_folds(dates: pd.Series, embargo: int, first_test_year: int = FIRST_TEST_YEAR) -> list[Fold]:
    """One fold per calendar year from `first_test_year` to the last year in `dates`.

    `embargo` is the number of trading days dropped at the end of the training
    window; use the forecast horizon.
    """
    calendar = np.sort(pd.to_datetime(pd.Series(dates).unique()))
    folds = []
    for year in range(first_test_year, pd.Timestamp(calendar[-1]).year + 1):
        in_year = calendar[(calendar >= np.datetime64(f"{year}-01-01")) & (calendar <= np.datetime64(f"{year}-12-31"))]
        before = calendar[calendar < np.datetime64(f"{year}-01-01")]
        if len(in_year) == 0 or len(before) <= embargo:
            continue
        folds.append(Fold(
            test_year=year,
            train_end=pd.Timestamp(before[-1 - embargo]),
            test_start=pd.Timestamp(in_year[0]),
            test_end=pd.Timestamp(in_year[-1]),
        ))
    return folds
