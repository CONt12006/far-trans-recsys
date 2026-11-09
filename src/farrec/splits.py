"""Temporal slicing: cutoff dates, target windows and train/valid/test assignment.

For a cutoff date t:
  history window = (-inf, t)            -> features, candidate generation
  target window  = [t, t + target_days) -> label = 1 if the customer bought the asset

Target windows of training cutoffs must END before the evaluation cutoff,
otherwise labels of the training set overlap the evaluated period.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data import FarTrans, listed_assets_asof


@dataclass
class Slice:
    cutoff: pd.Timestamp
    target_end: pd.Timestamp
    target: pd.DataFrame        # unique (customerID, ISIN) bought in the target window
    candidates: pd.Index        # assets recommendable at the cutoff (already listed)

    @property
    def eval_customers(self) -> pd.Index:
        return pd.Index(self.target.customerID.unique())


def all_cutoffs(first: str, last: str, freq: str = "MS") -> list[pd.Timestamp]:
    return list(pd.date_range(first, last, freq=freq))


def assign_splits(cutoffs: list[pd.Timestamp], target_days: int, n_valid: int = 1, n_test: int = 1) -> dict:
    """test = last n_test cutoffs; valid = latest cutoffs whose target window ends before
    the first test cutoff; train = cutoffs whose target window ends before the first valid one."""
    gap = pd.Timedelta(days=target_days)
    test = cutoffs[-n_test:]
    before_test = [c for c in cutoffs if c + gap <= test[0]]
    valid = before_test[-n_valid:]
    train = [c for c in cutoffs if c + gap <= valid[0]]
    return {"train": train, "valid": valid, "test": test}


def train_cutoffs_for(eval_cutoff: pd.Timestamp, cutoffs: list[pd.Timestamp], target_days: int) -> list[pd.Timestamp]:
    """All cutoffs usable for training a model evaluated at `eval_cutoff` (no label overlap)."""
    gap = pd.Timedelta(days=target_days)
    return [c for c in cutoffs if c + gap <= eval_cutoff]


def make_slice(ft: FarTrans, cutoff: pd.Timestamp, target_days: int, exclude_unlisted: bool = True) -> Slice:
    end = cutoff + pd.Timedelta(days=target_days)
    b = ft.buys
    w = b[(b.timestamp >= cutoff) & (b.timestamp < end)]
    candidates = listed_assets_asof(ft.price_wide, cutoff)
    if exclude_unlisted:
        # An asset with no price before the cutoff could not have been recommended.
        w = w[w.ISIN.isin(candidates)]
    target = w[["customerID", "ISIN"]].drop_duplicates().reset_index(drop=True)
    return Slice(cutoff, end, target, candidates)
