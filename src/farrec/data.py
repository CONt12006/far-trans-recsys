"""Loading and cleaning of the raw FAR-Trans tables.

Every function that returns information "as of" a date only uses rows with
timestamp strictly before that date. This is the single most important rule of
the project: nothing from the future may leak into features.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SNAPSHOT_DATE = pd.Timestamp("2018-01-02")

@dataclass
class FarTrans:
    transactions: pd.DataFrame
    customers: pd.DataFrame      # history of customer profiles (may change over time)
    assets: pd.DataFrame         # one row per ISIN (latest description)
    prices: pd.DataFrame         # long format: ISIN, timestamp, closePrice
    price_wide: pd.DataFrame     # date x ISIN, forward-filled

    @property
    def buys(self) -> pd.DataFrame:
        """Real purchase decisions (initial-holdings snapshot excluded)."""
        t = self.transactions
        return t[(t.transactionType == "Buy") & ~t.isSnapshot]


def _strip_predicted(s: pd.Series) -> pd.Series:
    """'Predicted_Balanced' -> 'Balanced'. Keeps the information in a separate flag."""
    return s.str.replace("Predicted_", "", regex=False)


def load(raw_dir: str | Path) -> FarTrans:
    raw_dir = Path(raw_dir)
    tx = pd.read_csv(raw_dir / "transactions.csv", parse_dates=["timestamp"])
    # transactionID is NOT unique: joint accounts share one transaction across several
    # customers. Each (customer, transaction) row is a real event, so only drop exact duplicates.
    tx = tx.drop_duplicates().sort_values("timestamp").reset_index(drop=True)
    # 2018-01-02 holds ~35k "buys" (vs ~50/day later): an initial-holdings snapshot,
    # not real purchase decisions. Keep it for portfolios, exclude it from buy events.
    tx["isSnapshot"] = tx.timestamp == SNAPSHOT_DATE

    cu = pd.read_csv(
        raw_dir / "customer_information.csv",
        parse_dates=["timestamp", "lastQuestionnaireDate"],
    )
    cu["riskPredicted"] = cu.riskLevel.str.startswith("Predicted_")
    cu["riskLevel"] = _strip_predicted(cu.riskLevel)
    cu["capPredicted"] = cu.investmentCapacity.str.startswith("Predicted_")
    cu["investmentCapacity"] = _strip_predicted(cu.investmentCapacity)
    cu = cu.sort_values(["timestamp", "customerID"]).reset_index(drop=True)

    a = pd.read_csv(raw_dir / "asset_information.csv", parse_dates=["timestamp"])
    # Static descriptors (category, sector) are treated as time-invariant; keep the latest row.
    a = a.sort_values("timestamp").drop_duplicates("ISIN", keep="last").reset_index(drop=True)

    p = pd.read_csv(raw_dir / "close_prices.csv", parse_dates=["timestamp"])
    p = p.drop_duplicates(["ISIN", "timestamp"], keep="last").sort_values(["ISIN", "timestamp"])
    # Forward fill only (never backward fill: that would pull future prices into the past).
    wide = p.pivot(index="timestamp", columns="ISIN", values="closePrice").sort_index().ffill()

    return FarTrans(tx, cu, a, p.reset_index(drop=True), wide)


def customer_profile_asof(cu: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """Latest known profile per customer using rows with timestamp < date."""
    hist = cu[cu.timestamp < date]
    return hist.drop_duplicates("customerID", keep="last").set_index("customerID")


def listed_assets_asof(price_wide: pd.DataFrame, date: pd.Timestamp) -> pd.Index:
    """Assets that already had a price before `date` (an asset not yet on the market
    cannot be recommended)."""
    hist = price_wide.loc[price_wide.index < date]
    return hist.columns[hist.notna().any()]


def price_asof(price_wide: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    """Last known close strictly before `date`."""
    hist = price_wide.loc[price_wide.index < date]
    return hist.iloc[-1] if len(hist) else pd.Series(dtype=float)


def forward_return(price_wide: pd.DataFrame, start: pd.Timestamp, days: int) -> pd.Series:
    """Return from last close before `start` to last close before `start + days`.

    Uses FUTURE prices on purpose. Allowed ONLY for evaluation, never for features.
    """
    p0 = price_asof(price_wide, start)
    p1 = price_asof(price_wide, start + pd.Timedelta(days=days))
    r = p1 / p0 - 1
    return r.replace([np.inf, -np.inf], np.nan)
