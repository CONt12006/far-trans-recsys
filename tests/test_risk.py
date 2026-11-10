import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from farrec.risk import asset_risk_levels  # noqa: E402


def test_frozen_stock_is_not_low_risk():
    dates = pd.bdate_range("2021-01-01", "2021-12-31")
    rng = np.random.default_rng(0)
    rows = []
    for isin, prices in {
        "FROZEN": np.full(len(dates), 3.0),                                   # suspended stock
        "BOND": 100 * np.cumprod(1 + rng.normal(0, 0.0005, len(dates))),     # ~0.8% annual vol
    }.items():
        rows += [{"ISIN": isin, "timestamp": d, "closePrice": p} for d, p in zip(dates, prices)]
    prices = pd.DataFrame(rows)
    assets = pd.DataFrame({"ISIN": ["FROZEN", "BOND"], "assetCategory": ["Stock", "Bond"]})
    lvl = asset_risk_levels(prices, assets, pd.Timestamp("2022-01-01"))
    assert lvl["FROZEN"] == 7
    assert lvl["BOND"] <= 2
