"""Evaluation protocol of the FAR-Trans paper (Sanz-Cruzado et al., 2024), re-implemented
to make our numbers comparable with the 11 methods reported there (Table 2).

  * recommendation dates t: every two weeks from 2019-08-01
  * test window: [t, t + 6 months)
  * relevant = assets acquired in the test window that the customer had NOT acquired before t
  * already-acquired assets are never recommended
  * customers: >= 1 acquisition before t and >= 1 relevant acquisition in the window
  * candidate assets: priced before t and still priced at the end of the window
  * metrics: nDCG@10 and ROI@10 = monthly average return of the equal-weighted top-10 over the window
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FarTrans, listed_assets_asof, price_asof
from .evaluate import _drop_seen, prior_pairs
from .metrics import ranking_metrics

PAPER_TABLE2 = {  # nDCG@10, ROI@10 as reported in the paper
    "Random": (0.0106, 0.0071), "Random forest (prices)": (0.0237, 0.0259),
    "Linear regression (prices)": (0.0215, 0.0249), "LightGBM (prices)": (0.0221, 0.0225),
    "Popularity": (0.2710, 0.0006), "LightGCN": (0.3404, 0.0004), "ARM": (0.2556, 0.0007),
    "MF": (0.1780, 0.0038), "UB kNN": (0.1599, 0.0119), "Hybrid-nDCG": (0.2313, 0.0063),
    "Hybrid-regression": (0.0261, 0.0132),
}
PAPER_MARKET_ROI, PAPER_CUSTOMER_ROI = 0.0079, 0.0018


def paper_dates(start="2019-08-01", n: int = 61) -> list[pd.Timestamp]:
    """61 bi-weekly dates as in the paper: 2019-08-01 .. 2022-01-13."""
    return list(pd.date_range(start, periods=n, freq="14D"))


def paper_slice(ft: FarTrans, t: pd.Timestamp, months: int = 6):
    end = t + pd.DateOffset(months=months)
    last_obs = ft.prices.groupby("ISIN").timestamp.max()
    alive = last_obs[last_obs >= end - pd.Timedelta(days=7)].index
    cands = listed_assets_asof(ft.price_wide, t).intersection(alive)

    seen = prior_pairs(ft, t)
    b = ft.buys
    w = b[(b.timestamp >= t) & (b.timestamp < end) & b.ISIN.isin(cands)][["customerID", "ISIN"]].drop_duplicates()
    w = w.merge(seen.assign(_s=1), on=["customerID", "ISIN"], how="left")
    target = w[w._s.isna()].drop(columns="_s")
    target = target[target.customerID.isin(seen.customerID.unique())].reset_index(drop=True)
    return target, cands, seen, end


def monthly_roi(ft: FarTrans, t, end, months: int = 6) -> pd.Series:
    p0, p1 = price_asof(ft.price_wide, t), price_asof(ft.price_wide, end)
    total = (p1 / p0).replace([np.inf, -np.inf], np.nan)
    return total ** (1 / months) - 1


def evaluate_paper(ft: FarTrans, dates, recommenders: dict, k: int = 10, depth: int = 200,
                   verbose: bool = True) -> pd.DataFrame:
    rows = []
    for t in dates:
        target, cands, seen, end = paper_slice(ft, t)
        if target.empty:
            continue
        users = pd.Index(target.customerID.unique())
        roi = monthly_roi(ft, t, end)
        mkt = roi.reindex(cands).mean()
        for name, rec in recommenders.items():
            r = _drop_seen(rec(ft, t, users, cands, depth), seen)
            m = ranking_metrics(r, target, ks=[k])
            top = r[r["rank"] <= k]
            u_roi = top.ISIN.map(roi).groupby(top.customerID).mean()
            rows.append({"date": t.date().isoformat(), "method": name, "n_users": len(users),
                         f"nDCG@{k}": m[f"NDCG@{k}"], f"ROI@{k}": float(u_roi.mean()), "ROI_market": float(mkt)})
        if verbose:
            print(f"  {t.date()} ({len(users)} users)", flush=True)
    return pd.DataFrame(rows)
