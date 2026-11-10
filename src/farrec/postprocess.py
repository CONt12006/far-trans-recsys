"""Post-processing of the ranker's output: risk constraints and return-aware re-ranking.

Works on a scored candidate frame (one row per customer x candidate with `score`
and stage-2 features). Scores are first converted to within-customer percentile
ranks, so lambda / alpha have the same meaning for every customer and model.

    final = alpha * pct(score) + (1 - alpha) * pct(expected_return) - lambda * risk_excess
    risk_excess = max(0, asset_risk_level - customer_max_level)

hard=True removes unsuitable assets instead of penalising them; customers whose
candidate list runs short are padded with the most popular suitable assets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SIGNALS = {
    "mom_90": lambda d: d.a_ret_90d,
    "mom_365": lambda d: d.a_ret_365d,
    "low_vol": lambda d: -d.a_vol_365d,
}


def pct_within(df: pd.DataFrame, values: pd.Series) -> pd.Series:
    return values.groupby(df.customerID).rank(pct=True).fillna(0.5)


def suitable_fillers(asset_level: pd.Series, popularity: pd.Series, levels=range(1, 8), n: int = 50) -> dict:
    """For every risk limit: the n most popular assets that do not exceed it."""
    out = {}
    for lim in levels:
        ok = asset_level[asset_level <= lim].index
        out[lim] = popularity.reindex(ok).fillna(0).sort_values(ascending=False).head(n).index
    return out


def rerank(df: pd.DataFrame, k: int, limit: pd.Series, risk_lambda: float = 0.0, hard: bool = False,
           alpha: float = 1.0, signal: str | None = None, fillers: dict | None = None) -> pd.DataFrame:
    d = df[["customerID", "ISIN", "score", "a_risk_level"]].copy()
    d["limit"] = d.customerID.map(limit).fillna(7)
    s = pct_within(d, d.score)
    if alpha < 1.0 and signal is not None:
        s = alpha * s + (1 - alpha) * pct_within(d, SIGNALS[signal](df))
    excess = (d.a_risk_level.fillna(7) - d.limit).clip(lower=0)
    d["final"] = s - risk_lambda * excess
    if hard:
        d = d[excess == 0]
        if fillers is not None:
            d = _pad(d, limit, fillers, k)
    d = d.sort_values(["customerID", "final"], ascending=[True, False])
    d["rank"] = d.groupby("customerID").cumcount() + 1
    return d.loc[d["rank"] <= k, ["customerID", "ISIN", "final", "rank"]].rename(columns={"final": "score"})


def _pad(d: pd.DataFrame, limit: pd.Series, fillers: dict, k: int) -> pd.DataFrame:
    counts = d.groupby("customerID").size().reindex(limit.index, fill_value=0)
    short = counts[counts < k].index
    if not len(short):
        return d
    rows = []
    for c in short:
        lim = int(limit.get(c, 7))
        have = set(d.loc[d.customerID == c, "ISIN"])
        extra = [i for i in fillers[lim] if i not in have][: k]
        rows.append(pd.DataFrame({"customerID": c, "ISIN": extra,
                                  "final": -1.0 - np.arange(len(extra)) * 1e-3}))
    return pd.concat([d, *rows], ignore_index=True)
