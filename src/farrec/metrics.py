"""Three groups of metrics: behaviour (ranking), financial outcome, suitability.

All functions take recommendations as a long DataFrame [customerID, ISIN, rank]
with rank starting at 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ranking_metrics(recs: pd.DataFrame, target: pd.DataFrame, ks=(1, 5, 10)) -> dict:
    """Precision/Recall/NDCG/HitRate@k averaged over customers that have a target."""
    users = target.customerID.unique()
    n_rel = target.groupby("customerID").size().reindex(users)
    r = recs[recs.customerID.isin(users)].merge(target.assign(hit=1), on=["customerID", "ISIN"], how="left")
    r["hit"] = r.hit.fillna(0)
    r["gain"] = r.hit / np.log2(r["rank"] + 1)
    out = {}
    for k in ks:
        top = r[r["rank"] <= k]
        hits = top.groupby("customerID").hit.sum().reindex(users, fill_value=0)
        dcg = top.groupby("customerID").gain.sum().reindex(users, fill_value=0)
        ideal_n = np.minimum(n_rel.values, k)
        idcg = np.array([np.sum(1 / np.log2(np.arange(2, n + 2))) for n in ideal_n])
        out[f"P@{k}"] = float((hits / k).mean())
        out[f"R@{k}"] = float((hits / n_rel).mean())
        out[f"NDCG@{k}"] = float((dcg.values / idcg).mean())
        out[f"HR@{k}"] = float((hits > 0).mean())
    out["n_users"] = int(len(users))
    return out


def ndcg_per_user(recs: pd.DataFrame, target: pd.DataFrame, k: int = 10) -> pd.Series:
    """NDCG@k for every customer with a target (for segment analysis / significance tests)."""
    users = target.customerID.unique()
    n_rel = target.groupby("customerID").size().reindex(users)
    r = recs[(recs["rank"] <= k) & recs.customerID.isin(users)].merge(
        target.assign(hit=1), on=["customerID", "ISIN"], how="left")
    dcg = (r.hit.fillna(0) / np.log2(r["rank"] + 1)).groupby(r.customerID).sum().reindex(users, fill_value=0)
    idcg = np.array([np.sum(1 / np.log2(np.arange(2, min(n, k) + 2))) for n in n_rel.values])
    return pd.Series(dcg.values / idcg, index=users, name=f"NDCG@{k}")


def financial_metrics(recs: pd.DataFrame, target: pd.DataFrame, fwd_ret: pd.Series,
                      universe: pd.Index, k: int = 5) -> dict:
    """Mean forward return of top-k recs vs. what customers actually bought vs. equal-weight market.

    `fwd_ret` uses prices AFTER the cutoff: valid for evaluation only.
    """
    users = target.customerID.unique()
    top = recs[(recs["rank"] <= k) & recs.customerID.isin(users)]
    rec_ret = top.ISIN.map(fwd_ret).groupby(top.customerID).mean()
    bought_ret = target.ISIN.map(fwd_ret).groupby(target.customerID).mean()
    mkt = fwd_ret.reindex(universe).dropna()
    return {
        f"Ret@{k}": float(rec_ret.mean()),
        "Ret_bought": float(bought_ret.mean()),
        "Ret_market_ew": float(mkt.mean()),
        f"Excess@{k}": float(rec_ret.mean() - mkt.mean()),
        f"WinRate@{k}": float((top.ISIN.map(fwd_ret) > 0).mean()),
    }


def suitability_metrics(recs: pd.DataFrame, asset_level: pd.Series, cust_max: pd.Series, k: int = 5) -> dict:
    """Share of top-k recommendations whose risk level exceeds the customer's limit."""
    top = recs[recs["rank"] <= k]
    lvl = top.ISIN.map(asset_level).fillna(7)
    lim = top.customerID.map(cust_max).fillna(7)
    excess = (lvl - lim).clip(lower=0)
    return {f"Unsuit@{k}": float((excess > 0).mean()), f"RiskExcess@{k}": float(excess.mean())}


def coverage(recs: pd.DataFrame, universe_size: int, k: int = 10) -> dict:
    top = recs[recs["rank"] <= k]
    return {f"Cov@{k}": top.ISIN.nunique() / max(universe_size, 1)}
