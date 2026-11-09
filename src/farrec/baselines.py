"""Non-personalised and lightly personalised baselines.

Every recommender has the same signature:
    fit_predict(ft, cutoff, customers, candidates, k) -> DataFrame[customerID, ISIN, score, rank]
and only looks at data strictly before `cutoff`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FarTrans, customer_profile_asof


def _broadcast(scores: pd.Series, customers: pd.Index, k: int) -> pd.DataFrame:
    """Same top-k list for every customer."""
    top = scores.sort_values(ascending=False).head(k)
    base = pd.DataFrame({"ISIN": top.index, "score": top.values, "rank": np.arange(1, len(top) + 1)})
    out = base.loc[base.index.repeat(len(customers))].reset_index(drop=True)
    out.insert(0, "customerID", np.tile(customers.values, len(top)))
    return out.sort_values(["customerID", "rank"]).reset_index(drop=True)


def _buys_before(ft: FarTrans, cutoff: pd.Timestamp, days: int | None) -> pd.DataFrame:
    b = ft.buys
    b = b[b.timestamp < cutoff]
    if days is not None:
        b = b[b.timestamp >= cutoff - pd.Timedelta(days=days)]
    return b


def random_rec(ft, cutoff, customers, candidates, k, seed=0):
    rng = np.random.default_rng(seed)
    s = pd.Series(rng.random(len(candidates)), index=candidates)
    return _broadcast(s, customers, k)


def popularity(ft, cutoff, customers, candidates, k, days=30):
    b = _buys_before(ft, cutoff, days)
    s = b.groupby("ISIN").customerID.nunique().reindex(candidates, fill_value=0)
    return _broadcast(s.astype(float), customers, k)


def segment_popularity(ft, cutoff, customers, candidates, k, days=90, segment="riskLevel"):
    """Popularity among customers with the same risk profile (profile known as of cutoff)."""
    prof = customer_profile_asof(ft.customers, cutoff)
    seg_of = prof[segment].reindex(customers).fillna("Not_Available")
    b = _buys_before(ft, cutoff, days)
    b = b.assign(seg=b.customerID.map(prof[segment]).fillna("Not_Available"))
    glob = b.groupby("ISIN").customerID.nunique().reindex(candidates, fill_value=0)
    parts = []
    for seg, custs in seg_of.groupby(seg_of):
        s = b[b.seg == seg].groupby("ISIN").customerID.nunique().reindex(candidates, fill_value=0)
        s = s + 1e-3 * glob  # tie-break by global popularity
        parts.append(_broadcast(s.astype(float), pd.Index(custs.index), k))
    return pd.concat(parts, ignore_index=True)


def momentum(ft, cutoff, customers, candidates, k, days=90):
    hist = ft.price_wide.loc[ft.price_wide.index < cutoff]
    p1 = hist.iloc[-1]
    p0 = hist.loc[hist.index < cutoff - pd.Timedelta(days=days)].iloc[-1]
    s = (p1 / p0 - 1).replace([np.inf, -np.inf], np.nan).reindex(candidates).fillna(-np.inf)
    return _broadcast(s, customers, k)


def repeat_popularity(ft, cutoff, customers, candidates, k, days=30):
    """Customer's own past buys first (recency-weighted), then global popularity.

    Strong in finance because repeat purchases (top-ups) are frequent.
    """
    b = _buys_before(ft, cutoff, None)
    b = b[b.customerID.isin(customers) & b.ISIN.isin(candidates)]
    age = (cutoff - b.timestamp).dt.days
    own = (np.exp(-age / 90.0)).groupby([b.customerID, b.ISIN]).sum().rename("own").reset_index()
    pop = _buys_before(ft, cutoff, days).groupby("ISIN").customerID.nunique().reindex(candidates, fill_value=0)
    pop = pop / (pop.max() or 1)
    glob = _broadcast(pop.astype(float), customers, k)[["customerID", "ISIN", "score"]]
    own = own.rename(columns={"own": "score"})
    own["score"] = own.score + 10.0  # own history always above global popularity
    allr = pd.concat([own, glob]).sort_values("score", ascending=False).drop_duplicates(["customerID", "ISIN"])
    allr["rank"] = allr.groupby("customerID").cumcount() + 1
    return allr[allr["rank"] <= k].sort_values(["customerID", "rank"]).reset_index(drop=True)


from .cf import als, item_knn  # noqa: E402

BASELINES = {
    "random": random_rec,
    "pop_30d": popularity,
    "pop_segment_90d": segment_popularity,
    "momentum_90d": momentum,
    "repeat_then_pop": repeat_popularity,
    "item_knn": item_knn,
    "als": als,
}
