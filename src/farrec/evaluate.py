"""One evaluation protocol for every recommender."""
from __future__ import annotations

from typing import Callable

import pandas as pd

from . import metrics
from .data import FarTrans, customer_profile_asof, forward_return
from .risk import asset_risk_levels, customer_max_level
from .splits import Slice

K_MAX = 50  # depth requested from recommenders (needed to re-rank after removing seen items)


def prior_pairs(ft: FarTrans, cutoff: pd.Timestamp) -> pd.DataFrame:
    """(customer, asset) pairs bought before the cutoff (snapshot holdings included)."""
    t = ft.transactions
    b = t[(t.transactionType == "Buy") & (t.timestamp < cutoff)]
    return b[["customerID", "ISIN"]].drop_duplicates()


def _drop_seen(recs: pd.DataFrame, seen: pd.DataFrame) -> pd.DataFrame:
    r = recs.merge(seen.assign(_seen=1), on=["customerID", "ISIN"], how="left")
    r = r[r._seen.isna()].drop(columns="_seen").sort_values(["customerID", "rank"])
    r["rank"] = r.groupby("customerID").cumcount() + 1
    return r


class SliceContext:
    """Everything about a slice that does not depend on the recommender (computed once)."""

    def __init__(self, ft: FarTrans, sl: Slice, cfg: dict):
        self.sl = sl
        self.fwd = forward_return(ft.price_wide, sl.cutoff, cfg["eval"]["return_horizon_days"])
        self.lvl = asset_risk_levels(ft.prices, ft.assets, sl.cutoff,
                                     cfg["risk"]["srri_bounds"], cfg["risk"]["vol_window_days"])
        prof = customer_profile_asof(ft.customers, sl.cutoff)
        self.lim = customer_max_level(prof, sl.eval_customers, cfg["risk"]["max_level"])
        self.seen = prior_pairs(ft, sl.cutoff)
        tn = sl.target.merge(self.seen.assign(_s=1), on=["customerID", "ISIN"], how="left")
        self.target_new = tn[tn._s.isna()].drop(columns="_s")


def evaluate_slice(ft: FarTrans, ctx: SliceContext, recommender: Callable, cfg: dict) -> dict:
    sl = ctx.sl
    recs = recommender(ft, sl.cutoff, sl.eval_customers, sl.candidates, K_MAX)

    out = {"cutoff": sl.cutoff.date().isoformat()}
    out.update(metrics.ranking_metrics(recs, sl.target, ks=cfg["eval"]["ks"]))
    out.update(metrics.financial_metrics(recs, sl.target, ctx.fwd, sl.candidates, k=5))
    out.update(metrics.suitability_metrics(recs, ctx.lvl, ctx.lim, k=5))
    out.update(metrics.coverage(recs, len(sl.candidates), k=10))

    # Discovery: hide everything the customer already bought, evaluate on first-time buys only.
    if len(ctx.target_new):
        new = metrics.ranking_metrics(_drop_seen(recs, ctx.seen), ctx.target_new, ks=[10])
        out.update({"NDCG@10_new": new["NDCG@10"], "HR@10_new": new["HR@10"], "n_users_new": new["n_users"]})
    return out


def evaluate(ft: FarTrans, slices: list[Slice], recommenders: dict, cfg: dict, verbose: bool = True) -> pd.DataFrame:
    rows = []
    for sl in slices:
        ctx = SliceContext(ft, sl, cfg)
        for name, rec in recommenders.items():
            r = evaluate_slice(ft, ctx, rec, cfg)
            r["method"] = name
            rows.append(r)
        if verbose:
            print(f"  {sl.cutoff.date()} done ({len(sl.eval_customers)} users)", flush=True)
    return pd.DataFrame(rows)
