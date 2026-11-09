"""One evaluation protocol for every recommender."""
from __future__ import annotations

from typing import Callable

import pandas as pd

from . import metrics
from .data import FarTrans, customer_profile_asof, forward_return
from .risk import asset_risk_levels, customer_max_level
from .splits import Slice


def evaluate_slice(ft: FarTrans, sl: Slice, recommender: Callable, cfg: dict, k_max: int = 10) -> dict:
    customers = sl.eval_customers
    recs = recommender(ft, sl.cutoff, customers, sl.candidates, k_max)

    out = {"cutoff": sl.cutoff.date().isoformat()}
    out.update(metrics.ranking_metrics(recs, sl.target, ks=cfg["eval"]["ks"]))

    fwd = forward_return(ft.price_wide, sl.cutoff, cfg["eval"]["return_horizon_days"])
    out.update(metrics.financial_metrics(recs, sl.target, fwd, sl.candidates, k=5))

    lvl = asset_risk_levels(ft.prices, ft.assets, sl.cutoff, cfg["risk"]["srri_bounds"], cfg["risk"]["vol_window_days"])
    prof = customer_profile_asof(ft.customers, sl.cutoff)
    lim = customer_max_level(prof, customers, cfg["risk"]["max_level"])
    out.update(metrics.suitability_metrics(recs, lvl, lim, k=5))
    out.update(metrics.coverage(recs, len(sl.candidates), k=10))
    return out


def evaluate(ft: FarTrans, slices: list[Slice], recommenders: dict, cfg: dict) -> pd.DataFrame:
    rows = []
    for name, rec in recommenders.items():
        for sl in slices:
            r = evaluate_slice(ft, sl, rec, cfg)
            r["method"] = name
            rows.append(r)
    return pd.DataFrame(rows)
