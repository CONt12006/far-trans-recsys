"""Small FastAPI service: top-k recommendations for a customer at a date.

    uvicorn service.app:app --port 8000
    GET /recommend?customer_id=00017496858921195E5A&date=2022-09-15&k=5&policy=strict

The date is snapped to the first day of its month (models are trained per monthly cutoff).
The first request for a month trains the ranker for it (~1-2 min) and caches it in memory;
later requests take a few seconds (candidate generation refits kNN/ALS on the history).
Unsuitable assets are removed (hard filter) and the list is padded with popular suitable ones.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Query

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farrec.data import customer_profile_asof, listed_assets_asof, load  # noqa: E402
from farrec.features import build_frame  # noqa: E402
from farrec.postprocess import rerank, suitable_fillers  # noqa: E402
from farrec.ranker import TwoStageRanker  # noqa: E402
from farrec.risk import asset_risk_levels, customer_max_level  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "configs/base.yaml"))
FT = load(ROOT / CFG["data"]["raw_dir"])
RANKER = TwoStageRanker(FT, CFG, "lgbm", cache_dir=ROOT / "data/processed/frames")
ASSETS = FT.assets.set_index("ISIN")
FIRST, LAST = pd.Timestamp("2020-06-01"), FT.transactions.timestamp.max()

app = FastAPI(title="FAR-Trans recommender", version="1.0")


@lru_cache(maxsize=24)
def _cutoff_context(cutoff: pd.Timestamp):
    lvl = asset_risk_levels(FT.prices, FT.assets, cutoff, CFG["risk"]["srri_bounds"])
    b = FT.buys[(FT.buys.timestamp < cutoff) & (FT.buys.timestamp >= cutoff - pd.Timedelta(days=30))]
    return listed_assets_asof(FT.price_wide, cutoff), lvl, suitable_fillers(lvl, b.groupby("ISIN").customerID.nunique())


@app.get("/health")
def health():
    return {"status": "ok", "customers": int(FT.transactions.customerID.nunique()), "assets": len(ASSETS)}


@app.get("/recommend")
def recommend(customer_id: str, date: str = "2022-09-01", k: int = Query(5, ge=1, le=20),
              policy: str = Query("strict", pattern="^(strict|moderate|none)$")):
    try:
        cutoff = pd.Timestamp(date).to_period("M").to_timestamp()
    except ValueError:
        raise HTTPException(400, "bad date")
    if not FIRST <= cutoff <= LAST:
        raise HTTPException(400, f"date must be between {FIRST.date()} and {LAST.date()}")
    if customer_id not in set(FT.transactions.customerID):
        raise HTTPException(404, "unknown customer")

    candidates, lvl, fillers = _cutoff_context(cutoff)
    df = build_frame(FT, cutoff, pd.Index([customer_id]), candidates, CFG)
    df["score"] = RANKER.score(df, cutoff)

    prof = customer_profile_asof(FT.customers, cutoff)
    limits = CFG["risk"]["policies"].get(policy, {r: 7 for r in CFG["risk"]["max_level"]})
    limit = customer_max_level(prof, pd.Index([customer_id]), limits)
    recs = rerank(df, k, limit, hard=policy != "none", fillers=fillers)

    info = df.set_index("ISIN")
    items = []
    for _, r in recs.iterrows():
        isin = r.ISIN
        a = ASSETS.loc[isin] if isin in ASSETS.index else None
        row = info.loc[isin] if isin in info.index else None
        items.append({
            "rank": int(r["rank"]), "isin": isin,
            "name": None if a is None else a.assetName,
            "category": None if a is None else a.assetCategory,
            "sector": None if a is None or pd.isna(a.sector) else a.sector,
            "risk_level": int(lvl.get(isin, 7)),
            "bought_before": bool(row is not None and row.p_n_buys > 0),
            "source": "model" if row is not None else "suitable_popular_fill",
        })
    removed = int((df.a_risk_level.fillna(7) > int(limit.iloc[0])).sum()) if policy != "none" else 0
    return {
        "customer_id": customer_id, "cutoff": str(cutoff.date()),
        "risk_profile": str(prof.riskLevel.get(customer_id, "Unknown")),
        "max_risk_level": int(limit.iloc[0]), "policy": policy,
        "candidates_considered": int(len(df)), "candidates_removed_by_risk": removed,
        "recommendations": items,
    }
