"""Stage 1: candidate generation.

Union of the top-N lists of several cheap recommenders. Each source's score and
rank are kept: they become features for the stage-2 ranker.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import popularity, repeat_popularity
from .cf import als, item_knn
from .data import FarTrans

SOURCES = {
    "repeat": repeat_popularity,
    "knn": item_knn,
    "als": als,
    "pop": popularity,
}
MISSING_RANK = 999


def generate(ft: FarTrans, cutoff: pd.Timestamp, customers: pd.Index, candidates: pd.Index,
             n_per_source: int = 50) -> pd.DataFrame:
    """One row per (customer, ISIN) candidate with src_<name>_score / src_<name>_rank columns."""
    frames = []
    for name, fn in SOURCES.items():
        r = fn(ft, cutoff, customers, candidates, n_per_source)
        frames.append(r[["customerID", "ISIN", "score", "rank"]]
                      .rename(columns={"score": f"src_{name}_score", "rank": f"src_{name}_rank"})
                      .set_index(["customerID", "ISIN"]))
    out = pd.concat(frames, axis=1, join="outer")
    for name in SOURCES:
        out[f"src_{name}_rank"] = out[f"src_{name}_rank"].fillna(MISSING_RANK)
        out[f"src_{name}_score"] = out[f"src_{name}_score"].fillna(0.0)
    out["n_sources"] = sum((out[f"src_{n}_rank"] < MISSING_RANK).astype(int) for n in SOURCES)
    out["best_src_rank"] = out[[f"src_{n}_rank" for n in SOURCES]].min(axis=1)
    return out.reset_index()


def candidate_recall(cands: pd.DataFrame, target: pd.DataFrame) -> dict:
    """Share of target (customer, asset) pairs that made it into the candidate set: the ceiling for stage 2."""
    hit = target.merge(cands[["customerID", "ISIN"]], on=["customerID", "ISIN"], how="inner")
    per_user = cands.groupby("customerID").size()
    return {"cand_recall": len(hit) / max(len(target), 1),
            "cands_per_user": float(per_user.mean()),
            "users_with_any_hit": float(hit.customerID.nunique() / max(target.customerID.nunique(), 1))}
