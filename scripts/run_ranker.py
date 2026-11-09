"""Evaluate the two-stage ranker against the strongest baselines on the last N cutoffs.

The ranker is retrained for every evaluation cutoff using only earlier cutoffs.

Usage: python scripts/run_ranker.py [--backend lgbm|catboost] [--n_eval 6]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farrec.baselines import BASELINES  # noqa: E402
from farrec.data import load  # noqa: E402
from farrec.evaluate import evaluate  # noqa: E402
from farrec.ranker import TwoStageRanker  # noqa: E402
from farrec.splits import all_cutoffs, make_slice  # noqa: E402

SHOW = ["NDCG@10", "HR@10", "R@10", "NDCG@10_new", "HR@10_new", "Ret@5", "Excess@5", "Unsuit@5", "Cov@10"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="lgbm", choices=["lgbm", "catboost"])
    ap.add_argument("--n_eval", type=int, default=6)
    ap.add_argument("--n_train", type=int, default=12)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs/base.yaml"))
    s = cfg["split"]
    ft = load(ROOT / cfg["data"]["raw_dir"])

    cuts = all_cutoffs(s["first_cutoff"], s["last_cutoff"], s["freq"])[-args.n_eval:]
    ranker = TwoStageRanker(ft, cfg, backend=args.backend, n_train_cutoffs=args.n_train,
                            cache_dir=ROOT / "data/processed/frames")
    methods = {m: BASELINES[m] for m in ["pop_30d", "repeat_then_pop", "item_knn", "als"]}
    methods[f"ranker_{args.backend}"] = ranker

    slices = [make_slice(ft, c, s["target_days"]) for c in cuts]
    res = evaluate(ft, slices, methods, cfg)
    out = ROOT / "reports"
    res.to_csv(out / f"ranker_{args.backend}_by_cutoff.csv", index=False)

    order = list(methods)
    pd.set_option("display.width", 250)
    print(f"\nMean over {len(cuts)} cutoffs ({cuts[0].date()}..{cuts[-1].date()}):")
    print(res.groupby("method")[SHOW].mean().reindex(order).round(4).to_string())
    piv = res.pivot(index="cutoff", columns="method", values="NDCG@10")[order]
    print("\nNDCG@10 by cutoff:")
    print(piv.round(4).to_string())

    imp = ranker.feature_importance(cuts[-1])
    imp.to_csv(out / f"ranker_{args.backend}_importance.csv")
    print("\nTop-20 features (test cutoff model):")
    print((imp / imp.sum()).head(20).round(4).to_string())


if __name__ == "__main__":
    main()
