"""Run all baselines and write the results tables.

  reports/baselines_by_cutoff.csv  every method x cutoff
  reports/baselines_summary.csv    mean/std over the backtest cutoffs + test cutoff

Usage: python scripts/run_baselines.py [--config configs/base.yaml] [--methods pop_30d,als]
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
from farrec.splits import all_cutoffs, assign_splits, make_slice  # noqa: E402

SHOW = ["NDCG@10", "HR@10", "R@10", "NDCG@10_new", "HR@10_new", "Ret@5", "Excess@5", "Unsuit@5", "Cov@10"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/base.yaml"))
    ap.add_argument("--methods", default=None, help="comma-separated subset of methods")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    s = cfg["split"]

    methods = BASELINES if not args.methods else {m: BASELINES[m] for m in args.methods.split(",")}
    ft = load(ROOT / cfg["data"]["raw_dir"])
    cuts = all_cutoffs(s["first_cutoff"], s["last_cutoff"], s["freq"])
    sp = assign_splits(cuts, s["target_days"], s["n_valid"], s["n_test"])
    backtest = cuts[-cfg["eval"]["backtest_n"]:]
    print("test:", [c.date().isoformat() for c in sp["test"]],
          "| backtest:", backtest[0].date(), "..", backtest[-1].date())

    slices = [make_slice(ft, c, s["target_days"]) for c in backtest]
    res = evaluate(ft, slices, methods, cfg)
    test_dates = {c.date().isoformat() for c in sp["test"]}
    res["is_test"] = res.cutoff.isin(test_dates)

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    res.to_csv(out / "baselines_by_cutoff.csv", index=False)

    order = list(methods)
    mean = res.groupby("method")[SHOW].mean().reindex(order)
    std = res.groupby("method")[SHOW].std().reindex(order)
    test = res[res.is_test].set_index("method")[SHOW].reindex(order)
    summary = pd.concat({"backtest_mean": mean, "backtest_std": std, "test": test}, axis=1)
    summary.to_csv(out / "baselines_summary.csv")

    pd.set_option("display.width", 250)
    print("\nBacktest mean over", len(backtest), "cutoffs:")
    print(mean.round(4).to_string())
    print("\nBacktest std:")
    print(std[["NDCG@10", "NDCG@10_new", "Ret@5"]].round(4).to_string())
    print("\nTest cutoff:")
    print(test.round(4).to_string())


if __name__ == "__main__":
    main()
