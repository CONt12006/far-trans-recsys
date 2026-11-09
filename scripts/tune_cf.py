"""Small grid search for CF hyper-parameters on VALIDATION cutoffs only.

Validation cutoffs = cutoffs whose target window ends before the test cutoff
(the test cutoff is never touched here). Metric: mean NDCG@10 and NDCG@10_new.

Usage: python scripts/tune_cf.py --model als|item_knn
"""
from __future__ import annotations

import argparse
import itertools
import sys
from functools import partial
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farrec.cf import als, item_knn  # noqa: E402
from farrec.data import load  # noqa: E402
from farrec.evaluate import evaluate  # noqa: E402
from farrec.splits import all_cutoffs, assign_splits, make_slice  # noqa: E402

GRIDS = {
    "als": {"factors": [32, 64, 128], "reg": [0.01, 0.1], "alpha": [5.0, 20.0, 50.0], "tau_days": [90.0, 365.0]},
    "item_knn": {"n_neighbors": [20, 50, 200], "shrink": [0.0, 10.0, 50.0], "tau_days": [90.0, 180.0, 365.0]},
}
FN = {"als": als, "item_knn": item_knn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(GRIDS))
    ap.add_argument("--n_cutoffs", type=int, default=6)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs/base.yaml"))
    s = cfg["split"]
    ft = load(ROOT / cfg["data"]["raw_dir"])
    cuts = all_cutoffs(s["first_cutoff"], s["last_cutoff"], s["freq"])
    sp = assign_splits(cuts, s["target_days"], s["n_valid"], s["n_test"])
    gap = pd.Timedelta(days=s["target_days"])
    val_cuts = [c for c in cuts if c + gap <= sp["test"][0]][-args.n_cutoffs:]
    print("validation cutoffs:", [c.date().isoformat() for c in val_cuts])
    slices = [make_slice(ft, c, s["target_days"]) for c in val_cuts]

    grid = GRIDS[args.model]
    recs = {}
    for vals in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), vals))
        recs[str(params)] = partial(FN[args.model], **params)
    res = evaluate(ft, slices, recs, cfg, verbose=False)
    tab = res.groupby("method")[["NDCG@10", "NDCG@10_new", "HR@10_new"]].mean()
    tab = tab.sort_values("NDCG@10_new", ascending=False)
    out = ROOT / "reports" / f"tune_{args.model}.csv"
    tab.to_csv(out)
    pd.set_option("display.width", 250, "display.max_colwidth", 120)
    print(tab.round(4).head(10).to_string())


if __name__ == "__main__":
    main()
