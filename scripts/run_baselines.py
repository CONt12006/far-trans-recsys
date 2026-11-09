"""Run all baselines on valid and test cutoffs and write reports/baselines.csv.

Usage: python scripts/run_baselines.py [--config configs/base.yaml]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/base.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    s = cfg["split"]

    ft = load(ROOT / cfg["data"]["raw_dir"])
    cuts = all_cutoffs(s["first_cutoff"], s["last_cutoff"], s["freq"])
    sp = assign_splits(cuts, s["target_days"], s["n_valid"], s["n_test"])
    print({k: [c.date().isoformat() for c in v][-3:] for k, v in sp.items()})

    frames = []
    for part in ("valid", "test"):
        slices = [make_slice(ft, c, s["target_days"]) for c in sp[part]]
        res = evaluate(ft, slices, BASELINES, cfg)
        res["split"] = part
        frames.append(res)
    res = pd.concat(frames)

    out = ROOT / "reports" / "baselines.csv"
    out.parent.mkdir(exist_ok=True)
    res.to_csv(out, index=False)
    cols = ["split", "method", "n_users", "P@5", "R@10", "NDCG@10", "HR@10",
            "Ret@5", "Ret_bought", "Ret_market_ew", "Unsuit@5", "Cov@10"]
    pd.set_option("display.width", 200)
    print(res[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
