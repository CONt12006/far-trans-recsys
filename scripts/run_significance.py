"""Is the ranker's improvement real? Paired cluster bootstrap over customers.

Per-customer NDCG@10 (all buys and first-time buys) for the ranker and the strongest
baselines on the last N cutoffs. A customer can appear in several cutoffs, so the
bootstrap resamples CUSTOMERS (clusters), not rows. Also reports a sign test over cutoffs.

Requires saved ranker scores (scripts/run_ranker.py).
Usage: python scripts/run_significance.py [--n_eval 6] [--n_boot 2000]
"""
from __future__ import annotations

import argparse
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farrec.baselines import BASELINES  # noqa: E402
from farrec.data import load  # noqa: E402
from farrec.evaluate import _drop_seen, prior_pairs  # noqa: E402
from farrec.metrics import ndcg_per_user  # noqa: E402
from farrec.splits import all_cutoffs, make_slice  # noqa: E402

BASES = ["repeat_then_pop", "item_knn", "als", "pop_30d"]


def per_user(ft, cfg, cuts, backend="lgbm") -> pd.DataFrame:
    rows = []
    for c in cuts:
        sl = make_slice(ft, c, cfg["split"]["target_days"])
        sc = pd.read_parquet(ROOT / f"data/processed/scores_{backend}/{c.date()}.parquet")
        sc = sc.sort_values(["customerID", "score"], ascending=[True, False])
        sc["rank"] = sc.groupby("customerID").cumcount() + 1
        recs = {"ranker": sc[sc["rank"] <= 50][["customerID", "ISIN", "score", "rank"]]}
        for m in BASES:
            recs[m] = BASELINES[m](ft, c, sl.eval_customers, sl.candidates, 50)
        seen = prior_pairs(ft, c)
        tn = sl.target.merge(seen.assign(_s=1), on=["customerID", "ISIN"], how="left")
        tn = tn[tn._s.isna()].drop(columns="_s")
        for m, r in recs.items():
            a = ndcg_per_user(r, sl.target).rename("ndcg")
            n = ndcg_per_user(_drop_seen(r, seen), tn).rename("ndcg_new")
            d = pd.concat([a, n], axis=1).reset_index().rename(columns={"index": "customerID"})
            d["cutoff"], d["method"] = c.date().isoformat(), m
            rows.append(d)
    return pd.concat(rows, ignore_index=True)


def cluster_bootstrap(diff: pd.DataFrame, col: str, n_boot: int, seed: int = 0):
    """diff: rows (customerID, cutoff, col=ranker-minus-baseline). Returns mean, 95% CI, p (one-sided, H0: <=0)."""
    d = diff.dropna(subset=[col])
    g = d.groupby("customerID")[col].agg(["sum", "count"])
    s, n = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    means = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    est = s.sum() / n.sum()
    return est, np.percentile(means, 2.5), np.percentile(means, 97.5), float((means <= 0).mean())


def sign_test_p(wins: int, n: int) -> float:
    """One-sided P(X >= wins) for X ~ Bin(n, 0.5)."""
    return sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_eval", type=int, default=6)
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs/base.yaml"))
    s = cfg["split"]
    ft = load(ROOT / cfg["data"]["raw_dir"])
    cuts = all_cutoffs(s["first_cutoff"], s["last_cutoff"], s["freq"])[-args.n_eval:]

    pu = per_user(ft, cfg, cuts)
    wide = pu.pivot_table(index=["customerID", "cutoff"], columns="method", values=["ndcg", "ndcg_new"])
    rows = []
    for b in BASES:
        for metric in ["ndcg", "ndcg_new"]:
            diff = (wide[(metric, "ranker")] - wide[(metric, b)]).rename("d").reset_index()
            est, lo, hi, p = cluster_bootstrap(diff, "d", args.n_boot)
            base_mean = wide[(metric, b)].mean()
            per_cut = diff.groupby("cutoff").d.mean()
            wins = int((per_cut > 0).sum())
            rows.append({"baseline": b, "metric": "NDCG@10" + ("_new" if metric == "ndcg_new" else ""),
                         "diff": est, "ci_low": lo, "ci_high": hi, "rel_%": 100 * est / base_mean,
                         "p_bootstrap": p, "cutoff_wins": f"{wins}/{len(per_cut)}",
                         "p_sign": sign_test_p(wins, len(per_cut))})
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "reports" / "significance.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"Ranker minus baseline, {len(cuts)} cutoffs, {pu.customerID.nunique()} customers, "
          f"cluster bootstrap n={args.n_boot}:")
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
