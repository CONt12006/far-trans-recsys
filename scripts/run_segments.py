"""Quality by customer segment: ranker vs. strongest baselines (per-customer NDCG@10).

Segments are defined from information available at the cutoff only.
Requires saved ranker scores (scripts/run_ranker.py).

Usage: python scripts/run_segments.py [--backend lgbm] [--n_eval 6]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farrec.baselines import BASELINES  # noqa: E402
from farrec.data import load  # noqa: E402
from farrec.evaluate import _drop_seen, prior_pairs  # noqa: E402
from farrec.metrics import ndcg_per_user  # noqa: E402
from farrec.splits import all_cutoffs, make_slice  # noqa: E402

COLORS = {"item_knn": "#2a78d6", "repeat_then_pop": "#eb6834", "ranker_lgbm": "#1baf7a"}


def segments(frame: pd.DataFrame) -> pd.DataFrame:
    c = frame.drop_duplicates("customerID").set_index("customerID")
    seg = pd.DataFrame(index=c.index)
    seg["Новички (≤2 покупок)"] = c.c_n_buys_total <= 2
    seg["Активные (≥10 покупок за 90д)"] = c.c_n_buys_90d >= 10
    seg["Консерваторы и Income"] = c.c_riskLevel.isin(["Conservative", "Income"])
    seg["Balanced и Aggressive"] = c.c_riskLevel.isin(["Balanced", "Aggressive"])
    seg["Mass"] = c.c_customerType == "Mass"
    seg["Premium"] = c.c_customerType == "Premium"
    seg["Все"] = True
    return seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="lgbm")
    ap.add_argument("--n_eval", type=int, default=6)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs/base.yaml"))
    s = cfg["split"]
    ft = load(ROOT / cfg["data"]["raw_dir"])
    cuts = all_cutoffs(s["first_cutoff"], s["last_cutoff"], s["freq"])[-args.n_eval:]
    name = f"ranker_{args.backend}"

    rows = []
    for c in cuts:
        sl = make_slice(ft, c, s["target_days"])
        sc = pd.read_parquet(ROOT / f"data/processed/scores_{args.backend}/{c.date()}.parquet")
        sc = sc.sort_values(["customerID", "score"], ascending=[True, False])
        sc["rank"] = sc.groupby("customerID").cumcount() + 1
        recs = {name: sc[["customerID", "ISIN", "score", "rank"]]}
        for m in ["repeat_then_pop", "item_knn"]:
            recs[m] = BASELINES[m](ft, c, sl.eval_customers, sl.candidates, 50)
        seen = prior_pairs(ft, c)
        tn = sl.target.merge(seen.assign(_s=1), on=["customerID", "ISIN"], how="left")
        tn = tn[tn._s.isna()].drop(columns="_s")
        seg = segments(sc)
        for m, r in recs.items():
            all_ = ndcg_per_user(r, sl.target).rename("ndcg")
            new = ndcg_per_user(_drop_seen(r, seen), tn).rename("ndcg_new")
            d = pd.concat([all_, new], axis=1).join(seg, how="left")
            for sname in seg.columns:
                g = d[d[sname].fillna(False).astype(bool)]
                rows.append({"cutoff": c, "method": m, "segment": sname, "n": len(g),
                             "NDCG@10": g.ndcg.mean(), "NDCG@10_new": g.ndcg_new.mean()})
    res = pd.DataFrame(rows)
    out = ROOT / "reports"
    res.to_csv(out / "segments_by_cutoff.csv", index=False)
    tab = res.groupby(["segment", "method"])[["n", "NDCG@10", "NDCG@10_new"]].mean().unstack("method")
    pd.set_option("display.width", 250)
    print(tab.round(4).to_string())

    # figure: NDCG@10 by segment
    order = list(segments(sc).columns)
    methods = ["repeat_then_pop", "item_knn", name]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, metric, title in zip(axes, ["NDCG@10", "NDCG@10_new"], ["NDCG@10 (все покупки)", "NDCG@10 (новые активы)"]):
        m = res.groupby(["segment", "method"])[metric].mean().unstack("method").reindex(order)[methods]
        y = np.arange(len(order))
        h = 0.26
        for i, meth in enumerate(methods):
            ax.barh(y + (i - 1) * h, m[meth], height=h - 0.03, color=COLORS.get(meth, "#1baf7a"),
                    label=meth, edgecolor="white", linewidth=1)
        ax.set_yticks(y, order)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(True, axis="x", color="#e6e5e0")
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle(f"Качество по сегментам клиентов, среднее по {len(cuts)} срезам", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "figures" / "segments.png", dpi=150)


if __name__ == "__main__":
    main()
