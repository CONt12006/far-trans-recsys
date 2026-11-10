"""Week 4: trade-off curves on top of the saved ranker scores.

  1. Risk: soft penalty lambda (and hard filter) under two suitability policies
  2. Return: alpha * P(buy) + (1 - alpha) * expected return, for several return signals

Requires scores from scripts/run_ranker.py (data/processed/scores_<backend>/*.parquet).
Outputs reports/tradeoff_risk.csv, reports/tradeoff_return.csv and two figures.

Usage: python scripts/run_tradeoffs.py [--backend lgbm] [--n_eval 6]
"""
from __future__ import annotations

import argparse
import copy
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

from farrec.data import load  # noqa: E402
from farrec.evaluate import evaluate  # noqa: E402
from farrec.postprocess import ScoredRecommender  # noqa: E402
from farrec.splits import all_cutoffs, make_slice  # noqa: E402

METRICS = ["NDCG@10", "NDCG@10_new", "HR@10", "Ret@5", "Excess@5", "WinRate@5", "Unsuit@5", "RiskExcess@5"]
LAMBDAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
ALPHAS = [1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.0]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (validated default palette)


def run_grid(ft, cfg, slices, recs: dict, policy: dict) -> pd.DataFrame:
    c = copy.deepcopy(cfg)
    c["risk"]["max_level"] = policy  # suitability metric follows the same policy
    res = evaluate(ft, slices, recs, c, verbose=False)
    return res.groupby("method")[METRICS].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="lgbm")
    ap.add_argument("--n_eval", type=int, default=6)
    ap.add_argument("--plot_only", action="store_true", help="redraw figures from saved CSVs")
    args = ap.parse_args()
    out = ROOT / "reports"
    if args.plot_only:
        plot_risk(pd.read_csv(out / "tradeoff_risk.csv"), out / "figures" / "risk_tradeoff.png")
        plot_return(pd.read_csv(out / "tradeoff_return.csv"), out / "figures" / "return_tradeoff.png")
        return
    cfg = yaml.safe_load(open(ROOT / "configs/base.yaml"))
    s = cfg["split"]
    ft = load(ROOT / cfg["data"]["raw_dir"])
    score_dir = ROOT / f"data/processed/scores_{args.backend}"
    cuts = all_cutoffs(s["first_cutoff"], s["last_cutoff"], s["freq"])[-args.n_eval:]
    slices = [make_slice(ft, c, s["target_days"]) for c in cuts]
    out = ROOT / "reports"
    pd.set_option("display.width", 250)

    # ---------------------------------------------------------------- risk
    rows = []
    for pname, policy in cfg["risk"]["policies"].items():
        recs = {f"lambda={l}": ScoredRecommender(ft, cfg, score_dir, policy, risk_lambda=l) for l in LAMBDAS}
        recs["hard"] = ScoredRecommender(ft, cfg, score_dir, policy, hard=True)
        r = run_grid(ft, cfg, slices, recs, policy)
        r["policy"] = pname
        rows.append(r)
        print(f"\nRisk policy: {pname}")
        print(r[METRICS].round(4).to_string())
    risk = pd.concat(rows).reset_index()
    risk.to_csv(out / "tradeoff_risk.csv", index=False)

    # ---------------------------------------------------------------- return
    strict = cfg["risk"]["policies"]["strict"]
    rows = []
    for sig in ["mom_90", "mom_365", "low_vol"]:
        recs = {f"alpha={a}": ScoredRecommender(ft, cfg, score_dir, strict, alpha=a, signal=sig) for a in ALPHAS}
        r = run_grid(ft, cfg, slices, recs, strict)
        r["signal"] = sig
        rows.append(r)
        print(f"\nReturn signal: {sig}")
        print(r[METRICS].round(4).to_string())
    ret = pd.concat(rows).reset_index()
    ret.to_csv(out / "tradeoff_return.csv", index=False)

    plot_risk(risk, out / "figures" / "risk_tradeoff.png")
    plot_return(ret, out / "figures" / "return_tradeoff.png")


def _style(ax):
    ax.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#9a9a93")
    ax.tick_params(colors="#55554f")


def plot_risk(risk: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for color, (pname, g) in zip(COLORS, risk.groupby("policy", sort=False)):
        soft = g[g.method != "hard"].copy()
        soft["lam"] = soft.method.str.split("=").str[1].astype(float)
        soft = soft.sort_values("lam")
        hard = g[g.method == "hard"]
        for ax, y in zip(axes, ["NDCG@10", "NDCG@10_new"]):
            ax.plot(soft["Unsuit@5"], soft[y], "-o", color=color, lw=2, ms=6, label=f"{pname}: штраф λ")
            ax.plot(hard["Unsuit@5"], hard[y], "s", color=color, ms=9, mec="white", mew=2, label=f"{pname}: жёсткий фильтр")
            if pname == "strict":  # label one curve only; both use the same lambda grid
                offs = {0.0: (-34, 6), 0.2: (8, -4), 0.5: (8, -2), 1.0: (6, -12)}
                for _, r in soft.iterrows():
                    if r.lam in offs:
                        ax.annotate(f"λ={r.lam:g}", (r["Unsuit@5"], r[y]), textcoords="offset points",
                                    xytext=offs[r.lam], fontsize=8, color="#55554f")
    for ax, t in zip(axes, ["NDCG@10 (все покупки)", "NDCG@10 (новые активы)"]):
        ax.set_xlabel("Доля неподходящих по риску в топ-5")
        ax.set_title(t, loc="left", fontsize=11, color="#1f1f1c")
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Компромисс: качество рекомендаций против соответствия риск-профилю", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_return(ret: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for color, (sig, g) in zip(COLORS, ret.groupby("signal", sort=False)):
        g = g.copy()
        g["alpha"] = g.method.str.split("=").str[1].astype(float)
        g = g.sort_values("alpha", ascending=False)
        axes[0].plot(g.alpha, g["NDCG@10"], "-o", color=color, lw=2, ms=6, label=sig)
        axes[1].plot(g.alpha, g["Excess@5"] * 100, "-o", color=color, lw=2, ms=6, label=sig)
    axes[1].axhline(0, color="#9a9a93", lw=1)
    axes[0].set_title("NDCG@10", loc="left", fontsize=11, color="#1f1f1c")
    axes[1].set_title("Доходность топ-5 сверх рынка за 60 дней, п.п.", loc="left", fontsize=11, color="#1f1f1c")
    for ax in axes:
        ax.set_xlabel("α (1 = только вероятность покупки, 0 = только сигнал доходности)")
        ax.invert_xaxis()
        _style(ax)
    axes[0].legend(frameon=False, fontsize=9, title="сигнал", title_fontsize=9)
    fig.suptitle("Финансовый скор: α·P(покупка) + (1−α)·ожидаемая доходность", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
