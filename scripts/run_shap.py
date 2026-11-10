"""SHAP analysis of the LightGBM ranker trained for the test cutoff.

Uses LightGBM's exact TreeSHAP (pred_contrib=True). Outputs:
  reports/figures/shap_summary.png       beeswarm, top-20 features
  reports/figures/shap_groups.png        share of |SHAP| by feature group
  reports/figures/shap_dependence.png    risk excess and past return vs. contribution
  reports/shap_importance.csv

Usage: python scripts/run_shap.py [--cutoff 2022-09-01] [--sample 30000]
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

from farrec.data import load  # noqa: E402
from farrec.ranker import TwoStageRanker  # noqa: E402

GROUPS = {"src_": "Скоры и ранги источников (CF)", "n_sources": "Скоры и ранги источников (CF)",
          "best_src": "Скоры и ранги источников (CF)", "p_": "Пара клиент-актив",
          "c_": "Клиент", "a_": "Актив"}
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f1f1c", "#55554f"


def group_of(f: str) -> str:
    return next(v for k, v in GROUPS.items() if f.startswith(k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2022-09-01")
    ap.add_argument("--sample", type=int, default=30000)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs/base.yaml"))
    ft = load(ROOT / cfg["data"]["raw_dir"])
    cutoff = pd.Timestamp(args.cutoff)

    rk = TwoStageRanker(ft, cfg, "lgbm", cache_dir=ROOT / "data/processed/frames")
    model = rk.fit(cutoff)
    df = rk.frame(cutoff).drop(columns="label")
    df = df.sample(min(args.sample, len(df)), random_state=0)
    X = rk._prep(df, cutoff)
    contrib = model.predict(X, pred_contrib=True)[:, :-1]  # last column = expected value
    feats = list(X.columns)
    sv = pd.DataFrame(contrib, columns=feats, index=X.index)

    imp = sv.abs().mean().sort_values(ascending=False)
    out = ROOT / "reports"
    imp.rename("mean_abs_shap").to_csv(out / "shap_importance.csv")
    grp = imp.groupby(imp.index.map(group_of)).sum().sort_values()
    grp = grp / grp.sum()
    print("Share of |SHAP| by group:")
    print(grp.round(3).to_string())
    print("\nTop-20 features by mean |SHAP|:")
    print(imp.head(20).round(4).to_string())

    # -- beeswarm (own implementation: shap's plot does not handle pandas categoricals)
    top = list(imp.index[:20])
    fig, ax = plt.subplots(figsize=(9, 7.5))
    rng = np.random.default_rng(0)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("bo", [BLUE, "#c3c2b7", ORANGE])
    for i, f in enumerate(top):
        v = sv[f].to_numpy()
        x = X[f]
        x = x.cat.codes.astype(float).where(x.cat.codes >= 0) if str(x.dtype) == "category" else x.astype(float)
        lo, hi = np.nanpercentile(x, 5), np.nanpercentile(x, 95)
        col = np.clip((x - lo) / (hi - lo + 1e-12), 0, 1)
        idx = rng.choice(len(v), size=min(4000, len(v)), replace=False)
        y = i + rng.normal(0, 0.12, len(idx))
        ax.scatter(v[idx], y, c=np.nan_to_num(col.to_numpy()[idx], nan=0.5), cmap=cmap, s=4, alpha=0.6, linewidths=0)
    ax.set_yticks(range(len(top)), top)
    ax.invert_yaxis()
    ax.axvline(0, color="#9a9a93", lw=1)
    ax.set_xlabel("SHAP (вклад в скор ранжирования)")
    ax.set_title("SHAP: топ-20 признаков (цвет: синий = низкое значение, оранжевый = высокое)",
                 loc="left", fontsize=11, color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "figures" / "shap_summary.png", dpi=150)

    # -- groups
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    ax.barh(grp.index, grp.values * 100, color=BLUE, height=0.6)
    for y, v in enumerate(grp.values):
        ax.text(v * 100 + 0.8, y, f"{v:.0%}", va="center", color=INK, fontsize=9)
    ax.set_xlabel("Доля суммарного |SHAP|, %")
    ax.set_title("Какие группы признаков решают", loc="left", fontsize=11, color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "figures" / "shap_groups.png", dpi=150)

    # -- dependence: does the model push risky assets / chase past returns?
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    d = pd.DataFrame({"x": X.p_risk_excess, "s": sv.p_risk_excess}).dropna()
    m = d.groupby("x").s.agg(["mean", "count"])
    m = m[m["count"] >= 50]
    axes[0].bar(m.index, m["mean"], color=[ORANGE if v > 0 else BLUE for v in m["mean"]], width=0.7)
    axes[0].axhline(0, color="#9a9a93", lw=1)
    axes[0].set_xlabel("Превышение уровня риска над лимитом клиента")
    axes[0].set_title("p_risk_excess: средний SHAP", loc="left", fontsize=11, color=INK)
    d = pd.DataFrame({"x": X.a_ret_90d, "s": sv.a_ret_90d}).dropna()
    d["bin"] = pd.qcut(d.x, 10, duplicates="drop")
    m = d.groupby("bin", observed=True).agg(x=("x", "median"), s=("s", "mean"))
    axes[1].plot(m.x * 100, m.s, "-o", color=BLUE, lw=2, ms=6)
    axes[1].axhline(0, color="#9a9a93", lw=1)
    axes[1].set_xlabel("Доходность актива за 90 дней до среза, % (децили)")
    axes[1].set_title("a_ret_90d: средний SHAP", loc="left", fontsize=11, color=INK)
    for ax in axes:
        ax.grid(True, color="#e6e5e0")
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "figures" / "shap_dependence.png", dpi=150)


if __name__ == "__main__":
    main()
