"""Compare with the FAR-Trans paper using its own protocol (see src/farrec/paper_protocol.py).

  --part baselines : all 61 bi-weekly dates, cheap baselines (validates the re-implementation)
  --part ranker    : monthly dates in [--from, --to], ranker vs. baselines on the same dates

Usage:
  python scripts/run_paper_protocol.py --part baselines
  python scripts/run_paper_protocol.py --part ranker --from 2021-06-01 --to 2022-05-01
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
from farrec.paper_protocol import PAPER_TABLE2, evaluate_paper, paper_dates  # noqa: E402
from farrec.ranker import TwoStageRanker  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["baselines", "ranker"], required=True)
    ap.add_argument("--from", dest="start", default="2021-06-01")
    ap.add_argument("--to", dest="end", default="2022-05-01")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs/base.yaml"))
    ft = load(ROOT / cfg["data"]["raw_dir"])
    base = {m: BASELINES[m] for m in ["random", "pop_30d", "item_knn", "als"]}

    if args.part == "baselines":
        dates, recs = paper_dates(), base
    else:
        dates = list(pd.date_range(args.start, args.end, freq="MS"))
        ranker = TwoStageRanker(ft, cfg, "lgbm", cache_dir=ROOT / "data/processed/frames")
        recs = {**{m: base[m] for m in ["pop_30d", "item_knn"]}, "ranker_lgbm": ranker.recommend_any}

    res = evaluate_paper(ft, dates, recs)
    out = ROOT / "reports" / f"paper_protocol_{args.part}.csv"
    res.to_csv(out, index=False)
    summ = res.groupby("method")[["nDCG@10", "ROI@10", "ROI_market"]].mean().reindex(list(recs))
    pd.set_option("display.width", 200)
    print(f"\nPaper protocol, {len(dates)} dates ({dates[0].date()}..{dates[-1].date()}):")
    print(summ.round(4).to_string())
    print("\nPaper Table 2 (2019-08..2022-05, 61 dates):")
    print(pd.DataFrame(PAPER_TABLE2, index=["nDCG@10", "ROI@10"]).T.to_string())


if __name__ == "__main__":
    main()
