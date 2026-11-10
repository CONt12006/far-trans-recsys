"""Filter after ranking vs. rank only among suitable assets (strict policy, 0% unsuitable in both).

  post_filter    : ranker trained on all candidates, unsuitable removed afterwards (hard filter)
  suitable_ranker: ranker trained and applied ONLY on candidates allowed for the customer
  pop_suitable   : most popular suitable assets (no personalisation)

All three lists are padded with popular suitable assets when a customer has too few candidates.
Evaluated twice: on all purchases, and on purchases of suitable assets only (what the policy allows
us to "hit" at all).

Requires scores of the standard ranker (scripts/run_ranker.py).
Usage: python scripts/run_suitable.py [--n_eval 6]
"""
from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farrec.data import customer_profile_asof, load  # noqa: E402
from farrec.evaluate import evaluate  # noqa: E402
from farrec.postprocess import ScoredRecommender, suitable_fillers  # noqa: E402
from farrec.ranker import TwoStageRanker  # noqa: E402
from farrec.risk import asset_risk_levels, customer_max_level  # noqa: E402
from farrec.splits import all_cutoffs, make_slice  # noqa: E402

SHOW = ["NDCG@10", "HR@10", "NDCG@10_new", "HR@10_new", "Ret@5", "Excess@5", "Unsuit@5", "n_users"]


def pop_suitable(ft, cfg, policy):
    def rec(_, cutoff, customers, candidates, k):
        lvl = asset_risk_levels(ft.prices, ft.assets, cutoff, cfg["risk"]["srri_bounds"])
        b = ft.buys[(ft.buys.timestamp < cutoff) & (ft.buys.timestamp >= cutoff - pd.Timedelta(days=30))]
        fill = suitable_fillers(lvl.reindex(candidates).dropna(), b.groupby("ISIN").customerID.nunique(), n=k)
        lim = customer_max_level(customer_profile_asof(ft.customers, cutoff), pd.Index(customers), policy)
        rows = [pd.DataFrame({"customerID": c, "ISIN": fill[int(l)], "score": 0.0,
                              "rank": range(1, len(fill[int(l)]) + 1)}) for c, l in lim.items()]
        return pd.concat(rows, ignore_index=True)
    return rec


def suitable_target(ft, cfg, sl, policy):
    lvl = asset_risk_levels(ft.prices, ft.assets, sl.cutoff, cfg["risk"]["srri_bounds"])
    lim = customer_max_level(customer_profile_asof(ft.customers, sl.cutoff), sl.eval_customers, policy)
    t = sl.target
    t = t[t.ISIN.map(lvl).fillna(7) <= t.customerID.map(lim)]
    return replace(sl, target=t.reset_index(drop=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_eval", type=int, default=6)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs/base.yaml"))
    s = cfg["split"]
    policy = cfg["risk"]["policies"]["strict"]
    c2 = copy.deepcopy(cfg)
    c2["risk"]["max_level"] = policy
    ft = load(ROOT / cfg["data"]["raw_dir"])
    cuts = all_cutoffs(s["first_cutoff"], s["last_cutoff"], s["freq"])[-args.n_eval:]

    # train the suitable-only ranker (writes its scores to data/processed/scores_lgbm_suitable)
    suit = TwoStageRanker(ft, cfg, "lgbm", cache_dir=ROOT / "data/processed/frames", suitable_only=True)
    for c in cuts:
        sl = make_slice(ft, c, s["target_days"])
        suit(ft, c, sl.eval_customers, sl.candidates, 50)

    recs = {
        "pop_suitable": pop_suitable(ft, cfg, policy),
        "post_filter": ScoredRecommender(ft, cfg, ROOT / "data/processed/scores_lgbm", policy, hard=True),
        "suitable_ranker": ScoredRecommender(ft, cfg, ROOT / "data/processed/scores_lgbm_suitable", policy, hard=True),
    }
    slices = [make_slice(ft, c, s["target_days"]) for c in cuts]
    out = []
    for name, sls in [("all purchases", slices),
                      ("suitable purchases", [suitable_target(ft, cfg, sl, policy) for sl in slices])]:
        sls = [x for x in sls if len(x.target)]
        r = evaluate(ft, sls, recs, c2, verbose=False)
        t = r.groupby("method")[SHOW].mean().reindex(list(recs))
        t.insert(0, "target", name)
        out.append(t)
    res = pd.concat(out)
    res.to_csv(ROOT / "reports" / "suitable_ranker.csv")
    pd.set_option("display.width", 220)
    print(res.round(4).to_string())


if __name__ == "__main__":
    main()
