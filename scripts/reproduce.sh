#!/usr/bin/env bash
# Reproduce every number and figure in README from scratch.
# Expects the raw data in data/raw/FAR-Trans (see README). Takes ~1-2 h on a laptop
# (CatBoost is the slowest step and is optional).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="python -X utf8"

rm -rf data/processed            # feature cache depends on the code: always rebuild

$PY -m pytest -q tests                                   # leakage / unit tests
$PY scripts/run_baselines.py                             # baselines, 12 cutoffs
$PY scripts/tune_cf.py --model item_knn                  # validation-only tuning
$PY scripts/tune_cf.py --model als
$PY scripts/run_ranker.py --backend lgbm                 # two-stage model, 6 cutoffs (+ saves scores)
$PY scripts/run_tradeoffs.py                             # risk lambda / return alpha curves
$PY scripts/run_segments.py                              # quality by segment
$PY scripts/run_shap.py                                  # SHAP
$PY scripts/run_paper_protocol.py --part baselines       # paper protocol, 61 dates
$PY scripts/run_paper_protocol.py --part ranker --from 2020-06-01 --to 2021-11-01
# optional, slow:
# $PY scripts/run_ranker.py --backend catboost --n_eval 3
