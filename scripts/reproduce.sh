#!/usr/bin/env bash
# Reproduce every number and figure in README from scratch.
# Expects the raw data in data/raw/FAR-Trans (see README).
# Takes ~3 h on a laptop, most of it the ranker on the 61 dates of the paper protocol (last step).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="python -X utf8"
export PYTHONUTF8=1

rm -rf data/processed            # feature cache and models depend on the code: always rebuild

$PY -m pytest -q tests                                   # leakage / unit tests (service test skips: no model yet)
$PY scripts/run_baselines.py                             # baselines, 12 cutoffs
$PY scripts/tune_cf.py --model item_knn                  # validation-only tuning (informative: defaults are in cf.py)
$PY scripts/tune_cf.py --model als
$PY scripts/tune_cf.py --model als_ext
$PY scripts/run_ranker.py --backend lgbm                 # two-stage model, 6 cutoffs (+ saves scores and models)
$PY scripts/run_significance.py                          # paired cluster bootstrap
$PY scripts/run_tradeoffs.py                             # risk lambda / return alpha curves
$PY scripts/run_suitable.py                              # post-filter vs. ranker trained on suitable assets only
$PY scripts/run_segments.py                              # quality by segment
$PY scripts/run_shap.py                                  # SHAP
$PY -m pytest -q tests/test_service.py                   # API test with the saved model
(cd notebooks && $PY -m jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb)
$PY scripts/run_paper_protocol.py --part baselines       # paper protocol, 61 dates
$PY scripts/run_paper_protocol.py --part ranker --paper_dates
# optional, slow (~1 h per cutoff):
# $PY scripts/run_ranker.py --backend catboost --n_eval 3
