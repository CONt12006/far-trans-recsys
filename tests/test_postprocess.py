import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from farrec.postprocess import rerank, suitable_fillers  # noqa: E402


def _frame():
    # customer A (limit 3): best-scored assets are risky; customer B (limit 7)
    rows = []
    for c in ["A", "B"]:
        for i, (lvl, sc) in enumerate([(7, 0.9), (6, 0.8), (3, 0.5), (2, 0.4), (7, 0.3)]):
            rows.append({"customerID": c, "ISIN": f"X{i}", "score": sc, "a_risk_level": lvl,
                         "a_ret_90d": i * 0.01, "a_ret_365d": 0.0, "a_vol_365d": 0.1})
    return pd.DataFrame(rows)


LIMIT = pd.Series({"A": 3, "B": 7})


def test_lambda_zero_keeps_model_order():
    r = rerank(_frame(), 5, LIMIT, risk_lambda=0.0)
    assert list(r[r.customerID == "A"].ISIN) == ["X0", "X1", "X2", "X3", "X4"]


def test_large_lambda_pushes_unsuitable_down():
    r = rerank(_frame(), 2, LIMIT, risk_lambda=10.0)
    assert set(r[r.customerID == "A"].ISIN) == {"X2", "X3"}
    assert list(r[r.customerID == "B"].ISIN) == ["X0", "X1"]  # aggressive customer unaffected


def test_hard_filter_pads_with_suitable_assets():
    lvl = pd.Series({"X0": 7, "X1": 6, "X2": 3, "X3": 2, "X4": 7, "S1": 1, "S2": 3})
    pop = pd.Series({"S1": 5, "S2": 10, "X0": 100})
    r = rerank(_frame(), 4, LIMIT, hard=True, fillers=suitable_fillers(lvl, pop))
    a = r[r.customerID == "A"]
    assert len(a) == 4
    assert a.ISIN.map(lvl).max() <= 3
    assert list(a.ISIN[:2]) == ["X2", "X3"]  # real candidates before fillers


def test_alpha_zero_ranks_by_signal():
    r = rerank(_frame(), 5, LIMIT, alpha=0.0, signal="mom_90")
    assert list(r[r.customerID == "B"].ISIN) == ["X4", "X3", "X2", "X1", "X0"]
