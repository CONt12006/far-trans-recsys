"""Leakage guards: features/candidates must not depend on data at or after the cutoff."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farrec.baselines import BASELINES  # noqa: E402
from farrec.data import FarTrans, load  # noqa: E402
from farrec.splits import all_cutoffs, assign_splits, make_slice  # noqa: E402

CUT = pd.Timestamp("2021-06-01")


@pytest.fixture(scope="module")
def ft():
    return load(ROOT / "data/raw/FAR-Trans")


def _truncate(ft: FarTrans, date) -> FarTrans:
    """Dataset with everything at/after `date` removed."""
    return FarTrans(
        ft.transactions[ft.transactions.timestamp < date],
        ft.customers[ft.customers.timestamp < date],
        ft.assets,
        ft.prices[ft.prices.timestamp < date],
        ft.price_wide.loc[ft.price_wide.index < date],
    )


@pytest.mark.parametrize("name", list(BASELINES))
def test_recs_identical_on_truncated_data(ft, name):
    sl = make_slice(ft, CUT, 60)
    custs = sl.eval_customers[:200]
    full = BASELINES[name](ft, CUT, custs, sl.candidates, 10)
    trunc = BASELINES[name](_truncate(ft, CUT), CUT, custs, sl.candidates, 10)
    pd.testing.assert_frame_equal(full.reset_index(drop=True), trunc.reset_index(drop=True))


def test_features_identical_on_truncated_data(ft):
    """Stage-2 features (incl. candidate generation) must not change when the future is deleted."""
    import yaml

    from farrec.features import build_frame

    cfg = yaml.safe_load(open(ROOT / "configs/base.yaml"))
    sl = make_slice(ft, CUT, 60)
    custs = sl.eval_customers[:300]
    full = build_frame(ft, CUT, custs, sl.candidates, cfg)
    trunc = build_frame(_truncate(ft, CUT), CUT, custs, sl.candidates, cfg)
    pd.testing.assert_frame_equal(full, trunc, check_exact=False, rtol=1e-5)


def test_target_windows_do_not_overlap_eval():
    cuts = all_cutoffs("2020-01-01", "2022-09-01")
    sp = assign_splits(cuts, 60)
    gap = pd.Timedelta(days=60)
    assert all(c + gap <= sp["valid"][0] for c in sp["train"])
    assert all(c + gap <= sp["test"][0] for c in sp["valid"])


def test_snapshot_not_counted_as_buys(ft):
    assert (ft.buys.timestamp == pd.Timestamp("2018-01-02")).sum() == 0
