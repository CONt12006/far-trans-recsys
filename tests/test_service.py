"""API tests. Skipped without the dataset; the full /recommend check runs only when the
ranker for 2022-09-01 is already saved (scripts/run_ranker.py), so pytest never trains a model."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/FAR-Trans/transactions.csv"
MODEL = ROOT / "data/processed/models/lgbm_2022-09-01.txt"
CUSTOMER = "DD1E7FA6C74787F13EC7"

pytestmark = pytest.mark.skipif(not DATA.exists(), reason="dataset not downloaded")


@pytest.fixture(scope="module")
def client():
    import sys

    sys.path.insert(0, str(ROOT))
    from fastapi.testclient import TestClient

    from service.app import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_validation_errors(client):
    assert client.get("/recommend", params={"customer_id": "nope"}).status_code == 404
    assert client.get("/recommend", params={"customer_id": CUSTOMER, "date": "2015-01-01"}).status_code == 400
    assert client.get("/recommend", params={"customer_id": CUSTOMER, "policy": "yolo"}).status_code == 422
    assert client.get("/recommend", params={"customer_id": CUSTOMER, "k": 0}).status_code == 422


@pytest.mark.skipif(not MODEL.exists(), reason="ranker for 2022-09-01 not trained yet")
def test_recommend_respects_risk_policy(client):
    r = client.get("/recommend", params={"customer_id": CUSTOMER, "date": "2022-09-15", "k": 5, "policy": "strict"})
    assert r.status_code == 200
    body = r.json()
    recs = body["recommendations"]
    assert body["cutoff"] == "2022-09-01"
    assert len(recs) == 5
    assert [x["rank"] for x in recs] == [1, 2, 3, 4, 5]
    assert len({x["isin"] for x in recs}) == 5
    assert all(x["risk_level"] <= body["max_risk_level"] for x in recs)

    free = client.get("/recommend", params={"customer_id": CUSTOMER, "date": "2022-09-15", "k": 5, "policy": "none"}).json()
    assert free["candidates_removed_by_risk"] == 0
