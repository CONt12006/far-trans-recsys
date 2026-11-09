"""Collaborative filtering baselines: Item-kNN and ALS.

Both are fit at every cutoff on buys strictly before it. Interactions are
recency-weighted: w = sum over buys of exp(-age_days / tau). Snapshot holdings
(2018-01-02) are included as interactions with a lower weight: they show
what a customer owns, but are not a purchase decision.

Customers with no history (cold start) get global popularity, and every list
is tie-broken / padded with popularity.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # implicit recommends this

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .data import FarTrans

SNAPSHOT_WEIGHT = 0.3


def interactions(ft: FarTrans, cutoff: pd.Timestamp, items: pd.Index, tau_days: float = 180.0):
    """User x item CSR matrix of recency-weighted buys before `cutoff`."""
    t = ft.transactions
    b = t[(t.transactionType == "Buy") & (t.timestamp < cutoff) & t.ISIN.isin(items)]
    age = (cutoff - b.timestamp).dt.days.to_numpy()
    w = np.exp(-age / tau_days) * np.where(b.isSnapshot, SNAPSHOT_WEIGHT, 1.0)
    users = pd.Index(b.customerID.unique())
    ui = users.get_indexer(b.customerID)
    ii = items.get_indexer(b.ISIN)
    X = sp.csr_matrix((w, (ui, ii)), shape=(len(users), len(items)), dtype=np.float32)
    X.sum_duplicates()
    return X, users


def _popularity(ft: FarTrans, cutoff, items: pd.Index, days: int = 30) -> np.ndarray:
    b = ft.buys
    b = b[(b.timestamp < cutoff) & (b.timestamp >= cutoff - pd.Timedelta(days=days))]
    pop = b.groupby("ISIN").customerID.nunique().reindex(items, fill_value=0).to_numpy(float)
    return pop / max(pop.max(), 1.0)


def _topk(scores: np.ndarray, customers: pd.Index, items: pd.Index, k: int) -> pd.DataFrame:
    k = min(k, scores.shape[1])
    idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    part = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-part, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    sc = np.take_along_axis(part, order, axis=1)
    return pd.DataFrame({
        "customerID": np.repeat(customers.values, k),
        "ISIN": items.values[idx.ravel()],
        "score": sc.ravel(),
        "rank": np.tile(np.arange(1, k + 1), len(customers)),
    })


def _assemble(user_scores: np.ndarray, users: pd.Index, customers: pd.Index, pop: np.ndarray,
              items: pd.Index, k: int) -> pd.DataFrame:
    """Scores for known customers (+ tiny popularity tie-break), popularity for cold ones."""
    S = np.tile(pop, (len(customers), 1)) * 1e-3
    pos = users.get_indexer(customers)
    known = pos >= 0
    if known.any():
        us = user_scores[pos[known]]
        # rescale each user's scores to [0, 1] so popularity stays a pure tie-breaker
        mx = np.abs(us).max(axis=1, keepdims=True)
        S[known] += us / np.where(mx > 0, mx, 1.0)
    S[~known] = pop
    return _topk(S, customers, items, k)


def item_knn(ft, cutoff, customers, candidates, k, n_neighbors: int = 200, tau_days: float = 180.0,
             shrink: float = 50.0):
    # defaults tuned on validation cutoffs (scripts/tune_cf.py), see reports/tune_item_knn.csv
    items = pd.Index(candidates)
    X, users = interactions(ft, cutoff, items, tau_days)
    Xb = X.copy()
    Xb.data = np.log1p(Xb.data)
    # cosine similarity with shrinkage (penalises pairs supported by few users)
    co = (Xb.T @ Xb).toarray()
    norms = np.sqrt(np.diag(co))
    support = ((X > 0).astype(np.float32).T @ (X > 0).astype(np.float32)).toarray()
    sim = co / (np.outer(norms, norms) + 1e-9) * support / (support + shrink)
    # keep only top-n neighbours per item (diagonal kept: repeat buys are common)
    if n_neighbors < sim.shape[0]:
        thr = -np.partition(-sim, n_neighbors, axis=0)[n_neighbors]
        sim[sim < thr] = 0.0
    pos = users.get_indexer(customers)
    sub = Xb[pos[pos >= 0]]
    us = np.zeros((len(users), len(items)), dtype=np.float32)
    us[pos[pos >= 0]] = (sub @ sim)
    return _assemble(us, users, pd.Index(customers), _popularity(ft, cutoff, items), items, k)


def als(ft, cutoff, customers, candidates, k, factors: int = 128, reg: float = 0.01, iters: int = 20,
        alpha: float = 50.0, tau_days: float = 90.0, seed: int = 0):
    # defaults tuned on validation cutoffs (scripts/tune_cf.py), see reports/tune_als.csv;
    # alpha=50 is the edge of the grid -> worth extending
    from implicit.als import AlternatingLeastSquares
    from implicit.nearest_neighbours import bm25_weight

    items = pd.Index(candidates)
    X, users = interactions(ft, cutoff, items, tau_days)
    Xw = bm25_weight(X, K1=100, B=0.8).tocsr().astype(np.float32)
    model = AlternatingLeastSquares(factors=factors, regularization=reg, iterations=iters,
                                    alpha=alpha, random_state=seed, calculate_training_loss=False)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1, "blas"):
        model.fit(Xw, show_progress=False)
    uf = np.asarray(model.user_factors)
    itf = np.asarray(model.item_factors)
    pos = users.get_indexer(customers)
    us = np.zeros((len(users), len(items)), dtype=np.float32)
    known = pos[pos >= 0]
    us[known] = uf[known] @ itf.T
    return _assemble(us, users, pd.Index(customers), _popularity(ft, cutoff, items), items, k)
