"""Reconstruct customer portfolios from the transaction log.

Caveat: transactions start in 2018, so anything bought earlier is invisible.
A sell of more units than we have seen bought therefore happens; we clip
holdings at zero and report how often that occurs (`n_negative_clipped`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import price_asof


def holdings_asof(tx: pd.DataFrame, date: pd.Timestamp, price_wide: pd.DataFrame | None = None) -> pd.DataFrame:
    """Net units (and market value if prices are given) per (customerID, ISIN) before `date`."""
    hist = tx[tx.timestamp < date]
    sign = np.where(hist.transactionType == "Buy", 1.0, -1.0)
    h = (
        hist.assign(netUnits=sign * hist.units, netCash=sign * hist.totalValue)
        .groupby(["customerID", "ISIN"], as_index=False)[["netUnits", "netCash"]]
        .sum()
    )
    h.attrs["n_negative_clipped"] = int((h.netUnits < -1e-9).sum())
    h = h[h.netUnits > 1e-9].copy()
    if price_wide is not None:
        px = price_asof(price_wide, date)
        h["price"] = h.ISIN.map(px)
        h["value"] = h.netUnits * h.price
        # Assets without a known price: fall back to invested cash
        h["value"] = h.value.fillna(h.netCash.clip(lower=0))
    return h.reset_index(drop=True)


def portfolio_summary(holdings: pd.DataFrame, assets: pd.DataFrame) -> pd.DataFrame:
    """Per-customer portfolio size, diversity and category shares."""
    h = holdings.merge(assets[["ISIN", "assetCategory"]], on="ISIN", how="left")
    h["assetCategory"] = h.assetCategory.fillna("Unknown")
    total = h.groupby("customerID").value.sum().rename("pf_value")
    shares = (
        h.pivot_table(index="customerID", columns="assetCategory", values="value", aggfunc="sum", fill_value=0)
        .div(total, axis=0)
        .add_prefix("pf_share_")
    )
    n = h.groupby("customerID").ISIN.nunique().rename("pf_n_assets")
    w = h.value / h.customerID.map(total)
    hhi = (w**2).groupby(h.customerID).sum().rename("pf_hhi")
    return pd.concat([total, n, hhi, shares], axis=1)
