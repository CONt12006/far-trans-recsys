"""Asset risk levels (SRRI-like, 1..7) and customer suitability limits."""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_BOUNDS = [0.005, 0.02, 0.05, 0.10, 0.15, 0.25]  # UCITS SRRI volatility buckets


def annual_volatility(prices: pd.DataFrame, date: pd.Timestamp, window_days: int = 365, min_obs: int = 40) -> pd.Series:
    """Annualised volatility per ISIN from observed closes in [date - window, date).

    Uses each asset's own observations (not the forward-filled grid) so that
    stale prices do not create artificial zero returns.
    """
    p = prices[(prices.timestamp < date) & (prices.timestamp >= date - pd.Timedelta(days=window_days))]
    p = p.sort_values(["ISIN", "timestamp"])
    r = p.groupby("ISIN").closePrice.pct_change()
    g = r.groupby(p.ISIN)
    vol = g.std() * np.sqrt(252)
    vol[g.count() < min_obs] = np.nan
    return vol.replace([np.inf, -np.inf], np.nan)


def stale_share(prices: pd.DataFrame, date: pd.Timestamp, window_days: int = 180) -> pd.Series:
    """Share of observations with an unchanged close in [date - window, date)."""
    p = prices[(prices.timestamp < date) & (prices.timestamp >= date - pd.Timedelta(days=window_days))]
    p = p.sort_values(["ISIN", "timestamp"])
    r = p.groupby("ISIN").closePrice.pct_change()
    return (r.abs() < 1e-12).groupby(p.ISIN).mean()


def asset_risk_levels(prices: pd.DataFrame, assets: pd.DataFrame, date: pd.Timestamp,
                      bounds=DEFAULT_BOUNDS, window_days: int = 365, max_stale: float = 0.5) -> pd.Series:
    vol = annual_volatility(prices, date, window_days)
    lvl = pd.Series(np.digitize(vol.values, bounds) + 1, index=vol.index, dtype=float)
    lvl[vol.isna()] = np.nan
    # Suspended / illiquid stocks have a frozen price -> ~zero volatility -> they would look like
    # the SAFEST assets (level 1; e.g. FOLLIE FOLLIE after the 2018 scandal). Their risk is unknown: level 7.
    stale = stale_share(prices, date)
    cat0 = assets.set_index("ISIN").assetCategory
    frozen = stale[(stale > max_stale) & (stale.index.map(cat0) == "Stock")].index
    lvl.loc[lvl.index.intersection(frozen)] = 7
    # Fallback for short history: median level of the asset's category, else the max level.
    cat = assets.set_index("ISIN").assetCategory
    all_isins = cat.index.union(lvl.index)
    lvl = lvl.reindex(all_isins)
    cat_med = lvl.groupby(cat.reindex(all_isins)).median()
    lvl = lvl.fillna(cat.reindex(all_isins).map(cat_med)).fillna(7)
    return lvl.round().astype(int).rename("riskLevelAsset")


def customer_max_level(profiles: pd.DataFrame, customers: pd.Index, max_level: dict) -> pd.Series:
    risk = profiles.riskLevel.reindex(customers).fillna("Not_Available")
    return risk.map(max_level).fillna(7).astype(int).rename("maxRiskLevel")
