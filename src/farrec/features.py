"""Stage 2 features for (customer, asset) candidate pairs at a cutoff.

Rule: every feature uses data strictly before the cutoff. The label is
computed from the target window and lives in a separate column (`label`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .candidates import generate
from .data import FarTrans, customer_profile_asof
from .portfolio import holdings_asof, portfolio_summary
from .risk import annual_volatility, asset_risk_levels, customer_max_level

CAT_FEATURES = ["c_riskLevel", "c_customerType", "c_investmentCapacity",
                "a_category", "a_subCategory", "a_sector", "a_industry", "a_market"]


def _days(cutoff, s: pd.Series) -> pd.Series:
    return (cutoff - s).dt.days.astype(float)


# ---------------------------------------------------------------- customers
def customer_features(ft: FarTrans, cutoff, customers: pd.Index, cfg: dict) -> pd.DataFrame:
    prof = customer_profile_asof(ft.customers, cutoff)
    f = pd.DataFrame(index=customers)
    f.index.name = "customerID"
    f["c_riskLevel"] = prof.riskLevel.reindex(customers).fillna("Unknown")
    f["c_customerType"] = prof.customerType.reindex(customers).fillna("Unknown")
    f["c_investmentCapacity"] = prof.investmentCapacity.reindex(customers).fillna("Unknown")
    f["c_riskPredicted"] = prof.riskPredicted.reindex(customers).astype(float)
    # NOT used: lastQuestionnaireDate is often AFTER the profile row timestamp (38.5% of rows),
    # i.e. profiles were back-filled with later questionnaires -> "days since questionnaire" leaks the future.
    f["c_max_risk_level"] = customer_max_level(prof, customers, cfg["risk"]["max_level"]).values

    t = ft.transactions
    t = t[(t.timestamp < cutoff) & t.customerID.isin(customers) & ~t.isSnapshot]
    buys, sells = t[t.transactionType == "Buy"], t[t.transactionType == "Sell"]
    for d in (30, 90, 180, 365):
        since = cutoff - pd.Timedelta(days=d)
        f[f"c_n_buys_{d}d"] = buys[buys.timestamp >= since].groupby("customerID").size().reindex(customers, fill_value=0)
    f["c_n_sells_90d"] = sells[sells.timestamp >= cutoff - pd.Timedelta(days=90)].groupby("customerID").size().reindex(customers, fill_value=0)
    f["c_n_buys_total"] = buys.groupby("customerID").size().reindex(customers, fill_value=0)
    f["c_n_assets_bought"] = buys.groupby("customerID").ISIN.nunique().reindex(customers, fill_value=0)
    f["c_days_since_last_tx"] = _days(cutoff, t.groupby("customerID").timestamp.max().reindex(customers))
    f["c_days_since_first_tx"] = _days(cutoff, t.groupby("customerID").timestamp.min().reindex(customers))
    f["c_mean_buy_value"] = buys.groupby("customerID").totalValue.mean().reindex(customers)
    f["c_share_internet"] = (t.channel == "Internet Banking").groupby(t.customerID).mean().reindex(customers)
    # share of past buys that were stocks
    cat = ft.assets.set_index("ISIN").assetCategory
    f["c_share_stock_buys"] = (buys.ISIN.map(cat) == "Stock").groupby(buys.customerID).mean().reindex(customers)

    h = holdings_asof(ft.transactions, cutoff, ft.price_wide)
    h = h[h.customerID.isin(customers)]
    pf = portfolio_summary(h, ft.assets).reindex(customers)
    for col in ["pf_value", "pf_n_assets", "pf_hhi", "pf_share_Stock", "pf_share_Bond", "pf_share_MTF"]:
        f["c_" + col] = pf[col] if col in pf else np.nan
    f[["c_pf_value", "c_pf_n_assets"]] = f[["c_pf_value", "c_pf_n_assets"]].fillna(0)
    return f


# ---------------------------------------------------------------- assets
def asset_features(ft: FarTrans, cutoff, items: pd.Index, cfg: dict) -> pd.DataFrame:
    a = ft.assets.set_index("ISIN").reindex(items)
    f = pd.DataFrame(index=items)
    f.index.name = "ISIN"
    f["a_category"] = a.assetCategory.fillna("Unknown")
    f["a_subCategory"] = a.assetSubCategory.fillna("Unknown")
    f["a_sector"] = a.sector.fillna("Unknown")
    f["a_industry"] = a.industry.fillna("Unknown")
    f["a_market"] = a.marketID.fillna("Unknown")

    hist = ft.price_wide.loc[ft.price_wide.index < cutoff, items]
    last = hist.iloc[-1]
    for d in (30, 90, 180, 365):
        prev = hist.loc[hist.index < cutoff - pd.Timedelta(days=d)]
        f[f"a_ret_{d}d"] = (last / prev.iloc[-1] - 1).replace([np.inf, -np.inf], np.nan) if len(prev) else np.nan
    win = hist.loc[hist.index >= cutoff - pd.Timedelta(days=365)]
    f["a_max_drawdown_365d"] = (win / win.cummax() - 1).min()
    f["a_vol_365d"] = annual_volatility(ft.prices, cutoff, 365).reindex(items)
    f["a_vol_90d"] = annual_volatility(ft.prices, cutoff, 90, min_obs=20).reindex(items)
    f["a_risk_level"] = asset_risk_levels(ft.prices, ft.assets, cutoff, cfg["risk"]["srri_bounds"],
                                          cfg["risk"]["vol_window_days"]).reindex(items).astype(float)
    first = ft.prices[ft.prices.timestamp < cutoff].groupby("ISIN").timestamp.min()
    f["a_days_listed"] = _days(cutoff, first.reindex(items))
    f["a_log_price"] = np.log(last.clip(lower=1e-6))

    b = ft.buys[ft.buys.timestamp < cutoff]
    for d in (7, 30, 90, 365):
        w = b[b.timestamp >= cutoff - pd.Timedelta(days=d)]
        f[f"a_buyers_{d}d"] = w.groupby("ISIN").customerID.nunique().reindex(items, fill_value=0)
    f["a_pop_trend"] = (f.a_buyers_7d + 1) / (f.a_buyers_30d / 4 + 1)
    f["a_buyers_30d_rank"] = f.a_buyers_30d.rank(ascending=False)
    s = ft.transactions
    s = s[(s.transactionType == "Sell") & (s.timestamp < cutoff) & (s.timestamp >= cutoff - pd.Timedelta(days=30))]
    f["a_sellers_30d"] = s.groupby("ISIN").customerID.nunique().reindex(items, fill_value=0)
    return f


# ---------------------------------------------------------------- pairs
def pair_features(ft: FarTrans, cutoff, pairs: pd.DataFrame, cust_f: pd.DataFrame, asset_f: pd.DataFrame) -> pd.DataFrame:
    p = pairs[["customerID", "ISIN"]].copy()
    customers = pd.Index(p.customerID.unique())
    t = ft.transactions
    t = t[(t.timestamp < cutoff) & t.customerID.isin(customers)]
    real = t[~t.isSnapshot]

    g = real[real.transactionType == "Buy"].groupby(["customerID", "ISIN"])
    past = pd.DataFrame({"p_n_buys": g.size(), "p_last_buy": g.timestamp.max(), "p_buy_value": g.totalValue.sum()})
    gs = real[real.transactionType == "Sell"].groupby(["customerID", "ISIN"])
    past = past.join(pd.DataFrame({"p_n_sells": gs.size(), "p_last_sell": gs.timestamp.max()}), how="outer")
    snap = t[t.isSnapshot & (t.transactionType == "Buy")][["customerID", "ISIN"]].drop_duplicates().assign(p_in_snapshot=1)
    p = p.merge(past.reset_index(), on=["customerID", "ISIN"], how="left").merge(snap, on=["customerID", "ISIN"], how="left")
    p["p_days_since_buy"] = _days(cutoff, p.pop("p_last_buy"))
    p["p_days_since_sell"] = _days(cutoff, p.pop("p_last_sell"))
    p[["p_n_buys", "p_n_sells", "p_buy_value", "p_in_snapshot"]] = p[["p_n_buys", "p_n_sells", "p_buy_value", "p_in_snapshot"]].fillna(0)

    h = holdings_asof(ft.transactions, cutoff, ft.price_wide)
    h = h[h.customerID.isin(customers)]
    p = p.merge(h[["customerID", "ISIN", "value"]].rename(columns={"value": "p_held_value"}), on=["customerID", "ISIN"], how="left")
    p["p_held_value"] = p.p_held_value.fillna(0)
    p["p_held"] = (p.p_held_value > 0).astype(int)
    pf_val = cust_f.c_pf_value
    p["p_held_share"] = p.p_held_value / p.customerID.map(pf_val).replace(0, np.nan)

    # portfolio share in the candidate's sector / category
    a = ft.assets.set_index("ISIN")
    hh = h.assign(sector=h.ISIN.map(a.sector).fillna("Unknown"), cat=h.ISIN.map(a.assetCategory).fillna("Unknown"))
    for key, col in (("sector", "a_sector"), ("cat", "a_category")):
        name = f"p_pf_share_same_{key}"
        sh = hh.groupby(["customerID", key]).value.sum()
        sh = (sh / sh.groupby(level=0).transform("sum")).rename(name).reset_index().rename(columns={key: "_k"})
        p["_k"] = p.ISIN.map(asset_f[col])
        p = p.merge(sh, on=["customerID", "_k"], how="left").drop(columns="_k")
        p[name] = p[name].fillna(0)

    # suitability
    p["p_risk_excess"] = p.ISIN.map(asset_f.a_risk_level) - p.customerID.map(cust_f.c_max_risk_level)

    # popularity inside the customer's risk segment (last 90 days)
    seg = cust_f.c_riskLevel
    prof_all = customer_profile_asof(ft.customers, cutoff).riskLevel
    b = ft.buys[(ft.buys.timestamp < cutoff) & (ft.buys.timestamp >= cutoff - pd.Timedelta(days=90))]
    b = b.assign(seg=b.customerID.map(prof_all).fillna("Unknown"))
    sp = b.groupby(["seg", "ISIN"]).customerID.nunique()
    sp = (sp / sp.index.get_level_values("seg").map(b.groupby("seg").customerID.nunique())).rename("p_seg_pop_90d").reset_index()
    p["seg"] = p.customerID.map(seg)
    p = p.merge(sp, on=["seg", "ISIN"], how="left").drop(columns="seg")
    p["p_seg_pop_90d"] = p.p_seg_pop_90d.fillna(0)
    return p


# ---------------------------------------------------------------- assemble
def build_frame(ft: FarTrans, cutoff, customers: pd.Index, candidates: pd.Index, cfg: dict,
                target: pd.DataFrame | None = None, n_per_source: int = 50) -> pd.DataFrame:
    cands = generate(ft, cutoff, customers, candidates, n_per_source)
    cust_f = customer_features(ft, cutoff, customers, cfg)
    asset_f = asset_features(ft, cutoff, pd.Index(candidates), cfg)
    pf = pair_features(ft, cutoff, cands, cust_f, asset_f)
    df = (cands.merge(pf, on=["customerID", "ISIN"], how="left")
               .merge(cust_f, left_on="customerID", right_index=True, how="left")
               .merge(asset_f, left_on="ISIN", right_index=True, how="left"))
    df.insert(0, "cutoff", cutoff)
    if target is not None:
        lab = target[["customerID", "ISIN"]].assign(label=1)
        df = df.merge(lab, on=["customerID", "ISIN"], how="left")
        df["label"] = df.label.fillna(0).astype(int)
    return df.sort_values(["customerID", "best_src_rank"]).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("cutoff", "customerID", "ISIN", "label")]
