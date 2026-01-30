# scripts/brent_options_analytics.py

from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from brent_options_iv_utils import (
    implied_vol_black76,
    black76_greeks,
    strike_from_delta_black76,
    year_fraction,
)


# ====================
# Paths / DBs
# ====================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPTIONS_DB = PROJECT_ROOT / "db" / "brent_options.db"
FUTURES_DB = PROJECT_ROOT / "db" / "brent_futures.db"


def get_conn_options() -> sqlite3.Connection:
    return sqlite3.connect(OPTIONS_DB)


def get_conn_futures() -> sqlite3.Connection:
    return sqlite3.connect(FUTURES_DB)


# ====================
# SOFR (FRED)
# ====================

@st.cache_data(ttl=60 * 60)
def load_sofr_series(days_back: int = 2000) -> pd.DataFrame:
    """
    Pull SOFR from FRED using pandas_datareader.
    If pandas_datareader isn't installed, returns empty DF.
    """
    try:
        from pandas_datareader.data import DataReader  # type: ignore
    except Exception:
        return pd.DataFrame()

    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=days_back)

    try:
        df = DataReader("SOFR", "fred", start, end)
        df = df.reset_index().rename(columns={"DATE": "date", "SOFR": "sofr"})
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["sofr"] = df["sofr"] / 100.0  # percent -> decimal
        return df.dropna()
    except Exception:
        return pd.DataFrame()


def sofr_rate_for(d: date, sofr_df: pd.DataFrame) -> Optional[float]:
    if sofr_df.empty:
        return None
    row = sofr_df[sofr_df["date"] <= d].sort_values("date").tail(1)
    if row.empty:
        return None
    return float(row["sofr"].iloc[0])


# ====================
# Data loading
# ====================

@st.cache_data(ttl=60)
def list_futures_codes_in_options() -> list[str]:
    with get_conn_options() as conn:
        df = pd.read_sql_query(
            "SELECT DISTINCT futures_code FROM brent_options ORDER BY futures_code",
            conn,
        )
    return df["futures_code"].dropna().astype(str).tolist()


@st.cache_data(ttl=60)
def load_contracts_metadata() -> pd.DataFrame:
    with get_conn_futures() as conn:
        df = pd.read_sql_query(
            """
            SELECT contract_code, month_code, year, last_trade_date
            FROM brent_contracts
            """,
            conn,
        )
    if not df.empty:
        df["last_trade_date"] = pd.to_datetime(df["last_trade_date"], errors="coerce").dt.date
    return df


@st.cache_data(ttl=60)
def list_report_dates_in_range(start_date: date, end_date: date) -> list[date]:
    with get_conn_options() as conn:
        df = pd.read_sql_query(
            """
            SELECT DISTINCT report_date
            FROM brent_options
            WHERE report_date >= ? AND report_date <= ?
            ORDER BY report_date
            """,
            conn,
            params=[start_date.isoformat(), end_date.isoformat()],
        )
    if df.empty:
        return []
    return pd.to_datetime(df["report_date"], errors="coerce").dropna().dt.date.tolist()


@st.cache_data(ttl=60)
def load_options_date_slice(futures_code: str, d: date) -> pd.DataFrame:
    q = """
        SELECT report_date, futures_code, strike, option_type,
               settle, close, total_volume, oi
        FROM brent_options
        WHERE futures_code = ?
          AND report_date = ?
        ORDER BY strike ASC
    """
    with get_conn_options() as conn:
        df = pd.read_sql_query(q, conn, params=[futures_code, d.isoformat()])

    if df.empty:
        return df

    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce").dt.date
    for c in ["strike", "settle", "close", "total_volume", "oi"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["option_type"] = df["option_type"].astype(str)
    return df.dropna(subset=["report_date", "strike", "option_type"])


@st.cache_data(ttl=120)
def load_futures_close_series(contract_code: str, start_date: date, end_date: date) -> pd.DataFrame:
    """
    Futures closes for a contract in date range.
    """
    q = """
        SELECT trade_date, close, volume
        FROM brent_prices
        WHERE contract_code = ?
          AND trade_date >= ?
          AND trade_date <= ?
        ORDER BY trade_date ASC
    """
    with get_conn_futures() as conn:
        df = pd.read_sql_query(q, conn, params=[contract_code, start_date.isoformat(), end_date.isoformat()])

    if df.empty:
        return df

    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    return df.dropna(subset=["trade_date", "close"]).reset_index(drop=True)


def ltd_for_contract(contracts_meta: pd.DataFrame, code: str) -> Optional[date]:
    row = contracts_meta[contracts_meta["contract_code"] == code]
    if row.empty:
        return None
    ltd = row["last_trade_date"].iloc[0]
    return None if pd.isna(ltd) else ltd


# ====================
# Front-month / Rank mapping + rolls (C1/C2/C3)
# ====================

def resolve_rank_mapping(
    report_dates: list[date],
    contracts_meta: pd.DataFrame,
    available_codes: set[str],
    rank: int = 1,
) -> dict[date, Optional[str]]:
    """
    For each report_date, choose the rank-th nearest last_trade_date >= report_date,
    restricted to codes that exist in options DB (available_codes).
    rank=1 -> C1, rank=2 -> C2, rank=3 -> C3
    """
    meta = contracts_meta.dropna(subset=["contract_code", "last_trade_date"]).copy()
    meta = meta[meta["contract_code"].isin(list(available_codes))].copy()
    meta = meta.sort_values("last_trade_date")

    mapping: dict[date, Optional[str]] = {}
    for d in sorted(report_dates):
        valid = meta[meta["last_trade_date"] >= d]
        if len(valid) < rank:
            mapping[d] = None
        else:
            mapping[d] = str(valid.iloc[rank - 1]["contract_code"])
    return mapping


def roll_events_from_mapping(mapping: dict[date, Optional[str]]) -> pd.DataFrame:
    dates = sorted(mapping.keys())
    rows = []
    prev = None
    for d in dates:
        cur = mapping[d]
        if cur is None:
            continue
        if prev is None:
            prev = cur
            continue
        if cur != prev:
            rows.append({"date": d, "prev": prev, "new": cur})
            prev = cur
    return pd.DataFrame(rows)


# ====================
# IV helpers
# ====================

def choose_opt_price(row: pd.Series, price_field: str) -> Optional[float]:
    v = row.get(price_field, None)
    try:
        if v is None or pd.isna(v):
            return None
        v = float(v)
        return v if v > 0 else None
    except Exception:
        return None


def compute_day_ivs(
    day_df: pd.DataFrame,
    F: float,
    T: float,
    r: float,
    price_field: str,
) -> pd.DataFrame:
    """
    Adds iv column (float) for rows where IV can be solved.
    """
    df = day_df.copy()

    def _iv(row):
        px = choose_opt_price(row, price_field)
        if px is None:
            return np.nan
        is_call = (row["option_type"] == "Call")
        iv = implied_vol_black76(px, F=F, K=float(row["strike"]), T=T, r=r, is_call=is_call)
        return np.nan if iv is None else float(iv)

    df["iv"] = df.apply(_iv, axis=1)
    return df


def atm_row_for_day(day_df: pd.DataFrame, F: float, opt_type: Optional[str] = None) -> Optional[pd.Series]:
    df = day_df.copy()
    if opt_type in ("Call", "Put"):
        df = df[df["option_type"] == opt_type]
    df = df.dropna(subset=["strike"])
    if df.empty:
        return None
    df["abs_m"] = (df["strike"] - F).abs()
    return df.sort_values("abs_m").head(1).iloc[0]


def atm_iv_from_day(day_df_with_iv: pd.DataFrame, F: float) -> Optional[float]:
    """
    Prefer strike with both call & put IV; else closest IV row.
    """
    df = day_df_with_iv.dropna(subset=["iv", "strike"]).copy()
    if df.empty:
        return None

    pivot = df.pivot_table(index="strike", columns="option_type", values="iv", aggfunc="mean")
    both = pivot.dropna(subset=["Call", "Put"], how="any").copy()
    if not both.empty:
        both["abs_m"] = (both.index.to_series() - F).abs()
        s = both.sort_values("abs_m").index[0]
        return float(0.5 * (both.loc[s, "Call"] + both.loc[s, "Put"]))

    df["abs_m"] = (df["strike"] - F).abs()
    return float(df.sort_values("abs_m").iloc[0]["iv"])


def nearest_iv_at_strike(day_df_with_iv: pd.DataFrame, strike: float, opt_type: str) -> Optional[float]:
    df = day_df_with_iv[(day_df_with_iv["option_type"] == opt_type)].dropna(subset=["iv", "strike"]).copy()
    if df.empty:
        return None
    i = (df["strike"] - strike).abs().argsort()[:1]
    return float(df.iloc[i]["iv"].iloc[0])


def compute_25d_metrics_for_date(
    day_df_with_iv: pd.DataFrame,
    F: float,
    T: float,
    r: float,
) -> dict[str, Optional[float]]:
    """
    Returns dict with:
      rr_25d = IV(25Δ call) - IV(25Δ put)
      bf_25d = 0.5*(IV(25C)+IV(25P)) - IV(ATM)
      rr_25d_norm = rr_25d / vega_atm
      iv_25c, iv_25p, iv_atm, vega_atm
    """
    out = {k: None for k in ["rr_25d", "bf_25d", "rr_25d_norm", "iv_25c", "iv_25p", "iv_atm", "vega_atm"]}

    df = day_df_with_iv.dropna(subset=["iv", "strike"]).copy()
    if df.empty:
        return out

    sigma0 = atm_iv_from_day(df, F)
    if sigma0 is None or sigma0 <= 0:
        return out

    K_low = max(0.01, F * 0.3)
    K_high = F * 3.0

    K_25c = strike_from_delta_black76(
        target_delta=0.25, F=F, T=T, r=r, sigma=sigma0, is_call=True,
        K_low=K_low, K_high=K_high
    )
    K_25p = strike_from_delta_black76(
        target_delta=-0.25, F=F, T=T, r=r, sigma=sigma0, is_call=False,
        K_low=K_low, K_high=K_high
    )

    if K_25c is None or K_25p is None:
        return out

    iv_25c = nearest_iv_at_strike(df, K_25c, "Call")
    iv_25p = nearest_iv_at_strike(df, K_25p, "Put")
    iv_atm = atm_iv_from_day(df, F)

    if iv_25c is None or iv_25p is None or iv_atm is None:
        return out

    rr = float(iv_25c - iv_25p)
    bf = float(0.5 * (iv_25c + iv_25p) - iv_atm)

    # Vega at ATM (use call as proxy; near ATM call/put vega similar)
    # pick nearest actual strike around F
    strikes = df["strike"].dropna().to_numpy()
    K_atm = float(strikes[np.argmin(np.abs(strikes - F))])
    g = black76_greeks(F=F, K=K_atm, T=T, r=r, sigma=float(iv_atm), is_call=True)
    vega = float(g.vega)

    out.update({
        "rr_25d": rr,
        "bf_25d": bf,
        "iv_25c": float(iv_25c),
        "iv_25p": float(iv_25p),
        "iv_atm": float(iv_atm),
        "vega_atm": vega,
        "rr_25d_norm": (rr / vega) if (vega is not None and vega > 0) else None,
    })
    return out


# ====================
# Realised volatility
# ====================

def compute_log_returns(px: pd.Series) -> pd.Series:
    return np.log(px / px.shift(1))


def realised_vol(returns: pd.Series, window: int, annualise: bool = True, trading_days: int = 252) -> pd.Series:
    rv = returns.rolling(window).std()
    if annualise:
        rv = rv * np.sqrt(trading_days)
    return rv


# ====================
# Signals / regimes
# ====================

def rolling_zscore(x: pd.Series, window: int = 60) -> pd.Series:
    mu = x.rolling(window).mean()
    sd = x.rolling(window).std()
    return (x - mu) / sd


def rolling_percentile(x: pd.Series, window: int = 252) -> pd.Series:
    def _pct(arr):
        s = pd.Series(arr)
        return (s.rank(pct=True).iloc[-1])
    return x.rolling(window).apply(_pct, raw=False)


def regime_label(iv: float, rv: float, iv_rv_z: float, rv_slope: float) -> str:
    """
    Simple interpretability-first regime:
      - vol rich/cheap via z-score
      - vol rising/falling via RV slope
    """
    if pd.isna(iv) or pd.isna(rv) or pd.isna(iv_rv_z):
        return "Unknown"
    rich = "Rich vol" if iv_rv_z > 1.0 else "Cheap vol" if iv_rv_z < -1.0 else "Neutral vol"
    trend = "RV rising" if rv_slope > 0 else "RV falling" if rv_slope < 0 else "RV flat"
    return f"{rich} | {trend}"


# ====================
# Streamlit UI
# ====================

st.set_page_config(page_title="Brent Options — Analytics", layout="wide")
st.title("Brent Options — Volatility, Skew, RV, Forward Vol (Black–76 + SOFR)")

# ---- sidebar ----
today = date.today()

st.sidebar.header("Mode")

mode = st.sidebar.selectbox("Maturity mode", ["Specific contract", "Front month (C1)"], index=0)

st.sidebar.header("Date range")
range_mode = st.sidebar.selectbox("Range", ["Last 180 days", "Last 365 days", "Custom"], index=1)
if range_mode == "Last 180 days":
    start_d = today - timedelta(days=180)
    end_d = today
elif range_mode == "Last 365 days":
    start_d = today - timedelta(days=365)
    end_d = today
else:
    start_d, end_d = st.sidebar.date_input(
        "Custom (start, end)",
        value=(today - timedelta(days=365), today)
    )

price_field = st.sidebar.selectbox("Option price field", ["settle", "close"], index=0)
x_axis = st.sidebar.radio("Smile X-axis", ["Strike", "Moneyness (K/F)"], index=0)

st.sidebar.header("Rates")
use_sofr = st.sidebar.checkbox("Use SOFR (FRED)", value=True)
manual_rate = st.sidebar.number_input("Manual r (decimal)", value=0.05, step=0.005)

st.sidebar.header("Performance")
surface_max_strikes = st.sidebar.slider("Max strikes/day (surface)", 10, 200, 60, step=10)
signals_z_window = st.sidebar.slider("Signals z-score window", 20, 200, 60, step=10)
skew_pct_window = st.sidebar.slider("Skew percentile window", 60, 400, 252, step=21)

sofr_df = load_sofr_series()
contracts_meta = load_contracts_metadata()
available_codes = set(list_futures_codes_in_options())

if mode == "Specific contract":
    futures_code_fixed = st.sidebar.selectbox("Options maturity (futures_code)", sorted(available_codes), index=0)
else:
    futures_code_fixed = None
    st.sidebar.caption("C1 is resolved per date from last_trade_date (LTD).")

# report dates
report_dates = list_report_dates_in_range(start_d, end_d)
if not report_dates:
    st.warning("No options report_dates found in the selected range.")
    st.stop()

def rate_for(d: date) -> float:
    r = sofr_rate_for(d, sofr_df) if (use_sofr and not sofr_df.empty) else None
    return float(r) if r is not None else float(manual_rate)

# mappings for rank series
if mode == "Specific contract":
    map_c1 = {d: futures_code_fixed for d in report_dates}
    map_c2 = {d: None for d in report_dates}
    map_c3 = {d: None for d in report_dates}
else:
    map_c1 = resolve_rank_mapping(report_dates, contracts_meta, available_codes, rank=1)
    map_c2 = resolve_rank_mapping(report_dates, contracts_meta, available_codes, rank=2)
    map_c3 = resolve_rank_mapping(report_dates, contracts_meta, available_codes, rank=3)

rolls_c1 = roll_events_from_mapping(map_c1) if mode != "Specific contract" else pd.DataFrame()


def futures_close_on_date(code: str, d: date) -> Optional[float]:
    # quick point lookup via series cache (per contract)
    df = load_futures_close_series(code, start_d, end_d)
    if df.empty:
        return None
    row = df[df["trade_date"] == d]
    if row.empty:
        return None
    return float(row["close"].iloc[0])


def build_atm_iv_series(mapping: dict[date, Optional[str]], opt_type: str) -> pd.DataFrame:
    rows = []
    for d in report_dates:
        code = mapping.get(d)
        if code is None:
            continue

        ltd = ltd_for_contract(contracts_meta, code)
        if ltd is None:
            continue

        F = futures_close_on_date(code, d)
        if F is None:
            continue

        T = year_fraction(d, ltd, basis=365)
        if T <= 0:
            continue

        r = rate_for(d)
        day = load_options_date_slice(code, d)
        if day.empty:
            continue

        row_atm = atm_row_for_day(day, F, opt_type=opt_type)
        if row_atm is None:
            continue

        px = choose_opt_price(row_atm, price_field)
        if px is None:
            continue

        is_call = (opt_type == "Call")
        iv = implied_vol_black76(px, F=F, K=float(row_atm["strike"]), T=T, r=r, is_call=is_call)
        if iv is None:
            continue

        rows.append({
            "date": d,
            "active_code": code,
            "F": F,
            "K_atm": float(row_atm["strike"]),
            "opt_px": float(px),
            "iv": float(iv),
            "r": float(r),
            "T": float(T),
            "ltd": ltd,
        })

    return pd.DataFrame(rows)


def add_roll_lines(fig: go.Figure, rolls: pd.DataFrame):
    if rolls is None or rolls.empty:
        return
    for _, rr in rolls.iterrows():
        fig.add_vline(
            x=rr["date"],
            line_dash="dot",
            annotation_text=f"Roll {rr['prev']}→{rr['new']}",
            annotation_position="top left",
        )


tabs = st.tabs([
    "ATM IV (time series)",
    "IV vs RV",
    "Smile (single date)",
    "IV Surface (3D)",
    "Skew (RR/BF/Norm + Rolls)",
    "Forward Vol (C1→C2 etc.)",
    "Liquidity Diagnostics",
    "Delta Key Levels",
    "Signals & Regimes",
    "Sticky Strike vs Sticky Delta",
    "IV + Greeks Calculator",
])

# ====================
# TAB 1: ATM IV time series
# ====================

with tabs[0]:
    st.subheader("ATM Implied Volatility Over Time")

    opt_type = st.selectbox("Option type for ATM series", ["Call", "Put"], index=0)
    df_atm = build_atm_iv_series(map_c1, opt_type=opt_type)

    if df_atm.empty:
        st.info("No ATM IV points computed (missing futures closes/LTDs/prices).")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_atm["date"], y=df_atm["iv"], mode="lines", name="ATM IV"))
        if mode != "Specific contract":
            add_roll_lines(fig, rolls_c1)
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Implied Vol (decimal)",
            hovermode="x unified",
            height=520,
        )
        st.plotly_chart(fig, use_container_width=True)

        if mode != "Specific contract":
            st.markdown("#### C1 active contract")
            st.dataframe(df_atm[["date", "active_code"]].drop_duplicates(), use_container_width=True)

        st.markdown("#### Recent points")
        st.dataframe(df_atm.tail(60), use_container_width=True)

# ====================
# TAB 2: IV vs RV
# ====================

with tabs[1]:
    st.subheader("Implied vs Realised Volatility")

    windows = st.multiselect("Realised vol windows (days)", [10, 20, 30, 60, 90], default=[20, 60])
    annualise = st.checkbox("Annualise realised vol (×√252)", value=True)

    # build ATM IV series (call ATM is standard)
    df_iv = build_atm_iv_series(map_c1, opt_type="Call")
    if df_iv.empty:
        st.info("No IV series available.")
        st.stop()

    # Build realised vol from futures (active code each date)
    # For C1 mode, we compute RV on each active contract separately then align by date
    rv_rows = []
    for code in sorted(df_iv["active_code"].unique().tolist()):
        df_f = load_futures_close_series(code, start_d, end_d)
        if df_f.empty:
            continue
        df_f = df_f.sort_values("trade_date").copy()
        df_f["ret"] = compute_log_returns(df_f["close"])
        for w in windows:
            df_f[f"rv_{w}"] = realised_vol(df_f["ret"], w, annualise=annualise)
        keep = ["trade_date"] + [f"rv_{w}" for w in windows]
        df_f = df_f[keep].copy()
        df_f["active_code"] = code
        rv_rows.append(df_f)

    if not rv_rows:
        st.info("No futures data available to compute realised volatility.")
        st.stop()

    df_rv_all = pd.concat(rv_rows, ignore_index=True)

    # join RV to IV using date+active_code
    df_join = df_iv.merge(
        df_rv_all,
        left_on=["date", "active_code"],
        right_on=["trade_date", "active_code"],
        how="left",
    ).drop(columns=["trade_date"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_join["date"], y=df_join["iv"], mode="lines", name="ATM IV (Call)"))
    for w in windows:
        fig.add_trace(go.Scatter(x=df_join["date"], y=df_join[f"rv_{w}"], mode="lines", name=f"RV {w}d"))

    if mode != "Specific contract":
        add_roll_lines(fig, rolls_c1)

    fig.update_layout(
        title="ATM IV vs Realised Volatility",
        xaxis_title="Date",
        yaxis_title="Vol (decimal)",
        hovermode="x unified",
        height=560,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df_join.tail(80), use_container_width=True)

# ====================
# TAB 3: Smile
# ====================

with tabs[2]:
    st.subheader("Implied Volatility Smile")

    sel_date = st.date_input(
        "Report date",
        value=report_dates[-1],
        min_value=report_dates[0],
        max_value=report_dates[-1],
        key="smile_date",
    )

    code = map_c1.get(sel_date) if mode != "Specific contract" else futures_code_fixed
    if code is None:
        st.warning("No active contract for this date.")
        st.stop()

    ltd = ltd_for_contract(contracts_meta, code)
    if ltd is None:
        st.warning("Missing LTD for active contract.")
        st.stop()

    F = futures_close_on_date(code, sel_date)
    if F is None:
        st.warning("Missing futures close for this date.")
        st.stop()

    T = year_fraction(sel_date, ltd, basis=365)
    if T <= 0:
        st.warning("T<=0 on selected date.")
        st.stop()

    r = rate_for(sel_date)
    day = load_options_date_slice(code, sel_date)
    if day.empty:
        st.info("No options for this date.")
        st.stop()

    day_iv = compute_day_ivs(day, F=F, T=T, r=r, price_field=price_field).dropna(subset=["iv"])
    if day_iv.empty:
        st.info("No IVs computed for this date.")
        st.stop()

    day_iv["moneyness"] = day_iv["strike"] / F

    show_calls = st.checkbox("Show calls", value=True)
    show_puts = st.checkbox("Show puts", value=True)

    xcol = "strike" if x_axis == "Strike" else "moneyness"
    xlab = "Strike" if x_axis == "Strike" else "Moneyness (K/F)"

    fig = go.Figure()
    if show_calls:
        calls = day_iv[day_iv["option_type"] == "Call"].sort_values(xcol)
        fig.add_trace(go.Scatter(x=calls[xcol], y=calls["iv"], mode="lines+markers", name="Call IV"))
    if show_puts:
        puts = day_iv[day_iv["option_type"] == "Put"].sort_values(xcol)
        fig.add_trace(go.Scatter(x=puts[xcol], y=puts["iv"], mode="lines+markers", name="Put IV"))

    if x_axis == "Strike":
        fig.add_vline(x=F, line_dash="dash", annotation_text=f"F={F:.2f}", annotation_position="top left")
    else:
        fig.add_vline(x=1.0, line_dash="dash", annotation_text="ATM (K/F=1)", annotation_position="top left")

    fig.update_layout(
        title=f"Smile on {sel_date.isoformat()} (active {code})",
        xaxis_title=xlab,
        yaxis_title="IV (decimal)",
        hovermode="x unified",
        height=560,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(day_iv[["option_type", "strike", "moneyness", price_field, "iv", "total_volume", "oi"]], use_container_width=True)

# ====================
# TAB 4: Surface (3D)
# ====================

with tabs[3]:
    st.subheader("IV Surface (Date × Strike/Moneyness × IV)")

    surface_type = st.selectbox("Surface option type", ["Call", "Put"], index=0)
    surface_x = st.selectbox("Surface X axis", ["Strike", "Moneyness (K/F)"], index=0)

    rows = []
    for d in report_dates:
        code = map_c1.get(d)
        if code is None:
            continue
        ltd = ltd_for_contract(contracts_meta, code)
        if ltd is None:
            continue
        F = futures_close_on_date(code, d)
        if F is None:
            continue
        T = year_fraction(d, ltd, basis=365)
        if T <= 0:
            continue
        r = rate_for(d)

        day = load_options_date_slice(code, d)
        if day.empty:
            continue
        day = day[day["option_type"] == surface_type].copy()
        if day.empty:
            continue

        # speed: keep near-ATM strikes
        day["abs_m"] = (day["strike"] - F).abs()
        day = day.sort_values("abs_m").head(surface_max_strikes)

        day_iv = compute_day_ivs(day, F=F, T=T, r=r, price_field=price_field).dropna(subset=["iv"])
        if day_iv.empty:
            continue
        day_iv["moneyness"] = day_iv["strike"] / F

        for _, rr in day_iv.iterrows():
            rows.append({
                "date": d,
                "x": float(rr["strike"] if surface_x == "Strike" else rr["moneyness"]),
                "iv": float(rr["iv"]),
            })

    df_surf = pd.DataFrame(rows)
    if df_surf.empty:
        st.info("No surface points computed.")
    else:
        pivot = df_surf.pivot_table(index="date", columns="x", values="iv", aggfunc="mean").sort_index()
        Z = pivot.to_numpy()
        X = pivot.columns.to_numpy()
        Y = np.array([pd.to_datetime(d).toordinal() for d in pivot.index])

        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
        fig.update_layout(
            scene=dict(
                xaxis_title=("Strike" if surface_x == "Strike" else "Moneyness (K/F)"),
                yaxis_title="Date (ordinal)",
                zaxis_title="IV",
            ),
            height=700,
        )
        st.plotly_chart(fig, use_container_width=True)

# ====================
# TAB 5: Skew metrics
# ====================

with tabs[4]:
    st.subheader("Skew Metrics (25Δ RR / 25Δ BF / Vega-normalised RR) + Rolls")

    metric = st.selectbox(
        "Skew metric",
        ["25Δ Risk Reversal (Call-Put)", "25Δ Butterfly", "25Δ RR / ATM Vega"],
        index=0,
    )

    rows = []
    for d in report_dates:
        code = map_c1.get(d)
        if code is None:
            continue
        ltd = ltd_for_contract(contracts_meta, code)
        if ltd is None:
            continue
        F = futures_close_on_date(code, d)
        if F is None:
            continue
        T = year_fraction(d, ltd, basis=365)
        if T <= 0:
            continue
        r = rate_for(d)

        day = load_options_date_slice(code, d)
        if day.empty:
            continue

        day_iv = compute_day_ivs(day, F=F, T=T, r=r, price_field=price_field)
        m = compute_25d_metrics_for_date(day_iv, F=F, T=T, r=r)

        val = None
        if metric.startswith("25Δ Risk Reversal"):
            val = m["rr_25d"]
        elif metric.startswith("25Δ Butterfly"):
            val = m["bf_25d"]
        else:
            val = m["rr_25d_norm"]

        if val is None:
            continue

        rows.append({
            "date": d,
            "active_code": code,
            "value": float(val),
            "rr_25d": m["rr_25d"],
            "bf_25d": m["bf_25d"],
            "iv_atm": m["iv_atm"],
            "iv_25c": m["iv_25c"],
            "iv_25p": m["iv_25p"],
            "vega_atm": m["vega_atm"],
        })

    df_sk = pd.DataFrame(rows)
    if df_sk.empty:
        st.info("No skew points computed.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_sk["date"], y=df_sk["value"], mode="lines", name="Metric"))
        if mode != "Specific contract":
            add_roll_lines(fig, rolls_c1)
        fig.update_layout(
            title=metric,
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode="x unified",
            height=560,
        )
        st.plotly_chart(fig, use_container_width=True)

        if mode != "Specific contract":
            st.markdown("#### Roll events (C1)")
            if rolls_c1.empty:
                st.write("No rolls detected.")
            else:
                st.dataframe(rolls_c1, use_container_width=True)

        st.dataframe(df_sk.tail(80), use_container_width=True)

# ====================
# TAB 6: Forward vol (C1→C2 etc.)
# ====================

with tabs[5]:
    st.subheader("Forward Implied Volatility")

    if mode == "Specific contract":
        st.info("Forward vol needs multiple maturities. Switch to Front month (C1) mode.")
        st.stop()

    pair = st.selectbox("Forward pair", ["C1→C2", "C2→C3", "C1→C3"], index=0)

    if pair == "C1→C2":
        m1, m2 = map_c1, map_c2
    elif pair == "C2→C3":
        m1, m2 = map_c2, map_c3
    else:
        m1, m2 = map_c1, map_c3

    # Build ATM IV for each leg
    df1 = build_atm_iv_series(m1, opt_type="Call")
    df2 = build_atm_iv_series(m2, opt_type="Call")

    if df1.empty or df2.empty:
        st.info("Not enough data to compute forward vol (missing C2/C3 coverage or LTDs).")
        st.stop()

    df = df1.merge(df2, on="date", how="inner", suffixes=("_1", "_2"))
    if df.empty:
        st.info("No overlapping dates between the two legs.")
        st.stop()

    # forward variance formula
    # sigma_fwd^2 = (T2*s2^2 - T1*s1^2)/(T2-T1)
    df = df.dropna(subset=["iv_1", "iv_2", "T_1", "T_2"]).copy()
    df = df[df["T_2"] > df["T_1"]].copy()
    if df.empty:
        st.info("No valid dates where T2 > T1.")
        st.stop()

    df["fwd_var"] = (df["T_2"] * df["iv_2"]**2 - df["T_1"] * df["iv_1"]**2) / (df["T_2"] - df["T_1"])
    df["fwd_iv"] = np.sqrt(np.maximum(df["fwd_var"], 0.0))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["fwd_iv"], mode="lines", name="Forward IV"))
    fig.update_layout(
        title=f"Forward IV ({pair})",
        xaxis_title="Date",
        yaxis_title="Forward IV (decimal)",
        hovermode="x unified",
        height=560,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df[["date", "active_code_1", "active_code_2", "iv_1", "iv_2", "T_1", "T_2", "fwd_iv"]].tail(80), use_container_width=True)

# ====================
# TAB 7: Liquidity Diagnostics
# ====================

with tabs[6]:
    st.subheader("Liquidity Diagnostics (Volume/OI weighted IV, tradeability)")

    sel_date = st.date_input(
        "Report date (liquidity diagnostics)",
        value=report_dates[-1],
        min_value=report_dates[0],
        max_value=report_dates[-1],
        key="liq_date",
    )

    code = map_c1.get(sel_date)
    if code is None:
        st.warning("No active contract on this date.")
        st.stop()

    ltd = ltd_for_contract(contracts_meta, code)
    F = futures_close_on_date(code, sel_date)
    if ltd is None or F is None:
        st.warning("Need LTD and futures close.")
        st.stop()

    T = year_fraction(sel_date, ltd, basis=365)
    if T <= 0:
        st.warning("T<=0.")
        st.stop()

    r = rate_for(sel_date)

    day = load_options_date_slice(code, sel_date)
    if day.empty:
        st.info("No options rows for this date.")
        st.stop()

    day_iv = compute_day_ivs(day, F=F, T=T, r=r, price_field=price_field).dropna(subset=["iv"])
    if day_iv.empty:
        st.info("No IVs computed for this date.")
        st.stop()

    day_iv["moneyness"] = day_iv["strike"] / F
    min_vol = st.number_input("Min volume filter", value=0, step=1)
    min_oi = st.number_input("Min OI filter", value=0, step=1)

    filt = day_iv.copy()
    filt["total_volume"] = filt["total_volume"].fillna(0)
    filt["oi"] = filt["oi"].fillna(0)
    filt = filt[(filt["total_volume"] >= min_vol) & (filt["oi"] >= min_oi)].copy()

    if filt.empty:
        st.info("No strikes pass liquidity filters.")
        st.stop()

    # Weighted IV summary
    def wavg(x, w):
        w = np.asarray(w)
        x = np.asarray(x)
        if np.sum(w) <= 0:
            return np.nan
        return float(np.sum(w * x) / np.sum(w))

    summary_rows = []
    for typ in ["Call", "Put"]:
        sub = filt[filt["option_type"] == typ]
        if sub.empty:
            continue
        summary_rows.append({
            "type": typ,
            "count": len(sub),
            "IV mean": float(sub["iv"].mean()),
            "IV vol-weighted": wavg(sub["iv"], sub["total_volume"]),
            "IV OI-weighted": wavg(sub["iv"], sub["oi"]),
            "Volume total": float(sub["total_volume"].sum()),
            "OI total": float(sub["oi"].sum()),
        })
    st.table(pd.DataFrame(summary_rows))

    # Scatter: IV vs strike/moneyness with size by volume
    xcol = "strike" if x_axis == "Strike" else "moneyness"
    xlab = "Strike" if x_axis == "Strike" else "Moneyness (K/F)"
    fig = go.Figure()
    for typ in ["Call", "Put"]:
        sub = filt[filt["option_type"] == typ].copy()
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub[xcol],
                y=sub["iv"],
                mode="markers",
                name=f"{typ}",
                marker=dict(size=np.clip(sub["total_volume"].fillna(0).to_numpy() / 50.0 + 6, 6, 30)),
                text=[f"Strike={k:.2f}<br>Vol={v}<br>OI={oi}" for k, v, oi in zip(sub["strike"], sub["total_volume"], sub["oi"])],
                hovertemplate="%{text}<br>IV=%{y:.4f}<extra></extra>",
            )
        )
    if x_axis == "Strike":
        fig.add_vline(x=F, line_dash="dash", annotation_text=f"F={F:.2f}")
    else:
        fig.add_vline(x=1.0, line_dash="dash", annotation_text="ATM")
    fig.update_layout(
        title=f"IV scatter (size≈volume) on {sel_date.isoformat()} ({code})",
        xaxis_title=xlab,
        yaxis_title="IV",
        height=560,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(filt.sort_values(["option_type", "strike"]), use_container_width=True)

# ====================
# TAB 8: Delta key levels
# ====================

with tabs[7]:
    st.subheader("Delta Key Levels (Strikes & IVs)")

    sel_date = st.date_input(
        "Report date (delta levels)",
        value=report_dates[-1],
        min_value=report_dates[0],
        max_value=report_dates[-1],
        key="delta_levels_date",
    )

    code = map_c1.get(sel_date)
    if code is None:
        st.warning("No active contract for this date.")
        st.stop()

    ltd = ltd_for_contract(contracts_meta, code)
    F = futures_close_on_date(code, sel_date)
    if ltd is None or F is None:
        st.warning("Need LTD and futures close.")
        st.stop()

    T = year_fraction(sel_date, ltd, basis=365)
    if T <= 0:
        st.warning("T<=0.")
        st.stop()

    r = rate_for(sel_date)

    day = load_options_date_slice(code, sel_date)
    if day.empty:
        st.info("No options.")
        st.stop()

    day_iv = compute_day_ivs(day, F=F, T=T, r=r, price_field=price_field).dropna(subset=["iv"])
    if day_iv.empty:
        st.info("No IVs computed.")
        st.stop()

    sigma0 = atm_iv_from_day(day_iv, F)
    if sigma0 is None:
        st.info("Could not infer ATM IV to seed delta->strike.")
        st.stop()

    deltas = [0.10, 0.25, 0.50, 0.75, 0.90]
    rows = []
    K_low = max(0.01, F * 0.3)
    K_high = F * 3.0

    for dlt in deltas:
        # calls: +delta
        Kc = strike_from_delta_black76(dlt, F, T, r, sigma0, True, K_low=K_low, K_high=K_high)
        ivc = nearest_iv_at_strike(day_iv, Kc, "Call") if Kc is not None else None

        # puts: -delta (for 25Δ put etc.)
        Kp = strike_from_delta_black76(-dlt, F, T, r, sigma0, False, K_low=K_low, K_high=K_high)
        ivp = nearest_iv_at_strike(day_iv, Kp, "Put") if Kp is not None else None

        rows.append({
            "delta": dlt,
            "call_strike_est": Kc,
            "call_iv_nearest": ivc,
            "put_strike_est": Kp,
            "put_iv_nearest": ivp,
        })

    st.write(f"Active: **{code}** | F={F:.2f} | LTD={ltd} | T={T:.4f} | seed ATM IV={sigma0:.4%}")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ====================
# TAB 9: Signals & regimes
# ====================

with tabs[8]:
    st.subheader("Signals & Regimes")

    rv_window = st.selectbox("Primary RV window", [10, 20, 30, 60, 90], index=1)
    rv_slope_window = st.selectbox("RV slope window (days)", [10, 20, 30, 60], index=1)

    # IV series
    df_iv = build_atm_iv_series(map_c1, opt_type="Call")
    if df_iv.empty:
        st.info("No IV series available.")
        st.stop()

    # RV series per active contract
    rv_rows = []
    for code in sorted(df_iv["active_code"].unique().tolist()):
        df_f = load_futures_close_series(code, start_d, end_d)
        if df_f.empty:
            continue
        df_f = df_f.sort_values("trade_date").copy()
        df_f["ret"] = compute_log_returns(df_f["close"])
        df_f[f"rv_{rv_window}"] = realised_vol(df_f["ret"], rv_window, annualise=True)
        df_f["rv_slope"] = df_f[f"rv_{rv_window}"].diff(rv_slope_window)
        rv_rows.append(df_f[["trade_date", f"rv_{rv_window}", "rv_slope"]].assign(active_code=code))
    if not rv_rows:
        st.info("No futures data for RV.")
        st.stop()
    df_rv = pd.concat(rv_rows, ignore_index=True)

    df = df_iv.merge(
        df_rv,
        left_on=["date", "active_code"],
        right_on=["trade_date", "active_code"],
        how="left",
    ).drop(columns=["trade_date"])

    # Skew series (RR25) for signals
    sk_rows = []
    for d in report_dates:
        code = map_c1.get(d)
        if code is None:
            continue
        ltd = ltd_for_contract(contracts_meta, code)
        if ltd is None:
            continue
        F = futures_close_on_date(code, d)
        if F is None:
            continue
        T = year_fraction(d, ltd, basis=365)
        if T <= 0:
            continue
        r = rate_for(d)
        day = load_options_date_slice(code, d)
        if day.empty:
            continue
        day_iv = compute_day_ivs(day, F=F, T=T, r=r, price_field=price_field)
        m = compute_25d_metrics_for_date(day_iv, F=F, T=T, r=r)
        if m["rr_25d"] is None:
            continue
        sk_rows.append({"date": d, "active_code": code, "rr_25d": float(m["rr_25d"])})
    df_sk = pd.DataFrame(sk_rows)

    df = df.merge(df_sk, on=["date", "active_code"], how="left")

    df["iv_minus_rv"] = df["iv"] - df[f"rv_{rv_window}"]
    df["iv_rv_z"] = rolling_zscore(df["iv_minus_rv"], window=signals_z_window)
    df["skew_pct"] = rolling_percentile(df["rr_25d"], window=skew_pct_window)

    df["regime"] = [
        regime_label(iv, rv, z, slope)
        for iv, rv, z, slope in zip(df["iv"], df[f"rv_{rv_window}"], df["iv_rv_z"], df["rv_slope"])
    ]

    # Charts
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df["date"], y=df["iv_minus_rv"], mode="lines", name="IV - RV"))
    fig1.add_trace(go.Scatter(x=df["date"], y=df["iv_rv_z"], mode="lines", name="(IV - RV) z", yaxis="y2"))
    fig1.update_layout(
        title="IV - RV spread and z-score",
        xaxis_title="Date",
        yaxis_title="IV - RV",
        yaxis2=dict(title="z-score", overlaying="y", side="right"),
        hovermode="x unified",
        height=560,
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["date"], y=df["rr_25d"], mode="lines", name="25Δ RR"))
    fig2.add_trace(go.Scatter(x=df["date"], y=df["skew_pct"], mode="lines", name="Skew percentile", yaxis="y2"))
    fig2.update_layout(
        title="Skew level and percentile",
        xaxis_title="Date",
        yaxis_title="25Δ RR",
        yaxis2=dict(title="Percentile", overlaying="y", side="right", range=[0, 1]),
        hovermode="x unified",
        height=560,
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Regime table (latest)")
    st.dataframe(df[["date", "active_code", "iv", f"rv_{rv_window}", "iv_minus_rv", "iv_rv_z", "rr_25d", "skew_pct", "regime"]].tail(120), use_container_width=True)

# ====================
# TAB 10: Sticky strike vs sticky delta
# ====================

with tabs[9]:
    st.subheader("Sticky Strike vs Sticky Delta (diagnostic)")

    ref_date = st.date_input(
        "Reference date (defines strike and delta anchors)",
        value=report_dates[-1],
        min_value=report_dates[0],
        max_value=report_dates[-1],
        key="sticky_ref_date",
    )

    ref_code = map_c1.get(ref_date)
    if ref_code is None:
        st.warning("No active contract on reference date.")
        st.stop()

    ref_ltd = ltd_for_contract(contracts_meta, ref_code)
    ref_F = futures_close_on_date(ref_code, ref_date)
    if ref_ltd is None or ref_F is None:
        st.warning("Need LTD and futures close on reference date.")
        st.stop()

    ref_T = year_fraction(ref_date, ref_ltd, basis=365)
    if ref_T <= 0:
        st.warning("T<=0 on reference date.")
        st.stop()

    ref_r = rate_for(ref_date)
    ref_day = load_options_date_slice(ref_code, ref_date)
    ref_day_iv = compute_day_ivs(ref_day, F=ref_F, T=ref_T, r=ref_r, price_field=price_field).dropna(subset=["iv"])
    if ref_day_iv.empty:
        st.info("No IVs on reference date.")
        st.stop()

    # Sticky strike anchor: reference ATM strike (nearest strike)
    strikes = ref_day_iv["strike"].dropna().to_numpy()
    K_ref = float(strikes[np.argmin(np.abs(strikes - ref_F))])

    # Sticky delta anchor: reference 25Δ call strike estimate
    sigma0 = atm_iv_from_day(ref_day_iv, ref_F)
    if sigma0 is None:
        st.info("Could not infer ATM IV for reference date.")
        st.stop()

    K_low = max(0.01, ref_F * 0.3)
    K_high = ref_F * 3.0

    K_25c_ref = strike_from_delta_black76(
        0.25, ref_F, ref_T, ref_r, sigma0, True, K_low=K_low, K_high=K_high
    )
    if K_25c_ref is None:
        st.info("Could not compute 25Δ call strike on reference date.")
        st.stop()

    # Build time series:
    # - sticky strike: IV at K_ref each date (nearest strike)
    # - sticky delta: IV at that day's 25Δ call strike (nearest strike to K_25c(d))
    rows = []
    for d in report_dates:
        code = map_c1.get(d)
        if code is None:
            continue
        ltd = ltd_for_contract(contracts_meta, code)
        if ltd is None:
            continue
        F = futures_close_on_date(code, d)
        if F is None:
            continue
        T = year_fraction(d, ltd, basis=365)
        if T <= 0:
            continue
        r = rate_for(d)
        day = load_options_date_slice(code, d)
        if day.empty:
            continue

        day_iv = compute_day_ivs(day, F=F, T=T, r=r, price_field=price_field).dropna(subset=["iv"])
        if day_iv.empty:
            continue

        # sticky strike series uses the same absolute strike K_ref
        iv_sticky_strike = nearest_iv_at_strike(day_iv, K_ref, "Call")

        # sticky delta needs strike_from_delta using that day's ATM IV seed
        seed = atm_iv_from_day(day_iv, F)
        if seed is None:
            continue
        K_25c = strike_from_delta_black76(0.25, F, T, r, seed, True, K_low=max(0.01, F * 0.3), K_high=F * 3.0)
        iv_sticky_delta = nearest_iv_at_strike(day_iv, K_25c, "Call") if K_25c is not None else None

        rows.append({
            "date": d,
            "active_code": code,
            "F": F,
            "sticky_strike_iv": iv_sticky_strike,
            "sticky_delta_iv": iv_sticky_delta,
        })

    df = pd.DataFrame(rows).dropna(subset=["sticky_strike_iv", "sticky_delta_iv"])
    if df.empty:
        st.info("Not enough points to compare.")
        st.stop()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["sticky_strike_iv"], mode="lines", name=f"Sticky strike (K={K_ref:.2f})"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["sticky_delta_iv"], mode="lines", name="Sticky delta (25Δ call)"))
    if mode != "Specific contract":
        add_roll_lines(fig, rolls_c1)
    fig.update_layout(
        title="Sticky-Strike vs Sticky-Delta IV",
        xaxis_title="Date",
        yaxis_title="IV",
        hovermode="x unified",
        height=560,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Interpretation: if sticky-delta series is smoother/more stable than sticky-strike, market behaves more sticky-delta (common in FX/vol markets).")
    st.dataframe(df.tail(120), use_container_width=True)

# ====================
# TAB 11: Manual IV + Greeks
# ====================

with tabs[10]:
    st.subheader("IV Calculator + Greeks (Black–76)")

    col1, col2, col3 = st.columns(3)
    with col1:
        F_in = st.number_input("Futures price (F)", value=80.0, step=0.1)
        K_in = st.number_input("Strike (K)", value=80.0, step=0.5)
        opt_px_in = st.number_input("Option price", value=2.0, step=0.01)

    with col2:
        option_type_in = st.selectbox("Option type", ["Call", "Put"], index=0)
        val_date = st.date_input("Valuation date", value=today, key="calc_val_date")
        expiry = st.date_input("Expiry (LTD)", value=today + timedelta(days=60), key="calc_expiry")

    with col3:
        r_val = rate_for(val_date)
        st.write(f"Rate used: **r = {r_val:.4f}**")
        basis = st.selectbox("Day count basis", [365, 252], index=0)

    T_in = year_fraction(val_date, expiry, basis=int(basis))
    st.write(f"Time to expiry: **T = {T_in:.6f}** years")

    if st.button("Compute IV + Greeks"):
        iv = implied_vol_black76(
            price=float(opt_px_in),
            F=float(F_in),
            K=float(K_in),
            T=float(T_in),
            r=float(r_val),
            is_call=(option_type_in == "Call"),
        )
        if iv is None:
            st.error("IV could not be solved (check T, price bounds, or inputs).")
        else:
            g = black76_greeks(
                F=float(F_in),
                K=float(K_in),
                T=float(T_in),
                r=float(r_val),
                sigma=float(iv),
                is_call=(option_type_in == "Call"),
            )
            st.success(f"Implied volatility: **{iv:.4%}**")
            st.table(pd.DataFrame({
                "price": [g.price],
                "delta": [g.delta],
                "gamma": [g.gamma],
                "vega": [g.vega],
                "theta": [g.theta],
            }).T.rename(columns={0: "value"}))
