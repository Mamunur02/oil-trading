# scripts/brent_futures_analytics.py

import sqlite3
from pathlib import Path
from datetime import date

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np

# ==================== Paths & DB ====================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "db" / "brent_futures.db"


def get_conn():
    return sqlite3.connect(DB_PATH)


# ==================== Data Access ====================

def get_all_contract_codes():
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT DISTINCT contract_code FROM brent_prices ORDER BY contract_code",
        conn,
    )
    conn.close()
    return df["contract_code"].tolist()


def load_contracts_metadata() -> pd.DataFrame:
    """
    Load contract metadata. last_trade_date (LTD) is required for continuous futures and curve ordering.
    """
    conn = get_conn()
    df = pd.read_sql_query(
        """
        SELECT contract_code, month_code, year, last_trade_date
        FROM brent_contracts
        """,
        conn,
    )
    conn.close()

    if not df.empty:
        df["last_trade_date"] = pd.to_datetime(df["last_trade_date"], errors="coerce")

    return df


def load_futures_data(
    contract_codes,
    start_date=None,
    end_date=None,
):
    conn = get_conn()

    query = """
        SELECT
            trade_date,
            contract_code,
            open,
            high,
            low,
            close,
            volume
        FROM brent_prices
    """

    where = []
    params = []

    if contract_codes:
        placeholders = ",".join(["?"] * len(contract_codes))
        where.append(f"contract_code IN ({placeholders})")
        params.extend(contract_codes)

    if start_date:
        where.append("trade_date >= ?")
        params.append(start_date.isoformat())

    if end_date:
        where.append("trade_date <= ?")
        params.append(end_date.isoformat())

    if where:
        query += " WHERE " + " AND ".join(where)

    query += " ORDER BY trade_date ASC"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"])

    return df


# ==================== Indicators ====================

def add_moving_averages(df: pd.DataFrame, windows, price_col="close", group_col="contract_code"):
    """
    Adds SMA columns.
    """
    df = df.sort_values("trade_date")

    for w in windows:
        col = f"sma_{w}"
        if group_col:
            df[col] = (
                df.groupby(group_col)[price_col]
                  .transform(lambda x: x.rolling(w).mean())
            )
        else:
            df[col] = df[price_col].rolling(w).mean()

    return df


# ==================== Continuous Futures (Step 3) ====================

def build_continuous_mapping(dates: pd.Series, contracts_meta: pd.DataFrame, rank: int) -> pd.DataFrame:
    """
    For each trade_date, pick the 'rank'-th nearest contract by LTD among those with LTD >= trade_date.
    rank=1 => C1, rank=2 => C2, etc.
    Returns: trade_date, contract_code
    """
    if contracts_meta.empty:
        return pd.DataFrame(columns=["trade_date", "contract_code"])

    meta = contracts_meta.dropna(subset=["contract_code", "last_trade_date"]).copy()
    meta = meta.sort_values("last_trade_date")

    codes = meta["contract_code"].to_numpy()
    ltds = meta["last_trade_date"].to_numpy()

    out_dates = []
    out_codes = []

    for d in pd.to_datetime(dates).sort_values().unique():
        valid = (ltds >= d)
        if valid.sum() >= rank:
            valid_codes = codes[valid]  # already sorted by LTD
            out_dates.append(d)
            out_codes.append(valid_codes[rank - 1])
        else:
            out_dates.append(d)
            out_codes.append(None)

    return pd.DataFrame({"trade_date": out_dates, "contract_code": out_codes})


def build_continuous_series(df_prices: pd.DataFrame, contracts_meta: pd.DataFrame, rank: int) -> pd.DataFrame:
    """
    Join the mapping (trade_date -> contract_code) to the actual price rows.
    Output:
      trade_date, cont_code, contract_code, open, high, low, close, volume
    """
    if df_prices.empty:
        return df_prices

    mapping = build_continuous_mapping(df_prices["trade_date"], contracts_meta, rank=rank)

    df_join = mapping.merge(
        df_prices,
        on=["trade_date", "contract_code"],
        how="left",
        validate="one_to_one",
    )

    df_join["cont_code"] = f"C{rank}"
    df_join = df_join.dropna(subset=["contract_code", "close"]).copy()

    return df_join


def build_continuous_panel(df_prices_all: pd.DataFrame, contracts_meta: pd.DataFrame, ranks=(1, 2, 3)) -> pd.DataFrame:
    """
    Build C1/C2/C3 (or any ranks) and return a wide dataframe indexed by trade_date:
      trade_date, C1_close, C2_close, C3_close, C1_volume, ...
    """
    series = {}
    for r in ranks:
        df_c = build_continuous_series(df_prices_all, contracts_meta, rank=r)
        if df_c.empty:
            continue
        df_c = df_c.sort_values("trade_date")
        series[f"C{r}_close"] = df_c.set_index("trade_date")["close"]
        series[f"C{r}_volume"] = df_c.set_index("trade_date")["volume"]
        series[f"C{r}_contract"] = df_c.set_index("trade_date")["contract_code"]

    if not series:
        return pd.DataFrame()

    df_wide = pd.DataFrame(series).sort_index()
    df_wide = df_wide.reset_index().rename(columns={"index": "trade_date"})
    return df_wide


# ==================== Term Structure (Step 4) ====================

def get_curve_snapshot(
    trade_date: pd.Timestamp,
    df_prices_all: pd.DataFrame,
    contracts_meta: pd.DataFrame,
    max_contracts: int = 12,
) -> pd.DataFrame:
    """
    Curve snapshot for a given trade_date:
      - join prices (for that date) to LTD
      - keep contracts with LTD >= trade_date
      - sort by LTD
      - compute days_to_expiry
    """
    if df_prices_all.empty or contracts_meta.empty:
        return pd.DataFrame()

    d = pd.to_datetime(trade_date)

    px = df_prices_all[df_prices_all["trade_date"] == d][["trade_date", "contract_code", "close", "volume"]].copy()
    if px.empty:
        return pd.DataFrame()

    meta = contracts_meta[["contract_code", "last_trade_date"]].copy()
    meta = meta.dropna(subset=["last_trade_date"])

    out = px.merge(meta, on="contract_code", how="left")
    out = out.dropna(subset=["last_trade_date"])

    # Only keep contracts that haven't expired as of trade date
    out = out[out["last_trade_date"] >= d].copy()
    if out.empty:
        return pd.DataFrame()

    out["days_to_expiry"] = (out["last_trade_date"] - d).dt.days
    out = out.sort_values(["days_to_expiry", "contract_code"]).head(max_contracts)

    return out

# ==================== Spreads & Roll Analytics (Step 5) ====================

def compute_calendar_spread(
    df_prices: pd.DataFrame,
    near: str,
    far: str,
) -> pd.DataFrame:
    """
    Compute near - far calendar spread for two contract codes.
    """
    df_n = df_prices[df_prices["contract_code"] == near][["trade_date", "close"]].rename(
        columns={"close": "near_close"}
    )
    df_f = df_prices[df_prices["contract_code"] == far][["trade_date", "close"]].rename(
        columns={"close": "far_close"}
    )

    df = df_n.merge(df_f, on="trade_date", how="inner")
    df["spread"] = df["near_close"] - df["far_close"]

    return df.sort_values("trade_date")


def compute_roll_yield(
    df_cont_panel: pd.DataFrame,
    contracts_meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute a roll yield proxy from C1 and C2.
    Positive => backwardation
    Negative => contango
    """
    required_cols = {"C1_close", "C2_close", "C1_contract", "C2_contract"}
    if not required_cols.issubset(df_cont_panel.columns):
        return pd.DataFrame()

    meta = contracts_meta.set_index("contract_code")

    df = df_cont_panel.copy()
    df = df.dropna(subset=list(required_cols))

    trade_dates = pd.to_datetime(df["trade_date"])

    # Days to expiry (SAFE pandas way)
    df["C1_dte"] = (
        meta.loc[df["C1_contract"], "last_trade_date"].values
        - trade_dates
    ).dt.days

    df["C2_dte"] = (
        meta.loc[df["C2_contract"], "last_trade_date"].values
        - trade_dates
    ).dt.days

    # Avoid division by zero / nonsense
    df = df[(df["C2_dte"] > df["C1_dte"]) & (df["C1_dte"] > 0)]

    df["roll_yield"] = (df["C1_close"] - df["C2_close"]) / (
        df["C2_dte"] - df["C1_dte"]
    )

    return df[["trade_date", "roll_yield"]].dropna()

# ==================== Liquidity Analytics (Step 6) ====================

def compute_daily_dominant_contract(df_prices_all: pd.DataFrame) -> pd.DataFrame:
    """
    For each trade_date, find the contract with maximum volume.
    Returns: trade_date, dominant_contract, dominant_volume, total_volume, dominance_share
    """
    if df_prices_all.empty:
        return pd.DataFrame()

    df = df_prices_all.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    # total volume per day across all contracts
    total = (
        df.groupby("trade_date")["volume"]
          .sum(min_count=1)
          .rename("total_volume")
          .reset_index()
    )

    # dominant contract per day (max volume)
    idx = df.groupby("trade_date")["volume"].idxmax()
    dom = df.loc[idx, ["trade_date", "contract_code", "volume"]].rename(
        columns={"contract_code": "dominant_contract", "volume": "dominant_volume"}
    )

    out = dom.merge(total, on="trade_date", how="left")
    out["dominance_share"] = out["dominant_volume"] / out["total_volume"]
    return out.sort_values("trade_date")


def compute_volume_distribution_on_date(
    df_prices_all: pd.DataFrame,
    contracts_meta: pd.DataFrame,
    trade_date: pd.Timestamp,
    max_contracts: int = 12,
) -> pd.DataFrame:
    """
    Volume by maturity on a given date, ordered by LTD / days_to_expiry.
    Returns: contract_code, volume, close, last_trade_date, days_to_expiry
    """
    if df_prices_all.empty or contracts_meta.empty:
        return pd.DataFrame()

    d = pd.to_datetime(trade_date)

    px = df_prices_all[df_prices_all["trade_date"] == d][["contract_code", "volume", "close"]].copy()
    if px.empty:
        return pd.DataFrame()

    meta = contracts_meta[["contract_code", "last_trade_date"]].copy()
    meta["last_trade_date"] = pd.to_datetime(meta["last_trade_date"], errors="coerce")
    meta = meta.dropna(subset=["last_trade_date"])

    out = px.merge(meta, on="contract_code", how="left").dropna(subset=["last_trade_date"])
    out = out[out["last_trade_date"] >= d].copy()

    out["days_to_expiry"] = (out["last_trade_date"] - d).dt.days
    out = out.sort_values("days_to_expiry").head(max_contracts)

    return out


def compute_volume_migration(df_dom: pd.DataFrame) -> pd.DataFrame:
    """
    Identify dates when the dominant contract changes (liquidity roll points).
    Returns: trade_date, prev_dominant, new_dominant
    """
    if df_dom.empty:
        return pd.DataFrame()

    df = df_dom[["trade_date", "dominant_contract"]].copy()
    df["prev_dominant"] = df["dominant_contract"].shift(1)
    changes = df[df["dominant_contract"] != df["prev_dominant"]].dropna(subset=["prev_dominant"])
    changes = changes.rename(columns={"dominant_contract": "new_dominant"})
    return changes[["trade_date", "prev_dominant", "new_dominant"]]

# ==================== Volatility & Returns (Step 7) ====================

def compute_log_returns(df: pd.DataFrame, price_col="close", group_col=None):
    """
    Compute log returns.
    If group_col is provided, compute per group (e.g. per contract or cont_code).
    """
    df = df.sort_values("trade_date").copy()

    if group_col:
        df["log_return"] = (
            df.groupby(group_col)[price_col]
              .apply(lambda x: np.log(x / x.shift(1)))
              .reset_index(level=0, drop=True)
        )
    else:
        df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))

    return df


def compute_realised_vol(
    df: pd.DataFrame,
    window: int,
    group_col=None,
    annualise: bool = True,
    trading_days: int = 252,
):
    """
    Rolling realised volatility from log returns.
    """
    df = df.copy()

    if group_col:
        df[f"rv_{window}"] = (
            df.groupby(group_col)["log_return"]
              .transform(lambda x: x.rolling(window).std())
        )
    else:
        df[f"rv_{window}"] = df["log_return"].rolling(window).std()

    if annualise:
        df[f"rv_{window}"] *= np.sqrt(trading_days)

    return df

# ==================== Correlation & Regimes (Step 8) ====================

def compute_rolling_correlation(
    x: pd.Series,
    y: pd.Series,
    window: int,
) -> pd.Series:
    """
    Rolling Pearson correlation.
    """
    return x.rolling(window).corr(y)


def build_continuous_returns_panel(
    df_cont_panel: pd.DataFrame,
):
    """
    Build log returns for C1/C2/C3 in a wide format.
    """
    df = df_cont_panel.copy()
    df = df.sort_values("trade_date")

    for c in ["C1", "C2", "C3"]:
        col = f"{c}_close"
        if col in df.columns:
            df[f"{c}_ret"] = np.log(df[col] / df[col].shift(1))

    return df

# ==================== Simple Returns & Event Queries ====================

def compute_simple_returns(
    df: pd.DataFrame,
    price_col="close",
    group_col=None,
):
    """
    Compute simple percentage returns.
    """
    df = df.sort_values("trade_date").copy()

    if group_col:
        df["simple_return"] = (
            df.groupby(group_col)[price_col]
              .pct_change()
        )
    else:
        df["simple_return"] = df[price_col].pct_change()

    return df


def compute_simple_realised_vol(
    df: pd.DataFrame,
    window: int,
    group_col=None,
    annualise: bool = True,
    trading_days: int = 252,
):
    """
    Rolling realised volatility from simple returns.
    """
    df = df.copy()

    if group_col:
        df[f"srv_{window}"] = (
            df.groupby(group_col)["simple_return"]
              .transform(lambda x: x.rolling(window).std())
        )
    else:
        df[f"srv_{window}"] = df["simple_return"].rolling(window).std()

    if annualise:
        df[f"srv_{window}"] *= np.sqrt(trading_days)

    return df


# ==================== Streamlit Setup ====================

st.set_page_config(page_title="Brent Futures Analytics", layout="wide")
st.title("Brent Futures — Trading Analytics")

# ==================== Sidebar Filters ====================

st.sidebar.header("Filters")

all_contracts = get_all_contract_codes()

selected_contracts = st.sidebar.multiselect(
    "Raw futures contracts (for raw tabs)",
    options=all_contracts,
    default=all_contracts[:1] if all_contracts else [],
)

date_range = st.sidebar.date_input(
    "Date range",
    value=(date.today().replace(year=date.today().year - 1), date.today()),
)

if isinstance(date_range, tuple):
    start_date, end_date = date_range
else:
    start_date = end_date = None

# ---------- Technical indicator controls ----------

st.sidebar.markdown("### Technical Indicators (SMA)")

show_sma_20 = st.sidebar.checkbox("SMA 20", value=True)
show_sma_50 = st.sidebar.checkbox("SMA 50", value=False)
show_sma_100 = st.sidebar.checkbox("SMA 100", value=False)
show_sma_200 = st.sidebar.checkbox("SMA 200", value=False)

sma_windows = []
if show_sma_20:
    sma_windows.append(20)
if show_sma_50:
    sma_windows.append(50)
if show_sma_100:
    sma_windows.append(100)
if show_sma_200:
    sma_windows.append(200)

st.sidebar.markdown("### Liquidity")
liq_top_n = st.sidebar.slider("Top N contracts in volume distribution", 5, 30, 12)

st.sidebar.markdown("### Volatility")

vol_windows = st.sidebar.multiselect(
    "Volatility windows (days)",
    options=[10, 20, 30, 60, 90],
    default=[20],
)
st.sidebar.markdown("### Correlation & Regimes")

corr_window = st.sidebar.selectbox(
    "Rolling correlation window (days)",
    options=[10, 20, 30, 60],
    index=1,
)


# ---------- Continuous futures controls ----------

st.sidebar.markdown("### Continuous Futures (C1/C2/C3)")
cont_rank = st.sidebar.selectbox(
    "Continuous series",
    options=[1, 2, 3],
    index=0,
    format_func=lambda r: f"C{r}",
)

# ---------- Term structure controls ----------

st.sidebar.markdown("### Term Structure")
curve_max_contracts = st.sidebar.slider("Curve contracts to show", 5, 24, 12)

# ==================== Load Data ====================

# Raw contracts for Steps 1–2
if not selected_contracts:
    df_raw = pd.DataFrame()
else:
    df_raw = load_futures_data(
        contract_codes=selected_contracts,
        start_date=start_date,
        end_date=end_date,
    )

# Metadata for Step 3–4
contracts_meta = load_contracts_metadata()
missing_ltd = contracts_meta["last_trade_date"].isna().sum() if not contracts_meta.empty else 0
if missing_ltd > 0:
    st.warning(
        f"{missing_ltd} contract(s) in brent_contracts are missing last_trade_date. "
        "Continuous futures and curve ordering need LTD."
    )

# All prices in range for continuous/curve analytics
df_all_for_cont = load_futures_data(
    contract_codes=all_contracts if all_contracts else [],
    start_date=start_date,
    end_date=end_date,
)

df_dom = compute_daily_dominant_contract(df_all_for_cont)
df_dom_changes = compute_volume_migration(df_dom)

# Add SMAs to raw
if sma_windows and not df_raw.empty:
    df_raw = add_moving_averages(df_raw, sma_windows, price_col="close", group_col="contract_code")

# Build continuous (selected rank) for Step 3 tab
df_cont_selected = build_continuous_series(df_all_for_cont, contracts_meta, rank=int(cont_rank))
if sma_windows and not df_cont_selected.empty:
    df_cont_selected = add_moving_averages(df_cont_selected, sma_windows, price_col="close", group_col="cont_code")

# Build a continuous panel (C1/C2/C3) for spreads & term structure
df_cont_panel = build_continuous_panel(df_all_for_cont, contracts_meta, ranks=(1, 2, 3))

# ---- Volatility prep on continuous futures ----
df_vol = df_cont_selected.copy()

if not df_vol.empty:
    df_vol = compute_log_returns(
        df_vol,
        price_col="close",
        group_col="cont_code",
    )

    for w in vol_windows:
        df_vol = compute_realised_vol(
            df_vol,
            window=w,
            group_col="cont_code",
        )

# ---- Correlation prep ----
df_corr = build_continuous_returns_panel(df_cont_panel)

if not df_corr.empty and "C1_ret" in df_corr.columns:
    # Front spread
    if "C2_close" in df_corr.columns:
        df_corr["C1_C2_spread"] = df_corr["C1_close"] - df_corr["C2_close"]

    # Rolling correlations
    if "C2_ret" in df_corr.columns:
        df_corr["corr_C1_C2"] = compute_rolling_correlation(
            df_corr["C1_ret"], df_corr["C2_ret"], corr_window
        )

    if "C1_C2_spread" in df_corr.columns:
        df_corr["corr_ret_spread"] = compute_rolling_correlation(
            df_corr["C1_ret"], df_corr["C1_C2_spread"], corr_window
        )

# ---- Simple returns & volatility (continuous futures) ----

df_simple = df_cont_selected.copy()

if not df_simple.empty:
    df_simple = compute_simple_returns(
        df_simple,
        price_col="close",
        group_col="cont_code",
    )

    for w in vol_windows:
        df_simple = compute_simple_realised_vol(
            df_simple,
            window=w,
            group_col="cont_code",
        )


# ==================== Tabs ====================

tab_price, tab_tech, tab_cont, tab_curve, tab_spread, tab_liq, tab_vol, tab_corr, tab_events = st.tabs(
    [
        "Price & Volume (Raw)",
        "Technicals (Raw)",
        "Continuous Futures",
        "Term Structure",
        "Spreads & Rolls",
        "Liquidity",
        "Volatility & Returns",
        "Correlation & Regimes",
        "Returns Explorer",
    ]
)

# ==================== Step 1: Price & Volume (Raw) ====================

with tab_price:
    st.subheader("Price & Volume (Raw Contracts)")

    if df_raw.empty:
        st.info("No raw data loaded. Select raw contracts in the sidebar.")
    else:
        fig_price = go.Figure()
        for contract in selected_contracts:
            df_c = df_raw[df_raw["contract_code"] == contract]
            fig_price.add_trace(
                go.Scatter(x=df_c["trade_date"], y=df_c["close"], mode="lines", name=f"{contract} Close")
            )
        fig_price.update_layout(
            title="Futures Price (Close)",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x unified",
            height=500,
        )
        st.plotly_chart(fig_price, use_container_width=True)

        fig_vol = go.Figure()
        for contract in selected_contracts:
            df_c = df_raw[df_raw["contract_code"] == contract]
            fig_vol.add_trace(
                go.Bar(x=df_c["trade_date"], y=df_c["volume"], name=f"{contract} Volume")
            )
        fig_vol.update_layout(
            title="Trading Volume",
            xaxis_title="Date",
            yaxis_title="Volume",
            barmode="group",
            hovermode="x unified",
            height=350,
        )
        st.plotly_chart(fig_vol, use_container_width=True)

# ==================== Step 2: Technicals (Raw) ====================

with tab_tech:
    st.subheader("Technicals (Raw Contracts)")

    if df_raw.empty:
        st.info("No raw data loaded. Select raw contracts in the sidebar.")
    else:
        fig = go.Figure()
        for contract in selected_contracts:
            df_c = df_raw[df_raw["contract_code"] == contract]

            fig.add_trace(
                go.Scatter(
                    x=df_c["trade_date"], y=df_c["close"], mode="lines",
                    name=f"{contract} Close", line=dict(width=2)
                )
            )

            for w in sma_windows:
                col = f"sma_{w}"
                if col in df_c.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df_c["trade_date"], y=df_c[col], mode="lines",
                            name=f"{contract} SMA {w}", line=dict(dash="dot")
                        )
                    )

        fig.update_layout(
            title="Close with Moving Averages (Raw)",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x unified",
            height=550,
        )
        st.plotly_chart(fig, use_container_width=True)

# ==================== Step 3: Continuous Futures ====================

with tab_cont:
    st.subheader("Continuous Futures (Rolled by LTD)")

    if df_cont_selected.empty:
        st.info(
            "No continuous series could be built for this date range. "
            "Check brent_contracts.last_trade_date and price coverage."
        )
    else:
        cont_name = f"C{cont_rank}"
        latest = df_cont_selected["trade_date"].max()
        latest_contract = (
            df_cont_selected.loc[df_cont_selected["trade_date"] == latest, "contract_code"]
            .dropna()
            .iloc[0]
        )
        st.caption(f"{cont_name} on {latest.date().isoformat()} is using contract **{latest_contract}**.")

        fig_c = go.Figure()
        fig_c.add_trace(
            go.Scatter(
                x=df_cont_selected["trade_date"], y=df_cont_selected["close"],
                mode="lines", name=f"{cont_name} Close", line=dict(width=2)
            )
        )

        for w in sma_windows:
            col = f"sma_{w}"
            if col in df_cont_selected.columns:
                fig_c.add_trace(
                    go.Scatter(
                        x=df_cont_selected["trade_date"], y=df_cont_selected[col],
                        mode="lines", name=f"{cont_name} SMA {w}", line=dict(dash="dot")
                    )
                )

        fig_c.update_layout(
            title=f"{cont_name} Continuous Close",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x unified",
            height=550,
        )
        st.plotly_chart(fig_c, use_container_width=True)

        fig_cv = go.Figure()
        fig_cv.add_trace(
            go.Bar(x=df_cont_selected["trade_date"], y=df_cont_selected["volume"], name=f"{cont_name} Volume")
        )
        fig_cv.update_layout(
            title=f"{cont_name} Continuous Volume",
            xaxis_title="Date",
            yaxis_title="Volume",
            hovermode="x unified",
            height=350,
        )
        st.plotly_chart(fig_cv, use_container_width=True)

        st.markdown("#### Roll points (underlying contract changes)")
        df_roll = df_cont_selected[["trade_date", "contract_code"]].drop_duplicates().copy()
        df_roll["prev"] = df_roll["contract_code"].shift(1)
        rolls = df_roll[df_roll["contract_code"] != df_roll["prev"]].dropna(subset=["prev"])
        if rolls.empty:
            st.write("No rolls detected in the selected range.")
        else:
            st.dataframe(rolls[["trade_date", "prev", "contract_code"]], use_container_width=True)

# ==================== Step 4: Term Structure ====================

with tab_curve:
    st.subheader("Term Structure (Contango / Backwardation)")

    if df_all_for_cont.empty or contracts_meta.empty:
        st.info("Need price data + contract metadata (with LTD) to show term structure.")
        st.stop()

    # Use the latest available date in the loaded range as the default curve date
    available_dates = sorted(df_all_for_cont["trade_date"].dropna().unique())
    if not available_dates:
        st.info("No dates available in the selected range.")
        st.stop()

    default_curve_date = available_dates[-1]
    curve_date = st.date_input(
        "Curve snapshot date",
        value=default_curve_date.date(),
        min_value=available_dates[0].date(),
        max_value=available_dates[-1].date(),
    )
    curve_date_ts = pd.to_datetime(curve_date)

    # ---- Curve Snapshot ----
    curve_df = get_curve_snapshot(
        trade_date=curve_date_ts,
        df_prices_all=df_all_for_cont,
        contracts_meta=contracts_meta,
        max_contracts=int(curve_max_contracts),
    )

    if curve_df.empty:
        st.info("No curve snapshot could be built for that date (missing prices or LTDs).")
    else:
        # Plot curve by days_to_expiry
        fig_curve = go.Figure()
        fig_curve.add_trace(
            go.Scatter(
                x=curve_df["days_to_expiry"],
                y=curve_df["close"],
                mode="lines+markers",
                name="Curve",
                text=curve_df["contract_code"],
                hovertemplate="Contract=%{text}<br>DaysToExpiry=%{x}<br>Close=%{y}<extra></extra>",
            )
        )
        fig_curve.update_layout(
            title=f"Brent Futures Curve on {curve_date_ts.date().isoformat()}",
            xaxis_title="Days to expiry (from LTD)",
            yaxis_title="Close price",
            height=500,
        )
        st.plotly_chart(fig_curve, use_container_width=True)

        # A quick “state” readout using continuous panel (if available on the curve date)
        if not df_cont_panel.empty:
            row = df_cont_panel[df_cont_panel["trade_date"] == curve_date_ts]
            if not row.empty:
                c1 = row["C1_close"].iloc[0] if "C1_close" in row else None
                c2 = row["C2_close"].iloc[0] if "C2_close" in row else None

                if pd.notna(c1) and pd.notna(c2):
                    spread = float(c1 - c2)
                    state = "Backwardation" if spread > 0 else "Contango" if spread < 0 else "Flat"
                    st.caption(f"On {curve_date_ts.date().isoformat()}, **C1 − C2 = {spread:.4f}** → **{state}**.")

        st.markdown("#### Curve table")
        st.dataframe(
            curve_df[["contract_code", "last_trade_date", "days_to_expiry", "close", "volume"]],
            use_container_width=True,
        )

    # ---- Spreads Over Time (C1-C2, C1-C3) ----
    st.markdown("### Contango / Backwardation Spreads Over Time")

    if df_cont_panel.empty or ("C1_close" not in df_cont_panel.columns):
        st.info("Continuous panel (C1/C2/C3) not available — check LTD coverage.")
    else:
        df_sp = df_cont_panel.copy()
        df_sp["trade_date"] = pd.to_datetime(df_sp["trade_date"])

        if "C2_close" in df_sp.columns:
            df_sp["C1_minus_C2"] = df_sp["C1_close"] - df_sp["C2_close"]
        if "C3_close" in df_sp.columns:
            df_sp["C1_minus_C3"] = df_sp["C1_close"] - df_sp["C3_close"]

        fig_sp = go.Figure()
        if "C1_minus_C2" in df_sp.columns:
            fig_sp.add_trace(go.Scatter(x=df_sp["trade_date"], y=df_sp["C1_minus_C2"], mode="lines", name="C1 − C2"))
        if "C1_minus_C3" in df_sp.columns:
            fig_sp.add_trace(go.Scatter(x=df_sp["trade_date"], y=df_sp["C1_minus_C3"], mode="lines", name="C1 − C3"))

        fig_sp.update_layout(
            title="Front spreads (positive → backwardation, negative → contango)",
            xaxis_title="Date",
            yaxis_title="Spread",
            hovermode="x unified",
            height=450,
        )
        st.plotly_chart(fig_sp, use_container_width=True)

# ==================== Step 5: Spreads & Rolls ====================

with tab_spread:
    st.subheader("Spreads & Roll Analytics")

    if df_all_for_cont.empty:
        st.info("Need futures price data to compute spreads.")
        st.stop()

    # ---------- Calendar Spread ----------

    st.markdown("### Calendar Spread")

    col1, col2 = st.columns(2)
    with col1:
        near_contract = st.selectbox("Near contract", all_contracts, index=0)
    with col2:
        far_contract = st.selectbox("Far contract", all_contracts, index=1 if len(all_contracts) > 1 else 0)

    if near_contract == far_contract:
        st.warning("Near and far contracts must be different.")
    else:
        df_spread = compute_calendar_spread(
            df_prices=df_all_for_cont,
            near=near_contract,
            far=far_contract,
        )

        if df_spread.empty:
            st.info("No overlapping dates between selected contracts.")
        else:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df_spread["trade_date"],
                    y=df_spread["spread"],
                    mode="lines",
                    name=f"{near_contract} − {far_contract}",
                )
            )
            fig.update_layout(
                title=f"Calendar Spread: {near_contract} − {far_contract}",
                xaxis_title="Date",
                yaxis_title="Price spread",
                hovermode="x unified",
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ---------- Roll Yield ----------

    st.markdown("### Roll Yield (C1 → C2)")

    if df_cont_panel.empty:
        st.info("Continuous futures panel (C1/C2) not available.")
    else:
        df_roll_yield = compute_roll_yield(df_cont_panel, contracts_meta)

        if df_roll_yield.empty:
            st.info("Roll yield could not be computed.")
        else:
            fig_ry = go.Figure()
            fig_ry.add_trace(
                go.Scatter(
                    x=df_roll_yield["trade_date"],
                    y=df_roll_yield["roll_yield"],
                    mode="lines",
                    name="Roll yield",
                )
            )
            fig_ry.update_layout(
                title="Roll Yield Proxy (positive = backwardation)",
                xaxis_title="Date",
                yaxis_title="Roll yield",
                hovermode="x unified",
                height=450,
            )
            st.plotly_chart(fig_ry, use_container_width=True)

    # ---------- Roll Diagnostics ----------

    st.markdown("### Roll Events (C1)")

    if df_cont_selected.empty:
        st.info("Continuous series not available.")
    else:
        df_rolls = df_cont_selected[["trade_date", "contract_code"]].copy()
        df_rolls["prev"] = df_rolls["contract_code"].shift(1)
        roll_events = df_rolls[df_rolls["contract_code"] != df_rolls["prev"]].dropna(subset=["prev"])

        if roll_events.empty:
            st.write("No roll events in selected range.")
        else:
            st.dataframe(
                roll_events.rename(
                    columns={
                        "prev": "previous_contract",
                        "contract_code": "new_contract",
                    }
                ),
                use_container_width=True,
            )

# ==================== Step 6: Liquidity ====================

with tab_liq:
    st.subheader("Liquidity & Volume Dynamics")

    if df_all_for_cont.empty:
        st.info("No futures data available in the selected date range.")
        st.stop()

    # ---- Total market volume over time ----
    st.markdown("### Total Market Volume (all contracts)")

    df_total = (
        df_all_for_cont.groupby("trade_date")["volume"]
        .sum(min_count=1)
        .reset_index()
        .sort_values("trade_date")
    )

    fig_total = go.Figure()
    fig_total.add_trace(
        go.Scatter(
            x=df_total["trade_date"],
            y=df_total["volume"],
            mode="lines",
            name="Total volume",
        )
    )
    fig_total.update_layout(
        title="Total Volume Across All Contracts",
        xaxis_title="Date",
        yaxis_title="Volume",
        hovermode="x unified",
        height=420,
    )
    st.plotly_chart(fig_total, use_container_width=True)

    # ---- Dominant contract over time ----
    st.markdown("### Dominant Contract (by volume)")

    if df_dom.empty:
        st.info("Could not compute dominant contract series (check volume data).")
    else:
        # Plot dominance share (how concentrated liquidity is)
        fig_dom = go.Figure()
        fig_dom.add_trace(
            go.Scatter(
                x=df_dom["trade_date"],
                y=df_dom["dominance_share"],
                mode="lines",
                name="Dominance share",
            )
        )
        fig_dom.update_layout(
            title="Dominance Share: max(contract volume) / total volume",
            xaxis_title="Date",
            yaxis_title="Share",
            hovermode="x unified",
            height=420,
        )
        st.plotly_chart(fig_dom, use_container_width=True)

        st.markdown("#### Liquidity roll points (dominant contract changes)")
        if df_dom_changes.empty:
            st.write("No dominant-contract changes detected in this range.")
        else:
            st.dataframe(df_dom_changes, use_container_width=True)

    # ---- Volume distribution along the curve on a chosen date ----
    st.markdown("### Volume Distribution Along the Curve (snapshot)")

    available_dates = sorted(df_all_for_cont["trade_date"].dropna().unique())
    default_dt = available_dates[-1]
    snap_date = st.date_input(
        "Snapshot date for curve volume distribution",
        value=pd.to_datetime(default_dt).date(),
        min_value=pd.to_datetime(available_dates[0]).date(),
        max_value=pd.to_datetime(available_dates[-1]).date(),
        key="liq_snapshot_date",
    )

    vol_curve = compute_volume_distribution_on_date(
        df_prices_all=df_all_for_cont,
        contracts_meta=contracts_meta,
        trade_date=pd.to_datetime(snap_date),
        max_contracts=int(liq_top_n),
    )

    if vol_curve.empty:
        st.info("No volume distribution available for that date (check LTD + prices).")
    else:
        # Bar chart ordered by days_to_expiry
        fig_vc = go.Figure()
        fig_vc.add_trace(
            go.Bar(
                x=vol_curve["days_to_expiry"],
                y=vol_curve["volume"],
                text=vol_curve["contract_code"],
                hovertemplate="Contract=%{text}<br>DaysToExpiry=%{x}<br>Volume=%{y}<extra></extra>",
                name="Volume",
            )
        )
        fig_vc.update_layout(
            title=f"Volume by Maturity on {pd.to_datetime(snap_date).date().isoformat()}",
            xaxis_title="Days to expiry (from LTD)",
            yaxis_title="Volume",
            height=450,
        )
        st.plotly_chart(fig_vc, use_container_width=True)

        st.markdown("#### Snapshot table")
        st.dataframe(
            vol_curve[["contract_code", "last_trade_date", "days_to_expiry", "close", "volume"]],
            use_container_width=True,
        )

# ==================== Step 7: Volatility & Returns ====================

with tab_vol:
    st.subheader("Volatility & Returns")

    if df_vol.empty or "log_return" not in df_vol.columns:
        st.info("Not enough data to compute returns / volatility.")
        st.stop()

    cont_name = df_vol["cont_code"].iloc[0]

    # ---- Log returns time series ----
    st.markdown("### Log Returns")

    fig_ret = go.Figure()
    fig_ret.add_trace(
        go.Scatter(
            x=df_vol["trade_date"],
            y=df_vol["log_return"],
            mode="lines",
            name="Log return",
        )
    )
    fig_ret.update_layout(
        title=f"{cont_name} Log Returns",
        xaxis_title="Date",
        yaxis_title="Log return",
        hovermode="x unified",
        height=420,
    )
    st.plotly_chart(fig_ret, use_container_width=True)

    # ---- Realised volatility ----
    st.markdown("### Realised Volatility")

    fig_vol = go.Figure()
    for w in vol_windows:
        col = f"rv_{w}"
        if col in df_vol.columns:
            fig_vol.add_trace(
                go.Scatter(
                    x=df_vol["trade_date"],
                    y=df_vol[col],
                    mode="lines",
                    name=f"{w}d RV",
                )
            )

    fig_vol.update_layout(
        title=f"{cont_name} Rolling Realised Volatility (annualised)",
        xaxis_title="Date",
        yaxis_title="Volatility",
        hovermode="x unified",
        height=450,
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    # ---- Returns distribution ----
    st.markdown("### Returns Distribution")

    clean_rets = df_vol["log_return"].dropna()

    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Histogram(
            x=clean_rets,
            nbinsx=60,
            name="Returns",
        )
    )
    fig_hist.update_layout(
        title=f"{cont_name} Log Returns Distribution",
        xaxis_title="Log return",
        yaxis_title="Frequency",
        height=400,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ---- Quick stats ----
    st.markdown("### Summary Statistics")

    stats = {
        "Mean return": clean_rets.mean(),
        "Std dev": clean_rets.std(),
        "Skew": clean_rets.skew(),
        "Kurtosis": clean_rets.kurtosis(),
    }

    st.table(pd.DataFrame(stats, index=[cont_name]).T)

# ==================== Step 8: Correlation & Regimes ====================

with tab_corr:
    st.subheader("Correlation & Market Regimes")

    if df_corr.empty or "C1_ret" not in df_corr.columns:
        st.info("Not enough data to compute correlations.")
        st.stop()

    # ---- Rolling correlations ----
    st.markdown("### Rolling Correlations")

    fig_corr = go.Figure()

    if "corr_C1_C2" in df_corr.columns:
        fig_corr.add_trace(
            go.Scatter(
                x=df_corr["trade_date"],
                y=df_corr["corr_C1_C2"],
                mode="lines",
                name=f"C1 vs C2 returns ({corr_window}d)",
            )
        )

    if "corr_ret_spread" in df_corr.columns:
        fig_corr.add_trace(
            go.Scatter(
                x=df_corr["trade_date"],
                y=df_corr["corr_ret_spread"],
                mode="lines",
                name=f"C1 return vs C1–C2 spread ({corr_window}d)",
            )
        )

    fig_corr.update_layout(
        title="Rolling Correlations",
        xaxis_title="Date",
        yaxis_title="Correlation",
        hovermode="x unified",
        height=450,
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # ---- Static correlation matrix ----
    st.markdown("### Returns Correlation Matrix")

    ret_cols = [c for c in ["C1_ret", "C2_ret", "C3_ret"] if c in df_corr.columns]
    corr_matrix = df_corr[ret_cols].corr()

    st.dataframe(corr_matrix.style.format("{:.2f}"))

    # ---- Regime classification (simple & interpretable) ----
    st.markdown("### Simple Regime Signals")

    latest = df_corr.dropna().iloc[-1]

    regime_notes = []

    if "corr_C1_C2" in latest:
        if latest["corr_C1_C2"] > 0.8:
            regime_notes.append("High curve cohesion (directional regime)")
        elif latest["corr_C1_C2"] < 0.3:
            regime_notes.append("Curve decoupling (spread / stress regime)")

    if "corr_ret_spread" in latest:
        if latest["corr_ret_spread"] < -0.3:
            regime_notes.append("Mean-reverting carry dynamics")
        elif latest["corr_ret_spread"] > 0.3:
            regime_notes.append("Momentum aligned with curve")

    if not regime_notes:
        regime_notes.append("Neutral / mixed regime")

    for note in regime_notes:
        st.write(f"• {note}")

# ==================== Final: Returns Explorer ====================

with tab_events:
    st.subheader("Returns & Volatility Explorer")

    if df_simple.empty or "simple_return" not in df_simple.columns:
        st.info("Simple returns not available.")
        st.stop()

    cont_name = df_simple["cont_code"].iloc[0]

    # ---------- Controls ----------
    st.markdown("### Event Filter")

    col1, col2, col3 = st.columns(3)

    with col1:
        threshold_pct = st.number_input(
            "Return threshold (%)",
            min_value=0.5,
            max_value=20.0,
            value=5.0,
            step=0.5,
        ) / 100.0

    with col2:
        direction = st.selectbox(
            "Direction",
            options=["Both", "Up only", "Down only"],
        )

    with col3:
        vol_window = st.selectbox(
            "Volatility window (days)",
            options=vol_windows,
            index=0,
        )

    # ---------- Filter events ----------
    df_events = df_simple.copy()
    df_events = df_events.dropna(subset=["simple_return"])

    if direction == "Up only":
        df_events = df_events[df_events["simple_return"] >= threshold_pct]
    elif direction == "Down only":
        df_events = df_events[df_events["simple_return"] <= -threshold_pct]
    else:
        df_events = df_events[df_events["simple_return"].abs() >= threshold_pct]

    df_events = df_events.sort_values("trade_date")

    # ---------- Display ----------
    st.markdown(
        f"""
        **Events where |return| ≥ {threshold_pct*100:.1f}%**
        ({direction.lower()})
        """
    )

    if df_events.empty:
        st.write("No events found.")
    else:
        display_cols = [
            "trade_date",
            "contract_code",
            "close",
            "simple_return",
            f"srv_{vol_window}",
        ]
        display_cols = [c for c in display_cols if c in df_events.columns]

        df_out = df_events[display_cols].copy()
        df_out["simple_return"] = df_out["simple_return"] * 100

        st.dataframe(
            df_out.rename(
                columns={
                    "simple_return": "Return (%)",
                    f"srv_{vol_window}": f"{vol_window}d Vol (ann.)",
                }
            ),
            use_container_width=True,
        )

        # ---------- Context plot ----------
        st.markdown("### Price context for selected event")

        selected_date = st.selectbox(
            "Select a date to inspect",
            df_out["trade_date"],
        )

        window_days = 20
        mask = (
            (df_simple["trade_date"] >= selected_date - pd.Timedelta(days=window_days))
            & (df_simple["trade_date"] <= selected_date + pd.Timedelta(days=window_days))
        )

        df_ctx = df_simple[mask]

        fig_ctx = go.Figure()
        fig_ctx.add_trace(
            go.Scatter(
                x=df_ctx["trade_date"],
                y=df_ctx["close"],
                mode="lines",
                name="Close",
            )
        )
        event_dt = selected_date.to_pydatetime()
        
        fig_ctx.add_vline(
            x=event_dt,
            line_dash="dash",
            line_color="red",
            line_width=2,
        )

        fig_ctx.add_annotation(
            x=event_dt,
            y=1,
            yref="paper",
            text="Event day",
            showarrow=False,
            font=dict(color="red"),
            xanchor="left",
        )

        fig_ctx.update_layout(
            title=f"{cont_name} Price Around {selected_date.date()}",
            xaxis_title="Date",
            yaxis_title="Price",
            height=450,
        )
        st.plotly_chart(fig_ctx, use_container_width=True)
