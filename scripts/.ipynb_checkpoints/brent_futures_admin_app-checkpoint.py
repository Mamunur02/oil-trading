import sqlite3
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import streamlit as st

DB_PATH = Path(__file__).resolve().parents[1] / "db" / "brent_futures.db"


def get_conn():
    return sqlite3.connect(DB_PATH)


def get_contract_codes():
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT DISTINCT contract_code FROM brent_prices ORDER BY contract_code",
        conn,
    )
    conn.close()
    return df["contract_code"].tolist()


def load_data(
    contract_codes=None,
    date_mode="All",
    last_n_days=30,
    start_date=None,
    end_date=None,
    vol_min=None,
    vol_max=None,
    order_by="trade_date",
    order_dir="DESC",
    limit=500,
):
    conn = get_conn()

    # Base query
    query = "SELECT * FROM brent_prices"
    where_clauses = []
    params = []

    # Contract filter
    if contract_codes and len(contract_codes) > 0:
        placeholders = ",".join(["?"] * len(contract_codes))
        where_clauses.append(f"contract_code IN ({placeholders})")
        params.extend(contract_codes)

    # Date filters
    if date_mode == "Last N days":
        cutoff = (date.today() - timedelta(days=last_n_days)).isoformat()
        where_clauses.append("trade_date >= ?")
        params.append(cutoff)
    elif date_mode == "Custom range" and start_date and end_date:
        where_clauses.append("trade_date BETWEEN ? AND ?")
        params.append(start_date.isoformat())
        params.append(end_date.isoformat())

    # Volume filters
    if vol_min is not None:
        where_clauses.append("volume >= ?")
        params.append(vol_min)

    if vol_max is not None:
        where_clauses.append("volume <= ?")
        params.append(vol_max)

    # Assemble WHERE
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    # Ordering – whitelist columns
    orderable_cols = {
        "trade_date": "trade_date",
        "contract_code": "contract_code",
        "close": "close",
        "volume": "volume",
        "id": "id",
    }
    order_col_sql = orderable_cols.get(order_by, "trade_date")
    order_dir_sql = "DESC" if order_dir.upper().startswith("DESC") else "ASC"
    query += f" ORDER BY {order_col_sql} {order_dir_sql}"

    # Limit
    query += " LIMIT ?"
    params.append(limit)

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def update_row(row_id, contract_code, trade_date, open_, high, low, close, volume, change_pct):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE brent_prices
        SET contract_code = ?,
            trade_date    = ?,
            open          = ?,
            high          = ?,
            low           = ?,
            close         = ?,
            volume        = ?,
            change_pct    = ?
        WHERE id = ?
        """,
        (
            contract_code,
            trade_date,
            open_,
            high,
            low,
            close,
            volume,
            change_pct,
            row_id,
        ),
    )
    conn.commit()
    conn.close()


def insert_row(contract_code, trade_date, open_, high, low, close, volume, change_pct):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO brent_prices (
            contract_code, trade_date, open, high, low, close, volume, change_pct
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            contract_code,
            trade_date,
            open_,
            high,
            low,
            close,
            volume,
            change_pct,
        ),
    )
    conn.commit()
    conn.close()


def delete_row(row_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM brent_prices WHERE id = ?", (row_id,))
    conn.commit()
    conn.close()


# --- contracts helpers ---


def load_contracts():
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM brent_contracts ORDER BY year, month_code",
        conn,
    )
    conn.close()
    return df


def update_contract_ltd(contract_code: str, ltd: date):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE brent_contracts
        SET last_trade_date = ?
        WHERE contract_code = ?
        """,
        (ltd.isoformat(), contract_code),
    )
    conn.commit()
    conn.close()

# --- CSV parsing helpers (same logic as build script) ---

def parse_contract_code_from_name(filename: str) -> str:
    stem = Path(filename).stem  # e.g. 'Brent Futures X26' or 'Brent Futures (X26)'
    parts = stem.split()
    last = parts[-1]  # 'X26' or '(X26)'
    return last.strip("()")


def parse_volume(vol_str):
    if pd.isna(vol_str):
        return None
    if isinstance(vol_str, (int, float)):
        return int(vol_str)

    s = str(vol_str).strip()
    if s in ("", "-", "—"):
        return None

    multiplier = 1
    if s.endswith("K"):
        multiplier = 1_000
        s = s[:-1]
    elif s.endswith("M"):
        multiplier = 1_000_000
        s = s[:-1]

    try:
        return int(float(s.replace(",", "")) * multiplier)
    except ValueError:
        return None


def parse_change_pct(change_str):
    if pd.isna(change_str):
        return None

    s = str(change_str).strip()
    if s in ("", "-", "—"):
        return None
    if s.endswith("%"):
        s = s[:-1]

    try:
        return float(s)
    except ValueError:
        return None


def add_contract_if_missing(conn: sqlite3.Connection, contract_code: str):
    month_code = contract_code[0]
    year_suffix = int(contract_code[1:])
    year = 2000 + year_suffix

    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR IGNORE INTO brent_contracts (contract_code, month_code, year)
        VALUES (?, ?, ?)
        """,
        (contract_code, month_code, year),
    )
    conn.commit()

def load_uploaded_csv_into_db(file, filename: str) -> int:
    """
    Load an uploaded Brent futures CSV into the database.
    Returns the number of rows inserted/updated.
    """
    contract_code = parse_contract_code_from_name(filename)
    conn = get_conn()
    add_contract_if_missing(conn, contract_code)

    df = pd.read_csv(file)

    # Expect columns: Date,Price,Open,High,Low,Vol.,Change %
    df["trade_date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df["volume"] = df["Vol."].apply(parse_volume)
    df["change_pct"] = df["Change %"].apply(parse_change_pct)

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Price": "close",
        }
    )

    cur = conn.cursor()
    count = 0
    for _, row in df.iterrows():
        cur.execute(
            """
            INSERT INTO brent_prices (
                contract_code,
                trade_date,
                open,
                high,
                low,
                close,
                volume,
                change_pct
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(contract_code, trade_date) DO UPDATE SET
                open       = excluded.open,
                high       = excluded.high,
                low        = excluded.low,
                close      = excluded.close,
                volume     = excluded.volume,
                change_pct = excluded.change_pct
            """,
            (
                contract_code,
                row["trade_date"].date().isoformat(),
                float(row["open"]) if not pd.isna(row["open"]) else None,
                float(row["high"]) if not pd.isna(row["high"]) else None,
                float(row["low"]) if not pd.isna(row["low"]) else None,
                float(row["close"]) if not pd.isna(row["close"]) else None,
                int(row["volume"]) if not pd.isna(row["volume"]) else None,
                float(row["change_pct"])
                if not pd.isna(row["change_pct"])
                else None,
            ),
        )
        count += 1

    conn.commit()
    conn.close()
    return count


# ----------------- Streamlit UI -----------------

st.set_page_config(page_title="Brent Futures Admin", layout="wide")
st.title("Brent Futures Database Admin")

# Sidebar: global filters
st.sidebar.header("Filters")

contract_codes_all = get_contract_codes()
selected_contracts = st.sidebar.multiselect(
    "Contract codes",
    options=contract_codes_all,
    default=contract_codes_all,  # default: show all
)

date_mode = st.sidebar.selectbox(
    "Date filter mode",
    ["All", "Last N days", "Custom range"],
    index=0,
)

last_n_days = None
start_date = None
end_date = None

if date_mode == "Last N days":
    last_n_days = st.sidebar.slider("Last N days", min_value=1, max_value=365, value=30)
elif date_mode == "Custom range":
    start_date, end_date = st.sidebar.date_input(
        "Date range (start, end)",
        value=(date.today() - timedelta(days=30), date.today()),
    )

# Volume filters
st.sidebar.markdown("**Volume filter**")
vol_min_input = st.sidebar.number_input("Min volume", min_value=0, value=0, step=1)
vol_max_input = st.sidebar.number_input("Max volume (0 = no max)", min_value=0, value=0, step=1)
vol_min = vol_min_input if vol_min_input > 0 else None
vol_max = vol_max_input if vol_max_input > 0 else None

# Ordering
st.sidebar.markdown("**Ordering**")
order_by_label = st.sidebar.selectbox(
    "Order by",
    ["trade_date", "contract_code", "close", "volume", "id"],
    index=0,
)
order_dir = st.sidebar.selectbox(
    "Direction",
    ["Descending", "Ascending"],
    index=0,
)

# Row limit
limit = st.sidebar.slider("Max rows to load", min_value=50, max_value=5000, value=500, step=50)

# Load filtered price data once, reuse in tabs
df = load_data(
    contract_codes=selected_contracts,
    date_mode=date_mode,
    last_n_days=last_n_days or 30,
    start_date=start_date,
    end_date=end_date,
    vol_min=vol_min,
    vol_max=vol_max,
    order_by=order_by_label,
    order_dir=order_dir,
    limit=limit,
)

tab_view, tab_edit, tab_insert, tab_contracts, tab_upload = st.tabs(
    ["View data", "Edit / delete row", "Insert new row", "Contracts (LTD)", "Upload CSV"]
)

# ---------- VIEW TAB ----------
with tab_view:
    st.subheader("Current data")

    # Small summary of active filters
    st.caption(
        f"Contracts: {', '.join(selected_contracts) if selected_contracts else 'None'} | "
        f"Date mode: {date_mode} | "
        f"Order: {order_by_label} ({order_dir}) | "
        f"Rows loaded: {len(df)}"
    )

    if df.empty:
        st.info("No data matches the current filters.")
    else:
        st.dataframe(df, use_container_width=True)

# ---------- EDIT / DELETE TAB ----------
with tab_edit:
    st.subheader("Edit / delete a row")

    if df.empty:
        st.info("No rows available to edit under the current filters.")
    else:
        row_id = st.selectbox("Select row ID to edit/delete", df["id"])

        row = df[df["id"] == row_id].iloc[0]

        with st.form("edit_row_form"):
            edit_contract = st.text_input("Contract code", value=row["contract_code"])

            trade_date_val = pd.to_datetime(row["trade_date"]).date()
            edit_trade_date = st.date_input("Trade date", value=trade_date_val)

            edit_open = st.number_input(
                "Open",
                value=float(row["open"]) if pd.notna(row["open"]) else 0.0,
                key="edit_open",
            )
            edit_high = st.number_input(
                "High",
                value=float(row["high"]) if pd.notna(row["high"]) else 0.0,
                key="edit_high",
            )
            edit_low = st.number_input(
                "Low",
                value=float(row["low"]) if pd.notna(row["low"]) else 0.0,
                key="edit_low",
            )
            edit_close = st.number_input(
                "Close",
                value=float(row["close"]) if pd.notna(row["close"]) else 0.0,
                key="edit_close",
            )
            edit_volume = st.number_input(
                "Volume",
                value=int(row["volume"]) if pd.notna(row["volume"]) else 0,
                step=1,
                min_value=0,
                key="edit_volume",
            )
            edit_change_pct = st.number_input(
                "Change % (e.g. 0.48 for 0.48%)",
                value=float(row["change_pct"]) if pd.notna(row["change_pct"]) else 0.0,
                key="edit_change_pct",
            )

            col1, col2 = st.columns(2)
            with col1:
                submitted = st.form_submit_button("Save changes")
            with col2:
                delete_clicked = st.form_submit_button("Delete row", type="secondary")

        if submitted:
            update_row(
                row_id=row_id,
                contract_code=edit_contract,
                trade_date=edit_trade_date.isoformat(),
                open_=edit_open,
                high=edit_high,
                low=edit_low,
                close=edit_close,
                volume=int(edit_volume),
                change_pct=edit_change_pct,
            )
            st.success(f"Row {row_id} updated.")
            st.rerun()

        if delete_clicked:
            delete_row(row_id)
            st.warning(f"Row {row_id} deleted.")
            st.rerun()

# ---------- INSERT TAB ----------
with tab_insert:
    st.subheader("Insert new row")

    with st.form("insert_row_form"):
        new_contract = st.text_input("Contract code (e.g. G26, H26)")
        new_trade_date = st.date_input("Trade date", value=date.today())
        new_open = st.number_input("Open", value=0.0, key="new_open")
        new_high = st.number_input("High", value=0.0, key="new_high")
        new_low = st.number_input("Low", value=0.0, key="new_low")
        new_close = st.number_input("Close", value=0.0, key="new_close")
        new_volume = st.number_input(
            "Volume", value=0, min_value=0, step=1, key="new_volume"
        )
        new_change_pct = st.number_input(
            "Change % (e.g. 0.48 for 0.48%)",
            value=0.0,
            key="new_change_pct",
        )

        insert_submitted = st.form_submit_button("Insert row")

    if insert_submitted:
        if not new_contract:
            st.error("Contract code is required.")
        else:
            insert_row(
                contract_code=new_contract,
                trade_date=new_trade_date.isoformat(),
                open_=new_open,
                high=new_high,
                low=new_low,
                close=new_close,
                volume=int(new_volume),
                change_pct=new_change_pct,
            )
            st.success("New row inserted.")
            st.rerun()

# ---------- CONTRACTS (LTD) TAB ----------
with tab_contracts:
    st.subheader("Contract metadata (Last Trade Date / Expiry)")

    df_contracts = load_contracts()

    if df_contracts.empty:
        st.info("No contracts found. Run the build script to load futures data first.")
    else:
        st.dataframe(df_contracts, use_container_width=True)

        selected_code = st.selectbox(
            "Select contract to edit LTD",
            df_contracts["contract_code"],
        )

        row = df_contracts[df_contracts["contract_code"] == selected_code].iloc[0]

        with st.form("edit_ltd_form"):
            if pd.isna(row["last_trade_date"]):
                default_ltd = date.today()
            else:
                default_ltd = pd.to_datetime(row["last_trade_date"]).date()

            new_ltd = st.date_input(
                "Last Trade Date (LTD)",
                value=default_ltd,
            )

            submitted_ltd = st.form_submit_button("Save LTD")

        if submitted_ltd:
            update_contract_ltd(selected_code, new_ltd)
            st.success(f"LTD updated for {selected_code}")
            st.rerun()

# ---------- UPLOAD CSV TAB ----------
with tab_upload:
    st.subheader("Bulk upload Brent futures CSV")

    st.markdown(
        """
        Upload a CSV named like **Brent Futures X26.csv** or **Brent Futures (X26).csv**.
        The contract code (e.g. X26) is inferred from the file name.
        """
    )

    uploaded_file = st.file_uploader("Choose a Brent futures CSV", type=["csv"])

    if uploaded_file is not None:
        st.write(f"Detected filename: `{uploaded_file.name}`")

        try:
            contract_code = parse_contract_code_from_name(uploaded_file.name)
            st.write(f"Parsed contract code: **{contract_code}**")
        except Exception as e:
            st.error(f"Could not parse contract code from filename: {e}")
            contract_code = None

        if contract_code:
            if st.button("Process CSV and update database"):
                try:
                    rows = load_uploaded_csv_into_db(uploaded_file, uploaded_file.name)
                    st.success(f"Inserted/updated {rows} rows for contract {contract_code}.")
                    st.info("Go to the 'View data' tab and refresh filters to see the new data.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error while processing CSV: {e}")
