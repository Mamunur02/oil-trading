import io
import re
import sqlite3
from datetime import date
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import pdfplumber

# ------------ Paths ------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "db" / "brent_options.db"
DATA_DIR = PROJECT_ROOT / "data" / "brent_options"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ------------ Raw PDF parse layout ------------

RAW_COLS = [
    "commodity",
    "contract_month",   # e.g. 'Feb26'
    "strike",
    "pc",               # 'C' / 'P'
    "delta",
    "open",
    "high",
    "low",
    "close",
    "settle",
    "change",
    "total_volume",
    "oi",               # open interest
    "oi_change",
    "exercise_volume",
    "block_volume",
    "eoo_volume",
    "spread_volume",
]

NUMERIC_COLS = [
    "strike",
    "delta",
    "open",
    "high",
    "low",
    "close",
    "settle",
    "change",
    "total_volume",
    "oi",
    "oi_change",
    "exercise_volume",
    "block_volume",
    "eoo_volume",
    "spread_volume",
]

# ------------ DB schema / columns ------------

DB_TABLE = "brent_options"

# Exactly 18 columns (NOT including id)
DB_COLS = [
    "report_date",      # from filename B_YYYY_MM_DD
    "futures_code",     # e.g. G26
    "strike",
    "option_type",      # 'Call' / 'Put'
    "delta",
    "open",
    "high",
    "low",
    "close",
    "settle",
    "change",
    "total_volume",
    "oi",
    "oi_change",
    "exercise_volume",
    "block_volume",
    "eoo_volume",
    "spread_volume",
]


# ------------ Helpers ------------

def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def clean_num(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s in {"-", "—"}:
        return None
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def contract_month_to_futures_code(cm: str) -> Optional[str]:
    """
    Convert 'Feb26' -> 'G26', 'Nov25' -> 'X25', etc.
    """
    if not isinstance(cm, str) or len(cm) < 4:
        return None

    cm = cm.strip()
    # First three letters for month, last two chars for year
    month_part = cm[:3].upper()
    year_suffix = cm[-2:]

    month_map = {
        "JAN": "F",
        "FEB": "G",
        "MAR": "H",
        "APR": "J",
        "MAY": "K",
        "JUN": "M",
        "JUL": "N",
        "AUG": "Q",
        "SEP": "U",
        "OCT": "V",
        "NOV": "X",
        "DEC": "Z",
    }

    code = month_map.get(month_part)
    if code is None:
        return None
    return f"{code}{year_suffix}"


def parse_report_date_from_filename(filename: str) -> date:
    """
    From 'B_2025_12_05.pdf' (or .csv) extract 2025-12-05.
    """
    stem = Path(filename).stem  # 'B_2025_12_05'
    m = re.search(r"B_(\d{4})_(\d{2})_(\d{2})", stem)
    if not m:
        raise ValueError(f"Could not parse report date from filename: {filename}")
    year, month, day = map(int, m.groups())
    return date(year, month, day)


# ------------ PDF -> raw DataFrame ------------

def parse_brent_options_pdf(file_or_path: Union[str, Path, io.BytesIO]) -> pd.DataFrame:
    """
    Parse an ICE Brent options PDF into a raw DataFrame with blanks kept as NaN
    and NO token-based shifting.

    Expects rows to be aligned like:
      commodity | contract_month | strike | pc | delta | open | high | low |
      close | settle | change | total_volume | oi | oi_change |
      exercise_volume | block_volume | eoo_volume | spread_volume
    """
    records = []

    with pdfplumber.open(file_or_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables or []:
                for raw_row in table or []:
                    if not raw_row:
                        continue

                    # Normalise row length to exactly len(RAW_COLS)
                    row = list(raw_row)
                    if len(row) < len(RAW_COLS):
                        row = row + [None] * (len(RAW_COLS) - len(row))
                    elif len(row) > len(RAW_COLS):
                        row = row[: len(RAW_COLS)]

                    # Strip strings but KEEP empties so a missing cell stays missing
                    row = [
                        (cell.strip() if isinstance(cell, str) else cell)
                        for cell in row
                    ]

                    commodity = (row[0] or "")
                    pc = (row[3] or "").upper()

                    # Filter to only Brent options rows
                    if commodity != "B":
                        continue
                    if pc not in {"C", "P"}:
                        continue

                    records.append(dict(zip(RAW_COLS, row)))

    df = pd.DataFrame.from_records(records)

    # Clean numeric columns; blanks become NaN
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].apply(clean_num)

    return df


# ------------ Raw DF -> DB-ready DF ------------

def normalise_options_df_for_db(df_raw: pd.DataFrame, report_date: date) -> pd.DataFrame:
    """
    Takes the raw parsed DF and returns a DataFrame with columns = DB_COLS.
    - Drops 'commodity' and 'contract_month' if not needed.
    - Maps contract_month -> futures_code.
    - Maps 'C'/'P' -> 'Call'/'Put'.
    - Adds report_date from filename.
    """
    df = df_raw.copy()

    # futures_code from contract_month
    df["futures_code"] = df["contract_month"].apply(contract_month_to_futures_code)

    # Option type from pc
    df["option_type"] = df["pc"].map({"C": "Call", "P": "Put"}).fillna("")

    # Report date from filename
    df["report_date"] = report_date.isoformat()

    # Ensure numeric cols still numeric
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].apply(clean_num)

    # Select only DB columns, in the exact order
    df_db = df.reindex(columns=DB_COLS)

    return df_db


# ------------ DB schema / upsert ------------

def ensure_options_schema(conn: sqlite3.Connection):
    """
    Create the brent_options table if it doesn't exist.
    Unique key on (report_date, futures_code, option_type, strike).
    """
    cur = conn.cursor()
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {DB_TABLE} (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            report_date     DATE NOT NULL,
            futures_code    TEXT NOT NULL,
            strike          REAL NOT NULL,
            option_type     TEXT NOT NULL,  -- 'Call' or 'Put'
            delta           REAL,
            open            REAL,
            high            REAL,
            low             REAL,
            close           REAL,
            settle          REAL,
            change          REAL,
            total_volume    REAL,
            oi              REAL,
            oi_change       REAL,
            exercise_volume REAL,
            block_volume    REAL,
            eoo_volume      REAL,
            spread_volume   REAL,
            UNIQUE (report_date, futures_code, option_type, strike)
        );
        """
    )
    conn.commit()


def upsert_options_df_into_db(conn: sqlite3.Connection, df_db: pd.DataFrame) -> int:
    """
    Upsert rows into brent_options.
    Returns number of rows processed.
    """
    ensure_options_schema(conn)

    cols = DB_COLS
    placeholders = ",".join(["?"] * len(cols))  # 18 placeholders
    col_list_sql = ", ".join(cols)

    update_assignments = ", ".join(
        f"{col} = excluded.{col}"
        for col in cols
        if col not in ("report_date", "futures_code", "option_type", "strike")
    )

    sql = f"""
        INSERT INTO {DB_TABLE} ({col_list_sql})
        VALUES ({placeholders})
        ON CONFLICT(report_date, futures_code, option_type, strike) DO UPDATE SET
            {update_assignments};
    """

    cur = conn.cursor()
    rows = 0
    for _, r in df_db.iterrows():
        values = [r[col] for col in cols]  # exactly 18 values
        cur.execute(sql, values)
        rows += 1

    conn.commit()
    return rows


# ------------ Convenience: PDF -> DB (+ optional CSV) ------------

def process_brent_options_pdf_file(file_or_path: Union[str, Path, io.BytesIO], filename: str) -> int:
    """
    High-level helper:
      - parse report_date from filename
      - parse PDF into raw DF
      - normalise DF for DB
      - upsert into DB
      - save a CSV copy in data/brent_options/B_YYYY_MM_DD.csv

    Returns number of rows upserted.
    """
    report_date = parse_report_date_from_filename(filename)

    df_raw = parse_brent_options_pdf(file_or_path)
    df_db = normalise_options_df_for_db(df_raw, report_date)

    # Save CSV (optional but handy)
    csv_name = f"{Path(filename).stem}.csv"
    csv_path = DATA_DIR / csv_name
    df_db.to_csv(csv_path, index=False)

    with get_conn() as conn:
        rows = upsert_options_df_into_db(conn, df_db)

    return rows


# ------------ Query helper for the UI ------------

def load_options(
    conn: sqlite3.Connection,
    futures_codes=None,
    option_type: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    limit: int = 5000,
) -> pd.DataFrame:
    """
    Load options from DB with simple filters.
    """
    ensure_options_schema(conn)
    query = f"SELECT * FROM {DB_TABLE}"
    where = []
    params = []

    if futures_codes:
        placeholders = ",".join(["?"] * len(futures_codes))
        where.append(f"futures_code IN ({placeholders})")
        params.extend(futures_codes)

    if option_type in ("Call", "Put"):
        where.append("option_type = ?")
        params.append(option_type)

    if start_date:
        where.append("report_date >= ?")
        params.append(start_date.isoformat())

    if end_date:
        where.append("report_date <= ?")
        params.append(end_date.isoformat())

    if where:
        query += " WHERE " + " AND ".join(where)

    query += " ORDER BY report_date DESC, futures_code, option_type, strike"
    query += " LIMIT ?"
    params.append(limit)

    df = pd.read_sql_query(query, conn, params=params)
    return df
