import sqlite3
from pathlib import Path

import pandas as pd

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "brent_futures"
DB_PATH = PROJECT_ROOT / "db" / "brent_futures.db"


def ensure_schema(conn: sqlite3.Connection):
    """Create the brent_prices table if it doesn't exist."""
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS brent_prices (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            contract_code TEXT NOT NULL,
            trade_date    DATE NOT NULL,
            open          REAL,
            high          REAL,
            low           REAL,
            close         REAL,
            volume        INTEGER,
            change_pct    REAL,
            UNIQUE (contract_code, trade_date)
        );
        """
    )
    conn.commit()


def ensure_contracts_table(conn: sqlite3.Connection):
    """Create the brent_contracts table if it doesn't exist."""
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS brent_contracts (
            contract_code   TEXT PRIMARY KEY,  -- e.g. 'G26'
            month_code      TEXT,              -- 'G'
            year            INTEGER,           -- 2026
            last_trade_date DATE               -- LTD (expiry)
        );
        """
    )
    conn.commit()


def add_contract_if_missing(conn: sqlite3.Connection, contract_code: str):
    """
    Ensure a row exists in brent_contracts for this contract_code.
    month_code = first char, year = 2000 + last two digits.
    """
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


def parse_contract_code(filename: str) -> str:
    """
    Extract the futures code from the filename.

    Handles:
      - 'Brent Futures G26.csv'
      - 'Brent Futures (G26).csv'
    by taking the last 'word' before .csv and stripping brackets.
    """
    stem = Path(filename).stem  # e.g. 'Brent Futures G26' or 'Brent Futures (G26)'
    parts = stem.split()
    last = parts[-1]  # 'G26' or '(G26)'
    code = last.strip("()")  # remove any parentheses
    return code


def parse_volume(vol_str):
    """
    Convert volume strings like '258.41K' or '1.2M' to integers.
    Returns None if not parseable.
    """
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
    """
    Convert '0.48%' to 0.48 (float).
    """
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


def load_csv_into_db(conn: sqlite3.Connection, csv_path: Path):
    contract_code = parse_contract_code(csv_path.name)
    print(f"Processing {csv_path.name} (contract {contract_code})")

    # make sure this contract exists in brent_contracts
    add_contract_if_missing(conn, contract_code)

    df = pd.read_csv(csv_path)

    # Map columns from CSV to our schema
    # Expected CSV columns:
    # Date,Price,Open,High,Low,Vol.,Change %
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

    conn.commit()
    print(f"Inserted/updated {len(df)} rows for {contract_code}")


def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    ensure_schema(conn)
    ensure_contracts_table(conn)

    csv_files = sorted(DATA_DIR.glob("Brent Futures*.csv"))
    if not csv_files:
        print(f"No CSV files found in {DATA_DIR}")
        return

    for csv_path in csv_files:
        load_csv_into_db(conn, csv_path)

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
