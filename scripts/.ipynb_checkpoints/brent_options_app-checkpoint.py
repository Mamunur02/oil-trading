# scripts/brent_options_app.py

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from brent_options_utils import (
    parse_brent_options_pdf,
    normalise_options_df_for_db,
    upsert_options_df_into_db,
    get_conn,
    ensure_options_schema,
    load_options,
)

# ---------- Helper: extract report date from filename ----------

def extract_report_date_from_filename(filename: str) -> date:
    """
    Expect filenames of the form: B_YYYY_MM_DD.pdf
    e.g. B_2025_09_25.pdf -> date(2025, 9, 25)
    """
    stem = Path(filename).stem  # "B_2025_09_25"
    parts = stem.split("_")

    if len(parts) != 4 or parts[0] != "B":
        raise ValueError(
            f"Filename '{filename}' not in expected format B_YYYY_MM_DD.pdf"
        )

    year, month, day = map(int, parts[1:4])
    return date(year, month, day)


# ---------- Streamlit config ----------

st.set_page_config(page_title="Brent Options Admin", layout="wide")
st.title("Brent Options Database Admin")

# ---------- DB connection & schema ----------

conn = get_conn()
ensure_options_schema(conn)

# ---------- Sidebar filters for VIEW ----------

st.sidebar.header("View filters")

# Load once to get available futures codes & date range
df_all = load_options(conn, limit=100000)

if not df_all.empty:
    all_futures_codes = sorted(
        df_all["futures_code"].dropna().unique().tolist()
    )
else:
    all_futures_codes = []

selected_codes = st.sidebar.multiselect(
    "Futures codes",
    options=all_futures_codes,
    default=all_futures_codes,
)

option_type_filter = st.sidebar.selectbox(
    "Option type",
    ["All", "Call", "Put"],
    index=0,
)

date_mode = st.sidebar.selectbox(
    "Report date filter",
    ["All", "Last N days", "Custom range"],
    index=0,
)

start_date = None
end_date = None

if date_mode == "Last N days":
    n_days = st.sidebar.slider("Last N days", 1, 365, 30)
    end_date = date.today()
    start_date = end_date - timedelta(days=n_days)

elif date_mode == "Custom range":
    if not df_all.empty:
        min_date = pd.to_datetime(df_all["report_date"]).min().date()
        max_date = pd.to_datetime(df_all["report_date"]).max().date()
    else:
        min_date = date.today() - timedelta(days=30)
        max_date = date.today()
    start_date, end_date = st.sidebar.date_input(
        "Report date range (start, end)",
        value=(min_date, max_date),
    )

limit = st.sidebar.slider(
    "Max rows to load",
    min_value=100,
    max_value=50000,
    value=5000,
    step=100,
)

# ---------- Tabs ----------

tab_view, tab_upload = st.tabs(["View options data", "Upload options PDF(s)"])

# ---------- VIEW TAB ----------

with tab_view:
    st.subheader("Options database")

    opt_type = option_type_filter if option_type_filter != "All" else None

    df_view = load_options(
        conn,
        futures_codes=selected_codes,
        option_type=opt_type,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )

    st.caption(
        f"Futures codes: {', '.join(selected_codes) if selected_codes else 'None'} | "
        f"Option type: {option_type_filter} | "
        f"Date mode: {date_mode} | "
        f"Rows loaded: {len(df_view)}"
    )

    if df_view.empty:
        st.info("No options found for the current filters.")
    else:
        st.dataframe(df_view, use_container_width=True)


# ---------- UPLOAD TAB ----------

with tab_upload:
    st.subheader("Upload Brent options PDFs (batch supported)")

    st.markdown(
        """
        - Filenames must follow the pattern **B_YYYY_MM_DD.pdf**  
        - You can drop **one or many PDFs** at once.  
        - Each file will be parsed and upserted into the database.
        """
    )

    uploaded_files = st.file_uploader(
        "Choose one or more Brent options PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("Process all uploaded PDFs"):
            total_rows = 0

            for uploaded_file in uploaded_files:
                fname = uploaded_file.name
                st.write(f"Processing: `{fname}`")

                try:
                    # 1) Get the report date from the filename
                    report_date = extract_report_date_from_filename(fname)

                    # 2) Parse PDF -> raw options df
                    #    parse_brent_options_pdf must be able to handle a file-like object
                    df_raw = parse_brent_options_pdf(uploaded_file)

                    # 3) Normalise to DB schema:
                    #    - drop commodity if you do that in utils
                    #    - contract_month: Feb26 -> G26
                    #    - pc: C/P -> Call/Put
                    #    - add report_date from filename
                    df_db = normalise_options_df_for_db(df_raw, report_date)

                    # 4) Upsert into DB
                    rows = upsert_options_df_into_db(conn, df_db)
                    total_rows += rows

                    st.success(f"{fname}: inserted/updated {rows} rows.")

                except Exception as e:
                    st.error(f"{fname}: error while processing – {e}")

            # Commit all changes at the end of the batch
            conn.commit()

            st.info(
                f"Finished. Total rows inserted/updated across all PDFs: {total_rows}"
            )
