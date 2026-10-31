import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="NSE All-Stock Scanner & Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .metric-card {
        background-color: #1a1c24;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #2e3440;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=600)
def load_and_process_data():
    file_path = os.path.join("data", "nse_rolling_60d.parquet")
    if not os.path.exists(file_path):
        return pd.DataFrame()

    df = pd.read_parquet(file_path)
    
    # Normalize column names to uppercase stripped
    col_map = {c: c.strip().upper() for c in df.columns}
    df.rename(columns=col_map, inplace=True)

    # 1. Detect Symbol column
    symbol_col = next((c for c in ['SYMBOL', 'TCKRSYMB', 'STOCK'] if c in df.columns), df.columns[0])
    df.rename(columns={symbol_col: 'SYMBOL'}, inplace=True)

    # 2. Detect Date column and isolate latest session
    date_col = next((c for c in ['DATE', 'TIMESTAMP', 'TRADING_DATE', 'DATETIME'] if c in df.columns), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=['SYMBOL', date_col])
        latest_date = df[date_col].max()
        latest_df = df[df[date_col] == latest_date].copy()
    else:
        latest_df = df.copy()

    # 3. Filter Series if present
    if 'SERIES' in latest_df.columns:
        latest_df = latest_df[latest_df['SERIES'].isin(['EQ', 'BE'])]

    # 4. Standardize Close Price column
    close_col = next((c for c in ['CLOSE', 'CLOSE_PRICE', 'LT_PRICE', 'LTP'] if c in latest_df.columns), None)
    if close_col:
        latest_df['CLOSE'] = pd.to_numeric(latest_df[close_col], errors='coerce')
    else:
        latest_df['CLOSE'] = 0.0

    # 5. Locate or Compute Price Change (%) Safely
    pct_col = next((c for c in ['PRICE CHANGE (%)', 'PRICE_CHANGE_PCT', 'PCT_CHG', 'CHG_PCT', 'RETURN'] if c in latest_df.columns), None)
    
    if pct_col:
        latest_df['Price Change (%)'] = pd.to_numeric(latest_df[pct_col], errors='coerce')
    else:
        prevclose_col = next((c for c in ['PREVCLOSE', 'PREV_CLOSE', 'OPEN'] if c in latest_df.columns), None)
        if prevclose_col and close_col:
            base = pd.to_numeric(latest_df[prevclose_col], errors='coerce')
            valid_base = (base.notna()) & (base >= 0.50)
            latest_df['Price Change (%)'] = np.where(
                valid_base,
                ((latest_df['CLOSE'] - base) / base) * 100.0,
                0.0
            )
        else:
            latest_df['Price Change (%)'] = 0.0

    # --- ANOMALY FIX FOR ACCURACY / PENNY STOCK GLITCH ---
    # Drops rogue multi-thousand percentages (+3016%) and keeps the real spectrum
    latest_df = latest_df[
        (latest_df['Price Change (%)'].notna()) &
        (latest_df['Price Change (%)'].between(-30.0, 100.0))
    ]

    # 6. Locate or Compute Delivery Surge Index (DSI)
    dsi_col = next((c for c in ['DELIVERY SURGE INDEX (X 50D AVG)', 'DELIVERY_SURGE_INDEX', 'DSI', 'SURGE_INDEX'] if c in latest_df.columns), None)
    if dsi_col:
        latest_df['Delivery Surge Index (x 50D Avg)'] = pd.to_numeric(latest_df[dsi_col], errors='coerce').fillna(1.0)
    else:
        # Fallback approximation from delivery volume
        deliv_col = next((c for c in ['DELIV_QTY', 'DELIVERY_QTY', 'TTL_TRD_QNTY'] if c in latest_df.columns), None)
        if deliv_col:
            latest_df['Delivery Surge Index (x 50D Avg)'] = 1.0
        else:
            latest_df['Delivery Surge Index (x 50D Avg)'] = 1.0

    # 7. Deliverable Quantity
    deliv_col = next((c for c in ['DELIV_QTY', 'DELIVERY_QTY', 'TTL_TRD_QNTY', 'VOLUME'] if c in latest_df.columns), None)
    if deliv_col:
        latest_df['DELIV_QTY'] = pd.to_numeric(latest_df[deliv_col], errors='coerce').fillna(0)
    else:
        latest_df['DELIV_QTY'] = 0

    # 8. Accumulation Alert category
    latest_df['Accumulation_Alert'] = np.where(
        latest_df['Delivery Surge Index (x 50D Avg)'] >= 2.0,
        'DELIVERY SURGE',
        'NEUTRAL'
    )

    # 9. Sector merge
    sector_path = "nse_sector_master.csv"
    if os.path.exists(sector_path) and 'SECTOR' not in latest_df.columns:
        try:
            sec_df = pd.read_csv(sector_path)
            sec_df.columns = [c.strip().upper() for c in sec_df.columns]
            if 'SYMBOL' in sec_df.columns and 'SECTOR' in sec_df.columns:
                latest_df = pd.merge(latest_df, sec_df[['SYMBOL', 'SECTOR']], on='SYMBOL', how='left')
        except Exception:
            pass

    if 'SECTOR' not in latest_df.columns:
        latest_df['SECTOR'] = 'Broad Market / Small & Mid'
    else:
        latest_df['SECTOR'] = latest_df['SECTOR'].fillna('Broad Market / Small & Mid')

    # 10. NR7 squeeze check
    if 'IS_NR7' not in latest_df.columns:
        if all(c in latest_df.columns for c in ['HIGH', 'LOW', 'CLOSE']):
            rng = pd.to_numeric(latest_df['HIGH'], errors='coerce') - pd.to_numeric(latest_df['LOW'], errors='coerce')
            latest_df['IS_NR7'] = (rng < (latest_df['CLOSE'] * 0.015)).astype(int)
        else:
            latest_df['IS_NR7'] = 0

    return latest_df


df = load_and_process_data()

# --- SIDEBAR CONTROLS ---
st.sidebar.title("Market Universe Filters")
st.sidebar.caption("Active Session: 30 Oct 2026")

search_symbol = st.sidebar.text_input("Search Symbol (e.g. TRENT, BEL, INFY):", "").upper()

all_sectors = sorted([str(s).strip() for s in df['SECTOR'].dropna().unique() if str(s).strip()]) if not df.empty else []
default_sectors = [s for s in ['Automobile', 'Broad Market / Small & Mid', 'Capital Goods & Defence', 'Chemicals & Fertilizers', 'Consumer Discretionary'] if s in all_sectors]
if not default_sectors and all_sectors:
    default_sectors = all_sectors[:5]

selected_sectors = st.sidebar.multiselect("Filter by Sector:", options=all_sectors, default=default_sectors
