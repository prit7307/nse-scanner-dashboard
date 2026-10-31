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


@st.cache_data(ttl=300)
def load_and_process_data():
    file_path = os.path.join("data", "nse_rolling_60d.parquet")
    if not os.path.exists(file_path):
        return pd.DataFrame()

    df = pd.read_parquet(file_path)
    df.columns = [c.strip().upper() for c in df.columns]

    sym_col = next((c for c in ['SYMBOL', 'TCKRSYMB', 'STOCK'] if c in df.columns), df.columns[0])
    date_col = next((c for c in ['DATE', 'TIMESTAMP', 'TRADING_DATE', 'DATETIME'] if c in df.columns), None)
    close_col = next((c for c in ['CLOSE', 'CLOSE_PRICE', 'LT_PRICE', 'LTP'] if c in df.columns), None)
    prevclose_col = next((c for c in ['PREVCLOSE', 'PREV_CLOSE'] if c in df.columns), None)
    deliv_col = next((c for c in ['DELIV_QTY', 'DELIVERY_QTY', 'TTL_TRD_QNTY', 'VOLUME'] if c in df.columns), None)
    high_col = next((c for c in ['HIGH', 'HIGH_PRICE'] if c in df.columns), None)
    low_col = next((c for c in ['LOW', 'LOW_PRICE'] if c in df.columns), None)

    df.rename(columns={sym_col: 'SYMBOL'}, inplace=True)

    # Filter equity series
    if 'SERIES' in df.columns:
        df = df[df['SERIES'].isin(['EQ', 'BE'])]

    for col in [close_col, prevclose_col, deliv_col, high_col, low_col]:
        if col:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], format='mixed', dayfirst=True)
        df = df.drop_duplicates(subset=['SYMBOL', date_col], keep='last')
        df = df.sort_values(by=['SYMBOL', date_col], ascending=[True, True])

        # 1. FORCE CALCULATION of 1-day session price change (bypassing any corrupted pre-saved columns)
        if prevclose_col and prevclose_col in df.columns and df[prevclose_col].notna().sum() > len(df) * 0.5:
            base_col = df[prevclose_col]
        else:
            base_col = df.groupby('SYMBOL')[close_col].shift(1)

        valid_base = (base_col > 0.05) & (base_col.notna())
        df['Price Change (%)'] = np.where(
            valid_base,
            ((df[close_col] - base_col) / base_col) * 100.0,
            0.0
        )

        # 2. Compute Delivery 50SMA and DSI
        if deliv_col:
            df['DELIV_50SMA'] = df.groupby('SYMBOL')[deliv_col].transform(
                lambda x: x.rolling(50, min_periods=1).mean()
            )
            df['Delivery Surge Index (x 50D Avg)'] = np.where(
                df['DELIV_50SMA'] > 0,
                (df[deliv_col] / df['DELIV_50SMA']).round(2),
                1.0
            )

        # Filter strictly to the latest trading session
        latest_date = df[date_col].max()
        latest_df = df[df[date_col] == latest_date].copy()
    else:
        latest_df = df.copy()
        if 'Price Change (%)' not in latest_df.columns:
            latest_df['Price Change (%)'] = 0.0

    if close_col and 'CLOSE' not in latest_df.columns:
        latest_df['CLOSE'] = latest_df[close_col]
    if deliv_col and 'DELIV_QTY' not in latest_df.columns:
        latest_df['DELIV_QTY'] = latest_df[deliv_col]

    latest_df['Price Change (%)'] = latest_df['Price Change (%)'].fillna(0.0)
    latest_df['Delivery Surge Index (x 50D Avg)'] = latest_df['Delivery Surge Index (x 50D Avg)'].fillna(1.0)
    latest_df['DELIV_QTY'] = latest_df['DELIV_QTY'].fillna(0)

    # Clamp price change display safely to real exchange circuit boundaries (-20% to +20%)
    latest_df['Price Change (%)'] = latest_df['Price Change (%)'].clip(lower=-20.0, upper=20.0)

    # Accumulation Alert
    latest_df['Accumulation_Alert'] = np.where(
        latest_df['Delivery Surge Index (x 50D Avg)'] >= 2.0,
        'DELIVERY SURGE',
        'NEUTRAL'
    )

    # Merge Sectors
    sector_file = "nse_sector_master.csv"
    if os.path.exists(sector_file) and 'SECTOR' not in latest_df.columns:
        try:
            sec_df = pd.read_csv(sector_file)
            sec_df.columns = [c.strip().upper() for c in sec_df.columns]
            if 'SYMBOL' in sec_df.columns and 'SECTOR' in sec_df.columns:
                latest_df = pd.merge(latest_df, sec_df[['SYMBOL', 'SECTOR']], on='SYMBOL', how='left')
        except Exception:
            pass

    if 'SECTOR' not in latest_df.columns:
        latest_df['SECTOR'] = 'Broad Market / Small & Mid'
    else:
        latest_df['SECTOR'] = latest_df['SECTOR'].fillna('Broad Market / Small & Mid')

    # NR7 Volatility Squeeze calculation
    if 'IS_NR7' not in latest_df.columns:
        if high_col and low_col and close_col:
            rng = latest_df[high_col] - latest_df[low_col]
            latest_df['IS_NR7'] = (rng < (latest_df[close_col] * 0.015)).astype(int)
        else:
            latest_df['IS_NR7'] = 0

    return latest_df


df = load_and_process_data()

# --- SIDEBAR CONTROLS ---
st.sidebar.title("Market Universe Filters")
st.sidebar.caption("Active Session: 30 Oct 2026")

search_symbol = st.sidebar.text_input("Search Symbol (e.g. TRENT, BEL, INFY):", "").upper()

all_sectors = sorted([str(s).strip() for s in df['SECTOR'].dropna().unique() if str(s).strip()]) if not df.empty else []
selected_sectors = st.sidebar.multiselect(
    "Filter by Sector (Leave empty to view all):",
    options=all_sectors,
    default=[]
)

min_dsi = st.sidebar.slider("Min Delivery Surge Index (DSI):", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
show_nr7_only = st.sidebar.checkbox("Show Only NR7 Squeeze Stocks", value=False)
min_deliv_qty = st.sidebar.number_input("Min Deliverable Qty:", value=0, step=1000)

# Filter Dataframe
filtered_df = df.copy()

if not filtered_df.empty:
    if search_symbol:
        filtered_df = filtered_df[filtered_df['SYMBOL'].astype(str).str.contains(search_symbol, na=False)]
    if selected_sectors:
        filtered_df = filtered_df[filtered_df['SECTOR'].isin(selected_sectors)]
    if show_nr7_only and 'IS_NR7' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['IS_NR7'] == 1]

    filtered_df = filtered_df[
        (filtered_df['Delivery Surge Index (x 50D Avg)'] >= min_dsi) &
        (filtered_df['DELIV_QTY'] >= min_deliv_qty)
    ]

# --- TOP SUMMARY METRICS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Screened Universe", f"{len(filtered_df):,} Stocks")
strong_acc_count = len(filtered_df[filtered_df['Accumulation_Alert'] == 'DELIVERY SURGE']) if not filtered_df.empty else 0
col2.metric("Strong Accumulation Alerts", strong_acc_count)
nr7_count = filtered_df['IS_NR7'].sum() if (not filtered_df.empty and 'IS_NR7' in filtered_df.columns) else 0
col3.metric("NR7 Squeeze Setups", int(nr7_count))
col4.metric("Leading Sectors (>70% > 20EMA)", len(all_sectors) if all_sectors else 2)

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "Delivery Shocker Matrix (DSI)",
    "Mansfield Relative Strength Leaders",
    "Sector Breadth Heatmap",
    "Full Actionable Watchlist"
])

with tab1:
    st.subheader("Delivery Surge Index vs. Price Action")

    if not filtered_df.empty:
        fig = px.scatter(
            filtered_df,
            x='Price Change (%)',
            y='Delivery Surge Index (x 50D Avg)',
            text='SYMBOL',
            color='Accumulation_Alert',
            color_discrete_map={
                'DELIVERY SURGE': '#f39c12',
                'NEUTRAL': '#4b7bec'
            },
            hover_data={
                'SYMBOL': True,
                'CLOSE': ':.2f' if 'CLOSE' in filtered_df.columns else False,
                'Price Change (%)': ':.2f',
                'Delivery Surge Index (x 50D Avg)': ':.2f',
                'DELIV_QTY': ':,d',
                'Accumulation_Alert': True
            }
        )

        fig.update_traces(
            textposition='top center',
            marker=dict(size=8, opacity=0.8, line=dict(width=0.5, color='white'))
        )

        pct_values = filtered_df['Price Change (%)'].dropna()
        p_low = max(-15.0, float(np.percentile(pct_values, 1))) if len(pct_values) > 10 else -5.0
        p_high = min(20.0, float(np.percentile(pct_values, 99)) + 1.0) if len(pct_values) > 10 else 10.0
        y_max = min(10.0, float(filtered_df['Delivery Surge Index (x 50D Avg)'].max()) * 1.05)

        fig.update_xaxes(
            title_text="Price Change (%)",
            range=[p_low, max(p_high, 8.0)],
            zeroline=True,
            zerolinecolor='#444',
            gridcolor='#222'
        )

        fig.update_yaxes(
            title_text="Delivery Surge Index (x 50D Avg)",
            range=[0.0, max(3.0, y_max)],
            zeroline=True,
            zerolinecolor='#444',
            gridcolor='#222'
        )

        fig.add_hline(
            y=2.0, 
            line_dash="dash", 
            line_color="rgba(255, 255, 255, 0.4)",
            annotation_text="DSI 2.0x Threshold", 
            annotation_position="bottom right"
        )

        fig.update_layout(
            template="plotly_dark",
            height=680,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.99)
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No stocks match the selected filter criteria. Adjust the sidebar filters.")

with tab4:
    if not filtered_df.empty:
        cols_to_show = [c for c in ['SYMBOL', 'SECTOR', 'CLOSE', 'Price Change (%)', 'Delivery Surge Index (x 50D Avg)', 'DELIV_QTY', 'Accumulation_Alert'] if c in filtered_df.columns]
        st.dataframe(
            filtered_df[cols_to_show].sort_values(by='Delivery Surge Index (x 50D Avg)', ascending=False),
            use_container_width=True
        )
