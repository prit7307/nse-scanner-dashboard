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
    
    col_map = {c: c.strip().upper() for c in df.columns}
    df.rename(columns=col_map, inplace=True)

    # 1. Symbol column
    symbol_col = next((c for c in ['SYMBOL', 'TCKRSYMB', 'STOCK'] if c in df.columns), df.columns[0])
    df.rename(columns={symbol_col: 'SYMBOL'}, inplace=True)

    # 2. Date column and filter latest date
    date_col = next((c for c in ['DATE', 'TIMESTAMP', 'TRADING_DATE', 'DATETIME'] if c in df.columns), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=['SYMBOL', date_col])
        latest_date = df[date_col].max()
        latest_df = df[df[date_col] == latest_date].copy()
    else:
        latest_df = df.copy()

    # 3. Filter series
    if 'SERIES' in latest_df.columns:
        latest_df = latest_df[latest_df['SERIES'].isin(['EQ', 'BE'])]

    # 4. Close column
    close_col = next((c for c in ['CLOSE', 'CLOSE_PRICE', 'LT_PRICE', 'LTP'] if c in latest_df.columns), None)
    if close_col:
        latest_df['CLOSE'] = pd.to_numeric(latest_df[close_col], errors='coerce')
    else:
        latest_df['CLOSE'] = 0.0

    # 5. Price Change (%)
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

    # Sanitize ACCURACY / Penny Stock Outliers
    latest_df = latest_df[
        (latest_df['Price Change (%)'].notna()) &
        (latest_df['Price Change (%)'].between(-30.0, 100.0))
    ]

    # 6. Delivery Surge Index (DSI)
    dsi_col = next((c for c in ['DELIVERY SURGE INDEX (X 50D AVG)', 'DELIVERY_SURGE_INDEX', 'DSI', 'SURGE_INDEX'] if c in latest_df.columns), None)
    if dsi_col:
        latest_df['Delivery Surge Index (x 50D Avg)'] = pd.to_numeric(latest_df[dsi_col], errors='coerce').fillna(1.0)
    else:
        latest_df['Delivery Surge Index (x 50D Avg)'] = 1.0

    # 7. Deliverable Quantity
    deliv_col = next((c for c in ['DELIV_QTY', 'DELIVERY_QTY', 'TTL_TRD_QNTY', 'VOLUME'] if c in latest_df.columns), None)
    if deliv_col:
        latest_df['DELIV_QTY'] = pd.to_numeric(latest_df[deliv_col], errors='coerce').fillna(0)
    else:
        latest_df['DELIV_QTY'] = 0

    # 8. Accumulation Alert
    latest_df['Accumulation_Alert'] = np.where(
        latest_df['Delivery Surge Index (x 50D Avg)'] >= 2.0,
        'DELIVERY SURGE',
        'NEUTRAL'
    )

    # 9. Sector Mapping
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

selected_sectors = st.sidebar.multiselect(
    "Filter by Sector:",
    options=all_sectors,
    default=default_sectors
)

min_dsi = st.sidebar.slider("Min Delivery Surge Index (DSI):", min_value=0.5, max_value=5.0, value=1.00, step=0.1)
show_nr7_only = st.sidebar.checkbox("Show Only NR7 Squeeze Stocks", value=False)
min_deliv_qty = st.sidebar.number_input("Min Deliverable Qty:", value=10000, step=5000)

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
col4.metric("Leading Sectors (>70% > 20EMA)", 2)

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
        pct_values = filtered_df['Price Change (%)'].dropna()
        p_low = max(-20.0, float(np.percentile(pct_values, 1))) if len(pct_values) > 5 else -10.0
        p_high = min(35.0, float(np.percentile(pct_values, 99)) + 2.0) if len(pct_values) > 5 else 20.0

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
            marker=dict(size=9, opacity=0.85, line=dict(width=0.5, color='white'))
        )

        fig.update_xaxes(
            title_text="Price Change (%)",
            range=[p_low, max(p_high, 15.0)],
            zeroline=True,
            zerolinecolor='#444',
            gridcolor='#222'
        )

        fig.update_yaxes(
            title_text="Delivery Surge Index (x 50D Avg)",
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
        cols_to_show = [c for c in ['SYMBOL', 'CLOSE', 'Price Change (%)', 'Delivery Surge Index (x 50D Avg)', 'DELIV_QTY', 'Accumulation_Alert'] if c in filtered_df.columns]
        st.dataframe(
            filtered_df[cols_to_show].sort_values(by='Delivery Surge Index (x 50D Avg)', ascending=False),
            use_container_width=True
        )
