import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import io
import os

st.set_page_config(
    page_title="NSE All-Stock Scanner & Breadth Engine (2000+ Equities)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

PARQUET_PATH = "data/nse_rolling_60d.parquet"
SECTOR_PATH = "nse_sector_master.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
}

@st.cache_data(ttl=86400)
def load_all_nse_equities_master():
    """Fetches the official NSE master list of all ~2,100 listed equities."""
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            df_eq = pd.read_csv(io.StringIO(r.text))
            df_eq = df_eq[df_eq[" SERIES"] == "EQ"][["SYMBOL", "NAME OF COMPANY"]].copy()
            df_eq.columns = ["SYMBOL", "COMPANY_NAME"]
            return df_eq
    except Exception:
        pass
    return pd.DataFrame(columns=["SYMBOL", "COMPANY_NAME"])

@st.cache_data(ttl=3600)
def load_and_compute_metrics():
    # If the daily parquet exists from GitHub Actions, load all 2,000+ stocks
    if os.path.exists(PARQUET_PATH):
        df = pd.read_parquet(PARQUET_PATH)
        df["DATE"] = pd.to_datetime(df["DATE"])
    else:
        # Fallback generator pulling broad market data
        master_eq = load_all_nse_equities_master()
        symbols = master_eq["SYMBOL"].tolist() if not master_eq.empty else [
            "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "BHARTIARTL", "SBIN", "INFY", "LICI", "ITC", "HINDUNILVR",
            "LT", "BAJFINANCE", "HCLTECH", "MARUTI", "SUNPHARMA", "ADANIENT", "KOTAKBANK", "TITAN", "ONGC", "TATAMOTORS",
            "NTPC", "AXISBANK", "DMART", "ADANIGREEN", "ADANIPORTS", "ULTRACEMCO", "COALINDIA", "BAJAJFINSV", "POWERGRID",
            "M&M", "BEL", "HAL", "DIXON", "TRENT", "POLYCAB", "KALYANKJIL", "TATAPOWER", "CHAMBLFERT", "PERSISTENT", "CIPLA"
        ]
        
        # Build multi-day mock matrix across all stocks until update_data.py commits the rolling parquet
        np.random.seed(42)
        dates = pd.date_range(end=pd.Timestamp.today(), periods=30, freq='B')
        records = []
        for s in symbols[:200]: # Initial responsive slice
            base_p = np.random.uniform(50, 4000)
            for d in dates:
                fluct = np.random.normal(0, 0.02)
                p = base_p * (1 + fluct)
                vol = np.random.randint(50000, 5000000)
                deliv = int(vol * np.random.uniform(0.35, 0.65))
                records.append({
                    "DATE": d,
                    "SYMBOL": s,
                    "CLOSE": round(p, 2),
                    "HIGH": round(p * 1.015, 2),
                    "LOW": round(p * 0.985, 2),
                    "TOTTRDQTY": vol,
                    "DELIV_QTY": deliv
                })
        df = pd.DataFrame(records)

    df = df.sort_values(["SYMBOL", "DATE"])
    
    # 1. Delivery Surge Index (DSI)
    df["50D_AVG_DELIV"] = df.groupby("SYMBOL")["DELIV_QTY"].transform(lambda s: s.rolling(50, min_periods=5).mean())
    df["DSI"] = (df["DELIV_QTY"] / df["50D_AVG_DELIV"]).round(2)
    
    # 2. NR7 Volatility Contraction
    df["RANGE"] = df["HIGH"] - df["LOW"]
    df["MIN_7D_RANGE"] = df.groupby("SYMBOL")["RANGE"].transform(lambda s: s.rolling(7, min_periods=5).min())
    df["NR7"] = df["RANGE"] <= df["MIN_7D_RANGE"]
    
    # 3. Daily % Change & Moving Averages
    df["PREV_CLOSE"] = df.groupby("SYMBOL")["CLOSE"].shift(1)
    df["PCT_CHANGE"] = ((df["CLOSE"] - df["PREV_CLOSE"]) / df["PREV_CLOSE"] * 100).round(2)
    df["EMA20"] = df.groupby("SYMBOL")["CLOSE"].transform(lambda s: s.ewm(span=20, adjust=False).mean())
    df["EMA50"] = df.groupby("SYMBOL")["CLOSE"].transform(lambda s: s.ewm(span=50, adjust=False).mean())
    
    # 4. Mansfield Relative Strength (Universe-wide)
    benchmark_closes = df.groupby("DATE")["CLOSE"].transform("median")
    df["RAW_RS"] = df["CLOSE"] / benchmark_closes
    df["SMA50_RS"] = df.groupby("SYMBOL")["RAW_RS"].transform(lambda s: s.rolling(50, min_periods=5).mean())
    df["MANSFIELD_RS"] = (((df["RAW_RS"] / df["SMA50_RS"]) - 1) * 100).round(2)
    
    # Latest Day
    latest_date = df["DATE"].max()
    latest_df = df[df["DATE"] == latest_date].copy()
    
    # Merge Sector Reference
    if os.path.exists(SECTOR_PATH):
        sec_df = pd.read_csv(SECTOR_PATH)
        latest_df = latest_df.merge(sec_df[["SYMBOL", "SECTOR"]], on="SYMBOL", how="left")
    else:
        latest_df["SECTOR"] = "Unassigned"
        
    latest_df["SECTOR"] = latest_df["SECTOR"].fillna("Broad Market / Small & Mid Cap")
    
    # Categorize Institutional Signals
    latest_df["Accumulation_Alert"] = np.where(
        (latest_df["DSI"] >= 2.0) & (latest_df["PCT_CHANGE"] > 0) & (latest_df["NR7"]),
        "STRONG ACCUMULATION",
        np.where((latest_df["DSI"] >= 2.0) & (latest_df["PCT_CHANGE"] > 0), "DELIVERY SURGE", "NEUTRAL")
    )
    
    # Sector Breadth Aggregation
    sector_breadth = latest_df.groupby("SECTOR").agg(
        Total_Stocks=("SYMBOL", "count"),
        Above_20EMA=("CLOSE", lambda c: (c > latest_df.loc[c.index, "EMA20"]).sum()),
        Above_50EMA=("CLOSE", lambda c: (c > latest_df.loc[c.index, "EMA50"]).sum())
    ).reset_index()
    sector_breadth["Pct_Above_20EMA"] = ((sector_breadth["Above_20EMA"] / sector_breadth["Total_Stocks"]) * 100).round(1)
    sector_breadth["Pct_Above_50EMA"] = ((sector_breadth["Above_50EMA"] / sector_breadth["Total_Stocks"]) * 100).round(1)
    
    return latest_df, sector_breadth, latest_date

df_latest, df_breadth, session_date = load_and_compute_metrics()

# --- Sidebar Controls ---
st.sidebar.title("Market Universe Filters")
st.sidebar.caption(f"Active Session: **{session_date.strftime('%d %b %Y')}**")

# Quick search box for any stock
search_query = st.sidebar.text_input("🔍 Search Symbol (e.g. TRENT, BEL, INFY):").strip().upper()

selected_sectors = st.sidebar.multiselect(
    "Filter by Sector:",
    options=sorted(df_latest["SECTOR"].unique()),
    default=sorted(df_latest["SECTOR"].unique())
)

dsi_filter = st.sidebar.slider("Min Delivery Surge Index (DSI):", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
nr7_filter = st.sidebar.checkbox("Show Only NR7 Squeeze Stocks", value=False)
min_deliv_qty = st.sidebar.number_input("Min Deliverable Qty (Liquidity Filter):", min_value=0, value=25000, step=10000)

# Filter Data
filtered_df = df_latest[
    (df_latest["SECTOR"].isin(selected_sectors)) &
    (df_latest["DSI"] >= dsi_filter) &
    (df_latest["DELIV_QTY"] >= min_deliv_qty)
]

if nr7_filter:
    filtered_df = filtered_df[filtered_df["NR7"] == True]

if search_query:
    filtered_df = filtered_df[filtered_df["SYMBOL"].str.contains(search_query, na=False)]

# --- Top Header KPI Cards ---
st.title("NSE Universe Scanner & Institutional Breadth Engine")
st.markdown(f"Displaying **{len(filtered_df):,}** out of **{len(df_latest):,}** tracked listed securities.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Screened Universe", f"{len(filtered_df):,} Stocks")
c2.metric("Strong Accumulation Alerts", len(df_latest[df_latest["Accumulation_Alert"] == "STRONG ACCUMULATION"]))
c3.metric("NR7 Squeeze Setups", len(df_latest[df_latest["NR7"] == True]))
c4.metric("Leading Sectors (>70% > 20EMA)", len(df_breadth[df_breadth["Pct_Above_20EMA"] >= 70]))

st.divider()

tabs = st.tabs([
    "Delivery Shocker Matrix (DVS)",
    "Mansfield Relative Strength Leaders",
    "Sector Breadth Heatmap",
    "Full Actionable Watchlist"
])

# --- TAB 1: DVS SCATTER MATRIX ---
with tabs[0]:
    st.subheader("Delivery Surge Index vs. Price Action")
    # For performance across 2000+ stocks, plot top candidates
    plot_df = filtered_df.sort_values(by="DSI", ascending=False).head(250)
    fig_dvs = px.scatter(
        plot_df,
        x="PCT_CHANGE",
        y="DSI",
        size="DELIV_QTY",
        color="Accumulation_Alert",
        hover_name="SYMBOL",
        text="SYMBOL",
        color_discrete_map={
            "STRONG ACCUMULATION": "#00CC96",
            "DELIVERY SURGE": "#FFA15A",
            "NEUTRAL": "#636EFA"
        },
        labels={"PCT_CHANGE": "Price Change (%)", "DSI": "Delivery Surge Index (x 50D Avg)"}
    )
    fig_dvs.add_hline(y=2.0, line_dash="dash", line_color="gray", annotation_text="DSI 2.0x Threshold")
    fig_dvs.add_vline(x=0.0, line_dash="dash", line_color="gray")
    fig_dvs.update_traces(textposition="top center")
    fig_dvs.update_layout(height=560, template="plotly_white")
    st.plotly_chart(fig_dvs, use_container_width=True)

# --- TAB 2: MANSFIELD RS LEADERS ---
with tabs[1]:
    st.subheader("Top Relative Strength Outperformers vs. Underperformers")
    col_lead, col_lag = st.columns(2)
    top_leaders = filtered_df.sort_values(by="MANSFIELD_RS", ascending=False).head(20)
    top_laggards = filtered_df.sort_values(by="MANSFIELD_RS", ascending=True).head(20)
    
    with col_lead:
        st.markdown("**Top 20 Outperforming Stocks (Green)**")
        fig_lead = px.bar(
            top_leaders,
            x="MANSFIELD_RS",
            y="SYMBOL",
            orientation="h",
            color_discrete_sequence=["#00CC96"],
            text="MANSFIELD_RS"
        )
        fig_lead.update_layout(yaxis=dict(autorange="reversed"), height=500, template="plotly_white")
        st.plotly_chart(fig_lead, use_container_width=True)
        
    with col_lag:
        st.markdown("**Top 20 Lagging Stocks (Red)**")
        fig_lag = px.bar(
            top_laggards,
            x="MANSFIELD_RS",
            y="SYMBOL",
            orientation="h",
            color_discrete_sequence=["#EF553B"],
            text="MANSFIELD_RS"
        )
        fig_lag.update_layout(yaxis=dict(autorange="reversed"), height=500, template="plotly_white")
        st.plotly_chart(fig_lag, use_container_width=True)

# --- TAB 3: SECTOR BREADTH HEATMAP ---
with tabs[2]:
    st.subheader("Sector Breadth Diffusion Matrix (% Stocks Above Moving Averages)")
    breadth_matrix = df_breadth.set_index("SECTOR")[["Pct_Above_20EMA", "Pct_Above_50EMA"]]
    
    fig_breadth = px.imshow(
        breadth_matrix.values,
        labels=dict(x="Metric", y="Sector", color="% Breadth"),
        x=["% Above 20 EMA", "% Above 50 EMA"],
        y=breadth_matrix.index,
        color_continuous_scale="RdYlGn",
        text_auto=True,
        aspect="auto"
    )
    fig_breadth.update_layout(height=520, template="plotly_white")
    st.plotly_chart(fig_breadth, use_container_width=True)

# --- TAB 4: COMPLETE 2000+ STOCK DATA TABLE ---
with tabs[3]:
    st.subheader("Actionable Watchlist Data Table")
    display_cols = ["SYMBOL", "SECTOR", "CLOSE", "PCT_CHANGE", "DELIV_QTY", "DSI", "NR7", "MANSFIELD_RS", "Accumulation_Alert"]
    st.dataframe(
        filtered_df[display_cols].sort_values(by="DSI", ascending=False).style.format({
            "CLOSE": "₹{:,.2f}",
            "PCT_CHANGE": "{:+.2f}%",
            "DELIV_QTY": "{:,}",
            "DSI": "{:.2f}x",
            "MANSFIELD_RS": "{:+.2f}"
        }),
        use_container_width=True,
        height=600
    )
