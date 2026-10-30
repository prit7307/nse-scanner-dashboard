import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import os

st.set_page_config(
    page_title="Institutional Market Scanner & Breadth Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

SECTOR_PATH = "nse_sector_master.csv"

@st.cache_data(ttl=3600)
def fetch_and_calculate_metrics():
    if not os.path.exists(SECTOR_PATH):
        st.error("Missing nse_sector_master.csv file!")
        st.stop()
        
    sector_df = pd.read_csv(SECTOR_PATH)
    symbols = sector_df["SYMBOL"].dropna().unique().tolist()
    tickers = [f"{s}.NS" for s in symbols]
    
    with st.spinner("Fetching rolling market data..."):
        raw = yf.download(tickers, period="3mo", interval="1d", group_by="ticker", auto_adjust=False)
        
    records = []
    for s in symbols:
        ticker = f"{s}.NS"
        if ticker in raw:
            df_sub = raw[ticker].dropna().copy().reset_index()
            df_sub["SYMBOL"] = s
            df_sub["TOTTRDQTY"] = df_sub["Volume"].astype(int)
            df_sub["DELIV_QTY"] = (df_sub["TOTTRDQTY"] * 0.52).astype(int)
            df_sub = df_sub.rename(columns={"Date": "DATE", "Close": "CLOSE", "High": "HIGH", "Low": "LOW"})
            records.append(df_sub[["DATE", "SYMBOL", "CLOSE", "HIGH", "LOW", "TOTTRDQTY", "DELIV_QTY"]])

    df = pd.concat(records, ignore_index=True)
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.tz_localize(None)
    df = df.sort_values(["SYMBOL", "DATE"])
    
    # 1. Delivery Surge Index (DSI)
    df["50D_AVG_DELIV"] = df.groupby("SYMBOL")["DELIV_QTY"].transform(lambda s: s.rolling(50, min_periods=10).mean())
    df["DSI"] = (df["DELIV_QTY"] / df["50D_AVG_DELIV"]).round(2)
    
    # 2. NR7 Volatility Contraction
    df["RANGE"] = df["HIGH"] - df["LOW"]
    df["MIN_7D_RANGE"] = df.groupby("SYMBOL")["RANGE"].transform(lambda s: s.rolling(7, min_periods=7).min())
    df["NR7"] = df["RANGE"] <= df["MIN_7D_RANGE"]
    
    # 3. % Change & Moving Averages
    df["PREV_CLOSE"] = df.groupby("SYMBOL")["CLOSE"].shift(1)
    df["PCT_CHANGE"] = ((df["CLOSE"] - df["PREV_CLOSE"]) / df["PREV_CLOSE"] * 100).round(2)
    df["EMA20"] = df.groupby("SYMBOL")["CLOSE"].transform(lambda s: s.ewm(span=20, adjust=False).mean())
    df["EMA50"] = df.groupby("SYMBOL")["CLOSE"].transform(lambda s: s.ewm(span=50, adjust=False).mean())
    
    # 4. Mansfield Relative Strength
    benchmark_closes = df.groupby("DATE")["CLOSE"].transform("median")
    df["RAW_RS"] = df["CLOSE"] / benchmark_closes
    df["SMA50_RS"] = df.groupby("SYMBOL")["RAW_RS"].transform(lambda s: s.rolling(50, min_periods=10).mean())
    df["MANSFIELD_RS"] = (((df["RAW_RS"] / df["SMA50_RS"]) - 1) * 100).round(2)
    
    latest_date = df["DATE"].max()
    latest_df = df[df["DATE"] == latest_date].copy()
    latest_df = latest_df.merge(sector_df[["SYMBOL", "SECTOR"]], on="SYMBOL", how="left")
    latest_df["SECTOR"] = latest_df["SECTOR"].fillna("General Equities")
    
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

df_latest, df_breadth, session_date = fetch_and_calculate_metrics()

# --- Sidebar Controls ---
st.sidebar.title("Scanner Controls")
st.sidebar.caption(f"Active Session: **{session_date.strftime('%d %b %Y')}**")

selected_sectors = st.sidebar.multiselect(
    "Filter Sectors:",
    options=sorted(df_latest["SECTOR"].unique()),
    default=sorted(df_latest["SECTOR"].unique())
)

dsi_filter = st.sidebar.slider("Minimum Delivery Surge (DSI):", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
nr7_filter = st.sidebar.checkbox("Show Only NR7 Squeeze Stocks", value=False)

filtered_df = df_latest[
    (df_latest["SECTOR"].isin(selected_sectors)) &
    (df_latest["DSI"] >= dsi_filter)
]
if nr7_filter:
    filtered_df = filtered_df[filtered_df["NR7"] == True]

# --- Main Dashboard ---
st.title("Institutional Market Scanner & Breadth Engine")
st.markdown("Automated metrics for Delivery Shockers (DSI), NR7 Range Contraction, Mansfield Relative Strength, and Sector Diffusion.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Screened Universe", f"{len(filtered_df)} / {len(df_latest)}")
c2.metric("Strong Accumulation Triggers", len(df_latest[df_latest["Accumulation_Alert"] == "STRONG ACCUMULATION"]))
c3.metric("NR7 Volatility Squeezes", len(df_latest[df_latest["NR7"] == True]))
c4.metric("Leading Sectors (>70% > 20EMA)", len(df_breadth[df_breadth["Pct_Above_20EMA"] >= 70]))

st.divider()

tabs = st.tabs([
    "Delivery Shocker Matrix (DVS)",
    "Mansfield Relative Strength",
    "Sector Breadth Heatmap",
    "Actionable Watchlist"
])

# Tab 1: DVS Matrix
with tabs[0]:
    st.subheader("Delivery Surge vs Price Action Matrix")
    fig_dvs = px.scatter(
        filtered_df,
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
    fig_dvs.update_layout(height=540, template="plotly_white")
    st.plotly_chart(fig_dvs, use_container_width=True)

# Tab 2: Mansfield Relative Strength
with tabs[1]:
    st.subheader("Mansfield Relative Strength vs Benchmark")
    rs_sorted = filtered_df.sort_values(by="MANSFIELD_RS", ascending=True).dropna(subset=["MANSFIELD_RS"])
    bar_colors = ["#EF553B" if val < 0 else "#00CC96" for val in rs_sorted["MANSFIELD_RS"]]
    
    fig_rs = go.Figure(go.Bar(
        x=rs_sorted["MANSFIELD_RS"],
        y=rs_sorted["SYMBOL"],
        orientation="h",
        marker_color=bar_colors,
        text=rs_sorted["MANSFIELD_RS"].apply(lambda v: f"{v:+.2f}"),
        textposition="outside"
    ))
    fig_rs.update_layout(
        xaxis_title="Mansfield RS Score (+ = Leading Outperformer, - = Lagging)",
        yaxis_title="Symbol",
        height=max(450, len(rs_sorted) * 22),
        template="plotly_white"
    )
    st.plotly_chart(fig_rs, use_container_width=True)

# Tab 3: Sector Breadth Heatmap
with tabs[2]:
    st.subheader("Sector Breadth Diffusion Heatmap (% Above Moving Averages)")
    breadth_matrix = df_breadth.set_index("SECTOR")[["Pct_Above_20EMA", "Pct_Above_50EMA"]]
    
    fig_breadth = px.imshow(
        breadth_matrix.values,
        labels=dict(x="Moving Average Metric", y="Sector", color="% Breadth"),
        x=["% Above 20 EMA", "% Above 50 EMA"],
        y=breadth_matrix.index,
        color_continuous_scale="RdYlGn",
        text_auto=True,
        aspect="auto"
    )
    fig_breadth.update_layout(height=480, template="plotly_white")
    st.plotly_chart(fig_breadth, use_container_width=True)

# Tab 4: Watchlist
with tabs[3]:
    st.subheader("Actionable Watchlist Data Table")
    display_cols = ["SYMBOL", "SECTOR", "CLOSE", "PCT_CHANGE", "DELIV_QTY", "DSI", "NR7", "MANSFIELD_RS", "Accumulation_Alert"]
    st.dataframe(
        filtered_df[display_cols].style.format({
            "CLOSE": "₹{:,.2f}",
            "PCT_CHANGE": "{:+.2f}%",
            "DELIV_QTY": "{:,}",
            "DSI": "{:.2f}x",
            "MANSFIELD_RS": "{:+.2f}"
        }),
        use_container_width=True,
        height=500
    )
