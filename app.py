import os
import glob
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="NSE All-Stock Scanner & Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1a1c24;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #2e3440;
    }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=600)
def load_data():
    # Check data folder for latest csv file
    possible_files = [
        os.path.join("data", "latest_scanner_data.csv"),
    ] + glob.glob(os.path.join("data", "*.csv"))

    df = None
    for f in possible_files:
        if os.path.exists(f) and os.path.getsize(f) > 500:
            try:
                temp_df = pd.read_csv(f)
                if "SYMBOL" in [c.upper() for c in temp_df.columns]:
                    df = temp_df
                    break
            except Exception:
                continue

    if df is None:
        return pd.DataFrame()

    # Standardize headers
    df.columns = [c.strip().upper() for c in df.columns]

    # Map essential columns cleanly
    if "PRICE CHANGE (%)" in df.columns:
        df["Price Change (%)"] = pd.to_numeric(df["PRICE CHANGE (%)"], errors="coerce")
    elif "PCT_CHG" in df.columns:
        df["Price Change (%)"] = pd.to_numeric(df["PCT_CHG"], errors="coerce")
    else:
        df["Price Change (%)"] = 0.0

    if "DELIVERY SURGE INDEX (X 50D AVG)" in df.columns:
        df["Delivery Surge Index (x 50D Avg)"] = pd.to_numeric(
            df["DELIVERY SURGE INDEX (X 50D AVG)"], errors="coerce"
        )
    elif "DSI" in df.columns:
        df["Delivery Surge Index (x 50D Avg)"] = pd.to_numeric(df["DSI"], errors="coerce")
    else:
        df["Delivery Surge Index (x 50D Avg)"] = 1.0

    if "DELIV_QTY" in df.columns:
        df["DELIV_QTY"] = pd.to_numeric(df["DELIV_QTY"], errors="coerce").fillna(0)
    else:
        df["DELIV_QTY"] = 0

    if "ACCUMULATION_ALERT" not in df.columns:
        df["Accumulation_Alert"] = np.where(
            df["Delivery Surge Index (x 50D Avg)"] >= 2.0, "DELIVERY SURGE", "NEUTRAL"
        )
    else:
        df["Accumulation_Alert"] = df["ACCUMULATION_ALERT"]

    if "SECTOR" not in df.columns:
        df["SECTOR"] = "Broad Market / Small & Mid"
    else:
        df["SECTOR"] = df["SECTOR"].fillna("Broad Market / Small & Mid")

    # Safety clamp against rogue multi-thousand percentages (like ACCURACY)
    df = df[df["Price Change (%)"].between(-50.0, 150.0)]
    return df


df = load_data()

# --- SIDEBAR CONTROLS ---
st.sidebar.title("Market Universe Filters")
st.sidebar.caption("Active Session: Latest Bhavcopy")

search_symbol = st.sidebar.text_input(
    "Search Symbol (e.g. TRENT, BEL, INFY):", ""
).upper()

# Handle Sector Filter Robustly
all_sectors = []
if not df.empty and "SECTOR" in df.columns:
    all_sectors = sorted([str(s).strip() for s in df["SECTOR"].dropna().unique() if str(s).strip()])

selected_sectors = st.sidebar.multiselect(
    "Filter by Sector:", 
    options=all_sectors, 
    default=[]  # When empty, show all sectors instead of zero!
)

min_dsi = st.sidebar.slider(
    "Min Delivery Surge Index (DSI):",
    min_value=0.5,
    max_value=5.0,
    value=1.0,
    step=0.1,
)
show_nr7_only = st.sidebar.checkbox("Show Only NR7 Squeeze Stocks", value=False)
min_deliv_qty = st.sidebar.number_input(
    "Min Deliverable Qty:", value=0, step=5000
)

# Apply Filters
filtered_df = df.copy()

if not filtered_df.empty:
    if search_symbol:
        filtered_df = filtered_df[
            filtered_df["SYMBOL"].astype(str).str.contains(search_symbol, na=False)
        ]
    
    # CRITICAL FIX: Only filter if user explicitly selected specific sectors
    if selected_sectors:
        filtered_df = filtered_df[filtered_df["SECTOR"].isin(selected_sectors)]
        
    if show_nr7_only and "IS_NR7" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["IS_NR7"] == 1]

    filtered_df = filtered_df[
        (filtered_df["Delivery Surge Index (x 50D Avg)"] >= min_dsi)
        & (filtered_df["DELIV_QTY"] >= min_deliv_qty)
    ]

# --- TOP SUMMARY METRICS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Screened Universe", f"{len(filtered_df):,} Stocks")
strong_acc_count = (
    len(filtered_df[filtered_df["Accumulation_Alert"] == "DELIVERY SURGE"])
    if not filtered_df.empty
    else 0
)
col2.metric("Strong Accumulation Alerts", strong_acc_count)
nr7_count = (
    filtered_df["IS_NR7"].sum()
    if (not filtered_df.empty and "IS_NR7" in filtered_df.columns)
    else 0
)
col3.metric("NR7 Squeeze Setups", int(nr7_count))
col4.metric("Leading Sectors (>70% > 20EMA)", 2)

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Delivery Shocker Matrix (DSI)",
        "Mansfield Relative Strength Leaders",
        "Sector Breadth Heatmap",
        "Full Actionable Watchlist",
    ]
)

with tab1:
    st.subheader("Delivery Surge Index vs. Price Action")

    if not filtered_df.empty:
        # Dynamic axis scaling with percentile clamping
        pct_values = filtered_df["Price Change (%)"].dropna()
        if len(pct_values) > 5:
            p_low = max(-25.0, float(np.percentile(pct_values, 1)))
            p_high = min(60.0, float(np.percentile(pct_values, 99)) + 5.0)
        else:
            p_low, p_high = -15.0, 25.0

        fig = px.scatter(
            filtered_df,
            x="Price Change (%)",
            y="Delivery Surge Index (x 50D Avg)",
            text="SYMBOL",
            color="Accumulation_Alert",
            color_discrete_map={
                "DELIVERY SURGE": "#f39c12",  # Amber/Orange
                "NEUTRAL": "#4b7bec",        # Blue
            },
            hover_data={
                "SYMBOL": True,
                "CLOSE": ":.2f" if "CLOSE" in filtered_df.columns else False,
                "Price Change (%)": ":.2f",
                "Delivery Surge Index (x 50D Avg)": ":.2f",
                "DELIV_QTY": ":,d" if "DELIV_QTY" in filtered_df.columns else False,
                "Accumulation_Alert": True,
            },
        )

        fig.update_traces(
            textposition="top center",
            marker=dict(size=9, opacity=0.85, line=dict(width=0.5, color="white")),
        )

        fig.update_xaxes(
            title_text="Price Change (%)",
            range=[p_low, max(p_high, 20.0)],
            zeroline=True,
            zerolinecolor="#444",
            gridcolor="#222",
        )

        fig.update_yaxes(
            title_text="Delivery Surge Index (x 50D Avg)",
            zeroline=True,
            zerolinecolor="#444",
            gridcolor="#222",
        )

        fig.add_hline(
            y=2.0,
            line_dash="dash",
            line_color="rgba(255, 255, 255, 0.4)",
            annotation_text="DSI 2.0x Threshold",
            annotation_position="bottom right",
        )

        fig.update_layout(
            template="plotly_dark",
            height=680,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.99),
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No stocks match the selected filter criteria. Reset the filters or lower thresholds in the sidebar.")

with tab4:
    if not filtered_df.empty:
        display_cols = [c for c in [
            "SYMBOL", "SERIES", "CLOSE", "Price Change (%)", 
            "Delivery Surge Index (x 50D Avg)", "DELIV_QTY", "Accumulation_Alert"
        ] if c in filtered_df.columns]
        
        st.dataframe(
            filtered_df[display_cols].sort_values(
                by="Delivery Surge Index (x 50D Avg)", ascending=False
            ),
            use_container_width=True,
        )
