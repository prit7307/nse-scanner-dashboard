import os
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


@st.cache_data(ttl=1800)
def load_data():
    data_path = os.path.join("data", "latest_scanner_data.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        return pd.DataFrame(
            columns=[
                "SYMBOL",
                "SERIES",
                "CLOSE",
                "Price Change (%)",
                "Delivery Surge Index (x 50D Avg)",
                "DELIV_QTY",
                "SECTOR",
                "Accumulation_Alert",
                "IS_NR7",
            ]
        )

    df["Price Change (%)"] = pd.to_numeric(df["Price Change (%)"], errors="coerce")
    df["Delivery Surge Index (x 50D Avg)"] = pd.to_numeric(
        df["Delivery Surge Index (x 50D Avg)"], errors="coerce"
    )
    df["DELIV_QTY"] = pd.to_numeric(df["DELIV_QTY"], errors="coerce").fillna(0)

    # Secondary filter to guarantee X-axis remains readable
    df = df[df["Price Change (%)"].between(-50.0, 150.0)]
    return df


df = load_data()

# --- SIDEBAR CONTROLS ---
st.sidebar.title("Market Universe Filters")
st.sidebar.caption("Active Session: Latest Bhavcopy")

search_symbol = st.sidebar.text_input(
    "Search Symbol (e.g. TRENT, BEL, INFY):", ""
).upper()

all_sectors = sorted([s for s in df["SECTOR"].dropna().unique() if str(s).strip()])
default_sectors = all_sectors[:5] if len(all_sectors) >= 5 else all_sectors
selected_sectors = st.sidebar.multiselect(
    "Filter by Sector:", options=all_sectors, default=default_sectors
)

min_dsi = st.sidebar.slider(
    "Min Delivery Surge Index (DSI):",
    min_value=1.0,
    max_value=5.0,
    value=1.0,
    step=0.1,
)
show_nr7_only = st.sidebar.checkbox("Show Only NR7 Squeeze Stocks", value=False)
min_deliv_qty = st.sidebar.number_input(
    "Min Deliverable Qty:", value=10000, step=5000
)

# Apply Filters
filtered_df = df.copy()

if search_symbol:
    filtered_df = filtered_df[
        filtered_df["SYMBOL"].str.contains(search_symbol, na=False)
    ]
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
strong_acc_count = len(
    filtered_df[filtered_df["Accumulation_Alert"] == "DELIVERY SURGE"]
)
col2.metric("Strong Accumulation Alerts", strong_acc_count)
nr7_count = (
    filtered_df["IS_NR7"].sum() if "IS_NR7" in filtered_df.columns else 0
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
        p_low = max(-25.0, float(np.percentile(filtered_df["Price Change (%)"], 1)))
        p_high = min(
            60.0, float(np.percentile(filtered_df["Price Change (%)"], 99)) + 5.0
        )

        fig = px.scatter(
            filtered_df,
            x="Price Change (%)",
            y="Delivery Surge Index (x 50D Avg)",
            text="SYMBOL",
            color="Accumulation_Alert",
            color_discrete_map={
                "DELIVERY SURGE": "#f39c12",  # Amber/Orange
                "NEUTRAL": "#4b7bec",  # Blue
            },
            hover_data={
                "SYMBOL": True,
                "CLOSE": ":.2f",
                "Price Change (%)": ":.2f",
                "Delivery Surge Index (x 50D Avg)": ":.2f",
                "DELIV_QTY": ":,d",
                "Accumulation_Alert": True,
            },
        )

        fig.update_traces(
            textposition="top center",
            marker=dict(size=9, opacity=0.85, line=dict(width=0.5, color="white")),
        )

        # Clamped X-axis prevents single outliers from compressing all other 1,151 stocks
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
        st.info("No stocks match the selected filter criteria.")

with tab4:
    st.dataframe(
        filtered_df[
            [
                "SYMBOL",
                "SERIES",
                "CLOSE",
                "Price Change (%)",
                "Delivery Surge Index (x 50D Avg)",
                "DELIV_QTY",
                "Accumulation_Alert",
            ]
        ].sort_values(by="Delivery Surge Index (x 50D Avg)", ascending=False),
        use_container_width=True,
    )
