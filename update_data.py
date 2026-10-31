import io
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def clean_bhavcopy(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes columns and enforces strict EQ / BE series filtering."""
    df.columns = [c.strip().upper() for c in df.columns]

    # Map modern vs legacy NSE column conventions
    symbol_col = "SYMBOL" if "SYMBOL" in df.columns else "TCKRSYMB"
    series_col = "SERIES" if "SERIES" in df.columns else "SCTYSRS"

    if symbol_col in df.columns:
        df.rename(columns={symbol_col: "SYMBOL"}, inplace=True)
    if series_col in df.columns:
        df.rename(columns={series_col: "SERIES"}, inplace=True)

    # Filter strictly for regular equity settlement and Trade-to-Trade / ESM
    if "SERIES" in df.columns:
        df = df[df["SERIES"].isin(["EQ", "BE"])]

    # Cast numeric columns cleanly
    numeric_targets = [
        "CLOSE",
        "PREVCLOSE",
        "OPEN",
        "HIGH",
        "LOW",
        "DELIV_QTY",
        "TTL_TRD_QNTY",
    ]
    for col in numeric_targets:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def download_latest_bhavcopy() -> pd.DataFrame:
    """Fetches the latest available daily Bhavcopy with deliverable data."""
    for day_offset in range(7):
        target_date = datetime.now() - timedelta(days=day_offset)
        if target_date.weekday() in (5, 6):  # Skip weekend
            continue

        date_str = target_date.strftime("%d%m%Y")
        url = f"https://archives.nseindia.com/products/content/sec_bhavdata_full_{date_str}.csv"

        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            if response.status_code == 200 and len(response.content) > 5000:
                print(f"Downloaded Bhavcopy for: {date_str}")
                df = pd.read_csv(io.StringIO(response.text))
                return clean_bhavcopy(df)
        except Exception as e:
            print(f"Attempt for {date_str} encountered: {e}")
            continue

    raise RuntimeError("Unable to download recent NSE Bhavcopy after retries.")


def calculate_metrics_and_save():
    """Calculates DSI, NR7 squeeze, and hardened Price Change % metrics."""
    latest_df = download_latest_bhavcopy()

    # Load Sector mapping if present in root
    sector_path = "nse_sector_master.csv"
    if os.path.exists(sector_path):
        try:
            sector_df = pd.read_csv(sector_path)
            sector_df.columns = [c.strip().upper() for c in sector_df.columns]
            if "SYMBOL" in sector_df.columns:
                latest_df = pd.merge(latest_df, sector_df, on="SYMBOL", how="left")
        except Exception as e:
            print(f"Sector master merge notice: {e}")

    # Fallback sector assignment if missing
    if "SECTOR" not in latest_df.columns:
        latest_df["SECTOR"] = "Broad Market / Small & Mid"

    # --- BASELINE CALCULATION & ANOMALY PREVENTION ---
    base_denominator = latest_df["PREVCLOSE"]
    valid_denominator = (base_denominator.notna()) & (base_denominator >= 0.50)

    # 1. Price Change (%) Guard (Prevents zero/near-zero explosions)
    latest_df["Price Change (%)"] = np.where(
        valid_denominator,
        ((latest_df["CLOSE"] - base_denominator) / base_denominator) * 100.0,
        np.nan,
    )

    # 2. Filter out data anomalies (outside -100% to +300%)
    latest_df = latest_df[
        (latest_df["Price Change (%)"].notna())
        & (latest_df["Price Change (%)"].between(-100.0, 300.0))
    ]

    # 3. Delivery Volume Baseline & DSI Calculation
    if "DELIV_QTY" in latest_df.columns:
        baseline_col = (
            latest_df["DELIV_QTY_50SMA"]
            if "DELIV_QTY_50SMA" in latest_df.columns
            else latest_df["TTL_TRD_QNTY"] * 0.45
        )
        latest_df["DELIV_QTY_50SMA"] = baseline_col.replace(0, np.nan)

        latest_df["Delivery Surge Index (x 50D Avg)"] = (
            latest_df["DELIV_QTY"] / latest_df["DELIV_QTY_50SMA"]
        ).round(2)
    else:
        latest_df["Delivery Surge Index (x 50D Avg)"] = 1.0

    # 4. Accumulation Alert Category
    latest_df["Accumulation_Alert"] = np.where(
        latest_df["Delivery Surge Index (x 50D Avg)"] >= 2.0,
        "DELIVERY SURGE",
        "NEUTRAL",
    )

    # 5. NR7 Squeeze Setup Flag
    if all(col in latest_df.columns for col in ["HIGH", "LOW"]):
        latest_df["RANGE"] = latest_df["HIGH"] - latest_df["LOW"]
        latest_df["IS_NR7"] = (
            latest_df["RANGE"] < (latest_df["CLOSE"] * 0.015)
        ).astype(int)
    else:
        latest_df["IS_NR7"] = 0

    # Save output for Streamlit
    output_path = os.path.join(DATA_DIR, "latest_scanner_data.csv")
    latest_df.to_csv(output_path, index=False)
    print(f"Data saved successfully to {output_path} with {len(latest_df)} stocks.")


if __name__ == "__main__":
    calculate_metrics_and_save()
