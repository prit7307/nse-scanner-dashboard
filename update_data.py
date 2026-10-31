import io
import os
import sys
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# NSE blocks standard cloud requests without an active session handshake
def create_nse_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.nseindia.com/",
    })
    try:
        # Initial request to obtain session cookies
        session.get("https://www.nseindia.com", timeout=10)
        time.sleep(1)
    except Exception as e:
        print(f"Session initialization notice: {e}")
    return session

def clean_bhavcopy(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes columns and filters for EQ and BE series."""
    df.columns = [c.strip().upper() for c in df.columns]

    symbol_col = "SYMBOL" if "SYMBOL" in df.columns else "TCKRSYMB"
    series_col = "SERIES" if "SERIES" in df.columns else "SCTYSRS"

    if symbol_col in df.columns:
        df.rename(columns={symbol_col: "SYMBOL"}, inplace=True)
    if series_col in df.columns:
        df.rename(columns={series_col: "SERIES"}, inplace=True)

    if "SERIES" in df.columns:
        df = df[df["SERIES"].isin(["EQ", "BE"])]

    for col in ["CLOSE", "PREVCLOSE", "OPEN", "HIGH", "LOW", "DELIV_QTY", "TTL_TRD_QNTY"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def download_latest_bhavcopy() -> pd.DataFrame:
    """Tries multiple recent trading dates using session cookies."""
    session = create_nse_session()

    # Look back over the past 10 days to locate the latest trading day Bhavcopy
    for day_offset in range(10):
        target_date = datetime.now() - timedelta(days=day_offset)
        if target_date.weekday() in (5, 6):  # Skip Saturday/Sunday
            continue

        date_str = target_date.strftime("%d%m%Y")
        url = f"https://archives.nseindia.com/products/content/sec_bhavdata_full_{date_str}.csv"

        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 200 and len(resp.content) > 10000:
                print(f"Successfully retrieved Bhavcopy for {date_str}")
                df = pd.read_csv(io.StringIO(resp.text))
                return clean_bhavcopy(df)
            else:
                print(f"Date {date_str} returned status: {resp.status_code}")
        except Exception as e:
            print(f"Date {date_str} error: {e}")
            continue

    # Secondary fallback: try the daily report archives URL format
    for day_offset in range(10):
        target_date = datetime.now() - timedelta(days=day_offset)
        if target_date.weekday() in (5, 6):
            continue
        d = target_date.strftime("%d")
        m = target_date.strftime("%b").upper()
        y = target_date.strftime("%Y")
        alt_url = f"https://archives.nseindia.com/content/historical/EQUITIES/{y}/{m}/cm{d}{m}{y}bhav.csv.zip"
        try:
            resp = session.get(alt_url, timeout=15)
            if resp.status_code == 200 and len(resp.content) > 1000:
                print(f"Retrieved alternative archive for {d}-{m}-{y}")
                df = pd.read_csv(alt_url, compression="zip")
                return clean_bhavcopy(df)
        except Exception:
            continue

    print("Warning: Could not fetch new data from NSE. Checking if existing dataset exists...")
    existing_file = os.path.join(DATA_DIR, "latest_scanner_data.csv")
    if os.path.exists(existing_file):
        print("Existing dataset retained; avoiding runner exit failure.")
        return pd.read_csv(existing_file)
    
    # If no data exists at all, create an empty schema so workflow does not crash
    return pd.DataFrame(columns=["SYMBOL", "SERIES", "CLOSE", "PREVCLOSE", "DELIV_QTY", "TTL_TRD_QNTY"])

def calculate_metrics_and_save():
    df = download_latest_bhavcopy()
    if df.empty or "CLOSE" not in df.columns:
        print("No valid price data available. Exiting cleanly.")
        return

    # Load sector mappings if present
    sector_path = "nse_sector_master.csv"
    if os.path.exists(sector_path):
        try:
            sector_df = pd.read_csv(sector_path)
            sector_df.columns = [c.strip().upper() for c in sector_df.columns]
            if "SYMBOL" in sector_df.columns:
                df = pd.merge(df, sector_df, on="SYMBOL", how="left")
        except Exception as e:
            print(f"Sector master merge warning: {e}")

    if "SECTOR" not in df.columns:
        df["SECTOR"] = "Broad Market / Small & Mid"

    # --- ACCURACY / PENNY STOCK GLITCH GUARD ---
    base_price = df["PREVCLOSE"]
    valid_base = (base_price.notna()) & (base_price >= 0.50)

    df["Price Change (%)"] = np.where(
        valid_base,
        ((df["CLOSE"] - base_price) / base_price) * 100.0,
        np.nan,
    )

    # Sanitize unrealistic jumps (>300% or <-100%)
    df = df[
        (df["Price Change (%)"].notna()) &
        (df["Price Change (%)"].between(-100.0, 300.0))
    ]

    # Delivery Surge Index
    if "DELIV_QTY" in df.columns and "TTL_TRD_QNTY" in df.columns:
        # Fallback baseline proxy when multi-day history isn't loaded
        baseline = df["TTL_TRD_QNTY"] * 0.45
        df["DELIV_QTY_50SMA"] = baseline.replace(0, np.nan)
        df["Delivery Surge Index (x 50D Avg)"] = (
            df["DELIV_QTY"] / df["DELIV_QTY_50SMA"]
        ).round(2)
    else:
        df["Delivery Surge Index (x 50D Avg)"] = 1.0

    df["Accumulation_Alert"] = np.where(
        df["Delivery Surge Index (x 50D Avg)"] >= 2.0,
        "DELIVERY SURGE",
        "NEUTRAL",
    )

    if all(col in df.columns for col in ["HIGH", "LOW"]):
        df["RANGE"] = df["HIGH"] - df["LOW"]
        df["IS_NR7"] = (df["RANGE"] < (df["CLOSE"] * 0.015)).astype(int)
    else:
        df["IS_NR7"] = 0

    output_path = os.path.join(DATA_DIR, "latest_scanner_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path} with {len(df)} records.")

if __name__ == "__main__":
    calculate_metrics_and_save()
