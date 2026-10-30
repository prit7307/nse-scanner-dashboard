import os
import requests
import datetime
import pandas as pd
import numpy as np

DATA_DIR = "data"
PARQUET_PATH = os.path.join(DATA_DIR, "nse_rolling_60d.parquet")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
}

def fetch_nse_daily(target_date: datetime.date):
    """
    Downloads Bhavcopy and MTO (delivery) files for target_date.
    Returns (cleaned_df, True) if successful, or (None, False) if market was closed.
    """
    date_str = target_date.strftime("%d%m%Y")
    d_m_y = target_date.strftime("%d%b%Y").upper()  # e.g., 29OCT2026
    
    bhav_url = f"https://archives.nseindia.com/content/historical/EQUITIES/{target_date.year}/{target_date.strftime('%b').upper()}/cm{d_m_y}bhav.csv.zip"
    mto_url = f"https://archives.nseindia.com/archives/equities/mto/MTO_{date_str}.DAT"
    
    session = requests.Session()
    session.headers.update(HEADERS)
    
    # 1. Check Bhavcopy existence
    r_bhav = session.get(bhav_url, timeout=15)
    if r_bhav.status_code != 200:
        return None, False
        
    # 2. Check Delivery MTO file existence
    r_mto = session.get(mto_url, timeout=15)
    if r_mto.status_code != 200:
        return None, False

    try:
        # Parse Bhavcopy (all listed equities)
        df_bhav = pd.read_csv(bhav_url, compression="zip")
        # Strip potential trailing/leading whitespace in column names
        df_bhav.columns = [c.strip() for c in df_bhav.columns]
        
        # Filter for normal equity segment
        df_bhav = df_bhav[df_bhav["SERIES"] == "EQ"][["SYMBOL", "CLOSE", "HIGH", "LOW", "TOTTRDQTY"]].copy()
        df_bhav["SYMBOL"] = df_bhav["SYMBOL"].str.strip()
        
        # Parse Delivery MTO file
        lines = [line.strip().split(",") for line in r_mto.text.strip().split("\n") if line.startswith("20")]
        df_mto = pd.DataFrame(lines)[[2, 3, 5]]
        df_mto.columns = ["SYMBOL", "SERIES", "DELIV_QTY"]
        df_mto["SYMBOL"] = df_mto["SYMBOL"].str.strip()
        df_mto["SERIES"] = df_mto["SERIES"].str.strip()
        
        df_mto = df_mto[df_mto["SERIES"] == "EQ"]
        df_mto["DELIV_QTY"] = pd.to_numeric(df_mto["DELIV_QTY"], errors="coerce").fillna(0)
        
        # Merge pricing and deliverable volumes for all 2,000+ stocks
        merged = pd.merge(df_bhav, df_mto[["SYMBOL", "DELIV_QTY"]], on="SYMBOL", how="inner")
        merged["DATE"] = pd.to_datetime(target_date)
        return merged, True
    except Exception as e:
        print(f"Error parsing NSE files for {target_date}: {e}")
        return None, False


def update_rolling_history():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    master_df = None
    already_logged_dates = set()
    if os.path.exists(PARQUET_PATH):
        try:
            master_df = pd.read_parquet(PARQUET_PATH)
            master_df["DATE"] = pd.to_datetime(master_df["DATE"])
            already_logged_dates = set(master_df["DATE"].dt.date.unique())
        except Exception as e:
            print(f"Could not read existing parquet: {e}. Starting fresh.")

    today = datetime.date.today()
    found_data = None
    active_trading_date = None
    
    # Adaptive Search: checks today first, then steps back up to 7 calendar days
    # This automatically bypasses weekends and mid-week exchange holidays
    for lookback in range(0, 8):
        candidate_date = today - datetime.timedelta(days=lookback)
        
        # Skip Saturday (5) and Sunday (6)
        if candidate_date.weekday() >= 5:
            continue
            
        if candidate_date in already_logged_dates:
            print(f"Trading session {candidate_date} is already present in database. No update needed.")
            return

        print(f"Checking session availability for: {candidate_date}...")
        df_candidate, success = fetch_nse_daily(candidate_date)
        
        if success and df_candidate is not None and not df_candidate.empty:
            found_data = df_candidate
            active_trading_date = candidate_date
            print(f"Successfully downloaded active trading session: {candidate_date} ({len(found_data)} stocks)")
            break
        else:
            print(f"{candidate_date} was a market holiday or files are not yet posted.")

    if found_data is None:
        print("No new trading sessions found within the lookback window.")
        return

    # Combine with historical master
    if master_df is not None:
        combined_df = pd.concat([master_df, found_data], ignore_index=True)
    else:
        combined_df = found_data

    # Retain strictly the most recent 60 active trading dates
    unique_dates = sorted(combined_df["DATE"].drop_duplicates(), reverse=True)
    retained_dates = unique_dates[:60]
    final_df = combined_df[combined_df["DATE"].isin(retained_dates)].sort_values(["SYMBOL", "DATE"])
    
    # Save optimized Parquet file
    final_df.to_parquet(PARQUET_PATH, index=False)
    print(f"Database successfully updated. Retained {len(retained_dates)} sessions ({len(final_df)} records) in {PARQUET_PATH}.")


if __name__ == "__main__":
    update_rolling_history()
