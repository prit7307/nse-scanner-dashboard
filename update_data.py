import os
import requests
import datetime
import pandas as pd
import numpy as np

DATA_DIR = "data"
PARQUET_PATH = os.path.join(DATA_DIR, "nse_rolling_60d.parquet")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def fetch_nse_daily(target_date: datetime.date):
    date_str = target_date.strftime("%d%m%Y")
    d_m_y = target_date.strftime("%d%b%Y").upper()
    
    bhav_url = f"https://archives.nseindia.com/content/historical/EQUITIES/{target_date.year}/{target_date.strftime('%b').upper()}/cm{d_m_y}bhav.csv.zip"
    mto_url = f"https://archives.nseindia.com/archives/equities/mto/MTO_{date_str}.DAT"
    
    session = requests.Session()
    session.headers.update(HEADERS)
    
    r_bhav = session.get(bhav_url, timeout=15)
    if r_bhav.status_code != 200:
        return None, False
        
    r_mto = session.get(mto_url, timeout=15)
    if r_mto.status_code != 200:
        return None, False

    try:
        df_bhav = pd.read_csv(bhav_url, compression="zip")
        df_bhav.columns = [c.strip() for c in df_bhav.columns]
        df_bhav = df_bhav[df_bhav["SERIES"] == "EQ"][["SYMBOL", "CLOSE", "HIGH", "LOW", "TOTTRDQTY"]].copy()
        df_bhav["SYMBOL"] = df_bhav["SYMBOL"].str.strip()
        
        lines = [line.strip().split(",") for line in r_mto.text.strip().split("\n") if line.startswith("20")]
        df_mto = pd.DataFrame(lines)[[2, 3, 5]]
        df_mto.columns = ["SYMBOL", "SERIES", "DELIV_QTY"]
        df_mto["SYMBOL"] = df_mto["SYMBOL"].str.strip()
        df_mto["SERIES"] = df_mto["SERIES"].str.strip()
        df_mto = df_mto[df_mto["SERIES"] == "EQ"]
        df_mto["DELIV_QTY"] = pd.to_numeric(df_mto["DELIV_QTY"], errors="coerce").fillna(0)
        
        # Merge all equities
        merged = pd.merge(df_bhav, df_mto[["SYMBOL", "DELIV_QTY"]], on="SYMBOL", how="inner")
        merged["DATE"] = pd.to_datetime(target_date)
        return merged, True
    except Exception as e:
        print(f"Error: {e}")
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
        except Exception:
            pass

    today = datetime.date.today()
    found_data = None
    active_date = None
    
    for lookback in range(0, 8):
        candidate = today - datetime.timedelta(days=lookback)
        if candidate.weekday() >= 5:
            continue
        if candidate in already_logged_dates:
            print(f"Session {candidate} already present.")
            return

        df_candidate, success = fetch_nse_daily(candidate)
        if success and df_candidate is not None and not df_candidate.empty:
            found_data = df_candidate
            active_date = candidate
            break

    if found_data is None:
        return

    # Keep all listed equities without pruning
    if master_df is not None:
        combined_df = pd.concat([master_df, found_data], ignore_index=True)
    else:
        combined_df = found_data

    unique_dates = sorted(combined_df["DATE"].drop_duplicates(), reverse=True)
    retained_dates = unique_dates[:60]
    final_df = combined_df[combined_df["DATE"].isin(retained_dates)].sort_values(["SYMBOL", "DATE"])
    
    final_df.to_parquet(PARQUET_PATH, index=False)
    print(f"Updated with {len(final_df)} records across {len(retained_dates)} dates.")

if __name__ == "__main__":
    update_rolling_history()
