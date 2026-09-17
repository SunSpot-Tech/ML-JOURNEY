# fetch_rainfall.py (v2 — with retries and real error visibility)
import requests
import pandas as pd
import time
from zones import ZONES

START_DATE = "2019-01-01"
END_DATE = "2026-09-01"
OUTPUT_FILE = "lagos_rainfall.csv"
MAX_RETRIES = 3

HEADERS = {"User-Agent": "SoilToSignal-FloodRiskProject/1.0 (personal research project)"}


def fetch_rainfall(lat, lon, start, end):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": "precipitation_sum",
        "timezone": "Africa/Lagos",
    }
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_error = e
            print(f"    attempt {attempt}/{MAX_RETRIES} failed: {e}")
            time.sleep(3 * attempt)  # back off longer each retry
    raise last_error


def main():
    all_rows = []
    failed_zones = []

    for zone, (lat, lon) in ZONES.items():
        print(f"Fetching rainfall for {zone}...")
        try:
            data = fetch_rainfall(lat, lon, START_DATE, END_DATE)
            dates = data["daily"]["time"]
            precip = data["daily"]["precipitation_sum"]
            for d, p in zip(dates, precip):
                all_rows.append({"zone": zone, "date": d, "rainfall_mm": p})
            print(f"  OK — {len(dates)} days pulled")
        except Exception as e:
            print(f"  GAVE UP on {zone} after {MAX_RETRIES} attempts: {e}")
            failed_zones.append(zone)
        time.sleep(2)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df)} rows to {OUTPUT_FILE}")

    if failed_zones:
        print(f"\nStill failed after retries: {failed_zones}")
        print("Re-run the script — it will overwrite the file, so failed zones")
        print("just need another pass. If the same zones fail every time,")
        print("the issue is likely those specific coordinates, not the network.")
    else:
        print("\nAll zones succeeded.")


if __name__ == "__main__":
    main()