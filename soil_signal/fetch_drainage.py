# fetch_drainage.py (v2 — proper headers, retries, and a backup Overpass mirror)
import requests
import pandas as pd
import time
from zones import ZONES

RADIUS_M = 1000
OUTPUT_FILE = "lagos_drainage.csv"
MAX_RETRIES = 3

# Try the main server first, then fall back to a mirror if it keeps failing.
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

HEADERS = {"User-Agent": "SoilToSignal-FloodRiskProject/1.0 (personal research project)"}


def fetch_waterways(lat, lon, radius_m):
    query = f"""
    [out:json][timeout:25];
    (
      way["waterway"](around:{radius_m},{lat},{lon});
      way["natural"="water"](around:{radius_m},{lat},{lon});
      way["water"](around:{radius_m},{lat},{lon});
    );
    out center;
    """
    last_error = None
    for url in OVERPASS_URLS:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                r = requests.post(url, data={"data": query}, headers=HEADERS, timeout=60)
                r.raise_for_status()
                return r.json()["elements"]
            except Exception as e:
                last_error = e
                print(f"    {url} attempt {attempt}/{MAX_RETRIES} failed: {e}")
                time.sleep(3 * attempt)
    raise last_error


def main():
    rows = []
    failed_zones = []

    for zone, (lat, lon) in ZONES.items():
        print(f"Fetching drainage features for {zone}...")
        try:
            elements = fetch_waterways(lat, lon, RADIUS_M)
            rows.append({
                "zone": zone, "lat": lat, "lon": lon,
                "waterway_feature_count": len(elements),
            })
            print(f"  OK — {len(elements)} features found")
        except Exception as e:
            print(f"  GAVE UP on {zone}: {e}")
            rows.append({
                "zone": zone, "lat": lat, "lon": lon,
                "waterway_feature_count": None,
            })
            failed_zones.append(zone)
        time.sleep(3)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df)} rows to {OUTPUT_FILE}")

    if failed_zones:
        print(f"\nStill failed: {failed_zones}")
        print("If these keep failing even with a mirror, fetch them manually")
        print("at https://overpass-turbo.eu/ — paste the same query style,")
        print("run it for one zone at a time, and count features on the map.")
    else:
        print("\nAll zones succeeded.")


if __name__ == "__main__":
    main()