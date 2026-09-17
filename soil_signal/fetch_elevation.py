# fetch_elevation.py
# Downloads a single elevation value per zone centroid using Open-Meteo's
# free Elevation API. No API key, no signup. Good enough for a per-zone
# score. If you later want a full elevation raster for finer-grained terrain
# analysis, use OpenTopography instead (see fetch_elevation_dem.py below /
# the README for that path — it needs a free API key from
# https://portal.opentopography.org/).
#
# Docs: https://open-meteo.com/en/docs/elevation-api
#
# Install: pip install requests pandas

import requests
import pandas as pd
from zones import ZONES

OUTPUT_FILE = "lagos_elevation.csv"


def fetch_elevation(lat, lon):
    url = "https://api.open-meteo.com/v1/elevation"
    params = {"latitude": lat, "longitude": lon}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()["elevation"][0]


def main():
    rows = []
    for zone, (lat, lon) in ZONES.items():
        print(f"Fetching elevation for {zone}...")
        try:
            elev = fetch_elevation(lat, lon)
            rows.append({"zone": zone, "lat": lat, "lon": lon, "elevation_m": elev})
        except Exception as e:
            print(f"  Failed for {zone}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
