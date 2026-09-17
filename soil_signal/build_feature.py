# build_feature_table.py
# Summarizes the daily rainfall file into per-zone stats, then merges it
# with elevation and drainage into a single feature table: one row per zone.
#
# Install: pip install pandas

import pandas as pd

RAINFALL_FILE = "lagos_rainfall.csv"
ELEVATION_FILE = "lagos_elevation.csv"
DRAINAGE_FILE = "lagos_drainage.csv"
LAGOS_FLOOD_LABELS = "lagos_flood_labels.csv"
OUTPUT_FILE = "lagos_features.csv"

HEAVY_RAIN_THRESHOLD_MM = 20
RAINY_SEASON_MONTHS = [4, 5, 6, 7, 8, 9, 10]  # April to October


def summarize_rainfall(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df["month"] = df["date"].dt.month

    summary = df.groupby("zone").agg(
        avg_daily_rainfall_mm=("rainfall_mm", "mean"),
        max_daily_rainfall_mm=("rainfall_mm", "max"),
        heavy_rain_days=("rainfall_mm", lambda x: (x >= HEAVY_RAIN_THRESHOLD_MM).sum()),
    ).reset_index()

    rainy_season = df[df["month"].isin(RAINY_SEASON_MONTHS)]
    rainy_avg = rainy_season.groupby("zone")["rainfall_mm"].mean().reset_index()
    rainy_avg.columns = ["zone", "rainy_season_avg_mm"]

    return summary.merge(rainy_avg, on="zone", how="left")


def main():
    rainfall_summary = summarize_rainfall(RAINFALL_FILE)
    elevation = pd.read_csv(ELEVATION_FILE)[["zone", "elevation_m"]]
    drainage = pd.read_csv(DRAINAGE_FILE)[["zone", "waterway_feature_count"]]
    flood_labels = pd.read_csv(LAGOS_FLOOD_LABELS)[["zone", "flood_label"]]
    features = rainfall_summary.merge(elevation, on="zone", how="left")
    features = features.merge(drainage, on="zone", how="left")
    features = features.merge(flood_labels, on="zone", how="left")

    features.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(features)} zones with {len(features.columns)} columns to {OUTPUT_FILE}")
    print(features)


if __name__ == "__main__":
    main()