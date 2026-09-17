# ============================================================
# BabaPick — Local Cache Builder (2025/26 Season)
# Fetches last 5 matches per team and season stats
# Author: Powei Shadrack | SunSpot-Tech
# ============================================================
#
# USAGE:
#   python build_cache.py
#   git add cache/
#   git commit -m "Add 2025/26 data cache"
#   git push
# ============================================================

import os
import json
import time
import requests
import numpy as np
import pandas as pd
import urllib3
from io import StringIO
from bs4 import BeautifulSoup
from datetime import datetime

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

CACHE_DIR        = "cache"
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "da4fbeeb4b173da6bff493b5aca8d691")
os.makedirs(CACHE_DIR, exist_ok=True)

# ── League config ─────────────────────────────────────────────
# (fd_code, fd_season_folder, xgs_slug, apif_league_id, apif_season_year)
LEAGUES = {
    "EPL":          ("E0",  "2526", "epl",          39,  2025),
    "La Liga":      ("SP1", "2526", "la-liga",      140, 2025),
    "Serie A":      ("I1",  "2526", "serie-a",      135, 2025),
    "Bundesliga":   ("D1",  "2526", "bundesliga",   78,  2025),
    "Ligue 1":      ("F1",  "2526", "ligue-1",      61,  2025),
    "Eredivisie":   ("N1",  "2526", "eredivisie",   88,  2025),
    "Liga Portugal":("P1",  "2526", "liga-portugal",94,  2025),
    "Scottish Prem":("SC0", "2526", None,           179, 2025),
    "Belgian Pro":  ("B1",  "2526", None,           144, 2025),
    "Turkish SL":   ("T1",  "2526", None,           203, 2025),
    "Championship": ("E1",  "2526", None,           40,  2025),
    "Greek SL":     ("G1",  "2526", None,           197, 2025),
}

APIF_HEADERS = {
    "x-rapidapi-host": "v3.football.api-sports.io",
    "x-rapidapi-key":  API_FOOTBALL_KEY,
}

FD_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.5",
    "Referer":         "https://www.football-data.co.uk/englandm.php",
}

XGS_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

PROMOTED_DEFAULTS = {
    "avg_goals": 1.10, "avg_conceded": 1.45,
    "clean_sheet_pct": 20.0, "penalty_conv_pct": 75.0,
    "possession": 46.0, "pass_accuracy": 74.0,
    "avg_saves": 3.8, "avg_corners": 4.5,
    "xg": 1.10, "xga": 1.45,
}


def save(filename, data):
    path = os.path.join(CACHE_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  ✓ Saved → {path}")


def apif_get(endpoint, params=None):
    try:
        r = requests.get(
            f"https://v3.football.api-sports.io/{endpoint}",
            headers=APIF_HEADERS,
            params=params,
            timeout=15,
            verify=False,
        )
        if r.status_code == 200 and r.text.strip():
            data = r.json()
            errors = data.get("errors", {})
            if not errors:
                return data.get("response", [])
            print(f"    API-Football errors: {errors}")
    except Exception as e:
        print(f"    API-Football error ({endpoint}): {e}")
    return []


def fetch_fd_csv(fd_code, fd_folder):
    """
    Tries multiple URL patterns for football-data.co.uk.
    Returns a DataFrame or empty DataFrame.
    """
    urls = [
        f"https://www.football-data.co.uk/mmz4281/{fd_folder}/{fd_code}.csv",
        f"https://football-data.co.uk/mmz4281/{fd_folder}/{fd_code}.csv",
    ]
    for url in urls:
        try:
            r = requests.get(
                url,
                headers=FD_HEADERS,
                timeout=20,
                allow_redirects=True,
                verify=False,
            )
            if r.status_code == 200 and r.text.strip():
                # Check it is actually CSV not HTML
                if "<html" in r.text[:200].lower():
                    continue
                df = pd.read_csv(StringIO(r.text), on_bad_lines="skip")
                df.columns = [c.strip() for c in df.columns]
                if "HomeTeam" in df.columns and len(df) > 0:
                    return df
        except Exception as e:
            print(f"    URL failed ({url}): {e}")
    return pd.DataFrame()


def last5_from_df(df, team):
    """Extract last 5 matches for a team from a DataFrame."""
    if df.empty or "HomeTeam" not in df.columns:
        return []

    if "Date" in df.columns:
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    def sc(frame, col, default=1.0):
        if col not in frame.columns:
            return pd.Series([default] * len(frame), index=frame.index)
        return pd.to_numeric(frame[col], errors="coerce").fillna(default)

    home = df[df["HomeTeam"] == team].copy()
    away = df[df["AwayTeam"] == team].copy()

    home["goals_scored"]   = sc(home, "FTHG")
    home["goals_conceded"] = sc(home, "FTAG")
    home["sot"]            = sc(home, "HST", 4.0)
    home["shots_off"]      = sc(home, "HS", 7.0) - sc(home, "HST", 4.0)
    home["result"]         = home["FTR"].map({"H": "W", "D": "D", "A": "L"})
    home["date"]           = home["Date"] if "Date" in home.columns else pd.NaT

    away["goals_scored"]   = sc(away, "FTAG")
    away["goals_conceded"] = sc(away, "FTHG")
    away["sot"]            = sc(away, "AST", 4.0)
    away["shots_off"]      = sc(away, "AS", 7.0) - sc(away, "AST", 4.0)
    away["result"]         = away["FTR"].map({"A": "W", "D": "D", "H": "L"})
    away["date"]           = away["Date"] if "Date" in away.columns else pd.NaT

    cols     = ["date", "goals_scored", "goals_conceded", "sot", "shots_off", "result"]
    combined = pd.concat([home[cols], away[cols]]).sort_values("date")
    combined = combined.dropna(subset=["result"]).tail(5)

    records = []
    for _, row in combined.iterrows():
        records.append({
            "date":           str(row["date"].date()) if pd.notna(row.get("date")) else "",
            "goals_scored":   float(row["goals_scored"]),
            "goals_conceded": float(row["goals_conceded"]),
            "sot":            float(row["sot"]),
            "shots_off":      max(0.0, float(row["shots_off"])),
            "result":         str(row["result"]),
        })
    return records


# ============================================================
# STEP 1 — football-data.co.uk
# ============================================================
print("\n" + "="*60)
print("STEP 1 — football-data.co.uk (last 5 matches per team)")
print("="*60)

fd_last5   = {}
fd_league  = {}
fd_raw_dfs = {}

for league_name, (fd_code, fd_folder, xgs_slug, apif_id, apif_season) in LEAGUES.items():
    print(f"\n  {league_name} ({fd_code}, folder={fd_folder})...")
    df = fetch_fd_csv(fd_code, fd_folder)

    if df.empty:
        print(f"    No data retrieved.")
        fd_last5[league_name]  = {}
        fd_league[league_name] = []
        fd_raw_dfs[league_name] = pd.DataFrame()
        continue

    # Store full records for H2H and league average
    fd_league[league_name]  = df.to_dict(orient="records")
    fd_raw_dfs[league_name] = df

    # Extract teams and their last 5 matches
    teams = set()
    if "HomeTeam" in df.columns:
        teams = set(df["HomeTeam"].dropna().unique()) | set(df["AwayTeam"].dropna().unique())

    team_last5 = {}
    for team in sorted(teams):
        matches = last5_from_df(df, team)
        if matches:
            team_last5[team] = matches

    fd_last5[league_name] = team_last5
    print(f"    ✓ {len(team_last5)} teams, {len(df)} matches loaded")
    time.sleep(1.5)

save("fd_last5.json",  fd_last5)
# The application reads football_data.json; keep fd_league.json as a legacy alias.
save("football_data.json", fd_league)
save("fd_league.json", fd_league)


# ============================================================
# STEP 2 — xGscore.io
# ============================================================
print("\n" + "="*60)
print("STEP 2 — xGscore.io (xG, xGA, possession, pass accuracy)")
print("="*60)

xgs_cache = {}
for league_name, (fd_code, fd_folder, xgs_slug, apif_id, apif_season) in LEAGUES.items():
    if not xgs_slug:
        print(f"  {league_name}: no xGscore slug — skipping")
        xgs_cache[league_name] = {}
        continue
    try:
        url  = f"https://xgscore.io/xg-statistics/{xgs_slug}"
        r    = requests.get(url, headers=XGS_HEADERS, timeout=15)
        data = {}
        if r.status_code == 200 and r.text.strip():
            soup = BeautifulSoup(r.text, "html.parser")
            for table in soup.find_all("table"):
                for row in table.find_all("tr"):
                    cols = [td.get_text(strip=True) for td in row.find_all("td")]
                    if len(cols) >= 5:
                        try:
                            data[cols[0].strip()] = {
                                "xg":            float(cols[2]),
                                "xga":           float(cols[3]),
                                "possession":    float(cols[4].replace("%", "")),
                                "pass_accuracy": float(cols[5].replace("%", "")) if len(cols) > 5 else 75.0,
                            }
                        except Exception:
                            continue
        xgs_cache[league_name] = data
        status = f"{len(data)} teams cached" if data else "0 teams (blocked or no data)"
        print(f"  {league_name}: {status}")
        time.sleep(2)
    except Exception as e:
        print(f"  {league_name}: error — {e}")
        xgs_cache[league_name] = {}

save("xgscore.json", xgs_cache)


# ============================================================
# STEP 3 — API-Football season stats + last 5 fixture stats
# ============================================================
print("\n" + "="*60)
print("STEP 3 — API-Football (season stats + possession + saves)")
print("="*60)

apif_cache = {}

for league_name, (fd_code, fd_folder, xgs_slug, apif_id, apif_season) in LEAGUES.items():
    print(f"\n  {league_name} (league_id={apif_id}, season={apif_season})...")
    league_data = {}

    teams_resp = apif_get("teams", {"league": apif_id, "season": apif_season})
    time.sleep(0.5)

    if not teams_resp:
        print(f"    No teams returned from API-Football.")
        apif_cache[league_name] = {}
        continue

    for team_entry in teams_resp:
        team      = team_entry.get("team", {})
        team_id   = team.get("id")
        team_name = team.get("name")
        if not team_id or not team_name:
            continue

        print(f"    {team_name}...")
        team_data = {"id": team_id, "name": team_name}

        # ── Season stats ─────────────────────────────────────
        stats_resp = apif_get("teams/statistics", {
            "team": team_id, "league": apif_id, "season": apif_season
        })
        time.sleep(0.4)

        if stats_resp:
            s = stats_resp
            try:
                team_data["avg_goals"]       = float(s.get("goals", {}).get("for", {}).get("average", {}).get("total", 1.2) or 1.2)
                team_data["avg_conceded"]    = float(s.get("goals", {}).get("against", {}).get("average", {}).get("total", 1.2) or 1.2)
                cs = s.get("clean_sheet", {}).get("total", 0)
                mp = s.get("fixtures", {}).get("played", {}).get("total", 1)
                team_data["clean_sheet_pct"] = round((cs / max(mp, 1)) * 100, 2)
                ps = s.get("penalty", {}).get("scored", {}).get("total", 0)
                pm = s.get("penalty", {}).get("missed", {}).get("total", 0)
                team_data["penalty_conv_pct"] = round((ps / max(ps + pm, 1)) * 100, 2)
            except Exception:
                team_data.update({k: PROMOTED_DEFAULTS[k] for k in ["avg_goals","avg_conceded","clean_sheet_pct","penalty_conv_pct"]})
        else:
            team_data.update({k: PROMOTED_DEFAULTS[k] for k in ["avg_goals","avg_conceded","clean_sheet_pct","penalty_conv_pct"]})

        # ── Last 5 fixture stats ──────────────────────────────
        fix_resp = apif_get("fixtures", {
            "team": team_id, "league": apif_id,
            "season": apif_season, "status": "FT", "last": 5
        })
        time.sleep(0.4)

        poss, pacc, saves, corners = [], [], [], []
        for fix in fix_resp:
            fid = fix.get("fixture", {}).get("id")
            if not fid:
                continue
            sd = apif_get("fixtures/statistics", {"fixture": fid, "team": team_id})
            time.sleep(0.3)
            for ts in sd:
                if ts.get("team", {}).get("id") == team_id:
                    for s in ts.get("statistics", []):
                        t, val = s.get("type", ""), s.get("value")
                        try:
                            if t == "Ball Possession"   and val: poss.append(float(str(val).replace("%", "")))
                            elif t == "Passes %"        and val: pacc.append(float(str(val).replace("%", "")))
                            elif t == "Goalkeeper Saves"and val: saves.append(int(val))
                            elif t == "Corner Kicks"    and val: corners.append(int(val))
                        except Exception:
                            continue

        team_data["possession"]    = round(float(np.mean(poss))    if poss    else PROMOTED_DEFAULTS["possession"],    2)
        team_data["pass_accuracy"] = round(float(np.mean(pacc))    if pacc    else PROMOTED_DEFAULTS["pass_accuracy"], 2)
        team_data["avg_saves"]     = round(float(np.mean(saves))   if saves   else PROMOTED_DEFAULTS["avg_saves"],    2)
        team_data["avg_corners"]   = round(float(np.mean(corners)) if corners else PROMOTED_DEFAULTS["avg_corners"],  2)

        # ── Injury count ──────────────────────────────────────
        inj = apif_get("injuries", {"team": team_id, "season": apif_season})
        time.sleep(0.3)
        team_data["injuries"] = len(inj)

        # ── Merge xGscore data ────────────────────────────────
        xgs_data = xgs_cache.get(league_name, {})
        matched  = xgs_data.get(team_name)
        if not matched:
            for k in xgs_data:
                if team_name.lower() in k.lower() or k.lower() in team_name.lower():
                    matched = xgs_data[k]
                    break
        if matched:
            team_data["xg"]            = matched.get("xg",           PROMOTED_DEFAULTS["xg"])
            team_data["xga"]           = matched.get("xga",          PROMOTED_DEFAULTS["xga"])
            team_data["possession"]    = matched.get("possession",    team_data["possession"])
            team_data["pass_accuracy"] = matched.get("pass_accuracy", team_data["pass_accuracy"])
        else:
            team_data["xg"]  = PROMOTED_DEFAULTS["xg"]
            team_data["xga"] = PROMOTED_DEFAULTS["xga"]

        league_data[team_name] = team_data

    apif_cache[league_name] = league_data
    safe_name = league_name.lower().replace(" ", "_")
    save(f"apif_{safe_name}.json", league_data)
    print(f"  → {len(league_data)} teams cached for {league_name}")

save("apif_all.json", apif_cache)


# ============================================================
# STEP 4 — Meta
# ============================================================
meta = {
    "built_at":    datetime.utcnow().isoformat(),
    "season":      "2025/26",
    "fd_season":   "2526",
    "leagues":     list(LEAGUES.keys()),
    "fd_teams":    {k: list(v.keys()) for k, v in fd_last5.items()},
    "xgs_teams":   {k: list(v.keys()) for k, v in xgs_cache.items()},
    "apif_teams":  {k: list(v.keys()) for k, v in apif_cache.items()},
    "fd_matches":  {k: len(v) for k, v in fd_league.items()},
}
save("meta.json", meta)

print("\n" + "="*60)
print("Cache build complete.")
print(f"Files saved to: {os.path.abspath(CACHE_DIR)}/")
print("\nNext steps:")
print("  git add cache/")
print("  git commit -m 'Add 2025/26 data cache'")
print("  git push")
print("="*60)