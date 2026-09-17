#!/usr/bin/env python3
"""Build PredKing-compatible caches from current FBref competition pages.

Usage:
    python fbref_cache_builder.py --season 2026-2027
    python fbref_cache_builder.py --leagues EPL La Liga --season 2026-2027

The builder writes football_data.json, fd_league.json, fd_last5.json,
apif_all.json, xgscore.json, and meta.json into ./cache. It deliberately
uses the same full-match schema expected by cache_reader.py.

FBref is an HTML site, not a stable JSON API. The parser therefore searches
for tables by semantic column names, handles tables embedded in HTML comments,
and fails per league rather than destroying a previously good cache.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

CACHE_DIR = Path(__file__).resolve().parent / "cache"
BASE_URL = "https://fbref.com/en"
USER_AGENT = os.getenv(
    "FBREF_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
)

# FBref competition IDs and URL slugs. The season is passed separately.
LEAGUES: dict[str, dict[str, Any]] = {
    "EPL": {"comp_id": "9", "name": "Premier-League", "country": "England"},
    "La Liga": {"comp_id": "12", "name": "La-Liga", "country": "Spain"},
    "Serie A": {"comp_id": "11", "name": "Serie-A", "country": "Italy"},
    "Bundesliga": {"comp_id": "20", "name": "Bundesliga", "country": "Germany"},
    "Ligue 1": {"comp_id": "13", "name": "Ligue-1", "country": "France"},
    "Eredivisie": {"comp_id": "23", "name": "Eredivisie", "country": "Netherlands"},
    "Liga Portugal": {"comp_id": "32", "name": "Primeira-Liga", "country": "Portugal"},
    "Scottish Prem": {"comp_id": "40", "name": "Scottish-Premiership", "country": "Scotland"},
    "Belgian Pro": {"comp_id": "37", "name": "Belgian-Pro-League", "country": "Belgium"},
    "Turkish SL": {"comp_id": "26", "name": "Super-Lig", "country": "Türkiye"},
    "Championship": {"comp_id": "10", "name": "Championship", "country": "England", "schedule_short": True},
}

DEFAULT_TEAM_STATS = {
    "possession": 50.0,
    "pass_accuracy": 75.0,
    "avg_saves": 3.5,
    "avg_corners": 5.0,
    "penalty_conv_pct": 75.0,
}


def current_season() -> str:
    """Return the football season containing today, e.g. 2026-2027."""
    today = date.today()
    start = today.year if today.month >= 7 else today.year - 1
    return f"{start}-{start + 1}"


def clean_number(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().replace(",", "").replace("%", "")
    if text in {"", "-", "—", "nan", "None"}:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else None


def clean_team(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return re.sub(r"\s+\([^)]*\)$", "", text)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = [
            "_".join(str(part) for part in col if str(part) != "nan").strip("_")
            for col in result.columns
        ]
    result.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in result.columns]
    return result


class FBrefClient:
    def __init__(self, delay: float = 5.0, retries: int = 3):
        self.delay = max(1.0, delay)
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Referer": "https://fbref.com/en/",
        })
        self.last_request = 0.0

    def get(self, url: str) -> str:
        wait = self.delay - (time.monotonic() - self.last_request)
        if wait > 0:
            time.sleep(wait)
        error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self.session.get(url, timeout=30)
                self.last_request = time.monotonic()
                if response.status_code == 403:
                    raise RuntimeError(
                        "FBref returned HTTP 403. The network/IP or request profile is blocked; "
                        "this is not a parsing error. Try the browser-like default profile, "
                        "a different permitted network, or an approved data provider."
                    )
                if response.status_code == 429:
                    raise RuntimeError("FBref returned HTTP 429 (rate limited); increase --delay")
                response.raise_for_status()
                if "captcha" in response.text[:5000].lower():
                    raise RuntimeError("FBref returned a CAPTCHA page")
                return response.text
            except Exception as exc:
                error = exc
                if attempt + 1 < self.retries:
                    time.sleep(2 ** attempt * 3)
        raise RuntimeError(f"Unable to fetch {url}: {error}")


def html_tables(html: str) -> list[pd.DataFrame]:
    """Read normal tables and tables hidden inside HTML comments."""
    fragments = [html]
    soup = BeautifulSoup(html, "html.parser")
    fragments.extend(str(node) for node in soup.find_all(string=lambda x: isinstance(x, Comment)))
    tables: list[pd.DataFrame] = []
    for fragment in fragments:
        try:
            tables.extend(normalize_columns(table) for table in pd.read_html(StringIO(fragment)))
        except (ValueError, ImportError):
            continue
    return tables


def find_table(tables: list[pd.DataFrame], required: set[str], preferred: tuple[str, ...] = ()) -> pd.DataFrame | None:
    for table in tables:
        columns = {str(c).lower() for c in table.columns}
        if required.issubset(columns):
            if not preferred or any(any(p in c for c in columns) for p in preferred):
                return table
    for table in tables:
        columns = {str(c).lower() for c in table.columns}
        if required.issubset(columns):
            return table
    return None


def parse_schedule(html: str) -> list[dict[str, Any]]:
    tables = html_tables(html)
    table = find_table(tables, {"date", "home", "away", "score"})
    if table is None:
        raise RuntimeError("FBref schedule table was not found")

    records: list[dict[str, Any]] = []
    for _, row in table.iterrows():
        home, away = clean_team(row.get("Home")), clean_team(row.get("Away"))
        score = str(row.get("Score", "")).strip()
        match = re.search(r"(\d+)\s*[–-]\s*(\d+)", score)
        if not home or not away or not match:
            continue  # upcoming/unplayed fixture
        home_goals, away_goals = int(match.group(1)), int(match.group(2))
        result = "H" if home_goals > away_goals else "A" if home_goals < away_goals else "D"
        dt = pd.to_datetime(row.get("Date"), errors="coerce")
        records.append({
            "Date": dt.strftime("%d/%m/%Y") if not pd.isna(dt) else str(row.get("Date", "")),
            "HomeTeam": home,
            "AwayTeam": away,
            "FTHG": home_goals,
            "FTAG": away_goals,
            "FTR": result,
            "HS": None,
            "AS": None,
            "HST": None,
            "AST": None,
            "Referee": clean_team(row.get("Referee", "")),
            "Source": "FBref",
        })
    return records


def table_value(row: pd.Series, patterns: tuple[str, ...]) -> float | None:
    """Return the first numeric value from a column whose name matches patterns."""
    for column in row.index:
        label = str(column).lower().replace(" ", "")
        if all(pattern.lower().replace(" ", "") in label for pattern in patterns):
            value = clean_number(row[column])
            if value is not None:
                return value
    return None


def parse_fbref_team_tables(html: str) -> dict[str, dict[str, float]]:
    """Extract team-level values from FBref aggregate tables when available."""
    merged: dict[str, dict[str, float]] = {}
    for table in html_tables(html):
        team_col = next((c for c in table.columns if str(c).lower() in {"squad", "team"}), None)
        if team_col is None:
            continue
        columns = " ".join(str(c).lower() for c in table.columns)
        has_possession = "poss" in columns
        has_passing = "cmp%" in columns or "pass" in columns
        has_shooting = "sot" in columns or "g/sh" in columns
        has_keepers = "save%" in columns or "ga90" in columns
        for _, row in table.iterrows():
            team = clean_team(row.get(team_col))
            if not team or team.lower() in {"squad", "league total", "average"}:
                continue
            values = merged.setdefault(team, {})
            if has_possession:
                value = table_value(row, ("poss",))
                if value is not None:
                    values["possession"] = value
            if has_passing:
                value = table_value(row, ("cmp%",))
                if value is not None:
                    values["pass_accuracy"] = value
            if has_shooting:
                value = table_value(row, ("sot",))
                if value is not None:
                    values["sot_per90"] = value
            if has_keepers:
                value = table_value(row, ("save%",))
                if value is not None:
                    values["gk_save_pct"] = value
    return merged


def build_team_stats(matches: list[dict[str, Any]], fbref_stats: dict[str, dict[str, float]] | None = None) -> dict[str, dict[str, Any]]:
    fbref_stats = fbref_stats or {}
    if not matches:
        return {}
    frame = pd.DataFrame(matches)
    teams = sorted(set(frame["HomeTeam"]) | set(frame["AwayTeam"]))
    result: dict[str, dict[str, Any]] = {}
    league_goals = (frame["FTHG"].sum() + frame["FTAG"].sum()) / max(len(frame) * 2, 1)
    for team in teams:
        home = frame[frame["HomeTeam"] == team]
        away = frame[frame["AwayTeam"] == team]
        goals_for = pd.concat([home["FTHG"], away["FTAG"]])
        goals_against = pd.concat([home["FTAG"], away["FTHG"]])
        clean_sheets = int((goals_against == 0).sum())
        played = len(goals_for)
        result[team] = {
            "name": team,
            "avg_goals": round(float(goals_for.mean()), 4) if played else 1.2,
            "avg_conceded": round(float(goals_against.mean()), 4) if played else 1.2,
            "clean_sheet_pct": round(clean_sheets / max(played, 1) * 100, 2),
            "penalty_conv_pct": DEFAULT_TEAM_STATS["penalty_conv_pct"],
            "possession": round(fbref_stats.get(team, {}).get("possession", DEFAULT_TEAM_STATS["possession"]), 2),
            "pass_accuracy": round(fbref_stats.get(team, {}).get("pass_accuracy", DEFAULT_TEAM_STATS["pass_accuracy"]), 2),
            "avg_saves": DEFAULT_TEAM_STATS["avg_saves"],
            "avg_corners": DEFAULT_TEAM_STATS["avg_corners"],
            "gk_save_pct": round(fbref_stats.get(team, {}).get("gk_save_pct", 70.0), 2),
            "sot_per90": round(fbref_stats.get(team, {}).get("sot_per90", 0.0), 2),
            "injuries": 0,
            "xg": round(float(goals_for.mean()), 4) if played else 1.2,
            "xga": round(float(goals_against.mean()), 4) if played else 1.2,
            "source": "FBref results; possession/pass/saves/corners unavailable from schedule fallback",
            "league_avg_goals": round(float(league_goals), 4),
        }
    return result


def last5_cache(matches: list[dict[str, Any]], stats: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    frame = pd.DataFrame(matches)
    if frame.empty:
        return {team: [] for team in stats}
    frame["_date"] = pd.to_datetime(frame["Date"], dayfirst=True, errors="coerce")
    output: dict[str, list[dict[str, Any]]] = {}
    for team in stats:
        home = frame[frame["HomeTeam"] == team].copy()
        away = frame[frame["AwayTeam"] == team].copy()
        home["goals_scored"], home["goals_conceded"] = home["FTHG"], home["FTAG"]
        away["goals_scored"], away["goals_conceded"] = away["FTAG"], away["FTHG"]
        home["result"] = home["FTR"].map({"H": "W", "D": "D", "A": "L"})
        away["result"] = away["FTR"].map({"A": "W", "D": "D", "H": "L"})
        combined = pd.concat([home, away]).sort_values("_date").tail(5)
        output[team] = [
            {
                "date": row["_date"].strftime("%Y-%m-%d") if not pd.isna(row["_date"]) else "",
                "goals_scored": float(row["goals_scored"]),
                "goals_conceded": float(row["goals_conceded"]),
                "sot": 4.0,
                "shots_off": 3.0,
                "result": str(row["result"]),
            }
            for _, row in combined.iterrows()
            if str(row.get("result", "")) in {"W", "D", "L"}
        ]
    return output


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, default=str)
            handle.write("\n")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def build(selected: list[str], season: str, delay: float) -> None:
    client = FBrefClient(delay=delay)
    old_matches: dict[str, list[dict[str, Any]]] = {}
    old_stats: dict[str, dict[str, Any]] = {}
    for league in selected:
        try:
            cfg = LEAGUES[league]
            slug = cfg["name"]
            comp_id = cfg["comp_id"]
            if cfg.get("schedule_short"):
                schedule_url = f"{BASE_URL}/comps/{comp_id}/schedule/{slug}-Scores-and-Fixtures"
            else:
                schedule_url = f"{BASE_URL}/comps/{comp_id}/{season}/schedule/{season}-{slug}-Scores-and-Fixtures"
            stats_url = f"{BASE_URL}/comps/{comp_id}/{season}/{season}-{slug}-Stats"
            print(f"Fetching {league}: {schedule_url}")
            schedule_html = client.get(schedule_url)
            matches = parse_schedule(schedule_html)
            if not matches:
                raise RuntimeError("No completed matches found")
            fbref_stats: dict[str, dict[str, float]] = {}
            try:
                print(f"  Fetching team tables: {stats_url}")
                fbref_stats = parse_fbref_team_tables(client.get(stats_url))
                print(f"  parsed FBref team-stat rows: {len(fbref_stats)}")
            except Exception as stats_exc:
                print(f"  team tables unavailable; retaining explicit defaults: {stats_exc}")
            stats = build_team_stats(matches, fbref_stats)
            old_matches[league] = matches
            old_stats[league] = stats
            print(f"  loaded {len(matches)} completed matches and {len(stats)} teams")
        except Exception as exc:
            print(f"  FAILED {league}: {exc}")
            # Preserve an existing league cache instead of replacing good data with []
            previous_matches = {}
            previous_stats = {}
            try:
                previous_matches = json.loads((CACHE_DIR / "football_data.json").read_text()).get(league, [])
                previous_stats = json.loads((CACHE_DIR / "apif_all.json").read_text()).get(league, {})
            except Exception:
                pass
            old_matches[league] = previous_matches
            old_stats[league] = previous_stats

    if not any(old_matches.values()):
        raise RuntimeError("No league data was fetched and no previous cache was available")

    fd_last5 = {league: last5_cache(old_matches[league], old_stats[league]) for league in selected}
    xgscore = {
        league: {
            team: {"xg": data.get("xg"), "xga": data.get("xga"),
                   "possession": data.get("possession", 50.0),
                   "pass_accuracy": data.get("pass_accuracy", 75.0)}
            for team, data in old_stats[league].items()
        }
        for league in selected
    }
    meta = {
        "built_at": datetime.utcnow().isoformat() + "Z",
        "season": season,
        "source": "FBref",
        "leagues": selected,
        "fd_matches": {league: len(old_matches[league]) for league in selected},
        "teams": {league: len(old_stats[league]) for league in selected},
        "notes": "Results/form are from FBref. Fields unavailable on schedule pages use explicit defaults.",
    }
    atomic_json(CACHE_DIR / "football_data.json", old_matches)
    atomic_json(CACHE_DIR / "fd_league.json", old_matches)  # legacy compatibility
    atomic_json(CACHE_DIR / "fd_last5.json", fd_last5)
    atomic_json(CACHE_DIR / "apif_all.json", old_stats)
    atomic_json(CACHE_DIR / "xgscore.json", xgscore)
    atomic_json(CACHE_DIR / "meta.json", meta)
    print(f"Cache build complete: {CACHE_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default=current_season(), help="FBref season, e.g. 2026-2027")
    parser.add_argument("--leagues", nargs="+", choices=sorted(LEAGUES), default=list(LEAGUES))
    parser.add_argument("--delay", type=float, default=float(os.getenv("FBREF_DELAY_SECONDS", "5")))
    parser.add_argument("--user-agent", default=None, help="Optional permitted browser user-agent override")
    args = parser.parse_args()
    if args.user_agent:
        global USER_AGENT
        USER_AGENT = args.user_agent
    if not re.fullmatch(r"\d{4}-\d{4}", args.season):
        parser.error("--season must have the form YYYY-YYYY")
    build(args.leagues, args.season, args.delay)


if __name__ == "__main__":
    main()
