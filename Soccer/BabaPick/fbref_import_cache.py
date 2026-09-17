#!/usr/bin/env python3
"""Import FBref HTML saved manually from a normal browser into PredKing caches.

This is a compliant fallback for environments where direct automated FBref
requests return HTTP 403. It does not log in, solve CAPTCHAs, or bypass access
controls; the user must save pages they are allowed to view in a browser.

Example:
    python fbref_import_cache.py --league EPL --season 2026-2027 \
      --schedule-html EPL_schedule.html --stats-html EPL_stats.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fbref_cache_builder import (
    CACHE_DIR,
    LEAGUES,
    build_team_stats,
    last5_cache,
    parse_fbref_team_tables,
    parse_schedule,
    atomic_json,
)


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def import_league(league: str, season: str, schedule_path: Path, stats_path: Path | None) -> None:
    schedule_html = schedule_path.read_text(encoding="utf-8", errors="replace")
    matches = parse_schedule(schedule_html)
    if not matches:
        raise RuntimeError("No completed matches were found in the saved schedule HTML")

    fbref_stats = {}
    if stats_path:
        stats_html = stats_path.read_text(encoding="utf-8", errors="replace")
        fbref_stats = parse_fbref_team_tables(stats_html)

    team_stats = build_team_stats(matches, fbref_stats)
    team_last5 = last5_cache(matches, team_stats)

    all_matches = load_json(CACHE_DIR / "football_data.json", {})
    all_stats = load_json(CACHE_DIR / "apif_all.json", {})
    all_last5 = load_json(CACHE_DIR / "fd_last5.json", {})
    all_xgs = load_json(CACHE_DIR / "xgscore.json", {})

    all_matches[league] = matches
    all_stats[league] = team_stats
    all_last5[league] = team_last5
    all_xgs[league] = {
        team: {
            "xg": data.get("xg"),
            "xga": data.get("xga"),
            "possession": data.get("possession", 50.0),
            "pass_accuracy": data.get("pass_accuracy", 75.0),
        }
        for team, data in team_stats.items()
    }

    meta = load_json(CACHE_DIR / "meta.json", {})
    meta.update({
        "season": season,
        "source": "FBref browser-export import",
        "fd_matches": {key: len(value) for key, value in all_matches.items()},
        "teams": {key: len(value) for key, value in all_stats.items()},
        "last_imported_league": league,
    })

    # Write all related files atomically. Legacy aliases are retained.
    atomic_json(CACHE_DIR / "football_data.json", all_matches)
    atomic_json(CACHE_DIR / "fd_league.json", all_matches)
    atomic_json(CACHE_DIR / "fd_last5.json", all_last5)
    atomic_json(CACHE_DIR / "apif_all.json", all_stats)
    atomic_json(CACHE_DIR / "xgscore.json", all_xgs)
    atomic_json(CACHE_DIR / "meta.json", meta)
    print(f"Imported {league}: {len(matches)} completed matches, {len(team_stats)} teams")
    print(f"Cache updated: {CACHE_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", required=True, choices=sorted(LEAGUES))
    parser.add_argument("--season", required=True, help="Season in YYYY-YYYY format")
    parser.add_argument("--schedule-html", required=True, type=Path)
    parser.add_argument("--stats-html", type=Path, default=None)
    args = parser.parse_args()
    if not args.schedule_html.exists():
        parser.error(f"Schedule HTML not found: {args.schedule_html}")
    if args.stats_html and not args.stats_html.exists():
        parser.error(f"Stats HTML not found: {args.stats_html}")
    import_league(args.league, args.season, args.schedule_html, args.stats_html)


if __name__ == "__main__":
    main()
