"""Build cache/squad_context.json for PredKing.

The generator discovers teams from the imported football cache and merges
verified values from an optional squad_context_overrides.json file. It never
invents manager-tenure, retention, or transfer values: missing fields use the
backend's documented defaults and are marked with source=default.

Example:
  python build_squad_context.py --season 2026-2027 \
      --leagues EPL Championship \
      --overrides squad_context_overrides.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULTS = {
    "coach_years": 1,
    "starters_retained": 0,
    "new_signings": 0,
}
REQUIRED = tuple(DEFAULTS)


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def discover_teams(cache_dir: Path, leagues: list[str]) -> dict[str, list[str]]:
    candidates = [cache_dir / "football_data.json", cache_dir / "fd_league.json", cache_dir / "apif_all.json", cache_dir / "fd_last5.json"]
    data: dict[str, Any] = {}
    for path in candidates:
        loaded = read_json(path, {})
        if isinstance(loaded, dict) and any(loaded.get(league) for league in leagues):
            data = loaded
            break
    result: dict[str, list[str]] = {}
    for league in leagues:
        value = data.get(league, {}) if isinstance(data, dict) else {}
        if isinstance(value, dict):
            teams = list(value.keys())
        elif isinstance(value, list):
            teams = sorted({
                str(item.get("home_team")) for item in value if isinstance(item, dict) and item.get("home_team")
            } | {
                str(item.get("away_team")) for item in value if isinstance(item, dict) and item.get("away_team")
            })
        else:
            teams = []
        result[league] = sorted(teams)
    return result


def normalize_override(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, Any] = {}
    for key in REQUIRED:
        if key in value and value[key] is not None:
            try:
                number = int(value[key])
                if number >= 0:
                    out[key] = number
            except (TypeError, ValueError):
                pass
    for key in ("source", "confidence", "updated_at", "notes"):
        if value.get(key) is not None:
            out[key] = str(value[key])
    return out


def build_context(cache_dir: Path, season: str, leagues: list[str], overrides_path: Path | None) -> dict[str, Any]:
    discovered = discover_teams(cache_dir, leagues)
    overrides = read_json(overrides_path, {}) if overrides_path else {}
    if not isinstance(overrides, dict):
        overrides = {}
    # Accept either {season: {league: {team: {...}}}} or {league: {team: {...}}}.
    season_overrides = overrides.get(season, overrides)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    output: dict[str, Any] = {season: {}}
    for league in leagues:
        output[season][league] = {}
        league_overrides = season_overrides.get(league, {}) if isinstance(season_overrides, dict) else {}
        for team in discovered.get(league, []):
            item = {**DEFAULTS, "source": "default", "confidence": "low", "updated_at": now}
            override = normalize_override(league_overrides.get(team, {}) if isinstance(league_overrides, dict) else {})
            item.update(override)
            if all(key in override for key in REQUIRED):
                item.setdefault("source", "verified_override")
                item.setdefault("confidence", "medium")
            output[season][league][team] = item
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", required=True)
    parser.add_argument("--leagues", nargs="+", required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--overrides", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.cache_dir / "squad_context.json"
    context = build_context(args.cache_dir, args.season, args.leagues, args.overrides)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(json.dumps(context, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(output)
    counts = {league: len(context[args.season].get(league, {})) for league in args.leagues}
    print(f"Wrote {output}")
    print("Teams: " + ", ".join(f"{league}={count}" for league, count in counts.items()))
    print("Missing values are marked source=default; use --overrides for verified data.")


if __name__ == "__main__":
    main()


def _self_test() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); cache = root / "cache"; cache.mkdir()
        (cache / "football_data.json").write_text(json.dumps({"EPL": [{"home_team": "Arsenal", "away_team": "Chelsea"}], "Championship": []}))
        out = build_context(cache, "2026-2027", ["EPL", "Championship"], None)
        assert "Arsenal" in out["2026-2027"]["EPL"]
        assert out["2026-2027"]["EPL"]["Arsenal"]["source"] == "default"


if __name__ == "__main__":
    _self_test()
    main()
