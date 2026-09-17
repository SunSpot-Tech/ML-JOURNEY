# ============================================================
# BabaPick — Cache Reader
# Reads pre-built cache files instead of live scraping
# Author: Powei Shadrack | SunSpot-Tech
# ============================================================

import os
import json
import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

# ── Default fallbacks ───────────────────────────────────────
def _default_log():
    return pd.DataFrame({
        "date": pd.date_range(end=pd.Timestamp.today(), periods=5, freq="7D"),
        "goals_scored":   [1, 1, 1, 1, 1],
        "goals_conceded": [1, 1, 1, 1, 1],
        "sot":            [4, 4, 4, 4, 4],
        "shots_off":      [3, 3, 3, 3, 3],
        "result":         ["D", "D", "D", "D", "D"],
    })

def _default_season_stats():
    return {
        "avg_goals": 1.2,
        "avg_conceded": 1.2,
        "clean_sheet_pct": 30.0,
        "penalty_conv_pct": 75.0,
    }

def _default_match_stats():
    return {
        "possession": 50.0,
        "pass_accuracy": 75.0,
        "avg_saves": 3.5,
        "avg_corners": 5.0,
    }

def _default_gk():
    return {
        "gk_save_pct": 70.0,
        "gk_cs_pct": 30.0,
        "gk_pen_save_pct": 20.0,
    }

def _default_xgs():
    return {
        "xg": None,
        "xga": None,
        "possession": 50.0,
        "pass_accuracy": 75.0,
    }

# ── Cache Loader ────────────────────────────────────────────
def _load_cache(filename):
    path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  Could not load cache file {filename}: {exc}")
    return None


def _load_first_available(*filenames):
    """Load the first valid cache file, allowing old and new cache names."""
    for filename in filenames:
        data = _load_cache(filename)
        if data is not None:
            return data
    return None

class CacheReader:
    """
    Reads pre-built cache files.
    Falls back to defaults if cache not available.
    This is what runs on Render where live scraping is blocked.
    """

    def __init__(self):
        # build_cache.py historically wrote fd_league.json; accept both names.
        self._fd    = _load_first_available("football_data.json", "fd_league.json")
        self._xgs   = _load_cache("xgscore.json")
        self._apif  = _load_cache("apif_all.json")
        self._meta  = _load_cache("meta.json")

        if self._meta:
            print(f"  Cache loaded. Built at: {self._meta.get('built_at', 'unknown')}")
        else:
            print("  No cache found. Will attempt live data fetch.")

    @property
    def has_cache(self):
        return self._fd is not None

    def fd_matches(self, league_name):
        if not self._fd:
            return []
        return self._fd.get(league_name, [])

    def xgs_team(self, league_name, team_name):
        if not self._xgs:
            return _default_xgs()
        league_data = self._xgs.get(league_name, {})
        if team_name in league_data:
            return league_data[team_name]
        for key in league_data:
            if team_name.lower() in key.lower() or key.lower() in team_name.lower():
                return league_data[key]
        return _default_xgs()

    def apif_team(self, league_name, team_name):
        if not self._apif:
            return None
        league_data = self._apif.get(league_name, {})
        if team_name in league_data:
            return league_data[team_name]
        for key in league_data:
            if team_name.lower() in key.lower() or key.lower() in team_name.lower():
                return league_data[key]
        return None

    def team_match_log(self, league_name, team_name):
        matches = self.fd_matches(league_name)
        if not matches:
            return _default_log()
        df = pd.DataFrame(matches)
        if "HomeTeam" not in df.columns:
            return _default_log()

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

        home = df[df["HomeTeam"] == team_name].copy()
        away = df[df["AwayTeam"] == team_name].copy()

        def safe_col(frame, col, default=np.nan):
            return pd.to_numeric(frame[col], errors="coerce") if col in frame.columns else pd.Series([default]*len(frame))

        home["goals_scored"]   = safe_col(home, "FTHG", 1)
        home["goals_conceded"] = safe_col(home, "FTAG", 1)
        home["sot"]            = safe_col(home, "HST",  4)
        home["shots_off"]      = safe_col(home, "HS",   7) - safe_col(home, "HST", 4)
        home["result"]         = home["FTR"].map({"H": "W", "D": "D", "A": "L"})
        home["date"]           = home["Date"]

        away["goals_scored"]   = safe_col(away, "FTAG", 1)
        away["goals_conceded"] = safe_col(away, "FTHG", 1)
        away["sot"]            = safe_col(away, "AST",  4)
        away["shots_off"]      = safe_col(away, "AS",   7) - safe_col(away, "AST", 4)
        away["result"]         = away["FTR"].map({"A": "W", "D": "D", "H": "L"})
        away["date"]           = away["Date"]

        cols = ["date", "goals_scored", "goals_conceded", "sot", "shots_off", "result"]
        log  = pd.concat([home[cols], away[cols]]).sort_values("date").reset_index(drop=True)
        log  = log.dropna(subset=["result"])

        return log if len(log) >= 5 else _default_log()

    def h2h(self, league_name, home_team, away_team, perspective_team, n=5):
        matches = self.fd_matches(league_name)
        if not matches:
            return ["D"] * n
        df   = pd.DataFrame(matches)
        if "HomeTeam" not in df.columns:
            return ["D"] * n
        mask = (
            ((df["HomeTeam"] == home_team) & (df["AwayTeam"] == away_team)) |
            ((df["HomeTeam"] == away_team) & (df["AwayTeam"] == home_team))
        )
        h2h     = df[mask].tail(n)
        results = []
        for _, row in h2h.iterrows():
            try:
                if row["HomeTeam"] == perspective_team:
                    results.append({"H": "W", "D": "D", "A": "L"}[row["FTR"]])
                else:
                    results.append({"A": "W", "D": "D", "H": "L"}[row["FTR"]])
            except Exception:
                results.append("D")
        while len(results) < n:
            results.insert(0, "D")
        return results

    def last_match_date(self, league_name, team_name, before_date):
        matches = self.fd_matches(league_name)
        if not matches:
            return pd.to_datetime(before_date) - pd.Timedelta(days=7)
        df = pd.DataFrame(matches)
        if "Date" not in df.columns:
            return pd.to_datetime(before_date) - pd.Timedelta(days=7)
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        mask = (
            (df["HomeTeam"] == team_name) | (df["AwayTeam"] == team_name)
        ) & (df["Date"] < pd.to_datetime(before_date))
        filtered = df[mask].sort_values("Date")
        if filtered.empty:
            return pd.to_datetime(before_date) - pd.Timedelta(days=7)
        return filtered.iloc[-1]["Date"]

    def league_avg_goals(self, league_name):
        matches = self.fd_matches(league_name)
        if not matches:
            return 1.35
        df = pd.DataFrame(matches)
        try:
            total = pd.to_numeric(df["FTHG"], errors="coerce").sum() + \
                    pd.to_numeric(df["FTAG"], errors="coerce").sum()
            return round(float(total) / max(len(df), 1), 4)
        except Exception:
            return 1.35

    def apif_match_stats(self, league_name, team_name):
        data = self.apif_team(league_name, team_name)
        if not data:
            return _default_match_stats()
        return {
            "possession":    data.get("possession", 50.0),
            "pass_accuracy": data.get("pass_accuracy", 75.0),
            "avg_saves":     data.get("avg_saves", 3.5),
            "avg_corners":   data.get("avg_corners", 5.0),
        }

    def apif_injuries(self, league_name, team_name):
        data = self.apif_team(league_name, team_name)
        return data.get("injuries", 0) if data else 0

    def apif_season_stats(self, league_name, team_name):
        data = self.apif_team(league_name, team_name)
        if not data:
            return _default_season_stats()
        return {
            "avg_goals":        data.get("avg_goals", 1.2),
            "avg_conceded":     data.get("avg_conceded", 1.2),
            "clean_sheet_pct":  data.get("clean_sheet_pct", 30.0),
            "penalty_conv_pct": data.get("penalty_conv_pct", 75.0),
        }

    def apif_gk_stats(self, league_name, team_name):
        s        = self.apif_season_stats(league_name, team_name)
        m5       = self.apif_match_stats(league_name, team_name)
        avg_faced = m5["avg_saves"] + s["avg_conceded"]
        save_pct  = round((m5["avg_saves"] / max(avg_faced, 1)) * 100, 2)
        return {
            "gk_save_pct":     save_pct,
            "gk_cs_pct":       s["clean_sheet_pct"],
            "gk_pen_save_pct": 20.0,
        }


# ── Singleton cache instance ────────────────────────────────
_cache = None

def get_cache():
    global _cache
    if _cache is None:
        _cache = CacheReader()
    return _cache