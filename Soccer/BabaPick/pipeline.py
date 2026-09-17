# ============================================================
# BabaPick — Feature Pipeline Module v3 (Cache-Aware)
# Sources: Local cache (built by build_cache.py) with live fallback
# Author: Powei Shadrack | SunSpot-Tech
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
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── Config ────────────────────────────────────────────────────
API_FOOTBALL_KEY    = os.getenv("API_FOOTBALL_KEY", "da4fbeeb4b173da6bff493b5aca8d691")
FD_LEAGUE_CODE      = os.getenv("FD_LEAGUE_CODE", "E0")
FD_SEASON_CODE      = os.getenv("FD_SEASON_CODE", "2526")
API_FOOTBALL_LEAGUE = int(os.getenv("API_FOOTBALL_LEAGUE", "39"))
API_FOOTBALL_SEASON = int(os.getenv("API_FOOTBALL_SEASON", "2025"))
XGSCORE_LEAGUE_SLUG = os.getenv("XGSCORE_LEAGUE_SLUG", "epl")
CACHE_DIR           = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

COMPETITION_IMPORTANCE = {
    "Friendly": 1, "Domestic Cup": 2, "Nations League": 2,
    "Minor Cup": 2, "Continental Qualifier": 3, "Domestic League": 3,
    "World Cup Group Stage": 4, "Continental Championship": 4,
    "Champions League Group Stage": 4, "Europa League Group Stage": 4,
    "Champions League Knockout": 5, "Europa League Final": 5,
    "World Cup Knockout": 5, "Continental Knockout": 5, "Major Final": 5,
}

FOOTBALL_DATA_LEAGUES = {
    "EPL":          ("E0",  "epl",          39,  2025),
    "Championship": ("E1",  None,           40,  2025),
    "La Liga":      ("SP1", "la-liga",      140, 2025),
    "Serie A":      ("I1",  "serie-a",      135, 2025),
    "Bundesliga":   ("D1",  "bundesliga",   78,  2025),
    "Ligue 1":      ("F1",  "ligue-1",      61,  2025),
    "Eredivisie":   ("N1",  "eredivisie",   88,  2025),
    "Liga Portugal":("P1",  "liga-portugal",94,  2025),
    "Scottish Prem":("SC0", "SPFL",           179, 2025),
    "Belgian Pro":  ("B1",  None,           144, 2025),
    "Turkish SL":   ("T1",  None,           203, 2025),
    "Greek SL":     ("G1",  None,           197, 2025),
}

TEAM_COORDINATES = {
    "Arsenal": (51.5549,-0.1084), "Chelsea": (51.4816,-0.1910),
    "Manchester City": (53.4831,-2.2004), "Liverpool": (53.4308,-2.9608),
    "Tottenham": (51.6043,-0.0665), "Newcastle": (54.9756,-1.6218),
    "Aston Villa": (52.5090,-1.8847), "Brighton": (50.8618,-0.0834),
    "West Ham": (51.5386,-0.0164), "Brentford": (51.4882,-0.3087),
    "Fulham": (51.4749,-0.2217), "Crystal Palace": (51.3983,-0.0855),
    "Everton": (53.4388,-2.9662), "Wolverhampton": (52.5900,-2.1303),
    "Nottingham Forest": (52.9399,-1.1328), "Bournemouth": (50.7352,-1.8382),
    "Leicester City": (52.6204,-1.1422), "Burnley": (53.7892,-2.2297),
    "Luton Town": (51.8837,-0.4318), "Sheffield Utd": (53.3703,-1.4705),
    "Southampton": (50.9058,-1.3914), "Leeds United": (53.7772,-1.5724),
    "Ipswich": (52.0550,1.1450),
    "Barcelona": (41.3809,2.1228), "Real Madrid": (40.4530,-3.6883),
    "Atletico Madrid": (40.4361,-3.5995), "Sevilla": (37.3841,-5.9705),
    "Valencia": (39.4745,-0.3582), "Villarreal": (39.9447,-0.1031),
    "Athletic Club": (43.2641,-2.9490), "Real Sociedad": (43.3015,-1.9738),
    "Bayern Munich": (48.2188,11.6247), "Borussia Dortmund": (51.4926,7.4519),
    "RB Leipzig": (51.3456,12.3488), "Bayer Leverkusen": (51.0315,7.0023),
    "Eintracht Frankfurt": (50.0688,8.6452),
    "Borussia Monchengladbach": (51.1742,6.3855),
    "Juventus": (45.1096,7.6413), "Inter Milan": (45.4781,9.1240),
    "AC Milan": (45.4781,9.1240), "AS Roma": (41.9340,12.4547),
    "Napoli": (40.8279,14.1931), "Lazio": (41.9340,12.4547),
    "Atalanta": (45.7093,9.6802), "Fiorentina": (43.7809,11.2820),
    "Paris Saint-Germain": (48.8414,2.2530), "Lyon": (45.7653,4.9822),
    "Marseille": (43.2696,5.3960), "Monaco": (43.7274,7.4152),
    "Lille": (50.6117,3.1305), "Nice": (43.7047,7.1928),
}


# ============================================================
# DEFAULTS
# ============================================================

def _default_log():
    return pd.DataFrame({
        "date":           pd.date_range(end=pd.Timestamp.today(), periods=5, freq="7D"),
        "goals_scored":   [1,1,1,1,1],
        "goals_conceded": [1,1,1,1,1],
        "sot":            [4,4,4,4,4],
        "shots_off":      [3,3,3,3,3],
        "result":         ["D","D","D","D","D"],
    })

def _default_season():
    return {"avg_goals":1.2,"avg_conceded":1.2,"clean_sheet_pct":30.0,"penalty_conv_pct":75.0}

def _default_match():
    return {"possession":50.0,"pass_accuracy":75.0,"avg_saves":3.5,"avg_corners":5.0}

def _default_gk():
    return {"gk_save_pct":70.0,"gk_cs_pct":30.0,"gk_pen_save_pct":20.0}

def _default_xgs():
    return {"xg":None,"xga":None,"possession":50.0,"pass_accuracy":75.0}


# ============================================================
# CACHE READER
# ============================================================

def _load_json(filename):
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
        data = _load_json(filename)
        if data is not None:
            return data
    return None


class CacheReader:
    def __init__(self):
        # build_cache.py historically wrote fd_league.json; accept both names.
        self._fd   = _load_first_available("football_data.json", "fd_league.json")
        self._xgs  = _load_json("xgscore.json")
        self._apif = _load_json("apif_all.json")
        meta       = _load_json("meta.json")
        if meta:
            print(f"  Cache loaded — built at: {meta.get('built_at','unknown')}")
        else:
            print("  No cache found — will use live fetch fallback.")

    @property
    def has_cache(self):
        return self._fd is not None

    # ── Match logs ──────────────────────────────────────────
    def team_match_log(self, league, team):
        try:
            if not self._fd:
                return None
            matches = self._fd.get(league, [])
            if not matches:
                return None
            df = pd.DataFrame(matches)
            if "HomeTeam" not in df.columns:
                return None
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

            def sc(frame, col, default=np.nan):
                return pd.to_numeric(frame[col], errors="coerce").fillna(default) if col in frame.columns else pd.Series([default]*len(frame), index=frame.index)

            home = df[df["HomeTeam"] == team].copy()
            away = df[df["AwayTeam"] == team].copy()

            home["goals_scored"]   = sc(home, "FTHG", 1)
            home["goals_conceded"] = sc(home, "FTAG", 1)
            home["sot"]            = sc(home, "HST", 4)
            home["shots_off"]      = sc(home, "HS", 7) - sc(home, "HST", 4)
            home["result"]         = home["FTR"].map({"H":"W","D":"D","A":"L"})
            home["date"]           = home["Date"]

            away["goals_scored"]   = sc(away, "FTAG", 1)
            away["goals_conceded"] = sc(away, "FTHG", 1)
            away["sot"]            = sc(away, "AST", 4)
            away["shots_off"]      = sc(away, "AS", 7) - sc(away, "AST", 4)
            away["result"]         = away["FTR"].map({"A":"W","D":"D","H":"L"})
            away["date"]           = away["Date"]

            cols = ["date","goals_scored","goals_conceded","sot","shots_off","result"]
            log  = pd.concat([home[cols], away[cols]]).sort_values("date").reset_index(drop=True)
            log  = log.dropna(subset=["result"])
            return log if len(log) >= 5 else None
        except Exception:
            return None

    # ── H2H ────────────────────────────────────────────────
    def h2h(self, league, home_team, away_team, perspective, n=5):
        try:
            if not self._fd:
                return ["D"]*n
            matches = self._fd.get(league, [])
            if not matches:
                return ["D"]*n
            df   = pd.DataFrame(matches)
            mask = (
                ((df["HomeTeam"]==home_team)&(df["AwayTeam"]==away_team)) |
                ((df["HomeTeam"]==away_team)&(df["AwayTeam"]==home_team))
            )
            h2h     = df[mask].tail(n)
            results = []
            for _, row in h2h.iterrows():
                try:
                    if row["HomeTeam"] == perspective:
                        results.append({"H":"W","D":"D","A":"L"}[row["FTR"]])
                    else:
                        results.append({"A":"W","D":"D","H":"L"}[row["FTR"]])
                except Exception:
                    results.append("D")
            while len(results) < n:
                results.insert(0, "D")
            return results
        except Exception:
            return ["D"]*n

    # ── Last match date ─────────────────────────────────────
    def last_match_date(self, league, team, before_date):
        try:
            if not self._fd:
                return pd.to_datetime(before_date) - timedelta(days=7)
            matches = self._fd.get(league, [])
            if not matches:
                return pd.to_datetime(before_date) - timedelta(days=7)
            df = pd.DataFrame(matches)
            if "Date" not in df.columns:
                return pd.to_datetime(before_date) - timedelta(days=7)
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            mask = (
                (df["HomeTeam"]==team)|(df["AwayTeam"]==team)
            ) & (df["Date"] < pd.to_datetime(before_date))
            filtered = df[mask].sort_values("Date")
            if filtered.empty:
                return pd.to_datetime(before_date) - timedelta(days=7)
            return filtered.iloc[-1]["Date"]
        except Exception:
            return pd.to_datetime(before_date) - timedelta(days=7)

    # ── League avg goals ────────────────────────────────────
    def league_avg_goals(self, league):
        try:
            if not self._fd:
                return 1.35
            matches = self._fd.get(league, [])
            if not matches:
                return 1.35
            df    = pd.DataFrame(matches)
            total = pd.to_numeric(df.get("FTHG",pd.Series()), errors="coerce").sum() + \
                    pd.to_numeric(df.get("FTAG",pd.Series()), errors="coerce").sum()
            return round(float(total)/max(len(df),1), 4)
        except Exception:
            return 1.35

    # ── xGscore ─────────────────────────────────────────────
    def xgs_team(self, league, team):
        try:
            if not self._xgs:
                return _default_xgs()
            ld = self._xgs.get(league, {})
            if team in ld:
                return ld[team]
            for k in ld:
                if team.lower() in k.lower() or k.lower() in team.lower():
                    return ld[k]
        except Exception:
            pass
        return _default_xgs()

    # ── API-Football ────────────────────────────────────────
    def _apif_team(self, league, team):
        try:
            if not self._apif:
                return None
            ld = self._apif.get(league, {})
            if team in ld:
                return ld[team]
            for k in ld:
                if team.lower() in k.lower() or k.lower() in team.lower():
                    return ld[k]
        except Exception:
            pass
        return None

    def season_stats(self, league, team):
        d = self._apif_team(league, team)
        if not d:
            return _default_season()
        return {
            "avg_goals":        d.get("avg_goals", 1.2),
            "avg_conceded":     d.get("avg_conceded", 1.2),
            "clean_sheet_pct":  d.get("clean_sheet_pct", 30.0),
            "penalty_conv_pct": d.get("penalty_conv_pct", 75.0),
        }

    def match_stats(self, league, team):
        d = self._apif_team(league, team)
        if not d:
            return _default_match()
        return {
            "possession":    d.get("possession", 50.0),
            "pass_accuracy": d.get("pass_accuracy", 75.0),
            "avg_saves":     d.get("avg_saves", 3.5),
            "avg_corners":   d.get("avg_corners", 5.0),
        }

    def injuries(self, league, team):
        d = self._apif_team(league, team)
        return d.get("injuries", 0) if d else 0

    def gk_stats(self, league, team):
        s        = self.season_stats(league, team)
        m        = self.match_stats(league, team)
        faced    = m["avg_saves"] + s["avg_conceded"]
        save_pct = round((m["avg_saves"]/max(faced,1))*100, 2)
        return {"gk_save_pct":save_pct,"gk_cs_pct":s["clean_sheet_pct"],"gk_pen_save_pct":20.0}


# ── Singleton ────────────────────────────────────────────────
_cache_instance = None

def get_cache():
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheReader()
    return _cache_instance


# ============================================================
# LIVE FETCHERS (fallback when no cache)
# ============================================================

class FootballDataFetcher:
    BASE = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

    def __init__(self, league_code=FD_LEAGUE_CODE, season=FD_SEASON_CODE):
        self.url = self.BASE.format(season=season, league=league_code)
        self._df = None

    def fetch(self):
        if self._df is not None:
            return self._df
        try:
            r = requests.get(self.url, headers={"User-Agent":"Mozilla/5.0"}, timeout=20)
            if r.status_code != 200 or not r.text.strip():
                self._df = pd.DataFrame()
                return self._df
            self._df = pd.read_csv(StringIO(r.text), on_bad_lines="skip")
            self._df.columns = [c.strip() for c in self._df.columns]
            if "Date" in self._df.columns:
                self._df["Date"] = pd.to_datetime(self._df["Date"], dayfirst=True, errors="coerce")
        except Exception as e:
            print(f"  football-data live fetch failed: {e}")
            self._df = pd.DataFrame()
        return self._df

    def team_match_log(self, team):
        try:
            df = self.fetch()
            if df.empty or "HomeTeam" not in df.columns:
                return _default_log()

            def sc(frame, col, default=np.nan):
                return pd.to_numeric(frame[col], errors="coerce").fillna(default) if col in frame.columns else pd.Series([default]*len(frame), index=frame.index)

            home = df[df["HomeTeam"]==team].copy()
            away = df[df["AwayTeam"]==team].copy()

            home["goals_scored"]   = sc(home,"FTHG",1)
            home["goals_conceded"] = sc(home,"FTAG",1)
            home["sot"]            = sc(home,"HST",4)
            home["shots_off"]      = sc(home,"HS",7)-sc(home,"HST",4)
            home["result"]         = home["FTR"].map({"H":"W","D":"D","A":"L"})
            home["date"]           = home["Date"]

            away["goals_scored"]   = sc(away,"FTAG",1)
            away["goals_conceded"] = sc(away,"FTHG",1)
            away["sot"]            = sc(away,"AST",4)
            away["shots_off"]      = sc(away,"AS",7)-sc(away,"AST",4)
            away["result"]         = away["FTR"].map({"A":"W","D":"D","H":"L"})
            away["date"]           = away["Date"]

            cols = ["date","goals_scored","goals_conceded","sot","shots_off","result"]
            log  = pd.concat([home[cols],away[cols]]).sort_values("date").reset_index(drop=True)
            log  = log.dropna(subset=["result"])
            return log if len(log)>=5 else _default_log()
        except Exception:
            return _default_log()

    def h2h_results(self, home_team, away_team, perspective, n=5):
        try:
            df = self.fetch()
            if df.empty or "HomeTeam" not in df.columns:
                return ["D"]*n
            mask = (
                ((df["HomeTeam"]==home_team)&(df["AwayTeam"]==away_team))|
                ((df["HomeTeam"]==away_team)&(df["AwayTeam"]==home_team))
            )
            h2h = df[mask].sort_values("Date").tail(n)
            results = []
            for _,row in h2h.iterrows():
                try:
                    if row["HomeTeam"]==perspective:
                        results.append({"H":"W","D":"D","A":"L"}[row["FTR"]])
                    else:
                        results.append({"A":"W","D":"D","H":"L"}[row["FTR"]])
                except Exception:
                    results.append("D")
            while len(results)<n:
                results.insert(0,"D")
            return results
        except Exception:
            return ["D"]*n

    def last_match_date(self, team, before_date):
        try:
            df = self.fetch()
            if df.empty or "HomeTeam" not in df.columns:
                return pd.to_datetime(before_date)-timedelta(days=7)
            mask = ((df["HomeTeam"]==team)|(df["AwayTeam"]==team)) & (df["Date"]<pd.to_datetime(before_date))
            filtered = df[mask].sort_values("Date")
            if filtered.empty:
                return pd.to_datetime(before_date)-timedelta(days=7)
            return filtered.iloc[-1]["Date"]
        except Exception:
            return pd.to_datetime(before_date)-timedelta(days=7)

    def league_avg_goals(self):
        try:
            df = self.fetch()
            if df.empty or "FTHG" not in df.columns:
                return 1.35
            total = pd.to_numeric(df["FTHG"],errors="coerce").sum()+pd.to_numeric(df["FTAG"],errors="coerce").sum()
            return round(float(total)/max(len(df),1),4)
        except Exception:
            return 1.35


class APIFootballFetcher:
    BASE = "https://v3.football.api-sports.io"

    def __init__(self, api_key=None, league=API_FOOTBALL_LEAGUE, season=API_FOOTBALL_SEASON):
        self.api_key = api_key or API_FOOTBALL_KEY
        self.headers = {
            "x-rapidapi-host": "v3.football.api-sports.io",
            "x-rapidapi-key":  self.api_key,
        }
        self.league      = league
        self.season      = season
        self._team_ids   = {}
        self._team_stats = {}
        self._has_key    = self.api_key not in ["YOUR_API_FOOTBALL_KEY","",None]

    def _get(self, endpoint, params=None):
        if not self._has_key:
            return {"response":[]}
        try:
            r = requests.get(
                f"{self.BASE}/{endpoint}",
                headers=self.headers,
                params=params,
                timeout=15,
                verify=False,
            )
            if r.status_code in [401,403]:
                self._has_key = False
                return {"response":[]}
            if r.status_code!=200 or not r.text.strip():
                return {"response":[]}
            data   = r.json()
            errors = data.get("errors",{})
            if errors:
                return {"response":[]}
            time.sleep(0.4)
            return data
        except Exception as e:
            print(f"  API-Football ({endpoint}): {e}")
            return {"response":[]}

    def get_team_id(self, team_name):
        if team_name not in self._team_ids:
            data  = self._get("teams",{"name":team_name,"league":self.league,"season":self.season})
            teams = data.get("response",[])
            self._team_ids[team_name] = teams[0]["team"]["id"] if teams else None
        return self._team_ids[team_name]

    def team_season_stats(self, team_name):
        if team_name in self._team_stats:
            return self._team_stats[team_name]
        if not self._has_key:
            return _default_season()
        team_id = self.get_team_id(team_name)
        if not team_id:
            return _default_season()
        try:
            data = self._get("teams/statistics",{"team":team_id,"league":self.league,"season":self.season})
            resp = data.get("response",{})
            stats = {}
            stats["avg_goals"]        = float(resp.get("goals",{}).get("for",{}).get("average",{}).get("total",1.2) or 1.2)
            stats["avg_conceded"]     = float(resp.get("goals",{}).get("against",{}).get("average",{}).get("total",1.2) or 1.2)
            cs = resp.get("clean_sheet",{}).get("total",0)
            mp = resp.get("fixtures",{}).get("played",{}).get("total",1)
            stats["clean_sheet_pct"]  = round((cs/max(mp,1))*100,2)
            ps = resp.get("penalty",{}).get("scored",{}).get("total",0)
            pm = resp.get("penalty",{}).get("missed",{}).get("total",0)
            stats["penalty_conv_pct"] = round((ps/max(ps+pm,1))*100,2)
            self._team_stats[team_name] = stats
            return stats
        except Exception:
            return _default_season()

    def team_last5_stats(self, team_name):
        if not self._has_key:
            return _default_match()
        team_id = self.get_team_id(team_name)
        if not team_id:
            return _default_match()
        try:
            data = self._get("fixtures",{"team":team_id,"league":self.league,"season":self.season,"last":5})
            fixtures = data.get("response",[])
            poss,pacc,saves,corners = [],[],[],[]
            for fix in fixtures:
                fid = fix["fixture"]["id"]
                sd  = self._get("fixtures/statistics",{"fixture":fid,"team":team_id})
                time.sleep(0.2)
                for ts in sd.get("response",[]):
                    if ts.get("team",{}).get("id")==team_id:
                        for s in ts.get("statistics",[]):
                            t,val = s.get("type",""),s.get("value")
                            try:
                                if t=="Ball Possession" and val: poss.append(float(str(val).replace("%","")))
                                elif t=="Passes %" and val:     pacc.append(float(str(val).replace("%","")))
                                elif t=="Goalkeeper Saves" and val: saves.append(int(val))
                                elif t=="Corner Kicks" and val:     corners.append(int(val))
                            except Exception: continue
            return {
                "possession":    round(float(np.mean(poss))    if poss    else 50.0,2),
                "pass_accuracy": round(float(np.mean(pacc))    if pacc    else 75.0,2),
                "avg_saves":     round(float(np.mean(saves))   if saves   else 3.5, 2),
                "avg_corners":   round(float(np.mean(corners)) if corners else 5.0, 2),
            }
        except Exception:
            return _default_match()

    def injuries(self, team_name):
        if not self._has_key: return 0
        try:
            team_id = self.get_team_id(team_name)
            if not team_id: return 0
            data = self._get("injuries",{"team":team_id,"season":self.season})
            return len(data.get("response",[]))
        except Exception:
            return 0

    def referee_card_rate(self, referee_name):
        if not self._has_key or not referee_name: return 2.3
        try:
            data = self._get("fixtures",{"referee":referee_name,"season":self.season,"league":self.league})
            fixtures = data.get("response",[])
            if not fixtures: return 2.3
            total = 0
            for fix in fixtures:
                ed = self._get("fixtures/events",{"fixture":fix["fixture"]["id"],"type":"Card"})
                total += len(ed.get("response",[]))
                time.sleep(0.2)
            return round(total/max(len(fixtures),1),4)
        except Exception:
            return 2.3

    def gk_stats(self, team_name):
        try:
            s     = self.team_season_stats(team_name)
            m5    = self.team_last5_stats(team_name)
            faced = m5["avg_saves"]+s["avg_conceded"]
            return {
                "gk_save_pct":     round((m5["avg_saves"]/max(faced,1))*100,2),
                "gk_cs_pct":       s["clean_sheet_pct"],
                "gk_pen_save_pct": 20.0,
            }
        except Exception:
            return _default_gk()


class XGScoreFetcher:
    BASE    = "https://xgscore.io/xg-statistics/{slug}"
    HEADERS = {
        "User-Agent":(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language":"en-US,en;q=0.9",
    }

    def __init__(self, league_slug=XGSCORE_LEAGUE_SLUG):
        self.url      = self.BASE.format(slug=league_slug) if league_slug else None
        self._data    = {}
        self._fetched = False

    def fetch(self):
        if self._fetched or not self.url: return
        try:
            r = requests.get(self.url, headers=self.HEADERS, timeout=15)
            if r.status_code!=200 or not r.text.strip():
                self._fetched=True; return
            soup = BeautifulSoup(r.text,"html.parser")
            for table in soup.find_all("table"):
                for row in table.find_all("tr"):
                    cols = [td.get_text(strip=True) for td in row.find_all("td")]
                    if len(cols)>=5:
                        try:
                            self._data[cols[0]] = {
                                "xg":    float(cols[2]),
                                "xga":   float(cols[3]),
                                "possession":   float(cols[4].replace("%","")),
                                "pass_accuracy":float(cols[5].replace("%","")) if len(cols)>5 else 75.0,
                            }
                        except Exception: continue
        except Exception: pass
        self._fetched = True

    def team_stats(self, team_name):
        try:
            self.fetch()
            if team_name in self._data: return self._data[team_name]
            for k in self._data:
                if team_name.lower() in k.lower() or k.lower() in team_name.lower():
                    return self._data[k]
        except Exception: pass
        return _default_xgs()


# ============================================================
# MASTER FEATURE BUILDER
# Uses cache if available, falls back to live fetch
# ============================================================

def build_features_from_request(req_home, req_away, req_league, req_date,
                                 req_competition, req_home_coach, req_home_starters,
                                 req_home_signings, req_away_coach, req_away_starters,
                                 req_away_signings, req_referee=None):

    cache       = get_cache()
    league_cfg  = FOOTBALL_DATA_LEAGUES.get(req_league, ("E0","epl",39,2025))
    fd_code, xgs_slug, apif_league_id, apif_season = league_cfg
    is_domestic = req_competition in ["Domestic League","Domestic Cup"]
    comp_score  = COMPETITION_IMPORTANCE.get(req_competition, 3)

    # ── Try cache first, fall back to live ──────────────────
    if cache.has_cache:
        home_log   = cache.team_match_log(req_league, req_home) or _default_log()
        away_log   = cache.team_match_log(req_league, req_away) or _default_log()
        h2h_home   = cache.h2h(req_league, req_home, req_away, req_home)
        h2h_away   = cache.h2h(req_league, req_home, req_away, req_away)
        last_home  = cache.last_match_date(req_league, req_home, req_date)
        last_away  = cache.last_match_date(req_league, req_away, req_date)
        league_avg = cache.league_avg_goals(req_league)
        home_s     = cache.season_stats(req_league, req_home)
        away_s     = cache.season_stats(req_league, req_away)
        home_m5    = cache.match_stats(req_league, req_home)
        away_m5    = cache.match_stats(req_league, req_away)
        home_gk    = cache.gk_stats(req_league, req_home)
        away_gk    = cache.gk_stats(req_league, req_away)
        home_miss  = cache.injuries(req_league, req_home)
        away_miss  = cache.injuries(req_league, req_away)
        home_xgs   = cache.xgs_team(req_league, req_home)
        away_xgs   = cache.xgs_team(req_league, req_away)
        src        = "cache"
    else:
        # Live fallback
        fd         = FootballDataFetcher(league_code=fd_code)
        home_log   = fd.team_match_log(req_home)
        away_log   = fd.team_match_log(req_away)
        h2h_home   = fd.h2h_results(req_home, req_away, req_home)
        h2h_away   = fd.h2h_results(req_home, req_away, req_away)
        last_home  = fd.last_match_date(req_home, req_date)
        last_away  = fd.last_match_date(req_away, req_date)
        league_avg = fd.league_avg_goals()
        apif       = APIFootballFetcher(league=apif_league_id, season=apif_season)
        home_s     = apif.team_season_stats(req_home)
        away_s     = apif.team_season_stats(req_away)
        home_m5    = apif.team_last5_stats(req_home)
        away_m5    = apif.team_last5_stats(req_away)
        home_gk    = apif.gk_stats(req_home)
        away_gk    = apif.gk_stats(req_away)
        home_miss  = apif.injuries(req_home)
        away_miss  = apif.injuries(req_away)
        xgs_f      = XGScoreFetcher(league_slug=xgs_slug) if xgs_slug else None
        home_xgs   = xgs_f.team_stats(req_home) if xgs_f else _default_xgs()
        away_xgs   = xgs_f.team_stats(req_away) if xgs_f else _default_xgs()
        src        = "live"

    # Referee
    ref_rate = 2.3
    if req_referee:
        try:
            apif_r   = APIFootballFetcher(league=apif_league_id, season=apif_season)
            ref_rate = apif_r.referee_card_rate(req_referee)
        except Exception:
            ref_rate = 2.3

    # xG / possession resolution
    home_xg_val  = home_xgs.get("xg")  or f_xg_from_log(home_log)
    away_xg_val  = away_xgs.get("xg")  or f_xg_from_log(away_log)
    home_xga_val = home_xgs.get("xga") or f_xga_from_log(home_log)
    away_xga_val = away_xgs.get("xga") or f_xga_from_log(away_log)
    home_poss    = home_xgs.get("possession",50.0)    or home_m5["possession"]
    away_poss    = away_xgs.get("possession",50.0)    or away_m5["possession"]
    home_pass    = home_xgs.get("pass_accuracy",75.0) or home_m5["pass_accuracy"]
    away_pass    = away_xgs.get("pass_accuracy",75.0) or away_m5["pass_accuracy"]

    features = {
        "home_form":                 f_form(home_log),
        "away_form":                 f_form(away_log),
        "home_xg":                   round(float(home_xg_val),4),
        "away_xg":                   round(float(away_xg_val),4),
        "home_xga":                  round(float(home_xga_val),4),
        "away_xga":                  round(float(away_xga_val),4),
        "home_xt":                   f_xt(home_log),
        "away_xt":                   f_xt(away_log),
        "home_sot":                  f_sot(home_log),
        "away_sot":                  f_sot(away_log),
        "home_attack_strength":      f_attack_strength(home_s["avg_goals"], league_avg),
        "away_attack_strength":      f_attack_strength(away_s["avg_goals"], league_avg),
        "home_defense_strength":     f_defense_strength(home_s["avg_conceded"], league_avg),
        "away_defense_strength":     f_defense_strength(away_s["avg_conceded"], league_avg),
        "home_rest_days":            f_rest_days(last_home, req_date),
        "away_rest_days":            f_rest_days(last_away, req_date),
        "home_missing_players":      home_miss,
        "away_missing_players":      away_miss,
        "home_average_possession":   f_possession(home_poss),
        "away_average_possession":   f_possession(away_poss),
        "home_clean_sheet_rate":     f_clean_sheet_rate(home_log),
        "away_clean_sheet_rate":     f_clean_sheet_rate(away_log),
        "home_head_to_head_rate":    f_h2h(h2h_home),
        "away_head_to_head_rate":    f_h2h(h2h_away),
        "referee_card_rate":         ref_rate,
        "home_goalkeeper_rating":    f_gk_rating(home_gk["gk_save_pct"],home_gk["gk_cs_pct"],home_gk["gk_pen_save_pct"]),
        "away_goalkeeper_rating":    f_gk_rating(away_gk["gk_save_pct"],away_gk["gk_cs_pct"],away_gk["gk_pen_save_pct"]),
        "home_finishing_efficiency": f_finishing(home_log, home_xgs.get("xg")),
        "away_finishing_efficiency": f_finishing(away_log, away_xgs.get("xg")),
        "home_penalty_rating":       f_penalty_rating(home_s["penalty_conv_pct"],home_gk["gk_pen_save_pct"]),
        "away_penalty_rating":       f_penalty_rating(away_s["penalty_conv_pct"],away_gk["gk_pen_save_pct"]),
        "home_set_piece_strength":   f_set_piece(home_m5["avg_corners"],home_s["avg_goals"]),
        "away_set_piece_strength":   f_set_piece(away_m5["avg_corners"],away_s["avg_goals"]),
        "home_press_resistance":     f_press_resistance(home_pass,home_poss),
        "away_press_resistance":     f_press_resistance(away_pass,away_poss),
        "home_squad_chemistry":      f_squad_chemistry(req_home_coach,req_home_starters,req_home_signings),
        "away_squad_chemistry":      f_squad_chemistry(req_away_coach,req_away_starters,req_away_signings),
        "home_travel_fatigue":       0,
        "away_travel_fatigue":       f_travel_fatigue(req_home, req_away, is_domestic),
        "competition_importance":    comp_score,
    }

    source_map = {k: src for k in features}
    return pd.DataFrame([features]), features, source_map


# ============================================================
# FEATURE CALCULATORS
# ============================================================

def haversine_km(c1,c2):
    R=6371
    lat1,lon1=map(radians,c1); lat2,lon2=map(radians,c2)
    dlat=lat2-lat1; dlon=lon2-lon1
    a=sin(dlat/2)**2+cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R*2*atan2(sqrt(a),sqrt(1-a))

def travel_fatigue_score(dist,is_domestic):
    if is_domestic:
        if dist<100: return 0
        elif dist<200: return 1
        elif dist<350: return 2
        elif dist<500: return 3
        else: return 4
    else:
        if dist<500: return 0
        elif dist<1500: return 1
        elif dist<3000: return 2
        elif dist<5000: return 3
        elif dist<8000: return 4
        else: return 5

def f_form(log):
    try:
        pm={"W":3,"D":1,"L":0}
        return round(float(log["result"].iloc[-5:].map(pm).sum())/15,4)
    except Exception: return 0.5

def f_xg_from_log(log):
    try: return round(float(log["goals_scored"].iloc[-5:].mean()),4)
    except Exception: return 1.2

def f_xga_from_log(log):
    try: return round(float(log["goals_conceded"].iloc[-5:].mean()),4)
    except Exception: return 1.2

def f_xt(log):
    try:
        sot=log["sot"].iloc[-5:].fillna(0)
        soff=log["shots_off"].iloc[-5:].fillna(0)
        return round(float(((sot*0.3)+(soff*0.1)).mean()),4)
    except Exception: return 1.5

def f_sot(log):
    try: return round(float(log["sot"].iloc[-5:].fillna(0).mean()),4)
    except Exception: return 4.0

def f_attack_strength(team_avg,league_avg):
    try: return round(float(team_avg)/float(league_avg),4) if league_avg else 1.0
    except Exception: return 1.0

def f_defense_strength(team_avg,league_avg):
    try: return round(float(team_avg)/float(league_avg),4) if league_avg else 1.0
    except Exception: return 1.0

def f_rest_days(last_date,next_date):
    try: return int((pd.to_datetime(next_date)-pd.to_datetime(last_date)).days)
    except Exception: return 7

def f_possession(poss_pct):
    try: return round(float(poss_pct)/100,4)
    except Exception: return 0.5

def f_clean_sheet_rate(log):
    try: return round(float((log["goals_conceded"].iloc[-5:]==0).sum())/5,4)
    except Exception: return 0.3

def f_h2h(results_list):
    try:
        pm={"W":1,"D":0.5,"L":0}
        return round(sum(pm[r] for r in results_list[-5:])/5,4)
    except Exception: return 0.5

def f_gk_rating(save_pct,clean_pct,pen_save_pct):
    try: return round(((0.5*float(save_pct))+(0.3*float(clean_pct))+(0.2*float(pen_save_pct)))/100*5,4)
    except Exception: return 2.5

def f_finishing(log,xg_real=None):
    try:
        goals_l5=float(log["goals_scored"].iloc[-5:].sum())
        xg_l5=(float(xg_real)*5) if xg_real else (f_xg_from_log(log)*5)
        if xg_l5==0: return 0.0
        return round((min(goals_l5/xg_l5,2.0)/2)*5,4)
    except Exception: return 2.5

def f_penalty_rating(conv_pct,gk_pen_save_pct):
    try: return round(((0.6*float(conv_pct))+(0.4*float(gk_pen_save_pct)))/100*5,4)
    except Exception: return 2.5

def f_set_piece(avg_corners,avg_goals):
    try:
        sp_goals=float(avg_goals)*0.25
        sp_att=max(float(avg_corners),1)
        return round(min((sp_goals/sp_att)*25,5.0),4)
    except Exception: return 2.5

def f_press_resistance(pass_acc_pct,poss_pct):
    try: return round(((0.5*float(pass_acc_pct))+(0.5*float(poss_pct)))/100*5,4)
    except Exception: return 3.1

def f_squad_chemistry(coach_years,starters_retained,new_signings):
    try:
        score=0
        if int(coach_years)>2: score+=2
        if int(starters_retained)>=8: score+=2
        if int(new_signings)<5: score+=1
        return score
    except Exception: return 3

def f_travel_fatigue(home_team,away_team,is_domestic):
    try:
        if home_team in TEAM_COORDINATES and away_team in TEAM_COORDINATES:
            dist=haversine_km(TEAM_COORDINATES[home_team],TEAM_COORDINATES[away_team])
        else:
            dist=200
        return travel_fatigue_score(dist,is_domestic)
    except Exception: return 0