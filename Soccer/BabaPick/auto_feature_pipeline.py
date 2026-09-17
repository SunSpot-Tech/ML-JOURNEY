# ============================================================
# AUTOMATIC FOOTBALL MATCH PREDICTION PIPELINE v2
# Sources:
#   1. football-data.co.uk  → form, SOT, shots, H2H, results
#   2. API-Football (free)  → possession, saves, injuries,
#                             referee, penalties, corners
#   3. xGscore.io           → xG, xGA, pass accuracy
#   4. Proxy formulas       → xT, set piece, finishing efficiency
#   5. Manual inputs        → squad chemistry, competition importance
#
# Author: Powei Shadrack | SunSpot-Tech
# ============================================================
#
# SETUP (run once):
#   pip install pandas numpy requests beautifulsoup4 joblib
#
# API-Football free tier:
#   Register at https://dashboard.api-football.com
#   Free plan: 100 requests/day, no credit card needed
#   Set your key in CONFIG below
# ============================================================

import os
import re
import time
import json
import joblib
import warnings
import requests
import pandas as pd
import numpy as np
from io import StringIO
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG — EDIT THESE
# ============================================================
MODEL_PATH         = "football_prediction_model.pkl"
API_FOOTBALL_KEY   = "7dbe3184d9b6ddd1141455603aad467e"   # get free at dashboard.api-football.com
FD_LEAGUE_CODE     = "E0"     # football-data.co.uk league code
FD_SEASON_CODE     = "2526"   # season code
API_FOOTBALL_LEAGUE = 39      # API-Football league ID (39 = EPL)
API_FOOTBALL_SEASON = 2025    # API-Football season year

# xGscore.io league URL slugs
XGSCORE_LEAGUE_SLUG = "epl"   # change per league

# League code mappings
FOOTBALL_DATA_LEAGUES = {
    "EPL":          ("E0",  "epl",          39,   2025),
    "Championship": ("E1",  None,           40,   2025),
    "La Liga":      ("SP1", "la-liga",      140,  2025),
    "Serie A":      ("I1",  "serie-a",      135,  2025),
    "Bundesliga":   ("D1",  "bundesliga",   78,   2025),
    "Ligue 1":      ("F1",  "ligue-1",      61,   2025),
    "Eredivisie":   ("N1",  "eredivisie",   88,   2025),
    "Liga Portugal":("P1",  "liga-portugal",94,   2025),
    "Scottish Prem":("SC0", None,           179,  2025),
    "Belgian Pro":  ("B1",  None,           144,  2025),
    "Turkish SL":   ("T1",  None,           203,  2025),
    "Greek SL":     ("G1",  None,           197,  2025),
}

COMPETITION_IMPORTANCE = {
    "Friendly": 1,
    "Domestic Cup": 2,
    "Nations League": 2,
    "Minor Cup": 2,
    "Continental Qualifier": 3,
    "Domestic League": 3,
    "World Cup Group Stage": 4,
    "Continental Championship": 4,
    "Champions League Group Stage": 4,
    "Europa League Group Stage": 4,
    "Champions League Knockout": 5,
    "Europa League Final": 5,
    "World Cup Knockout": 5,
    "Continental Knockout": 5,
    "Major Final": 5,
}

TEAM_COORDINATES = {
    "Arsenal": (51.5549, -0.1084),
    "Chelsea": (51.4816, -0.1910),
    "Manchester City": (53.4831, -2.2004),
    "Liverpool": (53.4308, -2.9608),
    "Tottenham": (51.6043, -0.0665),
    "Newcastle": (54.9756, -1.6218),
    "Aston Villa": (52.5090, -1.8847),
    "Brighton": (50.8618, -0.0834),
    "West Ham": (51.5386, -0.0164),
    "Brentford": (51.4882, -0.3087),
    "Fulham": (51.4749, -0.2217),
    "Crystal Palace": (51.3983, -0.0855),
    "Everton": (53.4388, -2.9662),
    "Wolverhampton": (52.5900, -2.1303),
    "Nottingham Forest": (52.9399, -1.1328),
    "Bournemouth": (50.7352, -1.8382),
    "Leicester City": (52.6204, -1.1422),
    "Burnley": (53.7892, -2.2297),
    "Luton Town": (51.8837, -0.4318),
    "Sheffield Utd": (53.3703, -1.4705),
    "Southampton": (50.9058, -1.3914),
    "Leeds United": (53.7772, -1.5724),
    "Ipswich": (52.0550,  1.1450),
    # La Liga
    "Barcelona": (41.3809, 2.1228),
    "Real Madrid": (40.4530, -3.6883),
    "Atletico Madrid": (40.4361, -3.5995),
    "Sevilla": (37.3841, -5.9705),
    "Valencia": (39.4745, -0.3582),
    # Bundesliga
    "Bayern Munich": (48.2188, 11.6247),
    "Borussia Dortmund": (51.4926, 7.4519),
    "RB Leipzig": (51.3456, 12.3488),
    # Serie A
    "Juventus": (45.1096, 7.6413),
    "Inter Milan": (45.4781, 9.1240),
    "AC Milan": (45.4781, 9.1240),
    "AS Roma": (41.9340, 12.4547),
    "Napoli": (40.8279, 14.1931),
    # Ligue 1
    "Paris Saint-Germain": (48.8414, 2.2530),
    "Lyon": (45.7653, 4.9822),
    "Marseille": (43.2696, 5.3960),
}


# ============================================================
# SECTION 1: FOOTBALL-DATA.CO.UK FETCHER
# → form, SOT, shots off target, H2H, results
# ============================================================

class FootballDataFetcher:
    BASE = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

    def __init__(self, league_code=FD_LEAGUE_CODE, season=FD_SEASON_CODE):
        self.url = self.BASE.format(season=season, league=league_code)
        self._df = None

    def fetch(self):
        if self._df is None:
            print("  [football-data.co.uk] Fetching match results...")
            r = requests.get(self.url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            r.raise_for_status()
            self._df = pd.read_csv(StringIO(r.text))
            self._df.columns = [c.strip() for c in self._df.columns]
            if "Date" in self._df.columns:
                self._df["Date"] = pd.to_datetime(
                    self._df["Date"], dayfirst=True, errors="coerce"
                )
            print(f"  Loaded {len(self._df)} matches.")
        return self._df

    def team_match_log(self, team):
        df   = self.fetch()
        home = df[df["HomeTeam"] == team].copy()
        away = df[df["AwayTeam"] == team].copy()

        home["goals_scored"]   = home["FTHG"]
        home["goals_conceded"] = home["FTAG"]
        home["sot"]            = home["HST"]  if "HST" in home.columns else np.nan
        home["shots_off"]      = (home["HS"] - home["HST"]) if "HS" in home.columns else np.nan
        home["result"]         = home["FTR"].map({"H": "W", "D": "D", "A": "L"})
        home["date"]           = home["Date"]

        away["goals_scored"]   = away["FTAG"]
        away["goals_conceded"] = away["FTHG"]
        away["sot"]            = away["AST"]  if "AST" in away.columns else np.nan
        away["shots_off"]      = (away["AS"] - away["AST"]) if "AS" in away.columns else np.nan
        away["result"]         = away["FTR"].map({"A": "W", "D": "D", "H": "L"})
        away["date"]           = away["Date"]

        cols = ["date", "goals_scored", "goals_conceded", "sot", "shots_off", "result"]
        log  = pd.concat([home[cols], away[cols]]).sort_values("date").reset_index(drop=True)
        return log.dropna(subset=["result"])

    def h2h_results(self, home_team, away_team, perspective_team, n=5):
        df   = self.fetch()
        mask = (
            ((df["HomeTeam"] == home_team) & (df["AwayTeam"] == away_team)) |
            ((df["HomeTeam"] == away_team) & (df["AwayTeam"] == home_team))
        )
        h2h     = df[mask].sort_values("Date").tail(n)
        results = []
        for _, row in h2h.iterrows():
            if row["HomeTeam"] == perspective_team:
                results.append({"H": "W", "D": "D", "A": "L"}[row["FTR"]])
            else:
                results.append({"A": "W", "D": "D", "H": "L"}[row["FTR"]])
        while len(results) < n:
            results.insert(0, "D")
        return results

    def last_match_date(self, team, before_date):
        df       = self.fetch()
        mask     = (
            (df["HomeTeam"] == team) | (df["AwayTeam"] == team)
        ) & (df["Date"] < pd.to_datetime(before_date))
        filtered = df[mask].sort_values("Date")
        if filtered.empty:
            return pd.to_datetime(before_date) - timedelta(days=7)
        return filtered.iloc[-1]["Date"]

    def league_avg_goals(self):
        df = self.fetch()
        if "FTHG" not in df.columns:
            return 1.35
        return round((df["FTHG"].sum() + df["FTAG"].sum()) / max(len(df), 1), 4)


# ============================================================
# SECTION 2: API-FOOTBALL FETCHER
# → possession, saves, injuries, referee card rate,
#   penalty stats, corners (set piece proxy)
# ============================================================

class APIFootballFetcher:
    BASE = "https://v3.football.api-sports.io"

    def __init__(self, api_key=API_FOOTBALL_KEY,
                 league=API_FOOTBALL_LEAGUE, season=API_FOOTBALL_SEASON):
        self.headers = {
            "x-rapidapi-host": "v3.football.api-sports.io",
            "x-rapidapi-key":  api_key,
        }
        self.league  = league
        self.season  = season
        self._team_ids    = {}
        self._team_stats  = {}
        self._fixture_ids = {}

    def _get(self, endpoint, params=None):
        url = f"{self.BASE}/{endpoint}"
        r   = requests.get(url, headers=self.headers, params=params, timeout=15)
        r.raise_for_status()
        time.sleep(0.5)   # respect rate limit
        return r.json()

    def get_team_id(self, team_name):
        if team_name not in self._team_ids:
            data = self._get("teams", {"name": team_name, "league": self.league, "season": self.season})
            teams = data.get("response", [])
            if teams:
                self._team_ids[team_name] = teams[0]["team"]["id"]
            else:
                print(f"  Warning: could not find team ID for '{team_name}'")
                return None
        return self._team_ids[team_name]

    def team_season_stats(self, team_name):
        """Full season stats for a team — possession, passes, saves, penalties etc."""
        if team_name in self._team_stats:
            return self._team_stats[team_name]

        team_id = self.get_team_id(team_name)
        if not team_id:
            return self._default_stats()

        print(f"  [API-Football] Fetching season stats for {team_name}...")
        data  = self._get("teams/statistics", {
            "team": team_id,
            "league": self.league,
            "season": self.season
        })
        resp  = data.get("response", {})
        stats = {}

        try:
            # Possession
            poss_str = resp.get("fixtures", {}).get("played", {})
            poss_val = resp.get("biggest", {}).get("goals", {})
            # API-Football stores avg possession differently — get from goals
            stats["possession"] = 50.0   # fallback; use xGscore for real possession

            # Goals scored/conceded per game
            goals_for  = resp.get("goals", {}).get("for", {}).get("average", {}).get("total", "1.2")
            goals_ag   = resp.get("goals", {}).get("against", {}).get("average", {}).get("total", "1.2")
            stats["avg_goals"]    = float(goals_for  or 1.2)
            stats["avg_conceded"] = float(goals_ag   or 1.2)

            # Clean sheets
            cs = resp.get("clean_sheet", {}).get("total", 0)
            mp = resp.get("fixtures", {}).get("played", {}).get("total", 1)
            stats["clean_sheet_pct"] = round((cs / max(mp, 1)) * 100, 2)

            # Failed to score
            fts = resp.get("failed_to_score", {}).get("total", 0)

            # Penalty stats
            pen_scored  = resp.get("penalty", {}).get("scored",  {}).get("total", 0)
            pen_missed  = resp.get("penalty", {}).get("missed",  {}).get("total", 0)
            pen_total   = pen_scored + pen_missed
            stats["penalty_conv_pct"] = round((pen_scored / max(pen_total, 1)) * 100, 2)

        except Exception as e:
            print(f"  Warning: partial stats for {team_name}: {e}")
            stats.update(self._default_stats())

        self._team_stats[team_name] = stats
        return stats

    def team_last5_stats(self, team_name):
        """
        Fetch last 5 fixture IDs for a team and aggregate
        possession, passes, saves, corners from match statistics.
        """
        team_id = self.get_team_id(team_name)
        if not team_id:
            return self._default_match_stats()

        print(f"  [API-Football] Fetching last 5 fixtures for {team_name}...")
        data     = self._get("fixtures", {
            "team": team_id,
            "league": self.league,
            "season": self.season,
            "last": 5,
        })
        fixtures = data.get("response", [])

        possessions   = []
        pass_accs     = []
        saves_list    = []
        corners_list  = []
        gk_pen_saves  = []

        for fixture in fixtures:
            fid = fixture["fixture"]["id"]
            time.sleep(0.3)
            stat_data = self._get("fixtures/statistics", {"fixture": fid, "team": team_id})
            stats_resp = stat_data.get("response", [])

            for team_stat in stats_resp:
                if team_stat.get("team", {}).get("id") == team_id:
                    for s in team_stat.get("statistics", []):
                        t   = s.get("type", "")
                        val = s.get("value")

                        if t == "Ball Possession" and val:
                            poss_num = float(str(val).replace("%", "") or 50)
                            possessions.append(poss_num)

                        elif t == "Passes %" and val:
                            pa = float(str(val).replace("%", "") or 75)
                            pass_accs.append(pa)

                        elif t == "Goalkeeper Saves" and val:
                            saves_list.append(int(val or 0))

                        elif t == "Corner Kicks" and val:
                            corners_list.append(int(val or 0))

        return {
            "possession":    round(np.mean(possessions)  if possessions  else 50.0, 2),
            "pass_accuracy": round(np.mean(pass_accs)    if pass_accs    else 75.0, 2),
            "avg_saves":     round(np.mean(saves_list)   if saves_list   else 3.5,  2),
            "avg_corners":   round(np.mean(corners_list) if corners_list else 5.0,  2),
        }

    def injuries(self, team_name, fixture_id=None):
        """Count injured/suspended players for upcoming fixture."""
        team_id = self.get_team_id(team_name)
        if not team_id:
            return 0
        params = {"team": team_id, "season": self.season}
        if fixture_id:
            params["fixture"] = fixture_id
        print(f"  [API-Football] Fetching injuries for {team_name}...")
        data    = self._get("injuries", params)
        players = data.get("response", [])
        return len(players)

    def referee_card_rate(self, referee_name):
        """
        API-Football does not expose referee career stats directly.
        Use fixtures endpoint filtered by referee to compute rate.
        """
        print(f"  [API-Football] Fetching referee stats for {referee_name}...")
        data     = self._get("fixtures", {
            "referee": referee_name,
            "season":  self.season,
            "league":  self.league,
        })
        fixtures = data.get("response", [])
        if not fixtures:
            return 2.3   # league average fallback

        total_cards   = 0
        total_matches = len(fixtures)
        for fix in fixtures:
            fid  = fix["fixture"]["id"]
            edata = self._get("fixtures/events", {"fixture": fid, "type": "Card"})
            total_cards += len(edata.get("response", []))
            time.sleep(0.2)

        return round(total_cards / max(total_matches, 1), 4)

    def gk_stats(self, team_name):
        """Goalkeeper save%, clean sheet%, penalty save% from season stats."""
        s = self.team_season_stats(team_name)
        m5 = self.team_last5_stats(team_name)

        avg_sot_faced = m5.get("avg_saves", 3.5) + (s.get("avg_conceded", 1.2))
        save_pct      = round((m5.get("avg_saves", 3.5) / max(avg_sot_faced, 1)) * 100, 2)
        cs_pct        = s.get("clean_sheet_pct", 30.0)
        pen_save_pct  = 20.0   # API-Football does not expose this on free plan

        return {
            "gk_save_pct":     save_pct,
            "gk_cs_pct":       cs_pct,
            "gk_pen_save_pct": pen_save_pct,
        }

    @staticmethod
    def _default_stats():
        return {
            "avg_goals": 1.2, "avg_conceded": 1.2,
            "clean_sheet_pct": 30.0,
            "penalty_conv_pct": 75.0,
        }

    @staticmethod
    def _default_match_stats():
        return {
            "possession": 50.0, "pass_accuracy": 75.0,
            "avg_saves": 3.5,   "avg_corners": 5.0,
        }


# ============================================================
# SECTION 3: XGSCORE.IO FETCHER
# → xG, xGA, pass accuracy (season averages per team)
# ============================================================

class XGScoreFetcher:
    BASE = "https://xgscore.io/xg-statistics/{slug}"

    # xGscore uses JavaScript rendering — we try requests first,
    # fall back to defaults if blocked
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    def __init__(self, league_slug=XGSCORE_LEAGUE_SLUG):
        self.url      = self.BASE.format(slug=league_slug)
        self._data    = {}   # team_name → {xg, xga, pass_accuracy}
        self._fetched = False

    def fetch(self):
        if self._fetched:
            return
        print(f"  [xGscore.io] Fetching xG / pass accuracy stats...")
        try:
            r = requests.get(self.url, headers=self.HEADERS, timeout=15)
            if r.status_code != 200:
                print(f"  Warning: xGscore returned {r.status_code}. Using defaults.")
                self._fetched = True
                return

            soup   = BeautifulSoup(r.text, "html.parser")
            tables = soup.find_all("table")

            for table in tables:
                rows = table.find_all("tr")
                for row in rows:
                    cols = [td.get_text(strip=True) for td in row.find_all("td")]
                    # xGscore table: Team | MP | xG | xGA | xGDiff | Poss% | Pass%
                    if len(cols) >= 6:
                        team_name = cols[0]
                        try:
                            xg_val   = float(cols[2])
                            xga_val  = float(cols[3])
                            # Possession and pass accuracy in cols 5 and 6
                            poss_val = float(cols[4].replace("%","")) if "%" in cols[4] else float(cols[4])
                            pass_val = float(cols[5].replace("%","")) if len(cols) > 5 else 75.0
                            self._data[team_name] = {
                                "xg":           xg_val,
                                "xga":          xga_val,
                                "possession":   poss_val,
                                "pass_accuracy": pass_val,
                            }
                        except (ValueError, IndexError):
                            continue

            if self._data:
                print(f"  Loaded xGscore stats for {len(self._data)} teams.")
            else:
                print("  Warning: xGscore table parse failed. Using defaults.")

        except Exception as e:
            print(f"  Warning: xGscore fetch failed ({e}). Using defaults.")

        self._fetched = True

    def team_stats(self, team_name):
        self.fetch()
        # Fuzzy match — try exact first, then partial
        if team_name in self._data:
            return self._data[team_name]
        for key in self._data:
            if team_name.lower() in key.lower() or key.lower() in team_name.lower():
                return self._data[key]
        # Return defaults if team not found
        return {"xg": None, "xga": None, "possession": 50.0, "pass_accuracy": 75.0}


# ============================================================
# SECTION 4: FEATURE CALCULATORS
# ============================================================

def haversine_km(c1, c2):
    R = 6371
    lat1, lon1 = map(radians, c1)
    lat2, lon2 = map(radians, c2)
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def travel_fatigue_score(dist, is_domestic):
    if is_domestic:
        if dist < 100:   return 0
        elif dist < 200: return 1
        elif dist < 350: return 2
        elif dist < 500: return 3
        else:            return 4
    else:
        if dist < 500:    return 0
        elif dist < 1500: return 1
        elif dist < 3000: return 2
        elif dist < 5000: return 3
        elif dist < 8000: return 4
        else:             return 5

def f_form(log):
    pm  = {"W": 3, "D": 1, "L": 0}
    pts = log["result"].iloc[-5:].map(pm).sum()
    return round(pts / 15, 4)

def f_xg_from_log(log):
    return round(log["goals_scored"].iloc[-5:].mean(), 4)

def f_xga_from_log(log):
    return round(log["goals_conceded"].iloc[-5:].mean(), 4)

def f_xt(log):
    sot  = log["sot"].iloc[-5:].fillna(0)
    soff = log["shots_off"].iloc[-5:].fillna(0)
    return round(((sot * 0.3) + (soff * 0.1)).mean(), 4)

def f_sot(log):
    return round(log["sot"].iloc[-5:].fillna(0).mean(), 4)

def f_attack_strength(team_avg, league_avg):
    return round(team_avg / league_avg, 4) if league_avg else 1.0

def f_defense_strength(team_avg, league_avg):
    return round(team_avg / league_avg, 4) if league_avg else 1.0

def f_rest_days(last_date, next_date):
    return (pd.to_datetime(next_date) - pd.to_datetime(last_date)).days

def f_possession(poss_pct):
    return round(float(poss_pct) / 100, 4)

def f_clean_sheet_rate(log):
    return round((log["goals_conceded"].iloc[-5:] == 0).sum() / 5, 4)

def f_h2h(results_list):
    pm = {"W": 1, "D": 0.5, "L": 0}
    return round(sum(pm[r] for r in results_list[-5:]) / 5, 4)

def f_gk_rating(save_pct, clean_pct, pen_save_pct):
    return round(((0.5*save_pct) + (0.3*clean_pct) + (0.2*pen_save_pct)) / 100 * 5, 4)

def f_finishing(log, xg_real=None):
    goals_l5 = log["goals_scored"].iloc[-5:].sum()
    xg_l5    = (xg_real * 5) if xg_real else (f_xg_from_log(log) * 5)
    if xg_l5 == 0:
        return 0.0
    return round((min(goals_l5 / xg_l5, 2.0) / 2) * 5, 4)

def f_penalty_rating(conv_pct, gk_pen_save_pct):
    return round(((0.6*conv_pct) + (0.4*gk_pen_save_pct)) / 100 * 5, 4)

def f_set_piece(avg_corners, avg_goals):
    """
    Proxy: set piece goals ≈ 25% of avg goals
           set piece attempts ≈ corners (direct free kicks + corners)
    """
    sp_goals    = avg_goals * 0.25
    sp_attempts = max(avg_corners, 1)
    return round(min((sp_goals / sp_attempts) * 25, 5.0), 4)

def f_press_resistance(pass_acc_pct, poss_pct):
    return round(((0.5*pass_acc_pct) + (0.5*poss_pct)) / 100 * 5, 4)

def f_squad_chemistry(coach_years, starters_retained, new_signings):
    score = 0
    if coach_years > 2:        score += 2
    if starters_retained >= 8: score += 2
    if new_signings < 5:       score += 1
    return score

def f_travel_fatigue(home_team, away_team, is_domestic):
    if home_team in TEAM_COORDINATES and away_team in TEAM_COORDINATES:
        dist = haversine_km(TEAM_COORDINATES[home_team], TEAM_COORDINATES[away_team])
    else:
        dist = 200
    return travel_fatigue_score(dist, is_domestic)


# ============================================================
# SECTION 5: MANUAL INPUTS
# ============================================================

def collect_manual_inputs(home_team, away_team):
    print("\n" + "="*55)
    print("MANUAL INPUTS REQUIRED")
    print("="*55)

    match_date  = input("Match date (YYYY-MM-DD): ").strip() or str(datetime.today().date())
    print(f"Competition options: {list(COMPETITION_IMPORTANCE.keys())}")
    competition = input("Competition type: ").strip() or "Domestic League"
    referee     = input("Referee name (press Enter to skip): ").strip() or None

    print(f"\n── {home_team} (HOME) ──")
    home_coach    = int(input("  Manager years in charge: ").strip() or 1)
    home_starters = int(input("  Regular starters retained from last season: ").strip() or 8)
    home_signings = int(input("  New signings last transfer window: ").strip() or 4)

    print(f"\n── {away_team} (AWAY) ──")
    away_coach    = int(input("  Manager years in charge: ").strip() or 1)
    away_starters = int(input("  Regular starters retained from last season: ").strip() or 8)
    away_signings = int(input("  New signings last transfer window: ").strip() or 4)

    return {
        "match_date":    match_date,
        "competition":   competition,
        "referee":       referee,
        "home_coach":    home_coach,
        "home_starters": home_starters,
        "home_signings": home_signings,
        "away_coach":    away_coach,
        "away_starters": away_starters,
        "away_signings": away_signings,
    }


# ============================================================
# SECTION 6: MASTER PREDICTION FUNCTION
# ============================================================

def predict_match(home_team, away_team):
    print(f"\n{'='*55}")
    print(f"  {home_team}  vs  {away_team}")
    print(f"{'='*55}")

    manual      = collect_manual_inputs(home_team, away_team)
    competition = manual["competition"]
    match_date  = manual["match_date"]
    is_domestic = competition in ["Domestic League", "Domestic Cup"]
    comp_score  = COMPETITION_IMPORTANCE.get(competition, 3)

    # ── Source 1: football-data.co.uk ────────────────────────
    fd         = FootballDataFetcher()
    home_log   = fd.team_match_log(home_team)
    away_log   = fd.team_match_log(away_team)
    h2h_home   = fd.h2h_results(home_team, away_team, home_team)
    h2h_away   = fd.h2h_results(home_team, away_team, away_team)
    last_home  = fd.last_match_date(home_team, match_date)
    last_away  = fd.last_match_date(away_team, match_date)
    league_avg = fd.league_avg_goals()

    if len(home_log) < 5:
        print(f"  Warning: only {len(home_log)} matches found for {home_team}.")
    if len(away_log) < 5:
        print(f"  Warning: only {len(away_log)} matches found for {away_team}.")

    # ── Source 2: API-Football ───────────────────────────────
    apif = APIFootballFetcher()

    home_season = apif.team_season_stats(home_team)
    away_season = apif.team_season_stats(away_team)
    home_m5     = apif.team_last5_stats(home_team)
    away_m5     = apif.team_last5_stats(away_team)
    home_gk     = apif.gk_stats(home_team)
    away_gk     = apif.gk_stats(away_team)

    # Missing players from API-Football injuries endpoint
    home_missing = apif.injuries(home_team)
    away_missing = apif.injuries(away_team)

    # Referee card rate
    if manual["referee"]:
        ref_rate = apif.referee_card_rate(manual["referee"])
    else:
        ref_rate = 2.3   # league average fallback

    # ── Source 3: xGscore.io ─────────────────────────────────
    xgs       = XGScoreFetcher()
    home_xgs  = xgs.team_stats(home_team)
    away_xgs  = xgs.team_stats(away_team)

    # Use xGscore xG if available, else fall back to goals proxy
    home_xg_val = home_xgs["xg"]  if home_xgs["xg"]  else f_xg_from_log(home_log)
    away_xg_val = away_xgs["xg"]  if away_xgs["xg"]  else f_xg_from_log(away_log)
    home_xga_val = home_xgs["xga"] if home_xgs["xga"] else f_xga_from_log(home_log)
    away_xga_val = away_xgs["xga"] if away_xgs["xga"] else f_xga_from_log(away_log)

    # Use xGscore possession if available, else API-Football
    home_poss = home_xgs["possession"]   if home_xgs["possession"] != 50.0 else home_m5["possession"]
    away_poss = away_xgs["possession"]   if away_xgs["possession"] != 50.0 else away_m5["possession"]
    home_pass = home_xgs["pass_accuracy"] if home_xgs["pass_accuracy"] != 75.0 else home_m5["pass_accuracy"]
    away_pass = away_xgs["pass_accuracy"] if away_xgs["pass_accuracy"] != 75.0 else away_m5["pass_accuracy"]

    # ── Assemble all 40 features ──────────────────────────────
    features = {
        # Form — football-data.co.uk
        "home_form":                 f_form(home_log),
        "away_form":                 f_form(away_log),

        # xG — xGscore (real) or goals proxy fallback
        "home_xg":                   round(home_xg_val, 4),
        "away_xg":                   round(away_xg_val, 4),

        # xGA — xGscore (real) or goals conceded proxy
        "home_xga":                  round(home_xga_val, 4),
        "away_xga":                  round(away_xga_val, 4),

        # xT — proxy formula from SOT + shots off target
        "home_xt":                   f_xt(home_log),
        "away_xt":                   f_xt(away_log),

        # SOT — football-data.co.uk
        "home_sot":                  f_sot(home_log),
        "away_sot":                  f_sot(away_log),

        # Attack / Defense Strength — API-Football season goals
        "home_attack_strength":      f_attack_strength(home_season.get("avg_goals", f_xg_from_log(home_log)), league_avg),
        "away_attack_strength":      f_attack_strength(away_season.get("avg_goals", f_xg_from_log(away_log)), league_avg),
        "home_defense_strength":     f_defense_strength(home_season.get("avg_conceded", f_xga_from_log(home_log)), league_avg),
        "away_defense_strength":     f_defense_strength(away_season.get("avg_conceded", f_xga_from_log(away_log)), league_avg),

        # Rest Days — football-data.co.uk dates
        "home_rest_days":            f_rest_days(last_home, match_date),
        "away_rest_days":            f_rest_days(last_away, match_date),

        # Missing Players — API-Football injuries
        "home_missing_players":      home_missing,
        "away_missing_players":      away_missing,

        # Possession — xGscore (preferred) or API-Football
        "home_average_possession":   f_possession(home_poss),
        "away_average_possession":   f_possession(away_poss),

        # Clean Sheet Rate — football-data.co.uk
        "home_clean_sheet_rate":     f_clean_sheet_rate(home_log),
        "away_clean_sheet_rate":     f_clean_sheet_rate(away_log),

        # H2H — football-data.co.uk
        "home_head_to_head_rate":    f_h2h(h2h_home),
        "away_head_to_head_rate":    f_h2h(h2h_away),

        # Referee Card Rate — API-Football or fallback
        "referee_card_rate":         ref_rate,

        # GK Rating — API-Football saves + clean sheet %
        "home_goalkeeper_rating":    f_gk_rating(home_gk["gk_save_pct"], home_gk["gk_cs_pct"], home_gk["gk_pen_save_pct"]),
        "away_goalkeeper_rating":    f_gk_rating(away_gk["gk_save_pct"], away_gk["gk_cs_pct"], away_gk["gk_pen_save_pct"]),

        # Finishing Efficiency — real xG from xGscore where available
        "home_finishing_efficiency": f_finishing(home_log, home_xgs["xg"]),
        "away_finishing_efficiency": f_finishing(away_log, away_xgs["xg"]),

        # Penalty Rating — API-Football conversion %
        "home_penalty_rating":       f_penalty_rating(home_season.get("penalty_conv_pct", 75.0), home_gk["gk_pen_save_pct"]),
        "away_penalty_rating":       f_penalty_rating(away_season.get("penalty_conv_pct", 75.0), away_gk["gk_pen_save_pct"]),

        # Set Piece Strength — API-Football corners as proxy
        "home_set_piece_strength":   f_set_piece(home_m5["avg_corners"], home_season.get("avg_goals", 1.2)),
        "away_set_piece_strength":   f_set_piece(away_m5["avg_corners"], away_season.get("avg_goals", 1.2)),

        # Press Resistance — xGscore pass accuracy + possession
        "home_press_resistance":     f_press_resistance(home_pass, home_poss),
        "away_press_resistance":     f_press_resistance(away_pass, away_poss),

        # Squad Chemistry — manual input
        "home_squad_chemistry":      f_squad_chemistry(manual["home_coach"], manual["home_starters"], manual["home_signings"]),
        "away_squad_chemistry":      f_squad_chemistry(manual["away_coach"], manual["away_starters"], manual["away_signings"]),

        # Travel Fatigue — GPS coordinates
        "home_travel_fatigue":       0,
        "away_travel_fatigue":       f_travel_fatigue(home_team, away_team, is_domestic),

        # Competition Importance — manual input
        "competition_importance":    comp_score,
    }

    df = pd.DataFrame([features])

    # ── Print feature summary ─────────────────────────────────
    print(f"\n{'─'*55}")
    print("COMPUTED FEATURES")
    print(f"{'─'*55}")
    print(f"  {'Feature':<35} {'Value':<12} Source")
    print(f"  {'─'*35} {'─'*12} {'─'*20}")

    source_map = {
        "home_form": "football-data.co.uk",
        "away_form": "football-data.co.uk",
        "home_xg": "xGscore.io" if home_xgs["xg"] else "goals proxy",
        "away_xg": "xGscore.io" if away_xgs["xg"] else "goals proxy",
        "home_xga": "xGscore.io" if home_xgs["xga"] else "conceded proxy",
        "away_xga": "xGscore.io" if away_xgs["xga"] else "conceded proxy",
        "home_xt": "formula proxy",
        "away_xt": "formula proxy",
        "home_sot": "football-data.co.uk",
        "away_sot": "football-data.co.uk",
        "home_attack_strength": "API-Football",
        "away_attack_strength": "API-Football",
        "home_defense_strength": "API-Football",
        "away_defense_strength": "API-Football",
        "home_rest_days": "football-data.co.uk",
        "away_rest_days": "football-data.co.uk",
        "home_missing_players": "API-Football",
        "away_missing_players": "API-Football",
        "home_average_possession": "xGscore.io / API-Football",
        "away_average_possession": "xGscore.io / API-Football",
        "home_clean_sheet_rate": "football-data.co.uk",
        "away_clean_sheet_rate": "football-data.co.uk",
        "home_head_to_head_rate": "football-data.co.uk",
        "away_head_to_head_rate": "football-data.co.uk",
        "referee_card_rate": "API-Football" if manual["referee"] else "default 2.3",
        "home_goalkeeper_rating": "API-Football",
        "away_goalkeeper_rating": "API-Football",
        "home_finishing_efficiency": "xGscore.io + football-data",
        "away_finishing_efficiency": "xGscore.io + football-data",
        "home_penalty_rating": "API-Football",
        "away_penalty_rating": "API-Football",
        "home_set_piece_strength": "API-Football corners proxy",
        "away_set_piece_strength": "API-Football corners proxy",
        "home_press_resistance": "xGscore.io / API-Football",
        "away_press_resistance": "xGscore.io / API-Football",
        "home_squad_chemistry": "manual input",
        "away_squad_chemistry": "manual input",
        "home_travel_fatigue": "GPS coordinates",
        "away_travel_fatigue": "GPS coordinates",
        "competition_importance": "manual input",
    }

    for k, v in features.items():
        src = source_map.get(k, "")
        print(f"  {k:<35} {str(v):<12} {src}")

    # ── Load model and predict ────────────────────────────────
    print(f"\n{'─'*55}")
    print("PREDICTION")
    print(f"{'─'*55}")

    if not os.path.exists(MODEL_PATH):
        print(f"\n  Model file '{MODEL_PATH}' not found.")
        print("  Save your model with: joblib.dump(Football_Model, 'football_prediction_model.pkl')")
        return df

    model       = joblib.load(MODEL_PATH)
    probs       = model.predict_proba(df)[0]
    pred        = model.predict(df)[0]
    outcome_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}

    print(f"\n  Away Win  ({away_team:<22}) :  {probs[0]*100:.1f}%")
    print(f"  Draw                           :  {probs[1]*100:.1f}%")
    print(f"  Home Win  ({home_team:<22}) :  {probs[2]*100:.1f}%")
    print(f"\n  ✓  Prediction → {outcome_map[pred]}")
    print(f"{'='*55}\n")

    return df, probs, pred


# ============================================================
# SECTION 7: BUILD TRAINING DATASET FROM REAL HISTORICAL DATA
# ============================================================

def build_training_dataset(output_path="training_data_real.csv"):
    print("\nBuilding training dataset...")
    fd         = FootballDataFetcher()
    df         = fd.fetch()
    league_avg = fd.league_avg_goals()
    apif       = APIFootballFetcher()
    xgs        = XGScoreFetcher()
    xgs.fetch()   # fetch once, reuse for all teams

    rows      = []
    df_sorted = df.sort_values("Date").reset_index(drop=True)

    for idx, match in df_sorted.iterrows():
        home = match.get("HomeTeam", "")
        away = match.get("AwayTeam", "")
        date = match.get("Date")

        if not home or not away or pd.isnull(date):
            continue

        home_log   = fd.team_match_log(home)
        away_log   = fd.team_match_log(away)
        home_prior = home_log[home_log["date"] < date]
        away_prior = away_log[away_log["date"] < date]

        if len(home_prior) < 5 or len(away_prior) < 5:
            continue

        h2h_home      = fd.h2h_results(home, away, home)
        h2h_away      = fd.h2h_results(home, away, away)
        last_home_d   = fd.last_match_date(home, str(date.date()))
        last_away_d   = fd.last_match_date(away, str(date.date()))
        home_s        = apif.team_season_stats(home)
        away_s        = apif.team_season_stats(away)
        home_m5       = apif.team_last5_stats(home)
        away_m5       = apif.team_last5_stats(away)
        home_gk       = apif.gk_stats(home)
        away_gk       = apif.gk_stats(away)
        home_xgs_data = xgs.team_stats(home)
        away_xgs_data = xgs.team_stats(away)

        home_xg_val  = home_xgs_data["xg"]  or f_xg_from_log(home_prior)
        away_xg_val  = away_xgs_data["xg"]  or f_xg_from_log(away_prior)
        home_xga_val = home_xgs_data["xga"] or f_xga_from_log(home_prior)
        away_xga_val = away_xgs_data["xga"] or f_xga_from_log(away_prior)
        home_poss    = home_xgs_data["possession"]    or home_m5["possession"]
        away_poss    = away_xgs_data["possession"]    or away_m5["possession"]
        home_pass    = home_xgs_data["pass_accuracy"] or home_m5["pass_accuracy"]
        away_pass    = away_xgs_data["pass_accuracy"] or away_m5["pass_accuracy"]

        ftr    = match.get("FTR", "D")
        result = {"H": 2, "D": 1, "A": 0}.get(ftr, 1)

        row = {
            "home_form":                 f_form(home_prior),
            "away_form":                 f_form(away_prior),
            "home_xg":                   round(home_xg_val, 4),
            "away_xg":                   round(away_xg_val, 4),
            "home_xga":                  round(home_xga_val, 4),
            "away_xga":                  round(away_xga_val, 4),
            "home_xt":                   f_xt(home_prior),
            "away_xt":                   f_xt(away_prior),
            "home_sot":                  f_sot(home_prior),
            "away_sot":                  f_sot(away_prior),
            "home_attack_strength":      f_attack_strength(home_s.get("avg_goals", 1.2), league_avg),
            "away_attack_strength":      f_attack_strength(away_s.get("avg_goals", 1.2), league_avg),
            "home_defense_strength":     f_defense_strength(home_s.get("avg_conceded", 1.2), league_avg),
            "away_defense_strength":     f_defense_strength(away_s.get("avg_conceded", 1.2), league_avg),
            "home_rest_days":            f_rest_days(last_home_d, date),
            "away_rest_days":            f_rest_days(last_away_d, date),
            "home_missing_players":      0,
            "away_missing_players":      0,
            "home_average_possession":   f_possession(home_poss),
            "away_average_possession":   f_possession(away_poss),
            "home_clean_sheet_rate":     f_clean_sheet_rate(home_prior),
            "away_clean_sheet_rate":     f_clean_sheet_rate(away_prior),
            "home_head_to_head_rate":    f_h2h(h2h_home),
            "away_head_to_head_rate":    f_h2h(h2h_away),
            "referee_card_rate":         2.3,
            "home_goalkeeper_rating":    f_gk_rating(home_gk["gk_save_pct"], home_gk["gk_cs_pct"], home_gk["gk_pen_save_pct"]),
            "away_goalkeeper_rating":    f_gk_rating(away_gk["gk_save_pct"], away_gk["gk_cs_pct"], away_gk["gk_pen_save_pct"]),
            "home_finishing_efficiency": f_finishing(home_prior, home_xgs_data["xg"]),
            "away_finishing_efficiency": f_finishing(away_prior, away_xgs_data["xg"]),
            "home_penalty_rating":       f_penalty_rating(home_s.get("penalty_conv_pct", 75.0), home_gk["gk_pen_save_pct"]),
            "away_penalty_rating":       f_penalty_rating(away_s.get("penalty_conv_pct", 75.0), away_gk["gk_pen_save_pct"]),
            "home_set_piece_strength":   f_set_piece(home_m5["avg_corners"], home_s.get("avg_goals", 1.2)),
            "away_set_piece_strength":   f_set_piece(away_m5["avg_corners"], away_s.get("avg_goals", 1.2)),
            "home_press_resistance":     f_press_resistance(home_pass, home_poss),
            "away_press_resistance":     f_press_resistance(away_pass, away_poss),
            "home_squad_chemistry":      3,
            "away_squad_chemistry":      3,
            "home_travel_fatigue":       0,
            "away_travel_fatigue":       f_travel_fatigue(home, away, True),
            "competition_importance":    3,
            "target":                    result,
        }
        rows.append(row)

        if idx % 50 == 0:
            print(f"  Processed {idx}/{len(df_sorted)} matches...")

    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    print(f"\n  Training dataset saved → {output_path}")
    print(f"  Shape: {out.shape}")
    print(f"  Class distribution:\n{out['target'].value_counts().sort_index()}")
    return out


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  FOOTBALL MATCH PREDICTION PIPELINE v2")
    print("  Powei Shadrack | SunSpot-Tech")
    print("="*55)
    print("\n  Sources:")
    print("    1. football-data.co.uk  → form, SOT, H2H")
    print("    2. API-Football         → possession, saves, injuries")
    print("    3. xGscore.io           → xG, xGA, pass accuracy")
    print("    4. Formula proxies      → xT, set piece")
    print("    5. Manual inputs        → chemistry, competition")
    print("\n  Modes:")
    print("    1. Predict a single match")
    print("    2. Build training dataset from real data")
    mode = input("\nEnter 1 or 2: ").strip()

    if mode == "1":
        home = input("Home team: ").strip()
        away = input("Away team: ").strip()
        predict_match(home, away)

    elif mode == "2":
        out_path = input("Output CSV filename (press Enter for default): ").strip()
        build_training_dataset(out_path or "training_data_real.csv")

    else:
        print("Invalid option. Enter 1 or 2.")