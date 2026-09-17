# ============================================================
# BabaPick — FastAPI Backend
# Smarter Insights, Better Decisions
# Author: Powei Shadrack | SunSpot-Tech
# ============================================================

import os
import time
from collections import defaultdict, deque
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from jose import JWTError, jwt
from hashlib import sha256
import hmac

from pipeline import (
    FootballDataFetcher, APIFootballFetcher, XGScoreFetcher,
    f_form, f_xg_from_log, f_xga_from_log, f_xt, f_sot,
    f_attack_strength, f_defense_strength, f_rest_days,
    f_possession, f_clean_sheet_rate, f_h2h, f_gk_rating,
    f_finishing, f_penalty_rating, f_set_piece, f_press_resistance,
    f_squad_chemistry, f_travel_fatigue, COMPETITION_IMPORTANCE,
    FOOTBALL_DATA_LEAGUES
)

app = FastAPI()

@app.get("/health")
def health_check():
    return {"model": "loaded"}

# ── Config ───────────────────────────────────────────────────
SECRET_KEY      = os.getenv("JWT_SECRET", "babapick-secret-change-in-production")
ALGORITHM       = "HS256"
ACCESS_TOKEN_TTL = 60  # minutes
MODEL_PATH      = os.getenv("MODEL_PATH", "football_prediction_model.pkl")
bearer = HTTPBearer(auto_error=False)

# Base44-to-Render proxy configuration. Keep this secret in Render environment variables.
BASE44_PROXY_KEY = os.getenv("BASE44_PROXY_KEY")
BASE44_RATE_LIMIT = int(os.getenv("BASE44_RATE_LIMIT", "30"))
BASE44_RATE_WINDOW_SECONDS = int(os.getenv("BASE44_RATE_WINDOW_SECONDS", "3600"))
_proxy_requests = defaultdict(deque)


def verify_base44_proxy_key(x_base44_proxy_key: Optional[str] = Header(default=None)):
    """Authenticate the low-cost Base44 proxy using a Render-only secret."""
    if not BASE44_PROXY_KEY:
        raise HTTPException(status_code=503, detail="Base44 proxy is not configured")
    if not x_base44_proxy_key or not hmac.compare_digest(x_base44_proxy_key, BASE44_PROXY_KEY):
        raise HTTPException(status_code=401, detail="Invalid Base44 proxy key")

    # Basic per-key rate limit. Use a persistent rate limiter for multi-instance deployments.
    now = time.time()
    events = _proxy_requests[x_base44_proxy_key]
    while events and now - events[0] >= BASE44_RATE_WINDOW_SECONDS:
        events.popleft()
    if len(events) >= BASE44_RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Base44 prediction rate limit reached")
    events.append(now)
    return True

# ── Load model at startup ─────────────────────────────────────
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"✓ Model loaded from {MODEL_PATH}")
else:
    print(f"⚠ Model not found at {MODEL_PATH} — predictions will fail")

# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="BabaPick API",
    description="AI-powered football match prediction. Smarter Insights, Better Decisions.",
    version="1.0.0",
    contact={"name": "Powei Shadrack", "email": "powei@sunspot-tech.com"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple password hashing — no bcrypt dependency
def hash_password(password: str) -> str:
    return sha256(password.encode()).hexdigest()

def verify_password(plain: str, hashed: str) -> bool:
    return hmac.compare_digest(hash_password(plain), hashed)

USERS = {
    "demo":  {"password": hash_password("babapick2026"), "tier": "pro"},
    "admin": {"password": hash_password("admin123"),     "tier": "admin"},
}

def create_token(username: str) -> str:
    expire  = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_TTL)
    payload = {"sub": username, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        payload  = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username or username not in USERS:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
# ── Pydantic Models ───────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str

class PredictRequest(BaseModel):
    home_team:             str   = Field(..., example="Arsenal")
    away_team:             str   = Field(..., example="Chelsea")
    league:                str   = Field(..., example="EPL")
    match_date:            str   = Field(..., example="2026-08-10")
    competition_type:      str   = Field("Domestic League", example="Domestic League")
    home_coach_years:      int   = Field(1,   ge=0, le=50,  example=3)
    home_starters_retained:int   = Field(8,   ge=0, le=26,  example=9)
    home_new_signings:     int   = Field(4,   ge=0, le=26,  example=2)
    away_coach_years:      int   = Field(1,   ge=0, le=50,  example=1)
    away_starters_retained:int   = Field(8,   ge=0, le=26,  example=7)
    away_new_signings:     int   = Field(4,   ge=0, le=26,  example=6)
    referee:               Optional[str] = Field(None, example="Michael Oliver")

    @validator("competition_type")
    def validate_competition(cls, v):
        if v not in COMPETITION_IMPORTANCE:
            raise ValueError(f"competition_type must be one of: {list(COMPETITION_IMPORTANCE.keys())}")
        return v

    @validator("match_date")
    def validate_date(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("match_date must be in YYYY-MM-DD format")
        return v

class BatchPredictRequest(BaseModel):
    fixtures: list[PredictRequest]

class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    expires_in:   int = ACCESS_TOKEN_TTL * 60

# ── Helper: build feature row ─────────────────────────────────

from cache_reader import get_cache

def build_features(req: PredictRequest) -> tuple[pd.DataFrame, dict, dict]:
    cache      = get_cache()
    league_cfg = FOOTBALL_DATA_LEAGUES.get(req.league, ("E0", "epl", 39, 2025))
    _, xgs_slug, apif_league_id, apif_season = league_cfg
    league     = req.league
    is_domestic = req.competition_type in ["Domestic League", "Domestic Cup"]
    comp_score  = COMPETITION_IMPORTANCE.get(req.competition_type, 3)

    # Match logs
    home_log = cache.team_match_log(league, req.home_team)
    away_log = cache.team_match_log(league, req.away_team)

    # H2H
    h2h_home = cache.h2h(league, req.home_team, req.away_team, req.home_team)
    h2h_away = cache.h2h(league, req.home_team, req.away_team, req.away_team)

    # Rest days
    last_home  = cache.last_match_date(league, req.home_team, req.match_date)
    last_away  = cache.last_match_date(league, req.away_team, req.match_date)
    league_avg = cache.league_avg_goals(league)

    # Season stats
    home_season = cache.apif_season_stats(league, req.home_team)
    away_season = cache.apif_season_stats(league, req.away_team)
    home_m5     = cache.apif_match_stats(league, req.home_team)
    away_m5     = cache.apif_match_stats(league, req.away_team)
    home_gk     = cache.apif_gk_stats(league, req.home_team)
    away_gk     = cache.apif_gk_stats(league, req.away_team)
    home_missing = cache.apif_injuries(league, req.home_team)
    away_missing = cache.apif_injuries(league, req.away_team)

    # xGscore
    home_xgs  = cache.xgs_team(league, req.home_team)
    away_xgs  = cache.xgs_team(league, req.away_team)

    home_xg_val  = home_xgs.get("xg")  or f_xg_from_log(home_log)
    away_xg_val  = away_xgs.get("xg")  or f_xg_from_log(away_log)
    home_xga_val = home_xgs.get("xga") or f_xga_from_log(home_log)
    away_xga_val = away_xgs.get("xga") or f_xga_from_log(away_log)
    home_poss    = home_xgs.get("possession", 50.0)    or home_m5["possession"]
    away_poss    = away_xgs.get("possession", 50.0)    or away_m5["possession"]
    home_pass    = home_xgs.get("pass_accuracy", 75.0) or home_m5["pass_accuracy"]
    away_pass    = away_xgs.get("pass_accuracy", 75.0) or away_m5["pass_accuracy"]

    ref_rate = 2.3
    if req.referee:
        apif = APIFootballFetcher(league=apif_league_id, season=apif_season)
        ref_rate = apif.referee_card_rate(req.referee)

    features = {
        "home_form":                 f_form(home_log),
        "away_form":                 f_form(away_log),
        "home_xg":                   round(float(home_xg_val), 4),
        "away_xg":                   round(float(away_xg_val), 4),
        "home_xga":                  round(float(home_xga_val), 4),
        "away_xga":                  round(float(away_xga_val), 4),
        "home_xt":                   f_xt(home_log),
        "away_xt":                   f_xt(away_log),
        "home_sot":                  f_sot(home_log),
        "away_sot":                  f_sot(away_log),
        "home_attack_strength":      f_attack_strength(home_season["avg_goals"], league_avg),
        "away_attack_strength":      f_attack_strength(away_season["avg_goals"], league_avg),
        "home_defense_strength":     f_defense_strength(home_season["avg_conceded"], league_avg),
        "away_defense_strength":     f_defense_strength(away_season["avg_conceded"], league_avg),
        "home_rest_days":            f_rest_days(last_home, req.match_date),
        "away_rest_days":            f_rest_days(last_away, req.match_date),
        "home_missing_players":      home_missing,
        "away_missing_players":      away_missing,
        "home_average_possession":   f_possession(home_poss),
        "away_average_possession":   f_possession(away_poss),
        "home_clean_sheet_rate":     f_clean_sheet_rate(home_log),
        "away_clean_sheet_rate":     f_clean_sheet_rate(away_log),
        "home_head_to_head_rate":    f_h2h(h2h_home),
        "away_head_to_head_rate":    f_h2h(h2h_away),
        "referee_card_rate":         ref_rate,
        "home_goalkeeper_rating":    f_gk_rating(home_gk["gk_save_pct"], home_gk["gk_cs_pct"], home_gk["gk_pen_save_pct"]),
        "away_goalkeeper_rating":    f_gk_rating(away_gk["gk_save_pct"], away_gk["gk_cs_pct"], away_gk["gk_pen_save_pct"]),
        "home_finishing_efficiency": f_finishing(home_log, home_xgs.get("xg")),
        "away_finishing_efficiency": f_finishing(away_log, away_xgs.get("xg")),
        "home_penalty_rating":       f_penalty_rating(home_season["penalty_conv_pct"], home_gk["gk_pen_save_pct"]),
        "away_penalty_rating":       f_penalty_rating(away_season["penalty_conv_pct"], away_gk["gk_pen_save_pct"]),
        "home_set_piece_strength":   f_set_piece(home_m5["avg_corners"], home_season["avg_goals"]),
        "away_set_piece_strength":   f_set_piece(away_m5["avg_corners"], away_season["avg_goals"]),
        "home_press_resistance":     f_press_resistance(home_pass, home_poss),
        "away_press_resistance":     f_press_resistance(away_pass, away_poss),
        "home_squad_chemistry":      f_squad_chemistry(req.home_coach_years, req.home_starters_retained, req.home_new_signings),
        "away_squad_chemistry":      f_squad_chemistry(req.away_coach_years, req.away_starters_retained, req.away_new_signings),
        "home_travel_fatigue":       0,
        "away_travel_fatigue":       f_travel_fatigue(req.home_team, req.away_team, is_domestic),
        "competition_importance":    comp_score,
    }

    source_map = {k: "cache" for k in features}
    return pd.DataFrame([features]), features, source_map

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "model": "loaded" if model else "missing", "timestamp": datetime.utcnow().isoformat()}


@app.post("/auth/login", response_model=TokenResponse, tags=["Auth"])
def login(req: LoginRequest):
    user = USERS.get(req.username)
    if not user or not verify_password(req.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return TokenResponse(access_token=create_token(req.username))


@app.get("/api/v1/leagues", tags=["Data"])
def get_leagues():
    return {
        "leagues": [
            {"code": code, "fd_code": cfg[0], "xgscore_slug": cfg[1],
             "apif_id": cfg[2], "apif_season": cfg[3]}
            for code, cfg in FOOTBALL_DATA_LEAGUES.items()
        ]
    }


@app.get("/api/v1/competition-types", tags=["Data"])
def get_competition_types():
    return {"competition_types": list(COMPETITION_IMPORTANCE.keys())}


@app.post("/api/v1/predict", tags=["Prediction"])
def predict(req: PredictRequest, user: str = Depends(get_current_user)):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded. Upload football_prediction_model.pkl.")

    if req.league not in FOOTBALL_DATA_LEAGUES:
        raise HTTPException(
            status_code=400,
            detail=f"League '{req.league}' not supported. Available: {list(FOOTBALL_DATA_LEAGUES.keys())}"
        )

    try:
        df, features, sources = build_features(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature engineering failed: {str(e)}")

    try:
        probs = model.predict_proba(df)[0]
        pred  = int(model.predict(df)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    outcome_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}

    return {
        "match":      f"{req.home_team} vs {req.away_team}",
        "league":     req.league,
        "date":       req.match_date,
        "prediction": outcome_map[pred],
        "probabilities": {
            "away_win":  round(float(probs[0]) * 100, 2),
            "draw":      round(float(probs[1]) * 100, 2),
            "home_win":  round(float(probs[2]) * 100, 2),
        },
        "features":       features,
        "data_sources":   sources,
        "model_version":  "1.0.0",
        "predicted_by":   user,
        "timestamp":      datetime.utcnow().isoformat(),
        "disclaimer":     "For informational purposes only. Not financial or betting advice.",
    }




@app.post("/api/v1/base44/predict", tags=["Prediction"])
def base44_predict(
    req: PredictRequest,
    _: bool = Depends(verify_base44_proxy_key),
):
    """Public-facing bridge for Base44 when Base44 server functions are unavailable.

    Base44 sends X-Base44-Proxy-Key. Render keeps the real API credentials and
    calls the existing prediction implementation without exposing a JWT.
    """
    return predict(req, user="base44")


@app.post("/api/v1/predict/batch", tags=["Prediction"])
def predict_batch(req: BatchPredictRequest, user: str = Depends(get_current_user)):
    if len(req.fixtures) > 10:
        raise HTTPException(status_code=400, detail="Batch limited to 10 fixtures per request.")
    results = []
    for fixture in req.fixtures:
        try:
            result = predict(fixture, user)
            results.append(result)
        except HTTPException as e:
            results.append({"error": e.detail, "match": f"{fixture.home_team} vs {fixture.away_team}"})
    return {"predictions": results, "count": len(results)}