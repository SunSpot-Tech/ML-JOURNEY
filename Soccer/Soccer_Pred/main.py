from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(
    title="Soccer Match Prediction API",
    description="Predicts match outcomes (Home Win, Draw, Away Win) using a trained RandomForest model.",
    version="1.0.0"
)

# ---------------------------------------------------------------------------
# Load model on startup
# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "Soccer_Prediction.pkl")

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None  # Will raise a clear error at prediction time

# Feature order must match training
FEATURES = [
    "home_form_5g", "away_form_5g",
    "home_xg", "away_xg",
    "home_xga", "away_xga",
    "home_xt", "away_xt",
    "home_sot", "away_sot",
    "home_attack_strength", "away_attack_strength",
    "home_defense_strength", "away_defense_strength",
    "rest_days_home", "rest_days_away",
    "travel_distance_away_km",
    "h2h_home_win_ratio",
    "missing_key_players_home", "missing_key_players_away",
    "referee_card_rate",
]

OUTCOME_MAP = {0: "Away Win", 1: "Draw", 2: "Home Win"}

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class MatchFeatures(BaseModel):
    home_form_5g: float = Field(..., ge=0, le=1, description="Home team form over last 5 games (0–1)")
    away_form_5g: float = Field(..., ge=0, le=1, description="Away team form over last 5 games (0–1)")
    home_xg: float = Field(..., ge=0, description="Expected goals for home team")
    away_xg: float = Field(..., ge=0, description="Expected goals for away team")
    home_xga: float = Field(..., ge=0, description="Expected goals against for home team")
    away_xga: float = Field(..., ge=0, description="Expected goals against for away team")
    home_xt: float = Field(..., ge=0, description="Expected threat score for home team")
    away_xt: float = Field(..., ge=0, description="Expected threat score for away team")
    home_sot: float = Field(..., ge=0, description="Shots on target for home team")
    away_sot: float = Field(..., ge=0, description="Shots on target for away team")
    home_attack_strength: float = Field(..., ge=0, description="Home attack strength index")
    away_attack_strength: float = Field(..., ge=0, description="Away attack strength index")
    home_defense_strength: float = Field(..., ge=0, description="Home defense strength index")
    away_defense_strength: float = Field(..., ge=0, description="Away defense strength index")
    rest_days_home: int = Field(..., ge=0, description="Rest days for home team since last match")
    rest_days_away: int = Field(..., ge=0, description="Rest days for away team since last match")
    travel_distance_away_km: float = Field(..., ge=0, description="Travel distance for away team (km)")
    h2h_home_win_ratio: float = Field(..., ge=0, le=1, description="Head-to-head home win ratio (0–1)")
    missing_key_players_home: int = Field(..., ge=0, description="Number of key players missing for home team")
    missing_key_players_away: int = Field(..., ge=0, description="Number of key players missing for away team")
    referee_card_rate: float = Field(..., ge=0, description="Referee's average cards per game")

    class Config:
        json_schema_extra = {
            "example": {
                "home_form_5g": 0.67,
                "away_form_5g": 0.73,
                "home_xg": 1.73,
                "away_xg": 2.03,
                "home_xga": 1.3,
                "away_xga": 1.0,
                "home_xt": 1.93,
                "away_xt": 0.96,
                "home_sot": 6.7,
                "away_sot": 5.3,
                "home_attack_strength": 1.13,
                "away_attack_strength": 0.68,
                "home_defense_strength": 1.05,
                "away_defense_strength": 1.4,
                "rest_days_home": 3,
                "rest_days_away": 4,
                "travel_distance_away_km": 0,
                "h2h_home_win_ratio": 0,
                "missing_key_players_home": 1,
                "missing_key_players_away": 2,
                "referee_card_rate": 4.74,
            }
        }


class PredictionResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]
    confidence: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Soccer Match Prediction API is running."}


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(match: MatchFeatures):
    """
    Predict the outcome of a soccer match.

    Returns:
    - **prediction**: "Home Win", "Draw", or "Away Win"
    - **probabilities**: probability for each outcome
    - **confidence**: probability of the predicted class
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Make sure '{MODEL_PATH}' exists in the working directory."
        )

    # Build input DataFrame in correct feature order
    input_data = pd.DataFrame([match.model_dump()])[FEATURES]

    probs = model.predict_proba(input_data)[0]
    pred_class = int(model.predict(input_data)[0])

    classes = model.classes_
    prob_dict = {OUTCOME_MAP[int(c)]: round(float(p), 4) for c, p in zip(classes, probs)}

    return PredictionResponse(
        prediction=OUTCOME_MAP[pred_class],
        probabilities=prob_dict,
        confidence=round(float(probs[list(classes).index(pred_class)]), 4),
    )


@app.get("/features", tags=["Info"])
def list_features():
    """Return the list of required input features and their order."""
    return {"features": FEATURES, "total": len(FEATURES)}
