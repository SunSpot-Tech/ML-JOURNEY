# Soccer Match Prediction API

FastAPI wrapper around the trained `Football_Model.pkl` XGBoost model from `betting.ipynb`.

---

## Project structure

```
.
├── main.py                          # FastAPI application
├── requirements.txt                 # Python dependencies
├── Football_Model.pkl            # Trained model (you generate this from the notebook)
└── soccer_match_dataset_1000_balanced.csv  # (only needed to retrain)
```

---

## Quick start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Make sure the model file is present
Run the notebook (`betting.ipynb`) through to the `joblib.dump` cell so that
`Football_Model.pkl` is saved in the same directory as `main.py`.

### 3. Run the API
```bash
uvicorn main:app --reload
```

The API is now live at **http://127.0.0.1:8000**

---

## Interactive docs

| URL | Description |
|-----|-------------|
| http://127.0.0.1:8000/docs | Swagger UI — try the endpoints directly |
| http://127.0.0.1:8000/redoc | ReDoc — clean reference docs |

---

## Endpoints

### `GET /health`
Check whether the API is running and the model is loaded.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "Football_Model.pkl"
}
```

---

### `POST /predict`
Predict a match outcome.

**Request body:**
```json
{
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
  "referee_card_rate": 4.74
}
```

**Response:**
```json
{
  "prediction": "Away Win",
  "probabilities": {
    "Away Win": 0.4812,
    "Draw": 0.2134,
    "Home Win": 0.3054
  },
  "confidence": 0.4812
}
```

---

### `GET /features`
Returns the ordered list of all 21 input features.

---

## Environment variable

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `Football_Model.pkl` | Path to the saved `.pkl` model file |

Example:
```bash
MODEL_PATH=/models/Football_Model.pkl uvicorn main:app
```

---

## Deploying to Render

1. Push `main.py` and `requirements.txt` to a GitHub repo.
2. Add your `Football_Model.pkl` to the repo (or load it from a cloud bucket).
3. On Render → **New Web Service** → connect the repo.
4. Set **Start command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Deploy.
