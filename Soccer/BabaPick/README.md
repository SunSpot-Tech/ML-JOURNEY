# BabaPick ⚽
**Smarter Insights, Better Decisions**

AI-powered football match prediction platform by Powei Shadrack | SunSpot-Tech

---

## Project Structure

```
babapick/
├── main.py                        ← FastAPI backend (all API endpoints)
├── pipeline.py                    ← Feature engineering pipeline (all 3 data sources)
├── streamlit_app.py               ← Streamlit frontend UI
├── football_prediction_model.pkl  ← XGBoost model (you must place this here)
├── requirements.txt               ← Python dependencies
├── render.yaml                    ← Render deployment config (2 services)
├── .env.example                   ← Environment variables template
└── README.md                      ← This file
```

---

## Setup (Local Development)

### 1. Clone and install
```bash
git clone <your-repo>
cd babapick
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Place your model
Copy your trained XGBoost model to the project root:
```bash
cp /path/to/football_prediction_model.pkl .
```

### 4. Run FastAPI backend
```bash
uvicorn main:app --reload --port 8000
```
API docs available at: http://localhost:8000/docs

### 5. Run Streamlit frontend (separate terminal)
```bash
streamlit run streamlit_app.py
```
UI available at: http://localhost:8501

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/` | No | Root / health check |
| GET | `/health` | No | Detailed health check |
| GET | `/docs` | No | Swagger UI |
| POST | `/auth/login` | No | Get JWT token |
| GET | `/api/v1/leagues` | No | List supported leagues |
| GET | `/api/v1/competition-types` | No | List competition types |
| POST | `/api/v1/predict` | Yes | Single match prediction |
| POST | `/api/v1/predict/batch` | Yes | Batch predictions (max 10) |

### Example: Login
```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "demo", "password": "babapick2026"}'
```

### Example: Predict
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "home_team": "Arsenal",
    "away_team": "Chelsea",
    "league": "EPL",
    "match_date": "2026-08-10",
    "competition_type": "Domestic League",
    "home_coach_years": 3,
    "home_starters_retained": 9,
    "home_new_signings": 2,
    "away_coach_years": 1,
    "away_starters_retained": 7,
    "away_new_signings": 6,
    "referee": "Michael Oliver"
  }'
```

---

## Deployment on Render

### Two services required:
1. **babapick-api** — FastAPI backend (Web Service)
2. **babapick-ui** — Streamlit frontend (Web Service)

### Steps:
1. Push code to GitHub
2. Go to https://render.com → New → Blueprint
3. Connect your GitHub repo
4. Render reads `render.yaml` and creates both services automatically
5. In the **babapick-api** service → Environment → add:
   - `API_FOOTBALL_KEY` = your key from dashboard.api-football.com
   - `JWT_SECRET` = any long random string
6. In the **babapick-ui** service → Environment → update:
   - `API_BASE_URL` = your babapick-api Render URL (e.g. https://babapick-api.onrender.com)
7. Upload `football_prediction_model.pkl` via Render disk or include in repo

### Important Render Notes:
- Free tier services spin down after 15 min inactivity — first request will be slow
- Upgrade to Starter ($7/month) for always-on
- Model file must be present at startup — include in repo or mount a Render disk

---

## Data Sources

| Source | What it provides | Leagues |
|--------|-----------------|---------|
| football-data.co.uk | Results, form, SOT, shots, H2H | 22 European divisions |
| API-Football v3 | Possession, saves, injuries, referee, penalties, corners | 1,200+ |
| xGscore.io | xG, xGA, pass accuracy, possession | Top 15 European leagues |

---

## Default Login (Demo)
- Username: `demo`
- Password: `babapick2026`

**Change these in `main.py → USERS` before production deployment.**

---

## Prediction Output

```json
{
  "match": "Arsenal vs Chelsea",
  "league": "EPL",
  "date": "2026-08-10",
  "prediction": "Home Win",
  "probabilities": {
    "away_win": 8.2,
    "draw": 23.1,
    "home_win": 68.7
  },
  "features": { ... all 40 feature values ... },
  "data_sources": { ... source per feature ... },
  "disclaimer": "For informational purposes only. Not financial or betting advice."
}
```

---

## Target Variable
- `0` = Away Win
- `1` = Draw
- `2` = Home Win

---

## Tech Stack
- **Backend:** FastAPI + Uvicorn + Python 3.11
- **Frontend:** Streamlit
- **ML:** XGBoost (multi:softprob)
- **Auth:** JWT (python-jose)
- **Deployment:** Render (2 web services)
- **Data:** football-data.co.uk + API-Football v3 + xGscore.io

---

*BabaPick v1.0 | Powei Shadrack | SunSpot-Tech*
*For informational purposes only. Not financial or betting advice.*
