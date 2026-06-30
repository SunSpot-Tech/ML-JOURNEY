import streamlit as st
import requests
import os
import csv
from datetime import datetime, timezone
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
API_BASE_URL = "https://soccer-prediction-api-w74s.onrender.com"

# ---------------------------------------------------------------------------
# Data logging config (for future retraining / drift monitoring)
# ---------------------------------------------------------------------------
# Primary store: a Google Sheet (persists across redeploys/restarts).
# Fallback store: local CSV, used only if the Sheets write fails so a
# prediction is never silently lost.
SHEET_NAME = "soccer_prediction_log"   # the Google Sheet's title
WORKSHEET_NAME = "predictions"          # the tab within that sheet

DATA_DIR = "prediction_logs"
LOG_PATH = os.path.join(DATA_DIR, "prediction_log.csv")

# Columns: raw model inputs + model outputs + metadata.
# `actual_result` is left blank at prediction time and can be back-filled
# later (e.g. once the real match result is known) to build a labeled
# dataset for retraining.
LOG_COLUMNS = [
    "timestamp_utc",
    "home_country",
    "away_country",
    "home_form_5g",
    "away_form_5g",
    "home_xg",
    "away_xg",
    "home_xga",
    "away_xga",
    "home_xt",
    "away_xt",
    "home_sot",
    "away_sot",
    "home_attack_strength",
    "away_attack_strength",
    "home_defense_strength",
    "away_defense_strength",
    "rest_days_home",
    "rest_days_away",
    "travel_distance_away_km",
    "h2h_home_win_ratio",
    "missing_key_players_home",
    "missing_key_players_away",
    "referee_card_rate",
    "predicted_outcome",
    "prob_home_win",
    "prob_draw",
    "prob_away_win",
    "confidence",
    "actual_result",  # to be filled in later for retraining/drift checks
]


def init_log_file():
    """Create the local CSV fallback file with a header row if it doesn't exist yet."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.isfile(LOG_PATH):
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            writer.writeheader()


def log_prediction_to_csv(row: dict):
    """Append one row to the local CSV fallback."""
    init_log_file()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writerow(row)


@st.cache_resource(show_spinner=False)
def get_sheets_worksheet():
    """
    Authenticate to Google Sheets with a service account and return the
    target worksheet, creating the sheet/tab/header if they don't exist yet.

    Expects credentials in st.secrets["gcp_service_account"] (the JSON key
    downloaded from Google Cloud Console for the service account), e.g. in
    .streamlit/secrets.toml:

        [gcp_service_account]
        type = "service_account"
        project_id = "..."
        private_key_id = "..."
        private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
        client_email = "...@....iam.gserviceaccount.com"
        client_id = "..."
        token_uri = "https://oauth2.googleapis.com/token"

    The Sheet itself must be shared (Editor access) with that
    client_email, or this will raise a permissions error.
    """
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=scopes
    )
    client = gspread.authorize(creds)

    try:
        sheet = client.open(SHEET_NAME)
    except gspread.SpreadsheetNotFound:
        sheet = client.create(SHEET_NAME)

    try:
        worksheet = sheet.worksheet(WORKSHEET_NAME)
    except gspread.WorksheetNotFound:
        worksheet = sheet.add_worksheet(
            title=WORKSHEET_NAME, rows=1000, cols=len(LOG_COLUMNS)
        )
        worksheet.append_row(LOG_COLUMNS)

    # Make sure the header row exists (e.g. worksheet existed but was empty).
    if worksheet.row_count == 0 or not worksheet.row_values(1):
        worksheet.append_row(LOG_COLUMNS)

    return worksheet


def log_prediction_to_sheets(row: dict):
    """Append one row to the Google Sheet, in LOG_COLUMNS order."""
    worksheet = get_sheets_worksheet()
    worksheet.append_row([row.get(col, "") for col in LOG_COLUMNS])


def log_prediction(payload: dict, home_country: str, away_country: str, data: dict):
    """
    Record one prediction (inputs + outputs) for future retraining.
    Tries Google Sheets first (persistent across redeploys); if that fails
    for any reason, falls back to the local CSV so the prediction is never
    silently lost.
    """
    probs = data.get("probabilities", {})
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "home_country": home_country,
        "away_country": away_country,
        "home_form_5g": payload["home_form_5g"],
        "away_form_5g": payload["away_form_5g"],
        "home_xg": payload["home_xg"],
        "away_xg": payload["away_xg"],
        "home_xga": payload["home_xga"],
        "away_xga": payload["away_xga"],
        "home_xt": payload["home_xt"],
        "away_xt": payload["away_xt"],
        "home_sot": payload["home_sot"],
        "away_sot": payload["away_sot"],
        "home_attack_strength": payload["home_attack_strength"],
        "away_attack_strength": payload["away_attack_strength"],
        "home_defense_strength": payload["home_defense_strength"],
        "away_defense_strength": payload["away_defense_strength"],
        "rest_days_home": payload["rest_days_home"],
        "rest_days_away": payload["rest_days_away"],
        "travel_distance_away_km": payload["travel_distance_away_km"],
        "h2h_home_win_ratio": payload["h2h_home_win_ratio"],
        "missing_key_players_home": payload["missing_key_players_home"],
        "missing_key_players_away": payload["missing_key_players_away"],
        "referee_card_rate": payload["referee_card_rate"],
        "predicted_outcome": data.get("prediction"),
        "prob_home_win": probs.get("Home Win"),
        "prob_draw": probs.get("Draw"),
        "prob_away_win": probs.get("Away Win"),
        "confidence": data.get("confidence"),
        "actual_result": "",  # filled in later once the real outcome is known
    }

    if "gcp_service_account" in st.secrets:
        try:
            log_prediction_to_sheets(row)
            return "sheets"
        except Exception as sheets_err:
            st.warning(
                f"Couldn't write to Google Sheets ({sheets_err}); "
                "saved to local CSV instead."
            )

    log_prediction_to_csv(row)
    return "csv"

st.set_page_config(
    page_title="Soccer Match Predictor",
    page_icon="⚽",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .header h1 { color: #e2e8f0; font-size: 2.2rem; margin: 0; }
    .header p  { color: #94a3b8; margin: 0.4rem 0 0; font-size: 1rem; }

    .team-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .team-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .home-label { color: #38bdf8; }
    .away-label { color: #fb923c; }

    .result-box {
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .result-home { background: linear-gradient(135deg, #0c4a6e, #075985); border: 1px solid #0369a1; }
    .result-draw { background: linear-gradient(135deg, #1c1917, #292524); border: 1px solid #57534e; }
    .result-away { background: linear-gradient(135deg, #7c2d12, #9a3412); border: 1px solid #c2410c; }

    .result-label { font-size: 0.8rem; color: #94a3b8; letter-spacing: 0.1em; text-transform: uppercase; }
    .result-value { font-size: 2.4rem; font-weight: 700; color: #f8fafc; margin: 0.3rem 0; }
    .result-conf  { font-size: 1rem; color: #cbd5e1; }

    .prob-bar-wrap { margin-top: 1.5rem; }
    .prob-row { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.6rem; }
    .prob-name { width: 90px; color: #94a3b8; font-size: 0.85rem; text-align: right; }
    .prob-bar-bg { flex: 1; background: #1e293b; border-radius: 99px; height: 10px; overflow: hidden; }
    .prob-bar-fill { height: 100%; border-radius: 99px; }
    .prob-pct { width: 50px; color: #e2e8f0; font-size: 0.85rem; font-weight: 600; }

    .api-badge { display: inline-block; background: #166534; color: #86efac; font-size: 0.7rem;
        font-weight: 700; padding: 0.2rem 0.6rem; border-radius: 99px; margin-bottom: 1rem; }
    .api-badge-err { background: #7f1d1d; color: #fca5a5; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="header">
    <h1>⚽ Welcome to the Home of Soccer Prediction</h1>
    <p>This is my first soccer prediction model. Hopefully, we will add other sports predictions soon. Enjoy, and happy predicting!</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# API health check
# ---------------------------------------------------------------------------
try:
    r = requests.get(f"{API_BASE_URL}/health", timeout=60)
    if r.status_code == 200 and r.json().get("model_loaded"):
        st.markdown('<span class="api-badge">● API connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="api-badge api-badge-err">⚠ API reachable but model not loaded</span>', unsafe_allow_html=True)
except Exception:
    st.markdown('<span class="api-badge api-badge-err">✕ API unreachable — Render may be waking up, refresh in 30s</span>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
home_country = st.text_input("Home Team / Country Name", "Nigeria")
away_country = st.text_input("Away Team / Country Name", "Ghana")
col_home, col_away = st.columns(2, gap="large")

with col_home:
    st.markdown(f'<div class="team-card"><div class="team-label home-label">🏠 {home_country}</div>', unsafe_allow_html=True)
    home_form    = st.slider("Form (last 5 games)", 0.0, 1.0, 0.67, 0.01, key="hf")
    home_xg      = st.number_input("xG (Expected Goals)", 0.0, 10.0, 1.73, 0.01, key="hxg")
    home_xga     = st.number_input("xGA (Expected Goals Against)", 0.0, 10.0, 1.3, 0.01, key="hxga")
    home_xt      = st.number_input("xT (Expected Threat)", 0.0, 10.0, 1.93, 0.01, key="hxt")
    home_sot     = st.number_input("Shots on Target", 0.0, 20.0, 6.7, 0.1, key="hsot")
    home_atk     = st.number_input("Attack Strength", 0.0, 3.0, 1.13, 0.01, key="hatk")
    home_def     = st.number_input("Defense Strength", 0.0, 3.0, 1.05, 0.01, key="hdef")
    rest_home    = st.number_input("Rest Days", 0, 30, 3, key="rh")
    missing_home = st.number_input("Missing Key Players", 0, 11, 1, key="mh")
    st.markdown('</div>', unsafe_allow_html=True)

with col_away:
    st.markdown(f'<div class="team-card"><div class="team-label away-label">✈️ {away_country}</div>', unsafe_allow_html=True)
    away_form    = st.slider("Form (last 5 games)", 0.0, 1.0, 0.73, 0.01, key="af")
    away_xg      = st.number_input("xG (Expected Goals)", 0.0, 10.0, 2.03, 0.01, key="axg")
    away_xga     = st.number_input("xGA (Expected Goals Against)", 0.0, 10.0, 1.0, 0.01, key="axga")
    away_xt      = st.number_input("xT (Expected Threat)", 0.0, 10.0, 0.96, 0.01, key="axt")
    away_sot     = st.number_input("Shots on Target", 0.0, 20.0, 5.3, 0.1, key="asot")
    away_atk     = st.number_input("Attack Strength", 0.0, 3.0, 0.68, 0.01, key="aatk")
    away_def     = st.number_input("Defense Strength", 0.0, 3.0, 1.4, 0.01, key="adef")
    rest_away    = st.number_input("Rest Days", 0, 30, 4, key="ra")
    missing_away = st.number_input("Missing Key Players", 0, 11, 2, key="ma")
    st.markdown('</div>', unsafe_allow_html=True)

# Match context
st.markdown("---")
ctx1, ctx2, ctx3 = st.columns(3)
with ctx1:
    travel_dist = st.number_input("Travel Distance Away (km)", 0.0, 10000.0, 0.0, 10.0)
with ctx2:
    h2h_ratio   = st.slider("H2H Home Win Ratio", 0.0, 1.0, 0.0, 0.01)
with ctx3:
    card_rate   = st.number_input("Referee Card Rate (per game)", 0.0, 15.0, 4.74, 0.01)

# ---------------------------------------------------------------------------
# Predict button
# ---------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("⚡ Predict Match Outcome", use_container_width=True, type="primary")

if predict_btn:
    payload = {
        "home_form_5g": home_form,
        "away_form_5g": away_form,
        "home_xg": home_xg,
        "away_xg": away_xg,
        "home_xga": home_xga,
        "away_xga": away_xga,
        "home_xt": home_xt,
        "away_xt": away_xt,
        "home_sot": home_sot,
        "away_sot": away_sot,
        "home_attack_strength": home_atk,
        "away_attack_strength": away_atk,
        "home_defense_strength": home_def,
        "away_defense_strength": away_def,
        "rest_days_home": int(rest_home),
        "rest_days_away": int(rest_away),
        "travel_distance_away_km": travel_dist,
        "h2h_home_win_ratio": h2h_ratio,
        "missing_key_players_home": int(missing_home),
        "missing_key_players_away": int(missing_away),
        "referee_card_rate": card_rate,
    }

    with st.spinner("Calling prediction API... (first call may take ~30s if Render is waking up)"):
        try:
            resp = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            prediction = data["prediction"]
            probs      = data["probabilities"]
            confidence = data["confidence"]

            # Persist this prediction (inputs + outputs) to CSV for future
            # retraining / drift monitoring. Failures here should never
            # break the user-facing prediction flow, so they're caught
            # and surfaced as a small warning only.
            try:
                log_prediction(payload, home_country, away_country, data)
            except Exception as log_err:
                st.warning(f"Prediction succeeded, but logging to CSV failed: {log_err}")

            display_label = {
                "Home Win": f"{home_country} Win",
                "Draw": "Draw",
                "Away Win": f"{away_country} Win",
            }.get(prediction, prediction)

            result_class = {
                "Home Win": "result-home",
                "Draw": "result-draw",
                "Away Win": "result-away",
            }.get(prediction, "")

            st.markdown(f"""
            <div class="result-box {result_class}">
                <div class="result-label">Predicted Outcome</div>
                <div class="result-value">{display_label}</div>
                <div class="result-conf">Confidence: {confidence*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            bar_colors = {"Home Win": "#38bdf8", "Draw": "#a8a29e", "Away Win": "#fb923c"}
            outcome_labels = {"Home Win": home_country, "Draw": "Draw", "Away Win": away_country}
            bars_html = '<div class="prob-bar-wrap">'
            for outcome, pct in probs.items():
                color = bar_colors.get(outcome, "#64748b")
                bars_html += f"""
                <div class="prob-row">
                    <div class="prob-name">{outcome_labels.get(outcome, outcome)}</div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width:{pct*100:.1f}%;background:{color};"></div>
                    </div>
                    <div class="prob-pct">{pct*100:.1f}%</div>
                </div>"""
            bars_html += "</div>"
            st.markdown(bars_html, unsafe_allow_html=True)

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot reach the API. Check that your Render service is running.")
        except requests.exceptions.Timeout:
            st.error("⏱ Request timed out. Render may still be waking up — wait 30 seconds and try again.")
        except requests.exceptions.HTTPError as e:
            st.error(f"API error: {e.response.status_code} — {e.response.text}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# ---------------------------------------------------------------------------
# Prediction log, download & back-fill
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📥 Prediction Log & Download")

using_sheets = "gcp_service_account" in st.secrets
data_source = st.radio(
    "Select data source:",
    options=["Google Sheets (Live Cloud Data)", "Local CSV Fallback"],
    horizontal=True,
)

log_df = None

if data_source == "Google Sheets (Live Cloud Data)":
    if using_sheets:
        try:
            worksheet = get_sheets_worksheet()
            records = worksheet.get_all_records()
            log_df = pd.DataFrame(records) if records else pd.DataFrame(columns=LOG_COLUMNS)
        except Exception as e:
            st.error(f"Could not read from Google Sheets: {e}")
    else:
        st.warning("Google Sheets not configured — showing local CSV instead.")
        if os.path.isfile(LOG_PATH):
            try:
                log_df = pd.read_csv(LOG_PATH)
            except Exception as e:
                st.error(f"Could not read local CSV: {e}")
else:
    if os.path.isfile(LOG_PATH):
        try:
            log_df = pd.read_csv(LOG_PATH)
        except Exception as e:
            st.error(f"Could not read local CSV: {e}")

if log_df is None:
    log_df = pd.DataFrame(columns=LOG_COLUMNS)

# ---- Download button FIRST — always visible ----
csv_bytes = log_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Prediction Log as CSV",
    data=csv_bytes,
    file_name=f"soccer_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv",
    use_container_width=True,
)

# ---- Preview below ----
if log_df.empty:
    st.info("No predictions logged yet — the downloaded CSV contains only column headers.")
else:
    st.caption(f"{len(log_df)} predictions logged.")
    st.dataframe(log_df.tail(10), use_container_width=True)

    # Back-fill actual result
    st.markdown("##### ✏️ Back-fill an actual match result")
    st.caption("Once a real result is known, record it here to build a labeled dataset for retraining.")
    row_options = (
        log_df["timestamp_utc"].astype(str)
        + " — " + log_df["home_country"].astype(str)
        + " vs " + log_df["away_country"].astype(str)
    )
    sel = st.selectbox("Select prediction to label", options=row_options.tolist()[::-1])
    actual = st.selectbox("Actual result", ["", "Home Win", "Draw", "Away Win"])
    if st.button("Save actual result"):
        if actual == "":
            st.warning("Pick an actual result before saving.")
        else:
            sel_ts = sel.split(" — ")[0]
            try:
                if using_sheets and data_source == "Google Sheets (Live Cloud Data)":
                    worksheet = get_sheets_worksheet()
                    cell = worksheet.find(sel_ts)
                    col_idx = LOG_COLUMNS.index("actual_result") + 1
                    worksheet.update_cell(cell.row, col_idx, actual)
                else:
                    log_df.loc[log_df["timestamp_utc"] == sel_ts, "actual_result"] = actual
                    log_df.to_csv(LOG_PATH, index=False)
                st.success("Saved. Re-select your data source above to see the update.")
            except Exception as e:
                st.error(f"Couldn't save the actual result: {e}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(f"Model: RandomForestClassifier · API: `{API_BASE_URL}` · Built with FastAPI + Streamlit")