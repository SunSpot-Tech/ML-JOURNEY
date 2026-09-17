# ============================================================
# BabaPick — Streamlit Frontend v2
# Smarter Insights, Better Decisions
# Author: Powei Shadrack | SunSpot-Tech
# ============================================================

import os
import json
from datetime import date

import requests
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

# ── 2025/26 Season Teams ─────────────────────────────────────
TEAMS_BY_LEAGUE = {
    "EPL": [
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", 
        "Brighton", "Chelsea", "Coventry", "Crystal Palace", 
        "Everton", "Fulham", "Hull", "Ipswich", "Leeds", "Liverpool", 
        "Manchester City", "Manchester Utd", "Newcastle", "Nottingham Forest", 
        "Sunderland", "Tottenham",
    ],
    "La Liga": [
        "Alaves", "Athletic Club", "Atletico Madrid", "Barcelona",
        "Celta Vigo", "Espanyol", "Getafe", "Girona", "Las Palmas",
        "Leganes", "Mallorca", "Osasuna", "Rayo Vallecano", "Real Betis",
        "Real Madrid", "Real Sociedad", "Sevilla", "Valencia",
        "Valladolid", "Villarreal",
    ],
    "Serie A": [
        "AC Milan", "Atalanta", "Bologna", "Cagliari", "Como",
        "Empoli", "Fiorentina", "Genoa", "Inter Milan", "Juventus",
        "Lazio", "Lecce", "Monza", "Napoli", "Parma",
        "AS Roma", "Torino", "Udinese", "Venezia", "Verona",
    ],
    "Bundesliga": [
        "Augsburg", "Bayer Leverkusen", "Bayern Munich", "Bochum",
        "Borussia Dortmund", "Borussia Monchengladbach", "Eintracht Frankfurt",
        "FC Heidenheim", "Freiburg", "Hoffenheim", "Holstein Kiel",
        "Mainz", "RB Leipzig", "SC Freiburg", "St. Pauli",
        "Stuttgart", "Union Berlin", "Werder Bremen", "Wolfsburg", "Karlsruher SC",
    ],
    "Ligue 1": [
        "Angers", "Auxerre", "Brest", "Le Havre", "Lens",
        "Lille", "Lyon", "Marseille", "Monaco", "Montpellier",
        "Nantes", "Nice", "Paris Saint-Germain", "Reims", "Rennes",
        "Saint-Etienne", "Strasbourg", "Toulouse",
    ],
    "Eredivisie": [
        "AZ Alkmaar", "Ajax", "Almere City", "Feyenoord", "Fortuna Sittard",
        "Go Ahead Eagles", "Groningen", "Heerenveen", "Heracles",
        "NEC Nijmegen", "PEC Zwolle", "PSV Eindhoven", "RKC Waalwijk",
        "Sparta Rotterdam", "Twente", "Utrecht", "Willem II",
    ],
    "Liga Portugal": [
        "Arouca", "AVS", "Benfica", "Boavista", "Braga",
        "Casa Pia", "Estoril", "Estrela Amadora", "Famalicao",
        "Gil Vicente", "Moreirense", "Nacional", "Porto",
        "Rio Ave", "Santa Clara", "Sporting CP",
        "Vitoria Guimaraes",
    ],
    "Scottish Prem": [
        "Aberdeen", "Celtic", "Dundee", "Dundee Utd", "Hearts",
        "Hibernian", "Kilmarnock", "Motherwell", "Rangers",
        "Ross County", "St Johnstone", "St Mirren",
    ],
    "Belgian Pro": [
        "Anderlecht", "Antwerp", "Beerschot", "Club Brugge", "Cercle Brugge",
        "Charleroi", "Gent", "Genk", "KV Mechelen", "KV Kortrijk",
        "Lierse", "OH Leuven", "Standard Liege", "St-Truiden",
        "Union SG", "Westerlo",
    ],
    "Turkish SL": [
        "Adana Demirspor", "Antalyaspor", "Basaksehir", "Besiktas",
        "Eyupspor", "Fenerbahce", "Galatasaray", "Gaziantep",
        "Hatayspor", "Kasimpasa", "Kayserispor", "Konyaspor",
        "Rizespor", "Samsunspor", "Sivasspor", "Trabzonspor",
        "Alanyaspor", "Bodrum",
    ],
    "Championship": [
        "Birmingham", "Blackburn", "Bolton", "Bristol City", "Burnley", 
        "Cardiff", "Charlton", "Derby", "Lincoln City", "Middlesbrough", 
        "Millwall", "Norwich", "Portsmouth", "Preston", "QPR", "Sheffield Utd", 
        "Southampton", "Stoke City", "Swansea", "Watford", "West Brom", "West Ham", 
        "Wolves", "Wrexham",
    ],
    "Greek SL": [
        "AEK Athens", "Aris", "Asteras Tripolis", "Atromitos",
        "Kallithea", "Lamia", "Levadiakos", "OFI Crete",
        "Olympiacos", "PAOK", "Panathinaikos", "Panetolikos",
        "Panserraikos", "Volos",
    ],
}

LEAGUES = list(TEAMS_BY_LEAGUE.keys())

COMPETITION_TYPES = [
    "Domestic League", "Domestic Cup", "Champions League Group Stage",
    "Champions League Knockout", "Europa League Group Stage",
    "Europa League Final", "Continental Qualifier",
    "World Cup Group Stage", "World Cup Knockout",
    "Continental Championship", "Continental Knockout",
    "Nations League", "Minor Cup", "Major Final", "Friendly",
]

BRAND = {
    "navy":   "#0F172A",
    "blue":   "#2563EB",
    "emerald":"#10B981",
    "gold":   "#D4AF37",
    "slate":  "#64748B",
    "bg":     "#F8FAFC",
}


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="BabaPick | Smarter Insights, Better Decisions",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(f"""
<style>
/* Global Background */
.stApp {{ background: {BRAND['bg']}; }}
#MainMenu, footer, header {{ visibility: hidden; }}

/* Standard Labels & Form Field Labels — visibility only */
label,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] *,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
[data-testid="stWidgetLabel"] div {{
    color: {BRAND['navy']} !important;
    opacity: 1 !important;
    visibility: visible !important;
    display: block !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    line-height: 1.35 !important;
}}

/* Streamlit Tabs Styling — keep Tab 1 and Tab 2 visible */
[data-baseweb="tab-list"],
button[data-baseweb="tab"],
button[data-baseweb="tab"] *,
button[data-baseweb="tab"] p {{
    visibility: visible !important;
    opacity: 1 !important;
}}
button[data-baseweb="tab"] {{
    color: {BRAND['navy']} !important;
    display: flex !important;
}}
button[data-baseweb="tab"] p {{
    color: {BRAND['navy']} !important;
    font-size: 16px !important;
    font-weight: 800 !important;
    display: block !important;
}}
button[data-baseweb="tab"][aria-selected="true"] p {{
    color: {BRAND['blue']} !important;
}}

/* Input Box Styling */
input, textarea {{ 
    color: #FFFFFF !important; 
    background-color: #1E293B !important;
}}

[data-baseweb="select"] {{
    background-color: #1E293B !important;
    border-radius: 10px !important;
}}
[data-baseweb="select"] * {{ 
    color: #FFFFFF !important; 
}}

[data-testid="stDateInput"] input {{
    color: #FFFFFF !important;
    background-color: #1E293B !important;
}}

[data-testid="stNumberInput"] {{
    background-color: #1E293B !important;
    border-radius: 10px !important;
}}
[data-testid="stNumberInput"] input {{
    color: #FFFFFF !important;
    background-color: #1E293B !important;
}}
[data-testid="stNumberInput"] button {{ color: #FFFFFF !important; }}
[data-testid="stNumberInput"] [data-testid="stWidgetLabel"],
[data-testid="stNumberInput"] [data-testid="stWidgetLabel"] *,
[data-testid="stNumberInput"] label,
[data-testid="stNumberInput"] label * {{
    color: #F8FAFC !important;
    opacity: 1 !important;
    visibility: visible !important;
    display: block !important;
}}

[data-testid="stTextInput"] input {{
    color: #FFFFFF !important;
    background-color: #1E293B !important;
    border-radius: 10px !important;
}}
[data-testid="stTextInput"] input::placeholder {{ color: #94A3B8 !important; }}

/* Section Headings */
.section-title {{
    color: {BRAND['navy']} !important;
    font-size: 20px;
    font-weight: 800;
    margin-top: 16px;
    margin-bottom: 12px;
    border-left: 4px solid {BRAND['blue']};
    padding-left: 10px;
}}

h1, h2, h3, h4, h5, h6 {{
    color: {BRAND['navy']} !important;
}}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {{
    color: {BRAND['navy']};
}}

/* Metric Cards & Labels */
div[data-testid="stMetric"] {{
    background: #FFFFFF;
    padding: 14px;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 8px rgba(15,23,42,0.05);
}}
div[data-testid="stMetricLabel"] label,
div[data-testid="stMetricLabel"] p {{
    color: {BRAND['slate']} !important;
    font-weight: 700 !important;
}}
div[data-testid="stMetricValue"] div {{
    color: {BRAND['navy']} !important;
    font-weight: 800 !important;
}}

/* Header */
.babapick-header {{
    background: linear-gradient(135deg, {BRAND['navy']} 0%, #1E3A5F 60%, {BRAND['blue']} 100%);
    padding: 28px 36px;
    border-radius: 16px;
    margin-bottom: 24px;
    border-bottom: 4px solid {BRAND['gold']};
    box-shadow: 0 8px 24px rgba(15,23,42,0.15);
}}
.babapick-title {{
    color: #FFFFFF !important;
    font-size: 36px;
    font-weight: 900;
    margin: 0;
    letter-spacing: -1px;
}}
.babapick-tagline {{
    color: {BRAND['gold']} !important;
    font-size: 15px;
    margin-top: 6px;
    font-style: italic;
}}

/* Probability Cards */
.prob-card {{
    background: #FFFFFF;
    border-radius: 14px;
    padding: 22px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(15,23,42,0.08);
    border: 1px solid #E2E8F0;
    border-top: 5px solid;
    height: 100%;
}}
.prob-pct {{
    font-size: 34px;
    font-weight: 900;
    margin: 8px 0 4px 0;
}}
.prob-label {{
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: {BRAND['slate']} !important;
}}
.prob-team {{
    font-size: 13px;
    font-weight: 600;
    color: {BRAND['navy']} !important;
    margin-top: 4px;
}}

/* Winner Badge */
.winner-badge {{
    display: inline-block;
    padding: 12px 28px;
    border-radius: 50px;
    font-weight: 800;
    font-size: 18px;
    letter-spacing: 0.3px;
    margin: 12px 0;
}}

/* Disclaimer */
.disclaimer {{
    background: #FFFBEB;
    border-left: 5px solid {BRAND['gold']};
    color: #78350F !important;
    padding: 14px 18px;
    border-radius: 8px;
    margin-top: 18px;
    font-size: 13px;
}}

/* Status Indicator */
.status-online {{ color: {BRAND['emerald']} !important; font-weight: 700; }}
.status-offline {{ color: #EF4444 !important; font-weight: 700; }}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

for key, default in [
    ("token", None),
    ("username", None),
    ("prediction_result", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
# API HELPERS
# ============================================================

def auth_headers():
    h = {"Content-Type": "application/json"}
    if st.session_state.token:
        h["Authorization"] = f"Bearer {st.session_state.token}"
    return h

def api_get(endpoint, params=None, auth=False, timeout=15):
    try:
        return requests.get(
            f"{API_BASE_URL}{endpoint}",
            params=params,
            headers=auth_headers() if auth else {},
            timeout=timeout,
        )
    except Exception:
        return None

def api_post(endpoint, payload=None, auth=False, timeout=90):
    try:
        return requests.post(
            f"{API_BASE_URL}{endpoint}",
            json=payload,
            headers=auth_headers() if auth else {},
            timeout=timeout,
        )
    except Exception:
        return None

def check_health():
    r = api_get("/health")
    if r is None:
        return False, "API unreachable"
    if r.status_code == 200:
        try:
            d = r.json()
            model = d.get("model", "unknown")
            return True, f"API online  •  Model {model}"
        except Exception:
            return True, "API online"
    return False, f"HTTP {r.status_code}"

def login(username, password):
    r = api_post("/auth/login", {"username": username, "password": password})
    if r is None:
        return False, "Cannot connect to API."
    if r.status_code == 200:
        try:
            d = r.json()
            st.session_state.token    = d["access_token"]
            st.session_state.username = username
            return True, "Login successful."
        except Exception:
            return False, "Unexpected response."
    try:
        err = r.json().get("detail", "Invalid credentials.")
    except Exception:
        err = "Invalid credentials."
    return False, err

def logout():
    st.session_state.token             = None
    st.session_state.username          = None
    st.session_state.prediction_result = None
    st.rerun()


# ============================================================
# LOGIN SCREEN
# ============================================================

if not st.session_state.token:
    st.markdown("""
    <div class="babapick-header">
        <div class="babapick-title">⚽ BabaPick</div>
        <div class="babapick-tagline">Smarter Insights, Better Decisions</div>
    </div>
    """, unsafe_allow_html=True)

    _, center, _ = st.columns([1, 1.4, 1])
    with center:
        st.markdown('<div class="section-title">Sign in to BabaPick</div>', unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")

        if st.button("Sign In", type="primary", use_container_width=True):
            if not username or not password:
                st.error("Please enter your username and password.")
            else:
                ok, msg = login(username, password)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

        st.markdown("""
        <div class="disclaimer">
            BabaPick provides football analysis for informational purposes only.
            Predictions are not guaranteed outcomes.
        </div>
        """, unsafe_allow_html=True)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(f"""
    <div style="font-size:26px;font-weight:900;color:{BRAND['navy']};margin-bottom:2px;">
        ⚽ BabaPick
    </div>
    <div style="color:{BRAND['slate']};font-size:12px;margin-bottom:16px;">
        Smarter Insights, Better Decisions
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 👤 Account")
    st.write(f"**User:** {st.session_state.username}")
    if st.button("Logout", use_container_width=True):
        logout()

    st.markdown("---")
    st.markdown("### 🔌 API Status")
    health_ok, health_msg = check_health()
    if health_ok:
        st.markdown(f'<span class="status-online">● {health_msg}</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="status-offline">● {health_msg}</span>', unsafe_allow_html=True)
    st.caption(API_BASE_URL)

    st.markdown("---")
    st.caption("BabaPick v1.0 | SunSpot-Tech")
    st.caption("⚠️ For informational purposes only.")


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="babapick-header">
    <div class="babapick-title">⚽ BabaPick</div>
    <div class="babapick-tagline">Smarter Insights, Better Decisions</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# TABS
# ============================================================

tab1, tab2 = st.tabs(["🎯 Predict Match", "📊 Feature Breakdown"])

with tab1:

    # ── Competition ──────────────────────────────────────────
    st.markdown('<div class="section-title">Competition</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        league = st.selectbox("League", LEAGUES, index=0)
    with c2:
        competition_type = st.selectbox("Competition Type", COMPETITION_TYPES, index=0)
    with c3:
        match_date = st.date_input("Match Date", value=date.today())

    # ── Teams ────────────────────────────────────────────────
    st.markdown('<div class="section-title">Teams</div>', unsafe_allow_html=True)
    teams = sorted(TEAMS_BY_LEAGUE.get(league, []))

    c1, c2 = st.columns(2)
    with c1:
        home_team = st.selectbox("🏠 Home Team", teams, index=0, key="home_sel")
    with c2:
        away_options = [t for t in teams if t != home_team]
        away_team = st.selectbox("✈️ Away Team", away_options, index=0, key="away_sel")

    # ── Referee ──────────────────────────────────────────────
    st.markdown('<div class="section-title">Match Details</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        referee = st.text_input("Referee Name (optional)", placeholder="e.g. Michael Oliver")
    with c2:
        st.empty()

    # ── Squad info ───────────────────────────────────────────
    st.markdown(f'<div class="section-title">🏠 {home_team} — Squad Info</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        home_coach = st.number_input("Manager Years in Charge", min_value=0, max_value=50, value=1, step=1, key="hc")
    with c2:
        home_starters = st.number_input("Starters Retained from Last Season", min_value=0, max_value=26, value=8, step=1, key="hs")
    with c3:
        home_signings = st.number_input("New Signings This Window", min_value=0, max_value=26, value=4, step=1, key="hsg")

    st.markdown(f'<div class="section-title">✈️ {away_team} — Squad Info</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        away_coach = st.number_input("Manager Years in Charge", min_value=0, max_value=50, value=1, step=1, key="ac")
    with c2:
        away_starters = st.number_input("Starters Retained from Last Season", min_value=0, max_value=26, value=8, step=1, key="as_")
    with c3:
        away_signings = st.number_input("New Signings This Window", min_value=0, max_value=26, value=4, step=1, key="asg")

    # ── Predict button ───────────────────────────────────────
    if st.button("⚡ Generate Prediction", type="primary", use_container_width=True):
        payload = {
            "home_team":               home_team,
            "away_team":               away_team,
            "league":                  league,
            "match_date":              str(match_date),
            "competition_type":        competition_type,
            "home_coach_years":        int(home_coach),
            "home_starters_retained":  int(home_starters),
            "home_new_signings":       int(home_signings),
            "away_coach_years":        int(away_coach),
            "away_starters_retained":  int(away_starters),
            "away_new_signings":       int(away_signings),
            "referee":                 referee or None,
        }

        with st.spinner(f"Analysing {home_team} vs {away_team}..."):
            r = api_post("/api/v1/predict", payload, auth=True)

        if r is None:
            st.error("Cannot connect to API. Is FastAPI running?")
        elif r.status_code == 200:
            st.session_state.prediction_result = r.json()
        else:
            try:
                detail = r.json().get("detail", "Unknown error")
            except Exception:
                detail = r.text[:300]
            st.error(f"Prediction failed: {detail}")

    # ── Results ──────────────────────────────────────────────
    if st.session_state.prediction_result:
        res   = st.session_state.prediction_result
        probs = res["probabilities"]
        pred  = res["prediction"]

        st.markdown("---")
        st.markdown(f"### {res['match']}  |  {res['league']}  |  {res['date']}")

        # Probability bars
        c1, c2, c3 = st.columns(3)

        with c1:
            pct   = probs["away_win"]
            color = BRAND["gold"]
            st.markdown(f"""
            <div class="prob-card" style="border-top-color:{color}">
                <div class="prob-label">Away Win</div>
                <div class="prob-pct" style="color:{color}">{pct:.1f}%</div>
                <div class="prob-team">{away_team}</div>
            </div>""", unsafe_allow_html=True)
            st.progress(pct / 100)

        with c2:
            pct   = probs["draw"]
            color = BRAND["slate"]
            st.markdown(f"""
            <div class="prob-card" style="border-top-color:{color}">
                <div class="prob-label">Draw</div>
                <div class="prob-pct" style="color:{color}">{pct:.1f}%</div>
                <div class="prob-team">—</div>
            </div>""", unsafe_allow_html=True)
            st.progress(pct / 100)

        with c3:
            pct   = probs["home_win"]
            color = BRAND["emerald"]
            st.markdown(f"""
            <div class="prob-card" style="border-top-color:{color}">
                <div class="prob-label">Home Win</div>
                <div class="prob-pct" style="color:{color}">{pct:.1f}%</div>
                <div class="prob-team">{home_team}</div>
            </div>""", unsafe_allow_html=True)
            st.progress(pct / 100)

        # Winner badge
        badge_colors = {
            "Home Win": BRAND["emerald"],
            "Draw":     BRAND["slate"],
            "Away Win": BRAND["gold"],
        }
        badge_color = badge_colors.get(pred, BRAND["blue"])

        st.markdown(f"""
        <div style="text-align:center; margin:20px 0;">
            <span class="winner-badge" style="background:{badge_color}; color:white;">
                ✓ Prediction: {pred}
            </span>
        </div>""", unsafe_allow_html=True)

        # Quick metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Home Form", f"{res['features'].get('home_form', 0):.2f}")
        m2.metric("Away Form", f"{res['features'].get('away_form', 0):.2f}")
        m3.metric("Home xG", f"{res['features'].get('home_xg', 0):.2f}")
        m4.metric("Away xG", f"{res['features'].get('away_xg', 0):.2f}")

        st.markdown(f"""
        <div class="disclaimer">
            ⚠️ {res.get('disclaimer', 'For informational purposes only. Not financial or betting advice.')}
        </div>""", unsafe_allow_html=True)


with tab2:
    if not st.session_state.prediction_result:
        st.info("Run a prediction first to see the full feature breakdown.")
    else:
        res      = st.session_state.prediction_result
        features = res.get("features", {})
        sources  = res.get("data_sources", {})

        st.markdown('<div class="section-title">All 40 Features — Values and Sources</div>',
                    unsafe_allow_html=True)

        feat_df = pd.DataFrame([{
            "Feature": k,
            "Value":   round(float(v), 4) if isinstance(v, (int, float)) else v,
            "Source":  sources.get(k, "—"),
        } for k, v in features.items()])

        def highlight(row):
            src = str(row["Source"]).lower()
            if any(x in src for x in ["proxy","default","manual","—"]):
                return ["background-color:#FFFBEB; color:#0F172A"]*3
            return ["background-color:#F0FDF4; color:#0F172A"]*3

        st.dataframe(
            feat_df.style.apply(highlight, axis=1),
            use_container_width=True,
            height=700,
        )

        st.markdown("""
        <div style="font-size:0.82rem; margin-top:6px;">
            🟢 <b>Green</b> = Real data from external source &nbsp;&nbsp;
            🟡 <b>Yellow</b> = Proxy formula, manual input, or default value
        </div>""", unsafe_allow_html=True)

        with st.expander("View raw API response (JSON)"):
            st.json(res)