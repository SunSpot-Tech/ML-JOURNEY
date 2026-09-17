"""
dashboard.py
------------
Streamlit dashboard for the Pipeline Integrity & Corrosion Risk project.
Run with: streamlit run dashboard.py

Requires segment_risk_matrix.csv to exist (run train_pipeline_models.py first).
"""

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Pipeline Integrity Risk Dashboard", layout="wide")

# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("segment_risk_matrix.csv")
    df["ReadingDate"] = pd.to_datetime(df["ReadingDate"])
    return df

risk_df = load_data()

TIER_COLORS = {"CRITICAL": "#8B0000", "HIGH": "#E07B00", "MEDIUM": "#D4AC0D", "LOW": "#2E8B57"}
TIER_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.title("Pipeline Integrity & Corrosion Risk Dashboard")
st.caption("Niger Delta Pipeline Network — Segment-Level Risk Monitoring")

# ---------------------------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")
regions = st.sidebar.multiselect(
    "Region", options=risk_df["Region"].unique(), default=list(risk_df["Region"].unique())
)
tiers = st.sidebar.multiselect(
    "Risk Tier", options=TIER_ORDER, default=TIER_ORDER
)

filtered = risk_df[risk_df["Region"].isin(regions) & risk_df["RiskTier"].isin(tiers)]

# ---------------------------------------------------------------------------
# TOP-LINE METRICS
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Segments Monitored", len(filtered))
col2.metric("Critical / High Risk", int(filtered["RiskTier"].isin(["CRITICAL", "HIGH"]).sum()))
col3.metric("Avg. Predicted Corrosion Rate", f"{filtered['PredictedCorrosionRate_mmyr'].mean():.2f} mm/yr")
col4.metric("Segments Flagged by Isolation Forest", int(filtered["IsoAnomalyFlag"].sum()))

st.divider()

# ---------------------------------------------------------------------------
# PRIORITY INSPECTION LIST
# ---------------------------------------------------------------------------
st.subheader("Priority Inspection List")
st.dataframe(
    filtered.sort_values(
        "RiskTier", key=lambda s: s.map({t: i for i, t in enumerate(TIER_ORDER)})
    )[[
        "SegmentID", "FieldName", "Region", "RiskTier",
        "PredictedCorrosionRate_mmyr", "RemainingWallThickness_mm",
        "EstimatedYearsToCritical", "CorrosionRiskFlag", "IsoAnomalyFlag",
    ]],
    use_container_width=True,
    hide_index=True,
)

st.divider()

# ---------------------------------------------------------------------------
# CHARTS
# ---------------------------------------------------------------------------
c1, c2 = st.columns(2)

with c1:
    tier_counts = filtered["RiskTier"].value_counts().reindex(TIER_ORDER).fillna(0)
    fig = px.bar(
        x=tier_counts.index, y=tier_counts.values,
        color=tier_counts.index, color_discrete_map=TIER_COLORS,
        labels={"x": "Risk Tier", "y": "Segment Count"},
        title="Segments by Risk Tier",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig2 = px.scatter(
        filtered, x="EstimatedYearsToCritical", y="PredictedCorrosionRate_mmyr",
        color="RiskTier", color_discrete_map=TIER_COLORS,
        hover_data=["SegmentID", "FieldName"],
        title="Urgency vs. Predicted Corrosion Rate",
        labels={"EstimatedYearsToCritical": "Years to Critical Threshold",
                "PredictedCorrosionRate_mmyr": "Predicted Corrosion Rate (mm/yr)"},
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

c3, c4 = st.columns(2)

with c3:
    region_avg = filtered.groupby("Region")["PredictedCorrosionRate_mmyr"].mean().sort_values(ascending=False)
    fig3 = px.bar(
        x=region_avg.values, y=region_avg.index, orientation="h",
        labels={"x": "Avg Predicted Corrosion Rate (mm/yr)", "y": "Region"},
        title="Average Corrosion Rate by Region",
    )
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    fig4 = px.histogram(
        filtered, x="RemainingWallThickness_mm", nbins=20,
        title="Remaining Wall Thickness Distribution",
        labels={"RemainingWallThickness_mm": "Remaining Wall Thickness (mm)"},
    )
    st.plotly_chart(fig4, use_container_width=True)

st.divider()
st.caption(
    "Corrosion risk flag: predicted wall loss projected forward 6 months, "
    "flagged if it breaches 15% of nominal wall thickness. "
    "Isolation Forest flag: unsupervised anomaly detection on sensor patterns "
    "(pressure, flow, vibration) — catches sudden events like third-party interference."
)
