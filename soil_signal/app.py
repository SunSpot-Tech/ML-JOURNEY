# app.py — Soil to Signal: Lagos Flood Risk Explorer
#
# Run: streamlit run app.py
# Needs lagos_features.csv and lagos_flood_labels.csv in the same folder.
#
# Install: pip install streamlit pandas numpy scikit-learn matplotlib seaborn pydeck

import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Soil to Signal", layout="wide")

# --- Zone coordinates (from zones.py, inlined here for a single-file app) ---
COORDS = {
    "Lekki Phase 1": (6.4304, 3.4753),
    "Ajegunle": (6.4441, 3.3313),
    "Makoko": (6.4926, 3.3872),
    "Iwaya": (6.5119, 3.3803),
    "Ketu": (6.5921, 3.3854),
    "Mushin": (6.5295, 3.3492),
    "Ajeromi-Ifelodun": (6.4562, 3.3378),
    "Ikorodu": (6.6018, 3.5106),
    "Surulere": (6.5027, 3.3541),
    "Apapa": (6.4498, 3.3592),
    "Eti-Osa (VI/Lekki axis)": (6.4281, 3.4219),
    "Ipaja": (6.6144, 3.2807),
}

FEATURE_COLS = [
    "avg_daily_rainfall_mm", "max_daily_rainfall_mm", "heavy_rain_days",
    "rainy_season_avg_mm", "elevation_m", "waterway_feature_count",
]


@st.cache_data
def load_data():
    features = pd.read_csv("soil_signal/Lagos_features.csv")
    labels = pd.read_csv("lagos_flood_labels.csv")
    df = features.merge(labels[["zone", "source_summary"]], on="zone", how="left")
    df["lat"] = df["zone"].map(lambda z: COORDS[z][0])
    df["lon"] = df["zone"].map(lambda z: COORDS[z][1])
    df["flood_binary"] = (df["flood_label"] == 2).astype(int)
    df["outcome_label"] = df["flood_binary"].map({1: "Confirmed damage", 0: "Flood-prone, no confirmed damage"})
    df["color"] = df["flood_binary"].map({1: [231, 76, 60], 0: [52, 152, 219]})
    return df


@st.cache_data
def run_model(df):
    X = df[FEATURE_COLS]
    y = df["flood_binary"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Full model, fit on everything, for coefficients
    full_model = LogisticRegression(max_iter=1000)
    full_model.fit(X_scaled, y)
    coef = pd.Series(full_model.coef_[0], index=FEATURE_COLS).sort_values(key=abs, ascending=False)

    # Leave-one-out CV for honest accuracy
    loo = LeaveOneOut()
    preds = []
    for train_idx, test_idx in loo.split(X_scaled):
        m = LogisticRegression(max_iter=1000)
        m.fit(X_scaled[train_idx], y.iloc[train_idx])
        preds.append(m.predict(X_scaled[test_idx])[0])
    loo_accuracy = (np.array(preds) == y.values).mean()
    baseline_accuracy = max(y.mean(), 1 - y.mean())

    # Full-model predicted probability per zone (for display only, heavily caveated)
    probs = full_model.predict_proba(X_scaled)[:, 1]

    # Train/test split evaluation (Powei's original approach, 75/25, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    split_scaler = StandardScaler()
    x_train_scaled = split_scaler.fit_transform(x_train)
    x_test_scaled = split_scaler.transform(x_test)
    split_model = LogisticRegression(random_state=42)
    split_model.fit(x_train_scaled, y_train)
    split_preds = split_model.predict(x_test_scaled)
    split_accuracy = accuracy_score(y_test, split_preds)
    split_test_size = len(y_test)

    return coef, loo_accuracy, baseline_accuracy, probs, split_accuracy, split_test_size


df = load_data()
coef, loo_accuracy, baseline_accuracy, probs, split_accuracy, split_test_size = run_model(df)
df["model_probability"] = probs

st.title("Soil to Signal")
st.caption("A 10-day build: flood risk signals for 12 Lagos zones, from raw data to (an honest) model.")

tab_map, tab_zone, tab_model, tab_charts = st.tabs(
    ["Map", "Zone Detail", "Model (experimental)", "EDA Charts"]
)

# --- MAP TAB ---
with tab_map:
    st.subheader("Zones by Confirmed Flood Outcome")
    st.caption("Red = confirmed structural damage on record. Blue = flood-prone, no specific damage incident found.")

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius=400,
        pickable=True,
    )
    view_state = pdk.ViewState(latitude=6.50, longitude=3.42, zoom=10.2)
    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "{zone}\n{outcome_label}"},
    ))

    st.dataframe(
        df[["zone", "outcome_label", "elevation_m", "waterway_feature_count"]]
        .rename(columns={"outcome_label": "flood outcome"}),
        use_container_width=True, hide_index=True,
    )

# --- ZONE DETAIL TAB ---
with tab_zone:
    zone = st.selectbox("Select a zone", sorted(df["zone"].unique()))
    row = df[df["zone"] == zone].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Elevation", f"{row['elevation_m']:.0f} m")
    c2.metric("Drainage features nearby", int(row["waterway_feature_count"]))
    c3.metric("Avg daily rainfall", f"{row['avg_daily_rainfall_mm']:.2f} mm")

    c4, c5 = st.columns(2)
    c4.metric("Heavy rain days (2019-2026)", int(row["heavy_rain_days"]))
    c5.metric("Rainy season avg rainfall", f"{row['rainy_season_avg_mm']:.2f} mm")

    st.markdown(f"**Flood outcome:** {row['outcome_label']}")
    st.markdown(f"**Source:** {row['source_summary']}")

# --- MODEL TAB ---
with tab_model:
    st.warning(
        f"**Read this before trusting anything on this tab.** "
        f"Leave-one-out cross-validation accuracy is **{loo_accuracy:.0%}**, "
        f"which is *below* the {baseline_accuracy:.0%} you'd get by just guessing "
        f"the majority outcome for every zone. With only 12 zones, this model "
        f"has not proven it can predict flood risk reliably. Treat everything "
        f"below as an experiment in progress, not a trustworthy risk score."
    )

    col1, col2 = st.columns(2)
    col1.metric("Leave-one-out accuracy", f"{loo_accuracy:.0%}")
    col2.metric("Naive baseline accuracy", f"{baseline_accuracy:.0%}", delta=f"{(loo_accuracy - baseline_accuracy):.0%}", delta_color="inverse")

    st.subheader("What the model currently predicts per zone")
    st.caption("Predicted probability of confirmed damage, from a model fit on all 12 zones. Shown for transparency, not as a usable score.")
    st.dataframe(
        df[["zone", "outcome_label", "model_probability"]]
        .rename(columns={"outcome_label": "actual outcome", "model_probability": "model probability"})
        .sort_values("model probability", ascending=False),
        use_container_width=True, hide_index=True,
    )

    st.subheader("Feature coefficients")
    st.caption("Standardized weights the model learned. Note elevation's sign below — flagged in Day 6.")
    coef_df = coef.to_frame("standardized coefficient")
    st.dataframe(coef_df, use_container_width=True)
    if coef["elevation_m"] > 0:
        st.info("Elevation has a positive coefficient here, meaning the model associates higher elevation with more damage. That's backwards from physical reality — a sign of overfitting on a small dataset, not a real terrain relationship.")

# --- EDA CHARTS TAB ---
with tab_charts:
    st.caption("Regenerated live from the current data.")

    fig1, ax1 = plt.subplots(figsize=(9, 5))
    df_sorted = df.sort_values("avg_daily_rainfall_mm")
    colors = sns.color_palette("husl", df["avg_daily_rainfall_mm"].nunique())
    rain_to_color = {v: colors[i] for i, v in enumerate(sorted(df["avg_daily_rainfall_mm"].unique()))}
    ax1.barh(df_sorted["zone"], df_sorted["avg_daily_rainfall_mm"],
             color=[rain_to_color[v] for v in df_sorted["avg_daily_rainfall_mm"]])
    ax1.set_xlabel("Average daily rainfall (mm)")
    ax1.set_title("Avg Daily Rainfall by Zone (same color = identical grid-cell value)")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    for label, marker, color, name in [(1, "o", "#e74c3c", "Confirmed damage"), (0, "s", "#3498db", "No confirmed damage")]:
        subset = df[df["flood_binary"] == label]
        ax2.scatter(subset["elevation_m"], subset["waterway_feature_count"], s=150, marker=marker,
                    color=color, label=name, edgecolor="black", alpha=0.85)
    ax2.set_xlabel("Elevation (m)")
    ax2.set_ylabel("Waterway feature count")
    ax2.set_title("Elevation vs. Drainage Density, by Flood Outcome")
    ax2.legend()
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(7, 6))
    corr = df[FEATURE_COLS + ["flood_binary"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, ax=ax3)
    ax3.set_title("Feature Correlation Matrix")
    st.pyplot(fig3)

    fig4, axes4 = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col, title in zip(axes4, ["elevation_m", "waterway_feature_count"],
                               ["Elevation by Outcome", "Drainage by Outcome"]):
        data_to_plot = [df[df["flood_binary"] == 0][col], df[df["flood_binary"] == 1][col]]
        bp = ax.boxplot(data_to_plot, tick_labels=["No damage", "Confirmed damage"], patch_artist=True)
        for patch, c in zip(bp["boxes"], ["#3498db", "#e74c3c"]):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.set_title(title)
    st.pyplot(fig4)