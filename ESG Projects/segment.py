# ---------------------------------------------------------------------------
# Reload a clean copy just for the columns get_dummies destroyed
# (FieldName, Region) -- your working `data` is now one-hot encoded
# ---------------------------------------------------------------------------
raw_data = pd.read_csv("pipeline_merged_dataset.csv")
raw_data['ReadingDate'] = pd.to_datetime(raw_data['ReadingDate'])

# ---------------------------------------------------------------------------
# Build predictions on the test set using your already-trained model + iso
# ---------------------------------------------------------------------------
df_test = raw_data.iloc[test_idx].copy()
df_test['PredictedCorrosionRate_mmyr'] = model.predict(x_test)
df_test['IsoAnomalyScore'] = -iso.decision_function(x_sensor_test)
df_test['IsoAnomalyFlag'] = (iso.predict(x_sensor_test) == -1).astype(int)

# --- Corrosion threshold flag (tuned: 15% critical, 6-month horizon) ---
CRITICAL_PCT = 0.15
HORIZON_MONTHS = 6
df_test['CriticalThickness_mm'] = CRITICAL_PCT * df_test['NominalWallThickness_mm']
df_test['ProjectedWallLoss_mm'] = df_test['PredictedCorrosionRate_mmyr'] / 12 * HORIZON_MONTHS
df_test['ProjectedRemainingThickness_mm'] = df_test['RemainingWallThickness_mm'] - df_test['ProjectedWallLoss_mm']
df_test['CorrosionRiskFlag'] = (df_test['ProjectedRemainingThickness_mm'] <= df_test['CriticalThickness_mm']).astype(int)

safe_margin = (df_test['RemainingWallThickness_mm'] - df_test['CriticalThickness_mm']).clip(lower=0)
df_test['EstimatedYearsToCritical'] = np.where(
    df_test['PredictedCorrosionRate_mmyr'] > 0.01,
    safe_margin / df_test['PredictedCorrosionRate_mmyr'], np.inf
)

# ---------------------------------------------------------------------------
# Take each segment's most recent reading + assign a combined risk tier
# ---------------------------------------------------------------------------
latest = df_test.sort_values('ReadingDate').groupby('SegmentID', as_index=False).tail(1).copy()

def risk_tier(row):
    if row['CorrosionRiskFlag'] == 1 and row['IsoAnomalyFlag'] == 1:
        return 'CRITICAL'
    elif row['CorrosionRiskFlag'] == 1 or row['IsoAnomalyFlag'] == 1:
        return 'HIGH'
    elif row['EstimatedYearsToCritical'] < 3:
        return 'MEDIUM'
    return 'LOW'

latest['RiskTier'] = latest.apply(risk_tier, axis=1)

risk_matrix = latest[[
    'SegmentID', 'FieldName', 'Region', 'ReadingDate',
    'PredictedCorrosionRate_mmyr', 'RemainingWallThickness_mm', 'NominalWallThickness_mm',
    'EstimatedYearsToCritical', 'CorrosionRiskFlag', 'IsoAnomalyFlag', 'IsoAnomalyScore', 'RiskTier'
]].sort_values(
    ['RiskTier', 'EstimatedYearsToCritical'],
    key=lambda col: col.map({'CRITICAL':0,'HIGH':1,'MEDIUM':2,'LOW':3}) if col.name=='RiskTier' else col
)

risk_matrix.to_csv("segment_risk_matrix.csv", index=False)
print(f"Saved {len(risk_matrix)} segments")
print(risk_matrix['RiskTier'].value_counts())