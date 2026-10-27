
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("data")

laps = pd.read_csv(DATA / "f1_session_laps.csv", low_memory=False)
results = pd.read_csv(DATA / "f1_session_results.csv")
weather = pd.read_csv(DATA / "f1_weather.csv")

# Check and normalize column names
laps.columns = laps.columns.str.strip()
results.columns = results.columns.str.strip()
weather.columns = weather.columns.str.strip()

print("Lap columns:", laps.columns.tolist())

# Convert lap and sector times to seconds
def to_seconds(value):
    if pd.isna(value):
        return np.nan

    try:
        if isinstance(value, (int, float)):
            return float(value)

        value = str(value).strip()

        if value.startswith("0 days "):
            value = value.replace("0 days ", "")

        return pd.to_timedelta(value).total_seconds()
    except (ValueError, TypeError):
        return np.nan

for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    if col in laps.columns:
        laps[col + "_seconds"] = laps[col].apply(to_seconds)

# Keep useful practice and qualifying sessions
valid_sessions = ["FP1", "FP2", "FP3", "Q"]

practice = laps[laps["Session"].isin(valid_sessions)].copy()

# Exclude missing and clearly invalid lap times
if "LapTime_seconds" in practice.columns:
    practice = practice[
        practice["LapTime_seconds"].between(50, 200)
    ]

# Aggregate pace by driver, team, and race
group_cols = ["Year", "Race", "Driver", "Team", "Session"]

available_metrics = [
    col for col in [
        "LapTime_seconds",
        "Sector1Time_seconds",
        "Sector2Time_seconds",
        "Sector3Time_seconds",
        "TyreLife"
    ]
    if col in practice.columns
]

pace = (
    practice.groupby(group_cols, dropna=False)[available_metrics]
    .agg(["median", "min", "count"])
    .reset_index()
)

pace.columns = [
    "_".join(str(x) for x in col if str(x) != "")
    if isinstance(col, tuple) else str(col)
    for col in pace.columns
]

pace.to_csv(DATA / "f1_pace_features.csv", index=False)

print("\nPace dataset saved!")
print("Rows:", len(pace))
print("Columns:", pace.columns.tolist())
print(pace.head())
print("\nNext: we will join pace with race results and train the model.")

