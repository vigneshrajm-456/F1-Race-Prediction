
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.base import clone

DATA = Path("data")

print("Loading datasets...")
pace = pd.read_csv(DATA / "f1_pace_features.csv")
results = pd.read_csv(DATA / "f1_session_results.csv")

# Only actual race results can define the winner.
race_results = results[
    results["Session"].eq("R")
].copy()

race_results["Position"] = pd.to_numeric(
    race_results["Position"], errors="coerce"
)

race_results = race_results.dropna(
    subset=["Year", "Race", "Abbreviation", "Position"]
)

race_results["Year"] = race_results["Year"].astype(int)
race_results["Winner"] = (race_results["Position"] == 1).astype(int)

# Aggregate practice pace and qualifying pace separately.
pace["Year"] = pd.to_numeric(pace["Year"], errors="coerce")
pace = pace.dropna(subset=["Year", "Driver", "Race"]).copy()
pace["Year"] = pace["Year"].astype(int)

# Use only sessions available before the race.
pace = pace[pace["Session"].isin(["FP1", "FP2", "FP3", "Q"])]

metrics = [
    "LapTime_seconds_median",
    "LapTime_seconds_min",
    "Sector1Time_seconds_min",
    "Sector2Time_seconds_min",
    "Sector3Time_seconds_min",
    "TyreLife_median",
]

metrics = [c for c in metrics if c in pace.columns]

# Separate qualifying from practice, then combine per driver/race.
practice = pace[pace["Session"].isin(["FP1", "FP2", "FP3"])]
qualifying = pace[pace["Session"].eq("Q")]

practice_features = (
    practice.groupby(["Year", "Race", "Driver"], as_index=False)[metrics]
    .mean()
)

practice_features = practice_features.rename(
    columns={c: f"Practice_{c}" for c in metrics}
)

qualifying_features = (
    qualifying.groupby(["Year", "Race", "Driver"], as_index=False)[metrics]
    .mean()
)

qualifying_features = qualifying_features.rename(
    columns={c: f"Qualifying_{c}" for c in metrics}
)

features = practice_features.merge(
    qualifying_features,
    on=["Year", "Race", "Driver"],
    how="outer"
)

# Match FastF1's three-letter abbreviation to the pace driver's abbreviation.
features["Abbreviation"] = features["Driver"].astype(str).str.upper()

# Merge team and finishing position from race results.
df = race_results.merge(
    features.drop(columns=["Driver"]),
    on=["Year", "Race", "Abbreviation"],
    how="left"
)

# Drop rows without a winner label.
df = df.dropna(subset=["Winner"])

# Remove fields that directly reveal the race outcome.
drop_cols = [
    "Position", "Winner", "Points", "Status", "Time",
    "Q1", "Q2", "Q3", "GridPosition",
    "FullName", "Abbreviation"
]

X = df.drop(
    columns=[c for c in drop_cols if c in df.columns],
    errors="ignore"
)

# Keep race/year/team and numeric pace features.
categorical = [
    c for c in ["TeamName", "Race"]
    if c in X.columns
]

numeric = [
    c for c in X.select_dtypes(include=np.number).columns
    if c not in ["Year"]
]

numeric += ["Year"] if "Year" in X.columns else []

# Remove columns that cannot be used as features.
X = X[categorical + numeric]
y = df["Winner"].astype(int)

if y.sum() < 2:
    raise ValueError(
        "Not enough race winners. Check whether race results were collected."
    )

# Hold out complete races, avoiding rows from the same race in both splits.
groups = df["Year"].astype(str) + "_" + df["Race"].astype(str)

splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.25, random_state=42
)

train_idx, test_idx = next(splitter.split(X, y, groups=groups))

categorical = [c for c in X.columns if X[c].dtype == "object"]
numeric = [c for c in X.columns if c not in categorical]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        min_samples_leaf=2,
        random_state=42
    ))
])

print("\nTraining...")
model.fit(X.iloc[train_idx], y.iloc[train_idx])

pred = model.predict(X.iloc[test_idx])

print("\nValidation results")
print("Held-out race count:", groups.iloc[test_idx].nunique())
print("Accuracy:", round(accuracy_score(y.iloc[test_idx], pred), 4))
print(classification_report(y.iloc[test_idx], pred, zero_division=0))

# Retrain the final model on all available races.
model.fit(X, y)

import joblib

joblib.dump({
    "model": model,
    "feature_columns": X.columns.tolist(),
    "categorical_columns": categorical,
    "numeric_columns": numeric,
    "training_years": sorted(df["Year"].unique().tolist())
}, DATA / "f1_winner_model.pkl")

print("\nSaved model: data/f1_winner_model.pkl")
print("Training rows:", len(X))
print("Winner rows:", int(y.sum()))
print("Years:", sorted(df["Year"].unique().tolist()))
print("\nTraining complete!")

