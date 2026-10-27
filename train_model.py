
import pandas as pd

# Load fresh F1 data
df = pd.read_csv("data/f1_results.csv")

# Convert positions to numeric
df["grid_position"] = pd.to_numeric(
    df["grid_position"], errors="coerce"
)
df["finish_position"] = pd.to_numeric(
    df["finish_position"], errors="coerce"
)

# Remove invalid rows
df = df.dropna(subset=[
    "year", "race", "driver", "team",
    "grid_position", "finish_position"
])

# Exclude 2026 from training for now.
# Keep it separate for future race prediction.
df["year"] = pd.to_numeric(df["year"], errors="coerce")
train_df = df[df["year"] <= 2025].copy()

# Create target
train_df["winner"] = (
    train_df["finish_position"] == 1
).astype(int)

# Save cleaned historical data
train_df.to_csv("data/f1_cleaned.csv", index=False)

print("DATA PREPARATION COMPLETE")
print("Training records:", len(train_df))
print("Years:", sorted(train_df["year"].unique()))
print("Winner records:", train_df["winner"].sum())
print("Saved: data/f1_cleaned.csv")

