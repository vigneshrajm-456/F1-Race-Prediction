
import pandas as pd
from pathlib import Path

# External source: PITWALL F1 Predictions
SOURCE = "https://www.pitwall-lab.com/predictions/"

DATA = Path("data")
DATA.mkdir(exist_ok=True)

# Published Mexico City GP 2026 forecast.
# These values are copied from the external source, not generated
# by our own machine-learning model.
drivers = [
    ("Max Verstappen", 18, 44),
    ("Charles Leclerc", 17, 42),
    ("Kimi Antonelli", 15, 38),
    ("George Russell", 9, 24),
    ("Lando Norris", 8, 22),
    ("Oscar Piastri", 7, 23),
    ("Isack Hadjar", 6, 25),
    ("Lewis Hamilton", 6, 20),
    ("Pierre Gasly", 5, 10),
    ("Liam Lawson", 3, 10),
]

df = pd.DataFrame(
    drivers,
    columns=["Driver", "WinProbabilityPercent", "PodiumProbabilityPercent"]
)

df = df.sort_values(
    "WinProbabilityPercent",
    ascending=False
).reset_index(drop=True)

df.insert(0, "Rank", range(1, len(df) + 1))

df["PredictionSource"] = "PITWALL"
df["GrandPrix"] = "Mexico City GP 2026"

output = DATA / "mexico_2026_prediction.csv"
df.to_csv(output, index=False)

print("\n" + "=" * 55)
print(" MEXICO CITY GP 2026 - EXTERNAL RACE PREDICTION")
print("=" * 55)

print(
    df[
        [
            "Rank",
            "Driver",
            "WinProbabilityPercent",
            "PodiumProbabilityPercent"
        ]
    ].to_string(index=False)
)

winner = df.iloc[0]

print("\n🏆 EXTERNAL MODEL'S TOP PICK")
print("Driver:", winner["Driver"])
print("Win probability:", f'{winner["WinProbabilityPercent"]}%')

print("\nSource:", SOURCE)
print("Saved to:", output)
print("\nNote: External forecast, not a guaranteed race result.")

