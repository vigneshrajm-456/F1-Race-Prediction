import fastf1
from pathlib import Path

cache_dir = Path("cache")
cache_dir.mkdir(parents=True, exist_ok=True)

fastf1.Cache.configure(cache_dir=str(cache_dir))

print("FastF1 version:", fastf1.__version__)

session = fastf1.get_session(2024, "Bahrain Grand Prix", "R")

print("Loading F1 race data... Please wait.")

session.load()

print("\nRace session loaded successfully!")
print("\nBahrain Grand Prix Results:")

columns = [
"Abbreviation",
"FullName",
"TeamName",
"Position",
"GridPosition",
"Points",
"Status",
]

print(session.results[columns].head(10).to_string(index=False))
