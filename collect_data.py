import fastf1
import pandas as pd
from pathlib import Path

CACHE_DIR = Path("cache")
DATA_DIR = Path("data")

CACHE_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

fastf1.Cache.enable_cache(str(CACHE_DIR))

YEARS = [2025, 2026]

SESSION_TYPES = {
    "FP1": "Practice 1",
    "FP2": "Practice 2",
    "FP3": "Practice 3",
    "Q": "Qualifying",
    "R": "Race",
}

all_laps = []
all_results = []
all_weather = []
all_pits = []

for year in YEARS:
    print(f"\n{'=' * 40}\nSeason: {year}\n{'=' * 40}")

    try:
        schedule = fastf1.get_event_schedule(year)
        schedule = schedule[schedule["RoundNumber"] > 0]

    except Exception as exc:
        print(f"Could not load {year} schedule: {exc}")
        continue

    for _, event in schedule.iterrows():
        round_number = int(event["RoundNumber"])
        race_name = event["EventName"]

        for session_code, session_name in SESSION_TYPES.items():
            print(f"\n{year} | {race_name} | {session_name}")

            try:
                session = fastf1.get_session(
                    year, round_number, session_code
                )

                session.load(
                    laps=True,
                    telemetry=False,
                    weather=True,
                    messages=False,
                )

                # Lap and tyre information
                laps = session.laps

                if laps is not None and not laps.empty:
                    lap_df = laps.copy()
                    lap_df["Year"] = year
                    lap_df["Race"] = race_name
                    lap_df["Session"] = session_code

                    keep = [
                        "Year", "Race", "Session",
                        "Driver", "DriverNumber", "Team",
                        "LapNumber", "LapTime",
                        "Sector1Time", "Sector2Time",
                        "Sector3Time", "Compound",
                        "TyreLife", "FreshTyre",
                        "Stint", "Position", "PitOutTime",
                        "PitInTime", "IsPersonalBest",
                    ]

                    lap_df = lap_df[
                        [c for c in keep if c in lap_df.columns]
                    ]

                    all_laps.append(lap_df)

                    # Pit-stop laps: timing is a proxy, not official pit duration
                    pit_laps = laps[laps["PitInTime"].notna()].copy()

                    if not pit_laps.empty:
                        pit_laps["Year"] = year
                        pit_laps["Race"] = race_name
                        pit_laps["Session"] = session_code

                        cols = [
                            "Year", "Race", "Session",
                            "Driver", "LapNumber",
                            "PitInTime", "PitOutTime",
                        ]

                        all_pits.append(
                            pit_laps[
                                [c for c in cols if c in pit_laps.columns]
                            ]
                        )

                # Session results: qualifying positions and race results
                results = session.results

                if results is not None and not results.empty:
                    result_df = results.copy()
                    result_df["Year"] = year
                    result_df["Race"] = race_name
                    result_df["Session"] = session_code

                    keep = [
                        "Year", "Race", "Session",
                        "Abbreviation", "FullName",
                        "TeamName", "Position",
                        "GridPosition", "Q1", "Q2", "Q3",
                        "Time", "Status", "Points",
                    ]

                    result_df = result_df[
                        [c for c in keep if c in result_df.columns]
                    ]

                    all_results.append(result_df)

                # Weather readings, when available
                weather = session.weather_data

                if weather is not None and not weather.empty:
                    weather_df = weather.copy()
                    weather_df["Year"] = year
                    weather_df["Race"] = race_name
                    weather_df["Session"] = session_code

                    all_weather.append(weather_df)

                print("  Collected available session data.")

            except Exception as exc:
                print(f"  Skipped: {exc}")

# Save only datasets for which data was collected.
datasets = {
    "f1_session_laps.csv": all_laps,
    "f1_session_results.csv": all_results,
    "f1_weather.csv": all_weather,
    "f1_pit_laps.csv": all_pits,
}

for filename, frames in datasets.items():
    if not frames:
        print(f"\nNo new data collected for {filename}")
        continue

    output = pd.concat(frames, ignore_index=True)
    path = DATA_DIR / filename

    # Save to a new filename, without overwriting existing files.
    if path.exists():
        path = path.with_name(
            f"{path.stem}_new{path.suffix}"
        )

    output.to_csv(path, index=False)

    print(f"\nSaved: {path} | Rows: {len(output)}")

print("\nCollection finished.")