import pandas as pd
from pathlib import Path

for file in Path("data").glob("*.csv"):
    print(f"\n{'=' * 50}")
    print(f"FILE: {file.name}")

    df = pd.read_csv(file)

    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print(df.head(3).to_string(index=False))