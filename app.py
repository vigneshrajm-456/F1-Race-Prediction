
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="F1 Mexico GP Predictor",
    page_icon="🏎️",
    layout="wide",
)

DATA_FILE = Path("data/mexico_2026_prediction.csv")

st.markdown("""
<style>
.stApp {
    background-color: #101014;
    color: #f5f5f5;
}
[data-testid="stMetric"] {
    background: #1d1d24;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #35353e;
}
h1, h2, h3 {
    color: #ff4545;
}
</style>
""", unsafe_allow_html=True)

st.title("🏎️ F1 RACE PREDICTOR")
st.subheader("2026 Mexico City Grand Prix")
st.caption("External forecast dashboard | Forecast, not confirmed race results")

if not DATA_FILE.exists():
    st.error(f"Prediction file not found: {DATA_FILE}")
    st.info("Run `python predict_mexico.py` first.")
    st.stop()

df = pd.read_csv(DATA_FILE)

required = {
    "Driver",
    "WinProbabilityPercent",
    "PodiumProbabilityPercent",
}

if not required.issubset(df.columns):
    st.error("CSV columns do not match the expected prediction format.")
    st.write("Available columns:", list(df.columns))
    st.stop()

df["WinProbabilityPercent"] = pd.to_numeric(
    df["WinProbabilityPercent"], errors="coerce"
)
df["PodiumProbabilityPercent"] = pd.to_numeric(
    df["PodiumProbabilityPercent"], errors="coerce"
)
df = df.dropna(subset=["Driver", "WinProbabilityPercent"])
df = df.sort_values("WinProbabilityPercent", ascending=False)

if df.empty:
    st.error("No valid driver predictions found in the CSV.")
    st.stop()

winner = df.iloc[0]

c1, c2, c3 = st.columns(3)
c1.metric("🏆 Forecast Favorite", winner["Driver"])
c2.metric("Win Probability", f'{winner["WinProbabilityPercent"]:.1f}%')
c3.metric("Drivers Listed", len(df))

st.divider()

st.header("📊 Win Probability")

fig = px.bar(
    df,
    x="WinProbabilityPercent",
    y="Driver",
    orientation="h",
    text="WinProbabilityPercent",
    labels={
        "WinProbabilityPercent": "Win Probability (%)",
        "Driver": "Driver",
    },
    template="plotly_dark",
)

fig.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside",
    marker_color="#ff4545",
)
fig.update_layout(
    yaxis={"categoryorder": "total ascending"},
    xaxis={"range": [0, max(25, df["WinProbabilityPercent"].max() + 8)]},
    height=max(400, len(df) * 45),
)
st.plotly_chart(fig, use_container_width=True)

st.header("🥇 Podium Probability")

podium_df = df.sort_values("PodiumProbabilityPercent", ascending=False)

fig2 = px.bar(
    podium_df,
    x="PodiumProbabilityPercent",
    y="Driver",
    orientation="h",
    text="PodiumProbabilityPercent",
    labels={
        "PodiumProbabilityPercent": "Podium Probability (%)",
        "Driver": "Driver",
    },
    template="plotly_dark",
)
fig2.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside",
    marker_color="#39c6a5",
)
fig2.update_layout(
    yaxis={"categoryorder": "total ascending"},
    xaxis={"range": [0, max(50, podium_df["PodiumProbabilityPercent"].max() + 8)]},
    height=max(400, len(podium_df) * 45),
)
st.plotly_chart(fig2, use_container_width=True)

st.header("📋 Full Prediction Table")

display_df = df.copy()
if "Rank" not in display_df.columns:
    display_df.insert(0, "Rank", range(1, len(display_df) + 1))
display_df = display_df.rename(columns={
    "WinProbabilityPercent": "Win Probability (%)",
    "PodiumProbabilityPercent": "Podium Probability (%)",
    "PredictionSource": "Source",
    "GrandPrix": "Grand Prix",
})

columns = [
    col for col in [
        "Rank",
        "Driver",
        "Win Probability (%)",
        "Podium Probability (%)",
        "Source",
        "Grand Prix",
    ]
    if col in display_df.columns
]

st.dataframe(
    display_df[columns],
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "⬇️ Download Predictions CSV",
    data=display_df.to_csv(index=False).encode("utf-8"),
    file_name="mexico_2026_predictions.csv",
    mime="text/csv",
)

st.caption(
    "Source: PITWALL forecast snapshot used by predict_mexico.py. "
    "Percentages are estimates from that source, not guarantees or "
    "independently validated predictions. Refresh the source data before race day."
)

