import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Gender Pay Gap Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned_main_sample.csv")
    df_lag = pd.read_csv("data/cleaned_lag_sample.csv")
    reg = pd.read_csv("data/regression_summary.csv")
    return df, df_lag, reg

df, df_lag, reg = load_data()

st.title("Gender Pay Gap and Prior Pay Dashboard")

st.subheader("Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Observations", f"{len(df):,}")
col2.metric("Individuals", f"{df['PUBID_1997'].nunique():,}")
col3.metric("Average Wage", f"{df['HRLY_WAGE'].mean():.2f}")

st.subheader("Filters")
year_min, year_max = int(df["Year"].min()), int(df["Year"].max())
year_range = st.slider("Year range", year_min, year_max, (year_min, year_max))

filtered = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])].copy()

summary = (
    filtered.groupby(["Year", "sex_label"], as_index=False)["HRLY_WAGE"]
    .mean()
    .rename(columns={"HRLY_WAGE": "mean_wage"})
)

fig = px.line(summary, x="Year", y="mean_wage", color="sex_label", markers=True,
              title="Average Hourly Wage by Gender Over Time")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Regression Summary")
st.dataframe(reg, use_container_width=True)
