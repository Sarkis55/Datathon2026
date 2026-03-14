# =========================================================
# Streamlit App: Datathon 2026 Dashboard
# Reads graduate-full.csv.zip directly from GitHub
# =========================================================

import io
import zipfile
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Datathon 2026 Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# GitHub raw ZIP URL
# -----------------------------
ZIP_URL = "https://raw.githubusercontent.com/Sarkis55/Datathon2026/main/graduate-full.csv.zip"

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=True)
def load_data_from_github_zip(zip_url: str) -> pd.DataFrame:
    response = requests.get(zip_url, timeout=120)
    response.raise_for_status()

    zip_bytes = io.BytesIO(response.content)

    with zipfile.ZipFile(zip_bytes, "r") as zf:
        csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) == 0:
            raise FileNotFoundError("No CSV file found inside the ZIP.")
        csv_name = csv_names[0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f)

    return df

@st.cache_data(show_spinner=True)
def preprocess_data(df_raw: pd.DataFrame):
    df = df_raw.copy()
    df.columns = [c.strip() for c in df.columns]

    numeric_cols = [
        "PUBID_1997", "SAMPLE_RACE_1997", "SAMPLE_SEX_1997", "Year",
        "Employed", "TENURE", "HRLY_WAGE", "HRLY_COMP", "HRS_WRK",
        "UID", "Code_1990", "marital_status", "HGC", "Region"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    date_cols = ["DOB", "Interview_Date", "StartDate", "StopDate"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Core cleaning
    if "Employed" in df.columns:
        df = df[df["Employed"] == 1].copy()

    df = df[df["HRLY_WAGE"].notna()].copy()
    df = df[df["HRLY_WAGE"] > 0].copy()

    # Trim extreme wages
    wage_low = df["HRLY_WAGE"].quantile(0.01)
    wage_high = df["HRLY_WAGE"].quantile(0.99)
    df = df[(df["HRLY_WAGE"] >= wage_low) & (df["HRLY_WAGE"] <= wage_high)].copy()

    if "HRS_WRK" in df.columns:
        df = df[(df["HRS_WRK"].isna()) | ((df["HRS_WRK"] > 0) & (df["HRS_WRK"] <= 120))].copy()

    if "TENURE" in df.columns:
        df = df[(df["TENURE"].isna()) | (df["TENURE"] >= 0)].copy()

    # Feature engineering
    if "SAMPLE_SEX_1997" in df.columns:
        df["female"] = np.where(df["SAMPLE_SEX_1997"] == 2, 1, 0)
        sex_map = {1: "Male", 2: "Female"}
        df["sex_label"] = df["SAMPLE_SEX_1997"].map(sex_map).fillna("Unknown")
    else:
        df["female"] = np.nan
        df["sex_label"] = "Unknown"

    df["ln_wage"] = np.log(df["HRLY_WAGE"])

    if "DOB" in df.columns and "Interview_Date" in df.columns:
        df["age"] = (df["Interview_Date"] - df["DOB"]).dt.days / 365.25
        df["age_sq"] = df["age"] ** 2
    else:
        df["age"] = np.nan
        df["age_sq"] = np.nan

    race_map = {
        1: "Hispanic",
        2: "Black",
        3: "Non-Black/Non-Hispanic",
        4: "Mixed/Other"
    }
    if "SAMPLE_RACE_1997" in df.columns:
        df["race_label"] = df["SAMPLE_RACE_1997"].map(race_map).fillna("Unknown")
    else:
        df["race_label"] = "Unknown"

    region_map = {
        1: "Northeast",
        2: "North Central",
        3: "South",
        4: "West"
    }
    if "Region" in df.columns:
        df["region_label"] = df["Region"].map(region_map).fillna("Unknown")
    else:
        df["region_label"] = "Unknown"

    for col in ["Occupation_Group2", "Industry_Group"]:
        if col not in df.columns:
            df[col] = "Unknown"
        else:
            df[col] = df[col].astype(str).fillna("Unknown")

    # Sort and build prior wage
    sort_cols = []
    if "PUBID_1997" in df.columns:
        sort_cols.append("PUBID_1997")
    if "Year" in df.columns:
        sort_cols.append("Year")
    if "Interview_Date" in df.columns:
        sort_cols.append("Interview_Date")

    if sort_cols:
        df = df.sort_values(sort_cols).copy()

    if "PUBID_1997" in df.columns:
        df["prior_wage"] = df.groupby("PUBID_1997")["HRLY_WAGE"].shift(1)
        df["ln_prior_wage"] = np.log(df["prior_wage"])
        df["prior_year"] = df.groupby("PUBID_1997")["Year"].shift(1) if "Year" in df.columns else np.nan
        df["year_gap"] = df["Year"] - df["prior_year"] if "Year" in df.columns else np.nan
    else:
        df["prior_wage"] = np.nan
        df["ln_prior_wage"] = np.nan
        df["prior_year"] = np.nan
        df["year_gap"] = np.nan

    df_lag = df[df["prior_wage"].notna()].copy()
    if "prior_wage" in df_lag.columns:
        df_lag = df_lag[df_lag["prior_wage"] > 0].copy()
    if "year_gap" in df_lag.columns:
        df_lag = df_lag[(df_lag["year_gap"].isna()) | (df_lag["year_gap"] <= 3)].copy()

    year_gender = (
        df.groupby(["Year", "sex_label"], as_index=False)
          .agg(
              mean_wage=("HRLY_WAGE", "mean"),
              median_wage=("HRLY_WAGE", "median"),
              n=("HRLY_WAGE", "size")
          )
        if "Year" in df.columns else pd.DataFrame()
    )

    occupation_gap = (
        df.groupby(["Occupation_Group2", "sex_label"], as_index=False)
          .agg(mean_wage=("HRLY_WAGE", "mean"), n=("HRLY_WAGE", "size"))
    )
    occupation_pivot = occupation_gap.pivot(index="Occupation_Group2", columns="sex_label", values="mean_wage")
    if "Female" in occupation_pivot.columns and "Male" in occupation_pivot.columns:
        occupation_pivot["female_to_male_ratio"] = occupation_pivot["Female"] / occupation_pivot["Male"]
    else:
        occupation_pivot["female_to_male_ratio"] = np.nan
    occupation_pivot = occupation_pivot.reset_index()

    industry_gap = (
        df.groupby(["Industry_Group", "sex_label"], as_index=False)
          .agg(mean_wage=("HRLY_WAGE", "mean"), n=("HRLY_WAGE", "size"))
    )
    industry_pivot = industry_gap.pivot(index="Industry_Group", columns="sex_label", values="mean_wage")
    if "Female" in industry_pivot.columns and "Male" in industry_pivot.columns:
        industry_pivot["female_to_male_ratio"] = industry_pivot["Female"] / industry_pivot["Male"]
    else:
        industry_pivot["female_to_male_ratio"] = np.nan
    industry_pivot = industry_pivot.reset_index()

    return df, df_lag, year_gender, occupation_pivot, industry_pivot

# -----------------------------
# Load
# -----------------------------
st.title("Datathon 2026 Dashboard")
st.caption("Gender pay gap and prior pay analysis dashboard")

with st.spinner("Loading data from GitHub..."):
    df_raw = load_data_from_github_zip(ZIP_URL)
    df, df_lag, year_gender, occupation_pivot, industry_pivot = preprocess_data(df_raw)

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters")

if "Year" in df.columns and df["Year"].notna().any():
    year_min = int(df["Year"].min())
    year_max = int(df["Year"].max())
    year_range = st.sidebar.slider("Year range", year_min, year_max, (year_min, year_max))
else:
    year_range = None

gender_options = sorted(df["sex_label"].dropna().unique().tolist())
selected_gender = st.sidebar.multiselect("Gender", gender_options, default=gender_options)

race_options = sorted(df["race_label"].dropna().unique().tolist())
selected_race = st.sidebar.multiselect("Race", race_options, default=race_options)

region_options = sorted(df["region_label"].dropna().unique().tolist())
selected_region = st.sidebar.multiselect("Region", region_options, default=region_options)

# -----------------------------
# Apply filters
# -----------------------------
filtered = df.copy()

if year_range is not None:
    filtered = filtered[(filtered["Year"] >= year_range[0]) & (filtered["Year"] <= year_range[1])]

filtered = filtered[filtered["sex_label"].isin(selected_gender)]
filtered = filtered[filtered["race_label"].isin(selected_race)]
filtered = filtered[filtered["region_label"].isin(selected_region)]

# -----------------------------
# KPI cards
# -----------------------------
male_mean = filtered.loc[filtered["sex_label"] == "Male", "HRLY_WAGE"].mean()
female_mean = filtered.loc[filtered["sex_label"] == "Female", "HRLY_WAGE"].mean()

if pd.notna(male_mean) and male_mean != 0 and pd.notna(female_mean):
    raw_gap_pct = 100 * (female_mean / male_mean - 1)
else:
    raw_gap_pct = np.nan

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Observations", f"{len(filtered):,}")
col2.metric("Individuals", f"{filtered['PUBID_1997'].nunique():,}" if "PUBID_1997" in filtered.columns else "N/A")
col3.metric("Average hourly wage", f"{filtered['HRLY_WAGE'].mean():.2f}" if len(filtered) else "N/A")
col4.metric("Male average wage", f"{male_mean:.2f}" if pd.notna(male_mean) else "N/A")
col5.metric("Female average wage", f"{female_mean:.2f}" if pd.notna(female_mean) else "N/A")

st.metric("Raw female vs male wage gap (%)", f"{raw_gap_pct:.2f}%" if pd.notna(raw_gap_pct) else "N/A")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Descriptive Analysis",
    "Prior Pay",
    "Occupations and Industries",
    "Data Preview"
])

# -----------------------------
# Tab 1: Overview
# -----------------------------
with tab1:
    st.subheader("Executive Summary")
    st.markdown("""
    - This dashboard loads the submitted dataset directly from the public GitHub repository.
    - The cleaned sample focuses on employed observations with positive hourly wages.
    - Prior wage is constructed from the previous observed wage of the same individual.
    - The dashboard is designed for quick datathon presentation and policy-oriented interpretation.
    """)

    if "Year" in filtered.columns:
        summary = (
            filtered.groupby(["Year", "sex_label"], as_index=False)["HRLY_WAGE"]
            .mean()
            .rename(columns={"HRLY_WAGE": "mean_wage"})
        )

        fig = px.line(
            summary,
            x="Year",
            y="mean_wage",
            color="sex_label",
            markers=True,
            title="Average Hourly Wage by Gender Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Tab 2: Descriptive Analysis
# -----------------------------
with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        fig_hist = px.histogram(
            filtered,
            x="HRLY_WAGE",
            color="sex_label",
            barmode="overlay",
            nbins=50,
            title="Hourly Wage Distribution by Gender",
            opacity=0.60
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        trimmed = filtered[filtered["HRLY_WAGE"] <= filtered["HRLY_WAGE"].quantile(0.95)].copy()
        fig_box = px.box(
            trimmed,
            x="sex_label",
            y="HRLY_WAGE",
            color="sex_label",
            title="Hourly Wage by Gender"
        )
        st.plotly_chart(fig_box, use_container_width=True)

# -----------------------------
# Tab 3: Prior Pay
# -----------------------------
with tab3:
    lag_filtered = df_lag.copy()

    if year_range is not None and "Year" in lag_filtered.columns:
        lag_filtered = lag_filtered[(lag_filtered["Year"] >= year_range[0]) & (lag_filtered["Year"] <= year_range[1])]

    lag_filtered = lag_filtered[lag_filtered["sex_label"].isin(selected_gender)]
    lag_filtered = lag_filtered[lag_filtered["race_label"].isin(selected_race)]
    lag_filtered = lag_filtered[lag_filtered["region_label"].isin(selected_region)]

    st.write(f"Lag sample size: {len(lag_filtered):,}")

    plot_lag = lag_filtered[
        (lag_filtered["HRLY_WAGE"] <= lag_filtered["HRLY_WAGE"].quantile(0.99)) &
        (lag_filtered["prior_wage"] <= lag_filtered["prior_wage"].quantile(0.99))
    ].copy()

    if len(plot_lag) > 8000:
        plot_lag = plot_lag.sample(8000, random_state=42)

    fig_scatter = px.scatter(
        plot_lag,
        x="prior_wage",
        y="HRLY_WAGE",
        color="sex_label",
        opacity=0.4,
        title="Prior Wage vs Current Wage"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    if len(lag_filtered) > 0:
        corr_value = lag_filtered[["prior_wage", "HRLY_WAGE"]].corr().iloc[0, 1]
        st.metric("Correlation: prior wage vs current wage", f"{corr_value:.3f}")

# -----------------------------
# Tab 4: Occupations and Industries
# -----------------------------
with tab4:
    col_c, col_d = st.columns(2)

    with col_c:
        occ_plot = occupation_pivot.dropna(subset=["female_to_male_ratio"]).copy()
        occ_plot = occ_plot.sort_values("female_to_male_ratio").tail(15)

        fig_occ = px.bar(
            occ_plot,
            x="female_to_male_ratio",
            y="Occupation_Group2",
            orientation="h",
            title="Female-to-Male Wage Ratio by Occupation Group"
        )
        fig_occ.add_vline(x=1.0, line_dash="dash")
        st.plotly_chart(fig_occ, use_container_width=True)

    with col_d:
        ind_plot = industry_pivot.dropna(subset=["female_to_male_ratio"]).copy()
        ind_plot = ind_plot.sort_values("female_to_male_ratio").tail(15)

        fig_ind = px.bar(
            ind_plot,
            x="female_to_male_ratio",
            y="Industry_Group",
            orientation="h",
            title="Female-to-Male Wage Ratio by Industry Group"
        )
        fig_ind.add_vline(x=1.0, line_dash="dash")
        st.plotly_chart(fig_ind, use_container_width=True)

# -----------------------------
# Tab 5: Data Preview
# -----------------------------
with tab5:
    st.subheader("Raw Data Preview")
    st.dataframe(df_raw.head(100), use_container_width=True)

    st.subheader("Cleaned Data Preview")
    st.dataframe(filtered.head(100), use_container_width=True)

    st.subheader("Column Names")
    st.write(df_raw.columns.tolist())
