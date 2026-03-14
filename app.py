# =========================================================
# Streamlit App: Datathon 2026 Dashboard
# Reads graduate-full.csv.zip directly from GitHub
# Full version with Q1-Q4, model presentation, policy view,
# AI-style wage simulator, and model comparison
# =========================================================

import io
import zipfile
import warnings
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

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
def safe_exp_pct(beta):
    if pd.isna(beta):
        return np.nan
    return (np.exp(beta) - 1) * 100

def star_from_p(p):
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""

def human_term(term):
    mapping = {
        "female": "Female",
        "ln_prior_wage": "Log prior wage",
        "ln_prior_wage:female": "Log prior wage × Female",
        "ln_prior_wage:post_2018": "Log prior wage × Post-2018",
        "female:post_2018": "Female × Post-2018",
        "ln_prior_wage:female:post_2018": "Log prior wage × Female × Post-2018",
        "Year_num": "Year trend"
    }
    return mapping.get(term, term)

@st.cache_data(show_spinner=True)
def load_data_from_github_zip(zip_url: str) -> pd.DataFrame:
    response = requests.get(zip_url, timeout=180)
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

    # Keep string categories stable
    for col in ["Occupation_Group2", "Industry_Group"]:
        if col not in df.columns:
            df[col] = "Unknown"
        else:
            df[col] = df[col].astype(str).fillna("Unknown")

    if "marital_status" not in df.columns:
        df["marital_status"] = -1
    df["marital_status"] = df["marital_status"].fillna(-1)

    df["Year_num"] = df["Year"] if "Year" in df.columns else np.nan
    df["post_2018"] = np.where(df["Year_num"] >= 2018, 1, 0)

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

def fit_ols(formula, data):
    try:
        model = smf.ols(formula=formula, data=data).fit(cov_type="HC3")
        return model
    except Exception:
        return None

@st.cache_resource(show_spinner=True)
def fit_models(df, df_lag):
    models = {}

    # Q1: fixed-effect style descriptive regression models
    models["M1"] = fit_ols(
        """
        ln_wage ~ female + age + age_sq + HGC + TENURE + HRS_WRK
                + C(race_label) + C(region_label) + C(Year)
        """,
        df
    )

    models["M2"] = fit_ols(
        """
        ln_wage ~ female + age + age_sq + HGC + TENURE + HRS_WRK
                + C(race_label) + C(region_label) + C(marital_status) + C(Year)
        """,
        df
    )

    models["M3"] = fit_ols(
        """
        ln_wage ~ female + age + age_sq + HGC + TENURE + HRS_WRK
                + C(race_label) + C(region_label) + C(marital_status)
                + C(Occupation_Group2) + C(Industry_Group) + C(Year)
        """,
        df
    )

    # Q2 / Q3
    models["M4"] = fit_ols(
        """
        ln_wage ~ ln_prior_wage + female + age + age_sq + HGC + TENURE + HRS_WRK
                + C(race_label) + C(region_label) + C(marital_status) + C(Year)
        """,
        df_lag
    )

    models["M5"] = fit_ols(
        """
        ln_wage ~ ln_prior_wage * female + age + age_sq + HGC + TENURE + HRS_WRK
                + C(race_label) + C(region_label) + C(marital_status)
                + C(Occupation_Group2) + C(Industry_Group) + C(Year)
        """,
        df_lag
    )

    models["M6"] = fit_ols(
        """
        ln_wage ~ ln_prior_wage * post_2018 + female + age + age_sq + HGC + TENURE + HRS_WRK
                + C(race_label) + C(region_label) + C(marital_status)
                + C(Occupation_Group2) + C(Industry_Group) + C(Year)
        """,
        df_lag
    )

    models["M7"] = fit_ols(
        """
        ln_wage ~ ln_prior_wage + female * post_2018 + age + age_sq + HGC + TENURE + HRS_WRK
                + C(race_label) + C(region_label) + C(marital_status)
                + C(Occupation_Group2) + C(Industry_Group) + C(Year)
        """,
        df_lag
    )

    models["M8"] = fit_ols(
        """
        ln_wage ~ ln_prior_wage * female * post_2018 + age + age_sq + HGC + TENURE + HRS_WRK
                + C(race_label) + C(region_label) + C(marital_status)
                + C(Occupation_Group2) + C(Industry_Group) + C(Year)
        """,
        df_lag
    )

    # Forecast-friendly trend models for simulator
    models["F_Q1"] = fit_ols(
        """
        ln_wage ~ female + age + age_sq + HGC + TENURE + HRS_WRK + Year_num
                + C(race_label) + C(region_label) + C(marital_status)
                + C(Occupation_Group2) + C(Industry_Group)
        """,
        df
    )

    models["F_Q2"] = fit_ols(
        """
        ln_wage ~ ln_prior_wage * female + age + age_sq + HGC + TENURE + HRS_WRK + Year_num
                + C(race_label) + C(region_label) + C(marital_status)
                + C(Occupation_Group2) + C(Industry_Group)
        """,
        df_lag
    )

    models["F_Q3"] = fit_ols(
        """
        ln_wage ~ ln_prior_wage * female * post_2018 + age + age_sq + HGC + TENURE + HRS_WRK + Year_num
                + C(race_label) + C(region_label) + C(marital_status)
                + C(Occupation_Group2) + C(Industry_Group)
        """,
        df_lag
    )

    return models

def extract_model_table(models):
    rows = []

    model_meta = {
        "M1": ("Q1", "Baseline gender gap model"),
        "M2": ("Q1", "Add marital status"),
        "M3": ("Q1", "Full controls model"),
        "M4": ("Q2", "Prior wage model"),
        "M5": ("Q2", "Prior wage × female model"),
        "M6": ("Q3", "Prior wage × post-2018 model"),
        "M7": ("Q3", "Female × post-2018 model"),
        "M8": ("Q3", "Triple interaction model"),
    }

    terms_to_keep = [
        "female",
        "ln_prior_wage",
        "ln_prior_wage:female",
        "ln_prior_wage:post_2018",
        "female:post_2018",
        "ln_prior_wage:female:post_2018"
    ]

    for model_name, meta in model_meta.items():
        model = models.get(model_name)
        if model is None:
            continue

        q_label, model_desc = meta

        for term in terms_to_keep:
            if term in model.params.index:
                coef = model.params.get(term, np.nan)
                se = model.bse.get(term, np.nan)
                p = model.pvalues.get(term, np.nan)
                rows.append({
                    "question": q_label,
                    "model": model_name,
                    "model_desc": model_desc,
                    "term": term,
                    "term_label": human_term(term),
                    "coefficient": coef,
                    "std_error": se,
                    "p_value": p,
                    "stars": star_from_p(p),
                    "coef_display": f"{coef:.4f}{star_from_p(p)}",
                    "nobs": int(model.nobs),
                    "r_squared": float(model.rsquared),
                    "pct_effect": safe_exp_pct(coef) if term == "female" else np.nan
                })

    return pd.DataFrame(rows)

def build_summary_cards(df_filtered):
    male_mean = df_filtered.loc[df_filtered["sex_label"] == "Male", "HRLY_WAGE"].mean()
    female_mean = df_filtered.loc[df_filtered["sex_label"] == "Female", "HRLY_WAGE"].mean()

    if pd.notna(male_mean) and male_mean != 0 and pd.notna(female_mean):
        raw_gap_pct = 100 * (female_mean / male_mean - 1)
    else:
        raw_gap_pct = np.nan

    return male_mean, female_mean, raw_gap_pct

def get_top_interpretations(results_table):
    q1_row = results_table[(results_table["model"] == "M3") & (results_table["term"] == "female")]
    q2_row = results_table[(results_table["model"] == "M4") & (results_table["term"] == "ln_prior_wage")]
    q2_row2 = results_table[(results_table["model"] == "M5") & (results_table["term"] == "ln_prior_wage:female")]
    q3_row = results_table[(results_table["model"] == "M6") & (results_table["term"] == "ln_prior_wage:post_2018")]

    messages = []

    if not q1_row.empty:
        coef = q1_row.iloc[0]["coefficient"]
        pct = safe_exp_pct(coef)
        messages.append(f"Q1: In the full model, the female coefficient is {coef:.4f}, implying about {pct:.2f}% lower wages for women with similar observed characteristics.")

    if not q2_row.empty:
        coef = q2_row.iloc[0]["coefficient"]
        messages.append(f"Q2: Current wage remains strongly related to prior wage. The log prior wage coefficient is {coef:.4f} in the main prior-pay model.")

    if not q2_row2.empty:
        coef = q2_row2.iloc[0]["coefficient"]
        p = q2_row2.iloc[0]["p_value"]
        messages.append(f"Q2: The prior wage × female interaction is {coef:.4f} with p-value {p:.3f}, so the slope difference by gender is limited in this specification.")

    if not q3_row.empty:
        coef = q3_row.iloc[0]["coefficient"]
        p = q3_row.iloc[0]["p_value"]
        messages.append(f"Q3: The post-2018 prior wage interaction is {coef:.4f} with p-value {p:.3f}. The prior-current wage relationship does not collapse to zero.")

    return messages

def filter_main_data(df, year_range, selected_gender, selected_race, selected_region):
    filtered = df.copy()

    if year_range is not None:
        filtered = filtered[(filtered["Year"] >= year_range[0]) & (filtered["Year"] <= year_range[1])]

    filtered = filtered[filtered["sex_label"].isin(selected_gender)]
    filtered = filtered[filtered["race_label"].isin(selected_race)]
    filtered = filtered[filtered["region_label"].isin(selected_region)]
    return filtered

def filter_lag_data(df_lag, year_range, selected_gender, selected_race, selected_region):
    lag_filtered = df_lag.copy()

    if year_range is not None and "Year" in lag_filtered.columns:
        lag_filtered = lag_filtered[(lag_filtered["Year"] >= year_range[0]) & (lag_filtered["Year"] <= year_range[1])]

    lag_filtered = lag_filtered[lag_filtered["sex_label"].isin(selected_gender)]
    lag_filtered = lag_filtered[lag_filtered["race_label"].isin(selected_race)]
    lag_filtered = lag_filtered[lag_filtered["region_label"].isin(selected_region)]
    return lag_filtered

def build_profile_row(
    year_value,
    sex_label,
    race_label,
    region_label,
    marital_status,
    age,
    hgc,
    tenure,
    hrs_wrk,
    occupation_group,
    industry_group,
    prior_wage
):
    female = 1 if sex_label == "Female" else 0
    post_2018 = 1 if year_value >= 2018 else 0

    row = pd.DataFrame([{
        "Year": year_value,
        "Year_num": year_value,
        "female": female,
        "sex_label": sex_label,
        "race_label": race_label,
        "region_label": region_label,
        "marital_status": marital_status,
        "age": age,
        "age_sq": age ** 2,
        "HGC": hgc,
        "TENURE": tenure,
        "HRS_WRK": hrs_wrk,
        "Occupation_Group2": occupation_group,
        "Industry_Group": industry_group,
        "prior_wage": prior_wage,
        "ln_prior_wage": np.log(prior_wage) if prior_wage and prior_wage > 0 else np.nan,
        "post_2018": post_2018
    }])
    return row

def predict_wage_path(model, horizons, base_inputs):
    output_rows = []
    for h in horizons:
        yr = int(base_inputs["year_value"] + h)
        row = build_profile_row(
            year_value=yr,
            sex_label=base_inputs["sex_label"],
            race_label=base_inputs["race_label"],
            region_label=base_inputs["region_label"],
            marital_status=base_inputs["marital_status"],
            age=base_inputs["age"] + h,
            hgc=base_inputs["hgc"],
            tenure=base_inputs["tenure"] + h,
            hrs_wrk=base_inputs["hrs_wrk"],
            occupation_group=base_inputs["occupation_group"],
            industry_group=base_inputs["industry_group"],
            prior_wage=base_inputs["prior_wage"]
        )
        pred_ln = float(model.predict(row)[0])
        pred_wage = float(np.exp(pred_ln))
        output_rows.append({
            "horizon_years": h,
            "projected_year": yr,
            "predicted_ln_wage": pred_ln,
            "predicted_wage": pred_wage
        })
    return pd.DataFrame(output_rows)

def compare_model_predictions(models, base_inputs):
    compare_rows = []

    mapping = {
        "Q1 Full Controls Forecast Model": "F_Q1",
        "Q2 Prior Wage Forecast Model": "F_Q2",
        "Q3 Policy Forecast Model": "F_Q3"
    }

    for display_name, model_key in mapping.items():
        model = models.get(model_key)
        if model is None:
            continue

        row = build_profile_row(
            year_value=base_inputs["year_value"],
            sex_label=base_inputs["sex_label"],
            race_label=base_inputs["race_label"],
            region_label=base_inputs["region_label"],
            marital_status=base_inputs["marital_status"],
            age=base_inputs["age"],
            hgc=base_inputs["hgc"],
            tenure=base_inputs["tenure"],
            hrs_wrk=base_inputs["hrs_wrk"],
            occupation_group=base_inputs["occupation_group"],
            industry_group=base_inputs["industry_group"],
            prior_wage=base_inputs["prior_wage"]
        )

        pred_ln = float(model.predict(row)[0])
        pred_wage = float(np.exp(pred_ln))

        compare_rows.append({
            "model": display_name,
            "predicted_wage": pred_wage,
            "r_squared": getattr(model, "rsquared", np.nan),
            "nobs": int(getattr(model, "nobs", np.nan))
        })

    return pd.DataFrame(compare_rows)

def same_profile_gender_comparison(model, base_inputs):
    rows = []
    for sex_label in ["Male", "Female"]:
        row = build_profile_row(
            year_value=base_inputs["year_value"],
            sex_label=sex_label,
            race_label=base_inputs["race_label"],
            region_label=base_inputs["region_label"],
            marital_status=base_inputs["marital_status"],
            age=base_inputs["age"],
            hgc=base_inputs["hgc"],
            tenure=base_inputs["tenure"],
            hrs_wrk=base_inputs["hrs_wrk"],
            occupation_group=base_inputs["occupation_group"],
            industry_group=base_inputs["industry_group"],
            prior_wage=base_inputs["prior_wage"]
        )
        pred_ln = float(model.predict(row)[0])
        pred_wage = float(np.exp(pred_ln))
        rows.append({
            "sex_label": sex_label,
            "predicted_wage": pred_wage
        })
    return pd.DataFrame(rows)

def prior_wage_policy_comparison(model_with_prior, model_without_prior, base_inputs):
    rows = []

    row_with = build_profile_row(
        year_value=base_inputs["year_value"],
        sex_label=base_inputs["sex_label"],
        race_label=base_inputs["race_label"],
        region_label=base_inputs["region_label"],
        marital_status=base_inputs["marital_status"],
        age=base_inputs["age"],
        hgc=base_inputs["hgc"],
        tenure=base_inputs["tenure"],
        hrs_wrk=base_inputs["hrs_wrk"],
        occupation_group=base_inputs["occupation_group"],
        industry_group=base_inputs["industry_group"],
        prior_wage=base_inputs["prior_wage"]
    )
    pred_with = float(np.exp(model_with_prior.predict(row_with)[0]))
    rows.append({
        "scenario": "With prior wage information",
        "predicted_wage": pred_with
    })

    row_without = build_profile_row(
        year_value=base_inputs["year_value"],
        sex_label=base_inputs["sex_label"],
        race_label=base_inputs["race_label"],
        region_label=base_inputs["region_label"],
        marital_status=base_inputs["marital_status"],
        age=base_inputs["age"],
        hgc=base_inputs["hgc"],
        tenure=base_inputs["tenure"],
        hrs_wrk=base_inputs["hrs_wrk"],
        occupation_group=base_inputs["occupation_group"],
        industry_group=base_inputs["industry_group"],
        prior_wage=max(base_inputs["prior_wage"], 1e-6)
    )

    # For the no-prior-wage comparator, use Q1 forecast model
    pred_without = float(np.exp(model_without_prior.predict(row_without)[0]))
    rows.append({
        "scenario": "Without prior wage information",
        "predicted_wage": pred_without
    })

    return pd.DataFrame(rows)

def pre_post_policy_comparison(model_policy, base_inputs):
    rows = []

    for yr in [2017, 2021]:
        row = build_profile_row(
            year_value=yr,
            sex_label=base_inputs["sex_label"],
            race_label=base_inputs["race_label"],
            region_label=base_inputs["region_label"],
            marital_status=base_inputs["marital_status"],
            age=base_inputs["age"],
            hgc=base_inputs["hgc"],
            tenure=base_inputs["tenure"],
            hrs_wrk=base_inputs["hrs_wrk"],
            occupation_group=base_inputs["occupation_group"],
            industry_group=base_inputs["industry_group"],
            prior_wage=base_inputs["prior_wage"]
        )
        pred = float(np.exp(model_policy.predict(row)[0]))
        rows.append({
            "policy_period": "Pre-2018 proxy" if yr < 2018 else "Post-2018 proxy",
            "year_used": yr,
            "predicted_wage": pred
        })

    return pd.DataFrame(rows)

# -----------------------------
# Load
# -----------------------------
st.title("Datathon 2026 Dashboard")
st.caption("Gender pay gap, prior pay analysis, policy interpretation, and AI-style wage projection")

with st.spinner("Loading data from GitHub and preparing the dashboard..."):
    df_raw = load_data_from_github_zip(ZIP_URL)
    df, df_lag, year_gender, occupation_pivot, industry_pivot = preprocess_data(df_raw)
    models = fit_models(df, df_lag)
    results_table = extract_model_table(models)

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Global Filters")

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

filtered = filter_main_data(df, year_range, selected_gender, selected_race, selected_region)
lag_filtered = filter_lag_data(df_lag, year_range, selected_gender, selected_race, selected_region)

male_mean, female_mean, raw_gap_pct = build_summary_cards(filtered)

# -----------------------------
# KPI cards
# -----------------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Observations", f"{len(filtered):,}")
k2.metric("Individuals", f"{filtered['PUBID_1997'].nunique():,}" if "PUBID_1997" in filtered.columns else "N/A")
k3.metric("Average hourly wage", f"{filtered['HRLY_WAGE'].mean():.2f}" if len(filtered) else "N/A")
k4.metric("Male average wage", f"{male_mean:.2f}" if pd.notna(male_mean) else "N/A")
k5.metric("Female average wage", f"{female_mean:.2f}" if pd.notna(female_mean) else "N/A")

st.metric("Raw female vs male wage gap (%)", f"{raw_gap_pct:.2f}%" if pd.notna(raw_gap_pct) else "N/A")

# -----------------------------
# Tabs
# -----------------------------
tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Q1 Gender Pay Gap",
    "Q2 Prior Salary and Ban",
    "Q3 Post-2018 Policy",
    "Q4 Hidden Factors and Limits",
    "AI Wage Simulator",
    "Data Preview"
])

# -----------------------------
# Overview
# -----------------------------
with tab0:
    st.subheader("Executive Summary")

    for msg in get_top_interpretations(results_table):
        st.markdown(f"- {msg}")

    if "Year" in filtered.columns and len(filtered) > 0:
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

    st.subheader("Key Model Table")
    display_cols = ["question", "model", "model_desc", "term_label", "coefficient", "std_error", "p_value", "nobs", "r_squared"]
    st.dataframe(results_table[display_cols], use_container_width=True)

# -----------------------------
# Q1 Gender Pay Gap
# -----------------------------
with tab1:
    st.subheader("Question 1")
    st.markdown("**Is there evidence of pay discrimination between male and female employees?**")

    q1_results = results_table[results_table["question"] == "Q1"].copy()

    q1_main = q1_results[(q1_results["model"] == "M3") & (q1_results["term"] == "female")]
    if not q1_main.empty:
        coef = q1_main.iloc[0]["coefficient"]
        pct = safe_exp_pct(coef)
        p_val = q1_main.iloc[0]["p_value"]
        st.success(
            f"Full model result: female coefficient = {coef:.4f}, approximately {pct:.2f}% lower wages for women with similar observed characteristics. p-value = {p_val:.4g}"
        )

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

    st.subheader("Q1 Regression Results")
    st.dataframe(
        q1_results[["model", "model_desc", "term_label", "coefficient", "std_error", "p_value", "nobs", "r_squared"]],
        use_container_width=True
    )

    st.subheader("Q1 Coefficient Plot")
    q1_plot = q1_results.copy()
    q1_plot["ci_low"] = q1_plot["coefficient"] - 1.96 * q1_plot["std_error"]
    q1_plot["ci_high"] = q1_plot["coefficient"] + 1.96 * q1_plot["std_error"]
    q1_plot["label"] = q1_plot["model"] + " | " + q1_plot["term_label"]

    fig_q1_coef = go.Figure()
    fig_q1_coef.add_trace(go.Scatter(
        x=q1_plot["coefficient"],
        y=q1_plot["label"],
        mode="markers",
        error_x=dict(
            type="data",
            symmetric=False,
            array=q1_plot["ci_high"] - q1_plot["coefficient"],
            arrayminus=q1_plot["coefficient"] - q1_plot["ci_low"]
        )
    ))
    fig_q1_coef.add_vline(x=0, line_dash="dash")
    fig_q1_coef.update_layout(
        title="Q1 Key Coefficients with 95% Confidence Intervals",
        xaxis_title="Coefficient",
        yaxis_title=""
    )
    st.plotly_chart(fig_q1_coef, use_container_width=True)

    st.info(
        "Interpretation: A negative and statistically significant female coefficient across specifications indicates a persistent gender wage gap after controlling for education, tenure, hours, race, region, marital status, occupation, industry, and year."
    )

# -----------------------------
# Q2 Prior Salary and Ban
# -----------------------------
with tab2:
    st.subheader("Question 2")
    st.markdown("**Is there evidence that asking about prior salary negatively impacts female employees? Should a similar ban be implemented nationally?**")

    q2_results = results_table[results_table["question"] == "Q2"].copy()

    q2_main = q2_results[(q2_results["model"] == "M4") & (q2_results["term"] == "ln_prior_wage")]
    q2_inter = q2_results[(q2_results["model"] == "M5") & (q2_results["term"] == "ln_prior_wage:female")]

    if not q2_main.empty:
        coef = q2_main.iloc[0]["coefficient"]
        st.success(
            f"Main prior-pay finding: log prior wage coefficient = {coef:.4f}. Current wages remain strongly anchored to prior wages."
        )

    if not q2_inter.empty:
        coef = q2_inter.iloc[0]["coefficient"]
        p_val = q2_inter.iloc[0]["p_value"]
        st.warning(
            f"Interaction result: prior wage × female = {coef:.4f}, p-value = {p_val:.4g}. This suggests limited evidence that the slope itself differs strongly by gender in the full interaction model."
        )

    st.subheader("Prior Wage vs Current Wage")
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
        opacity=0.40,
        title="Prior Wage vs Current Wage"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    if len(lag_filtered) > 0:
        corr_value = lag_filtered[["prior_wage", "HRLY_WAGE"]].corr().iloc[0, 1]
        st.metric("Correlation: prior wage vs current wage", f"{corr_value:.3f}")

    st.subheader("Q2 Regression Results")
    st.dataframe(
        q2_results[["model", "model_desc", "term_label", "coefficient", "std_error", "p_value", "nobs", "r_squared"]],
        use_container_width=True
    )

    st.info(
        "Policy interpretation: The dashboard supports the concern that reliance on prior wage can carry existing wage disparities into future pay. This supports the rationale for salary-history restrictions, while remaining careful not to overstate strict causality."
    )

# -----------------------------
# Q3 Post-2018 Policy
# -----------------------------
with tab3:
    st.subheader("Question 3")
    st.markdown("**Would the prior-pay and current-pay relationship weaken after the California law? Would it become zero?**")

    q3_results = results_table[results_table["question"] == "Q3"].copy()

    q3_main = q3_results[(q3_results["model"] == "M6") & (q3_results["term"] == "ln_prior_wage:post_2018")]
    q3_triple = q3_results[(q3_results["model"] == "M8") & (q3_results["term"] == "ln_prior_wage:female:post_2018")]

    if not q3_main.empty:
        coef = q3_main.iloc[0]["coefficient"]
        p_val = q3_main.iloc[0]["p_value"]
        st.success(
            f"Post-2018 interaction: log prior wage × post-2018 = {coef:.4f}, p-value = {p_val:.4g}."
        )

    if not q3_triple.empty:
        coef = q3_triple.iloc[0]["coefficient"]
        p_val = q3_triple.iloc[0]["p_value"]
        st.warning(
            f"Triple interaction: log prior wage × female × post-2018 = {coef:.4f}, p-value = {p_val:.4g}."
        )

    q3_plot = q3_results.copy()
    q3_plot["ci_low"] = q3_plot["coefficient"] - 1.96 * q3_plot["std_error"]
    q3_plot["ci_high"] = q3_plot["coefficient"] + 1.96 * q3_plot["std_error"]
    q3_plot["label"] = q3_plot["model"] + " | " + q3_plot["term_label"]

    fig_q3_coef = go.Figure()
    fig_q3_coef.add_trace(go.Scatter(
        x=q3_plot["coefficient"],
        y=q3_plot["label"],
        mode="markers",
        error_x=dict(
            type="data",
            symmetric=False,
            array=q3_plot["ci_high"] - q3_plot["coefficient"],
            arrayminus=q3_plot["coefficient"] - q3_plot["ci_low"]
        )
    ))
    fig_q3_coef.add_vline(x=0, line_dash="dash")
    fig_q3_coef.update_layout(
        title="Q3 Key Coefficients with 95% Confidence Intervals",
        xaxis_title="Coefficient",
        yaxis_title=""
    )
    st.plotly_chart(fig_q3_coef, use_container_width=True)

    st.subheader("Q3 Regression Results")
    st.dataframe(
        q3_results[["model", "model_desc", "term_label", "coefficient", "std_error", "p_value", "nobs", "r_squared"]],
        use_container_width=True
    )

    st.info(
        "Interpretation: Even if salary-history restrictions weaken direct employer reliance on past wages, the relationship between prior and current pay is not expected to become zero because prior wages also reflect accumulated experience, occupation trajectory, and labor-market sorting."
    )

# -----------------------------
# Q4 Hidden Factors and Limits
# -----------------------------
with tab4:
    st.subheader("Question 4")
    st.markdown("**Are there hidden or latent factors influencing the observed differences?**")

    st.markdown("""
    **Potential latent factors not fully observed in the dataset**
    - Unobserved worker productivity
    - Career interruptions
    - Caregiving burden and motherhood penalty
    - Negotiation behavior
    - Firm compensation policy
    - Union coverage
    - Local labor market conditions
    - Job-switch timing and motive
    - Unobserved job quality
    - Measurement error in wages or hours
    - Selection into employment
    """)

    col_c, col_d = st.columns(2)

    with col_c:
        occ_plot = occupation_pivot.dropna(subset=["female_to_male_ratio"]).copy()
        occ_plot = occ_plot.sort_values("female_to_male_ratio").head(15)

        fig_occ = px.bar(
            occ_plot,
            x="female_to_male_ratio",
            y="Occupation_Group2",
            orientation="h",
            title="Lowest Female-to-Male Wage Ratios by Occupation Group"
        )
        fig_occ.add_vline(x=1.0, line_dash="dash")
        st.plotly_chart(fig_occ, use_container_width=True)

    with col_d:
        ind_plot = industry_pivot.dropna(subset=["female_to_male_ratio"]).copy()
        ind_plot = ind_plot.sort_values("female_to_male_ratio").head(15)

        fig_ind = px.bar(
            ind_plot,
            x="female_to_male_ratio",
            y="Industry_Group",
            orientation="h",
            title="Lowest Female-to-Male Wage Ratios by Industry Group"
        )
        fig_ind.add_vline(x=1.0, line_dash="dash")
        st.plotly_chart(fig_ind, use_container_width=True)

    st.warning(
        "Limitation note: The dataset does not directly observe whether employers explicitly asked for salary history. The results therefore support the policy rationale and association patterns, but they do not by themselves provide strict causal identification of employer behavior."
    )

# -----------------------------
# AI Wage Simulator
# -----------------------------
with tab5:
    st.subheader("AI-Style Wage Projection Simulator")
    st.markdown("Use the controls below to simulate projected wages, compare models, and connect the predictions back to Q1-Q4.")

    sim_col1, sim_col2, sim_col3 = st.columns(3)

def clamp(value, min_value, max_value):
    if pd.isna(value):
        return min_value
    return max(min_value, min(float(value), max_value))

default_year = int(df["Year"].max()) if "Year" in df.columns else 2021
default_age = int(clamp(np.nanmedian(df["age"]) if "age" in df.columns and df["age"].notna().any() else 30, 18, 70))
default_hgc = float(clamp(np.nanmedian(df["HGC"]) if "HGC" in df.columns and df["HGC"].notna().any() else 16.0, 0.0, 25.0))
default_tenure = float(clamp(np.nanmedian(df["TENURE"]) if "TENURE" in df.columns and df["TENURE"].notna().any() else 2.0, 0.0, 40.0))
default_hours = float(clamp(np.nanmedian(df["HRS_WRK"]) if "HRS_WRK" in df.columns and df["HRS_WRK"].notna().any() else 40.0, 1.0, 120.0))
default_prior_wage = float(clamp(np.nanmedian(df_lag["prior_wage"]) if len(df_lag) > 0 else 25.0, 0.5, 500.0))

    occ_choices = sorted(df["Occupation_Group2"].dropna().unique().tolist())
    ind_choices = sorted(df["Industry_Group"].dropna().unique().tolist())
    race_choices = sorted(df["race_label"].dropna().unique().tolist())
    region_choices = sorted(df["region_label"].dropna().unique().tolist())
    marital_choices = sorted(df["marital_status"].dropna().unique().tolist())

    with sim_col1:
        sim_year = st.number_input("Base year", min_value=int(df["Year"].min()), max_value=int(df["Year"].max()) + 10, value=default_year, step=1)
        sim_sex = st.selectbox("Gender", ["Male", "Female"], index=1)
        sim_race = st.selectbox("Race", race_choices, index=0 if race_choices else None)
        sim_region = st.selectbox("Region", region_choices, index=0 if region_choices else None)

    with sim_col2:
        sim_marital = st.selectbox("Marital status code", marital_choices, index=0 if marital_choices else None)
        sim_age = st.number_input("Age", min_value=18, max_value=70, value=int(default_age), step=1)
        sim_hgc = st.number_input("Education (HGC)", min_value=0.0, max_value=25.0, value=float(default_hgc), step=0.5)
        sim_tenure = st.number_input("Tenure", min_value=0.0, max_value=40.0, value=float(default_tenure), step=0.5)

    with sim_col3:
        sim_hours = st.number_input("Weekly hours worked", min_value=1.0, max_value=120.0, value=float(default_hours), step=1.0)
        sim_prior_wage = st.number_input("Prior wage", min_value=0.5, max_value=500.0, value=float(default_prior_wage), step=0.5)
        sim_occ = st.selectbox("Occupation group", occ_choices, index=0 if occ_choices else None)
        sim_ind = st.selectbox("Industry group", ind_choices, index=0 if ind_choices else None)

    forecast_model_choice = st.selectbox(
        "Projection model",
        [
            "Q1 Full Controls Forecast Model",
            "Q2 Prior Wage Forecast Model",
            "Q3 Policy Forecast Model"
        ],
        index=1
    )

    horizon_set = st.multiselect("Projection horizons", [0, 1, 3, 5], default=[0, 1, 3, 5])

    base_inputs = {
        "year_value": int(sim_year),
        "sex_label": sim_sex,
        "race_label": sim_race,
        "region_label": sim_region,
        "marital_status": sim_marital,
        "age": float(sim_age),
        "hgc": float(sim_hgc),
        "tenure": float(sim_tenure),
        "hrs_wrk": float(sim_hours),
        "occupation_group": sim_occ,
        "industry_group": sim_ind,
        "prior_wage": float(sim_prior_wage)
    }

    forecast_model_map = {
        "Q1 Full Controls Forecast Model": models.get("F_Q1"),
        "Q2 Prior Wage Forecast Model": models.get("F_Q2"),
        "Q3 Policy Forecast Model": models.get("F_Q3")
    }

    selected_forecast_model = forecast_model_map.get(forecast_model_choice)

    if selected_forecast_model is not None and len(horizon_set) > 0:
        pred_path = predict_wage_path(selected_forecast_model, sorted(horizon_set), base_inputs)

        p1, p2, p3, p4 = st.columns(4)
        if 0 in pred_path["horizon_years"].values:
            p1.metric(
                "Predicted wage now",
                f"{pred_path.loc[pred_path['horizon_years'] == 0, 'predicted_wage'].iloc[0]:.2f}"
            )
        if 1 in pred_path["horizon_years"].values:
            p2.metric(
                "Predicted wage in 1 year",
                f"{pred_path.loc[pred_path['horizon_years'] == 1, 'predicted_wage'].iloc[0]:.2f}"
            )
        if 3 in pred_path["horizon_years"].values:
            p3.metric(
                "Predicted wage in 3 years",
                f"{pred_path.loc[pred_path['horizon_years'] == 3, 'predicted_wage'].iloc[0]:.2f}"
            )
        if 5 in pred_path["horizon_years"].values:
            p4.metric(
                "Predicted wage in 5 years",
                f"{pred_path.loc[pred_path['horizon_years'] == 5, 'predicted_wage'].iloc[0]:.2f}"
            )

        fig_pred = px.line(
            pred_path,
            x="projected_year",
            y="predicted_wage",
            markers=True,
            title="Projected Wage Path"
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        st.subheader("Same Profile, Different Gender")
        gender_compare = same_profile_gender_comparison(models.get("F_Q1"), base_inputs)
        fig_gender = px.bar(
            gender_compare,
            x="sex_label",
            y="predicted_wage",
            color="sex_label",
            title="Predicted Wage Under the Same Observed Profile"
        )
        st.plotly_chart(fig_gender, use_container_width=True)

        st.subheader("With vs Without Prior Wage Information")
        prior_compare = prior_wage_policy_comparison(models.get("F_Q2"), models.get("F_Q1"), base_inputs)
        fig_prior = px.bar(
            prior_compare,
            x="scenario",
            y="predicted_wage",
            color="scenario",
            title="Predicted Wage Under Alternative Prior-Wage Information Regimes"
        )
        st.plotly_chart(fig_prior, use_container_width=True)

        st.subheader("Pre-2018 vs Post-2018 Policy Proxy")
        policy_compare = pre_post_policy_comparison(models.get("F_Q3"), base_inputs)
        fig_policy = px.bar(
            policy_compare,
            x="policy_period",
            y="predicted_wage",
            color="policy_period",
            title="Predicted Wage Under Pre-2018 and Post-2018 Proxies"
        )
        st.plotly_chart(fig_policy, use_container_width=True)

        st.subheader("Model Comparison")
        compare_df = compare_model_predictions(models, base_inputs)
        st.dataframe(compare_df, use_container_width=True)

        st.info(
            "How to read this simulator: Q1 compares same-profile male and female predictions. Q2 compares scenarios with and without prior wage information. Q3 compares pre-2018 and post-2018 policy proxies. Q4 is reflected in the uncertainty and omitted-factor caveat below."
        )

        st.warning(
            "Uncertainty note: These projections are model-based scenario forecasts, not guaranteed future outcomes. They rely on observed variables and do not include unobserved productivity, caregiving burden, negotiation dynamics, employer pay-setting differences, or future labor market shocks."
        )

# -----------------------------
# Data Preview
# -----------------------------
with tab6:
    st.subheader("Raw Data Preview")
    st.dataframe(df_raw.head(100), use_container_width=True)

    st.subheader("Cleaned Data Preview")
    st.dataframe(filtered.head(100), use_container_width=True)

    st.subheader("Lag Sample Preview")
    st.dataframe(lag_filtered.head(100), use_container_width=True)

    st.subheader("Column Names")
    st.write(df_raw.columns.tolist())
