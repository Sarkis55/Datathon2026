# =========================================================
# Streamlit App: Datathon 2026 Dashboard
# Cloud-optimized full version
# - Loads graduate-full.csv.zip directly from GitHub
# - Answers Q1-Q4
# - Shows visualizations
# - Includes user-facing prediction and model comparison
# - Optimized for Streamlit Cloud performance
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
# Config
# -----------------------------
ZIP_URL = "https://raw.githubusercontent.com/Sarkis55/Datathon2026/main/graduate-full.csv.zip"

# Reduced to fit Streamlit Cloud limits while keeping same functionality
MAX_MODEL_ROWS_MAIN = 1000
MAX_MODEL_ROWS_LAG = 1000
MAX_SCATTER_POINTS = 1500
MAX_PREVIEW_ROWS = 50

# -----------------------------
# Utility helpers
# -----------------------------
def clamp(value, min_value, max_value):
    if value is None:
        return min_value
    try:
        if pd.isna(value):
            return min_value
    except Exception:
        pass
    return max(min_value, min(float(value), max_value))

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

def make_model_sample(df, max_rows, seed=42):
    if df is None or len(df) == 0:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    if len(df) <= max_rows:
        return df.copy()
    return df.sample(max_rows, random_state=seed).copy()

def maybe_sample_for_scatter(df, max_rows=MAX_SCATTER_POINTS, seed=42):
    if df is None or len(df) == 0:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    if len(df) <= max_rows:
        return df.copy()
    return df.sample(max_rows, random_state=seed).copy()

def safe_selectbox(label, options, default_index=0, key=None):
    if not options:
        st.warning(f"No available options for {label}.")
        return None
    idx = min(max(default_index, 0), len(options) - 1)
    return st.selectbox(label, options, index=idx, key=key)

# -----------------------------
# Data loading
# -----------------------------
@st.cache_data(show_spinner=True)
def load_data_from_github_zip(zip_url: str) -> pd.DataFrame:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/octet-stream"
    }

    last_error = None
    for timeout_sec in [60, 120, 180]:
        try:
            response = requests.get(zip_url, timeout=timeout_sec, headers=headers)
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
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Failed to load ZIP from GitHub: {last_error}")

# -----------------------------
# Preprocessing
# -----------------------------
@st.cache_data(show_spinner=True)
def preprocess_data(df_raw: pd.DataFrame):
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

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

    if "HRLY_WAGE" not in df.columns:
        raise KeyError("Required column 'HRLY_WAGE' not found in dataset.")

    if "Employed" in df.columns:
        df = df[df["Employed"] == 1].copy()

    df = df[df["HRLY_WAGE"].notna()].copy()
    df = df[df["HRLY_WAGE"] > 0].copy()

    if len(df) > 0:
        wage_low = df["HRLY_WAGE"].quantile(0.01)
        wage_high = df["HRLY_WAGE"].quantile(0.99)
        df = df[(df["HRLY_WAGE"] >= wage_low) & (df["HRLY_WAGE"] <= wage_high)].copy()

    if "HRS_WRK" in df.columns:
        df = df[(df["HRS_WRK"].isna()) | ((df["HRS_WRK"] > 0) & (df["HRS_WRK"] <= 120))].copy()

    if "TENURE" in df.columns:
        df = df[(df["TENURE"].isna()) | (df["TENURE"] >= 0)].copy()

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
            df[col] = df[col].fillna("Unknown").astype(str)

    if "marital_status" not in df.columns:
        df["marital_status"] = -1
    df["marital_status"] = df["marital_status"].fillna(-1)

    if "HGC" not in df.columns:
        df["HGC"] = np.nan
    if "TENURE" not in df.columns:
        df["TENURE"] = np.nan
    if "HRS_WRK" not in df.columns:
        df["HRS_WRK"] = np.nan

    df["Year_num"] = df["Year"] if "Year" in df.columns else np.nan
    df["post_2018"] = np.where(df["Year_num"] >= 2018, 1, 0)

    sort_cols = []
    if "PUBID_1997" in df.columns:
        sort_cols.append("PUBID_1997")
    if "Year" in df.columns:
        sort_cols.append("Year")
    if "Interview_Date" in df.columns:
        sort_cols.append("Interview_Date")

    if len(sort_cols) > 0:
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

    return df, df_lag

# -----------------------------
# Summary tables
# -----------------------------
@st.cache_data(show_spinner=True)
def build_summaries(df, df_lag):
    year_gender = pd.DataFrame()
    if "Year" in df.columns and "sex_label" in df.columns and "HRLY_WAGE" in df.columns and len(df) > 0:
        year_gender = (
            df.groupby(["Year", "sex_label"], as_index=False)
              .agg(
                  mean_wage=("HRLY_WAGE", "mean"),
                  median_wage=("HRLY_WAGE", "median"),
                  n=("HRLY_WAGE", "size")
              )
        )

    occupation_pivot = pd.DataFrame(columns=["Occupation_Group2", "female_to_male_ratio"])
    if {"Occupation_Group2", "sex_label", "HRLY_WAGE"}.issubset(df.columns):
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

    industry_pivot = pd.DataFrame(columns=["Industry_Group", "female_to_male_ratio"])
    if {"Industry_Group", "sex_label", "HRLY_WAGE"}.issubset(df.columns):
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

    prior_scatter = df_lag.copy()
    if len(prior_scatter) > 0 and {"HRLY_WAGE", "prior_wage"}.issubset(prior_scatter.columns):
        prior_scatter = prior_scatter[
            (prior_scatter["HRLY_WAGE"] <= prior_scatter["HRLY_WAGE"].quantile(0.99)) &
            (prior_scatter["prior_wage"] <= prior_scatter["prior_wage"].quantile(0.99))
        ].copy()
        prior_scatter = maybe_sample_for_scatter(prior_scatter, max_rows=MAX_SCATTER_POINTS)

    return year_gender, occupation_pivot, industry_pivot, prior_scatter

# -----------------------------
# Modeling
# -----------------------------
def fit_ols(formula, data):
    try:
        if data is None or len(data) == 0:
            return None
        model = smf.ols(formula=formula, data=data).fit()
        return model
    except Exception:
        return None

@st.cache_resource(show_spinner=True)
def fit_models(df, df_lag):
    """
    Cloud-safe model fitting:
    - smaller samples for Streamlit Cloud stability
    - same model family and same app functionality
    """
    models = {}

    df_m = make_model_sample(df, MAX_MODEL_ROWS_MAIN, seed=42)
    df_lag_m = make_model_sample(df_lag, MAX_MODEL_ROWS_LAG, seed=42)

    cat_cols = ["race_label", "region_label", "marital_status", "Occupation_Group2", "Industry_Group"]
    for c in cat_cols:
        if c in df_m.columns:
            df_m[c] = df_m[c].astype("category")
        if c in df_lag_m.columns:
            df_lag_m[c] = df_lag_m[c].astype("category")

    q1_needed = [
        "ln_wage", "female", "age", "age_sq", "HGC", "TENURE", "HRS_WRK", "Year_num",
        "race_label", "region_label", "marital_status", "Occupation_Group2", "Industry_Group"
    ]
    q2_needed = q1_needed + ["ln_prior_wage"]
    q3_needed = q2_needed + ["post_2018"]

    df_m_q1 = df_m.dropna(subset=[c for c in q1_needed if c in df_m.columns]).copy()
    df_lag_q2 = df_lag_m.dropna(subset=[c for c in q2_needed if c in df_lag_m.columns]).copy()
    df_lag_q3 = df_lag_m.dropna(subset=[c for c in q3_needed if c in df_lag_m.columns]).copy()

    if len(df_m_q1) < 200:
        models["M_Q1"] = None
    else:
        models["M_Q1"] = fit_ols(
            """
            ln_wage ~ female + age + age_sq + HGC + TENURE + HRS_WRK + Year_num
                    + C(race_label) + C(region_label) + C(marital_status)
                    + C(Occupation_Group2) + C(Industry_Group)
            """,
            df_m_q1
        )

    if len(df_lag_q2) < 200:
        models["M_Q2"] = None
        models["M_Q2_INT"] = None
    else:
        models["M_Q2"] = fit_ols(
            """
            ln_wage ~ ln_prior_wage + female + age + age_sq + HGC + TENURE + HRS_WRK + Year_num
                    + C(race_label) + C(region_label) + C(marital_status)
                    + C(Occupation_Group2) + C(Industry_Group)
            """,
            df_lag_q2
        )

        models["M_Q2_INT"] = fit_ols(
            """
            ln_wage ~ ln_prior_wage * female + age + age_sq + HGC + TENURE + HRS_WRK + Year_num
                    + C(race_label) + C(region_label) + C(marital_status)
                    + C(Occupation_Group2) + C(Industry_Group)
            """,
            df_lag_q2
        )

    if len(df_lag_q3) < 200:
        models["M_Q3"] = None
        models["M_Q3_TRIPLE"] = None
    else:
        models["M_Q3"] = fit_ols(
            """
            ln_wage ~ ln_prior_wage * post_2018 + female + age + age_sq + HGC + TENURE + HRS_WRK + Year_num
                    + C(race_label) + C(region_label) + C(marital_status)
                    + C(Occupation_Group2) + C(Industry_Group)
            """,
            df_lag_q3
        )

        models["M_Q3_TRIPLE"] = fit_ols(
            """
            ln_wage ~ ln_prior_wage * female * post_2018 + age + age_sq + HGC + TENURE + HRS_WRK + Year_num
                    + C(race_label) + C(region_label) + C(marital_status)
                    + C(Occupation_Group2) + C(Industry_Group)
            """,
            df_lag_q3
        )

    models["F_BASE"] = models["M_Q1"]
    models["F_PRIOR"] = models["M_Q2_INT"]
    models["F_POLICY"] = models["M_Q3_TRIPLE"]

    return models

def extract_model_table(models):
    rows = []

    model_meta = {
        "M_Q1": ("Q1", "Gender pay gap full model"),
        "M_Q2": ("Q2", "Prior wage model"),
        "M_Q2_INT": ("Q2", "Prior wage × female model"),
        "M_Q3": ("Q3", "Prior wage × post-2018 model"),
        "M_Q3_TRIPLE": ("Q3", "Triple interaction model"),
    }

    terms_to_keep = [
        "female",
        "ln_prior_wage",
        "ln_prior_wage:female",
        "ln_prior_wage:post_2018",
        "female:post_2018",
        "ln_prior_wage:female:post_2018",
        "Year_num"
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

# -----------------------------
# Filter helpers
# -----------------------------
def filter_main_data(df, year_range, selected_gender, selected_race, selected_region):
    filtered = df.copy()
    if year_range is not None and "Year" in filtered.columns:
        filtered = filtered[(filtered["Year"] >= year_range[0]) & (filtered["Year"] <= year_range[1])]

    if "sex_label" in filtered.columns and selected_gender:
        filtered = filtered[filtered["sex_label"].isin(selected_gender)]
    if "race_label" in filtered.columns and selected_race:
        filtered = filtered[filtered["race_label"].isin(selected_race)]
    if "region_label" in filtered.columns and selected_region:
        filtered = filtered[filtered["region_label"].isin(selected_region)]

    return filtered

def filter_lag_data(df_lag, year_range, selected_gender, selected_race, selected_region):
    filtered = df_lag.copy()
    if year_range is not None and "Year" in filtered.columns:
        filtered = filtered[(filtered["Year"] >= year_range[0]) & (filtered["Year"] <= year_range[1])]

    if "sex_label" in filtered.columns and selected_gender:
        filtered = filtered[filtered["sex_label"].isin(selected_gender)]
    if "race_label" in filtered.columns and selected_race:
        filtered = filtered[filtered["race_label"].isin(selected_race)]
    if "region_label" in filtered.columns and selected_region:
        filtered = filtered[filtered["region_label"].isin(selected_region)]

    return filtered

def build_summary_cards(df_filtered):
    male_mean = np.nan
    female_mean = np.nan
    raw_gap_pct = np.nan

    if len(df_filtered) == 0 or "sex_label" not in df_filtered.columns or "HRLY_WAGE" not in df_filtered.columns:
        return male_mean, female_mean, raw_gap_pct

    male_mean = df_filtered.loc[df_filtered["sex_label"] == "Male", "HRLY_WAGE"].mean()
    female_mean = df_filtered.loc[df_filtered["sex_label"] == "Female", "HRLY_WAGE"].mean()

    if pd.notna(male_mean) and male_mean != 0 and pd.notna(female_mean):
        raw_gap_pct = 100 * (female_mean / male_mean - 1)

    return male_mean, female_mean, raw_gap_pct

# -----------------------------
# Interpretations
# -----------------------------
def get_top_interpretations(results_table):
    messages = []

    try:
        q1_row = results_table[(results_table["model"] == "M_Q1") & (results_table["term"] == "female")]
        if not q1_row.empty:
            coef = q1_row.iloc[0]["coefficient"]
            pct = safe_exp_pct(coef)
            messages.append(
                f"Q1: In the full cloud-safe model, the female coefficient is {coef:.4f}, implying about {pct:.2f}% lower wages for women with similar observed characteristics."
            )
    except Exception:
        pass

    try:
        q2_row = results_table[(results_table["model"] == "M_Q2") & (results_table["term"] == "ln_prior_wage")]
        if not q2_row.empty:
            coef = q2_row.iloc[0]["coefficient"]
            messages.append(
                f"Q2: Current wage remains strongly related to prior wage. The log prior wage coefficient is {coef:.4f}."
            )
    except Exception:
        pass

    try:
        q2_row2 = results_table[(results_table["model"] == "M_Q2_INT") & (results_table["term"] == "ln_prior_wage:female")]
        if not q2_row2.empty:
            coef = q2_row2.iloc[0]["coefficient"]
            p = q2_row2.iloc[0]["p_value"]
            messages.append(
                f"Q2: The prior wage × female interaction is {coef:.4f} with p-value {p:.3f}, so the slope difference by gender is limited in this interactive approximation."
            )
    except Exception:
        pass

    try:
        q3_row = results_table[(results_table["model"] == "M_Q3") & (results_table["term"] == "ln_prior_wage:post_2018")]
        if not q3_row.empty:
            coef = q3_row.iloc[0]["coefficient"]
            p = q3_row.iloc[0]["p_value"]
            messages.append(
                f"Q3: The post-2018 prior wage interaction is {coef:.4f} with p-value {p:.3f}. The prior-current wage relationship does not collapse to zero."
            )
    except Exception:
        pass

    return messages

# -----------------------------
# Simulator helpers
# -----------------------------
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
        "ln_prior_wage": np.log(prior_wage) if pd.notna(prior_wage) and prior_wage > 0 else np.nan,
        "post_2018": post_2018
    }])
    return row

def predict_wage_path(model, horizons, base_inputs):
    rows = []
    if model is None:
        return pd.DataFrame()

    for h in horizons:
        year_value = int(base_inputs["year_value"] + h)
        row = build_profile_row(
            year_value=year_value,
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
        try:
            pred_ln = float(model.predict(row)[0])
            pred_wage = float(np.exp(pred_ln))
            rows.append({
                "horizon_years": h,
                "projected_year": year_value,
                "predicted_wage": pred_wage
            })
        except Exception:
            continue

    return pd.DataFrame(rows)

def same_profile_gender_comparison(model, base_inputs):
    if model is None:
        return pd.DataFrame()

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
        try:
            pred_ln = float(model.predict(row)[0])
            rows.append({
                "sex_label": sex_label,
                "predicted_wage": float(np.exp(pred_ln))
            })
        except Exception:
            continue

    return pd.DataFrame(rows)

def prior_wage_policy_comparison(model_with_prior, model_without_prior, base_inputs):
    if model_with_prior is None or model_without_prior is None:
        return pd.DataFrame()

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
    try:
        pred_with = float(np.exp(model_with_prior.predict(row_with)[0]))
        rows.append({"scenario": "With prior wage information", "predicted_wage": pred_with})
    except Exception:
        pass

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
    try:
        pred_without = float(np.exp(model_without_prior.predict(row_without)[0]))
        rows.append({"scenario": "Without prior wage information", "predicted_wage": pred_without})
    except Exception:
        pass

    return pd.DataFrame(rows)

def pre_post_policy_comparison(model_policy, base_inputs):
    if model_policy is None:
        return pd.DataFrame()

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
        try:
            pred = float(np.exp(model_policy.predict(row)[0]))
            rows.append({
                "policy_period": "Pre-2018 proxy" if yr < 2018 else "Post-2018 proxy",
                "year_used": yr,
                "predicted_wage": pred
            })
        except Exception:
            continue

    return pd.DataFrame(rows)

def compare_model_predictions(models, base_inputs):
    rows = []

    mapping = {
        "Baseline wage model": "F_BASE",
        "Prior wage model": "F_PRIOR",
        "Policy interaction model": "F_POLICY"
    }

    for label, key in mapping.items():
        model = models.get(key)
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
        try:
            pred = float(np.exp(model.predict(row)[0]))
            nobs_val = getattr(model, "nobs", np.nan)
            rows.append({
                "model": label,
                "predicted_wage": pred,
                "r_squared": getattr(model, "rsquared", np.nan),
                "nobs": int(nobs_val) if pd.notna(nobs_val) else np.nan
            })
        except Exception:
            continue

    return pd.DataFrame(rows)

# -----------------------------
# Load everything
# -----------------------------
st.title("Datathon 2026 Dashboard")
st.caption("Gender pay gap, prior salary effects, policy interpretation, and user-facing wage prediction")

try:
    with st.spinner("Loading data and summaries..."):
        df_raw = load_data_from_github_zip(ZIP_URL)
        df, df_lag = preprocess_data(df_raw)
        year_gender, occupation_pivot, industry_pivot, prior_scatter = build_summaries(df, df_lag)

    with st.spinner("Fitting cloud-safe models..."):
        models = fit_models(df, df_lag)
        results_table = extract_model_table(models)

except Exception as e:
    st.error(f"App failed during initialization: {type(e).__name__}: {e}")
    st.exception(e)
    st.stop()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Q1 Gender Pay Gap",
        "Q2 Prior Salary and Ban",
        "Q3 Post-2018 Policy",
        "Q4 Hidden Factors and Limits",
        "AI Wage Simulator",
        "Data Preview"
    ]
)

st.sidebar.header("Global Filters")

if "Year" in df.columns and df["Year"].notna().any():
    year_min = int(df["Year"].min())
    year_max = int(df["Year"].max())
    year_range = st.sidebar.slider("Year range", year_min, year_max, (year_min, year_max))
else:
    year_range = None

gender_options = sorted(df["sex_label"].dropna().unique().tolist()) if "sex_label" in df.columns else []
selected_gender = st.sidebar.multiselect("Gender", gender_options, default=gender_options)

race_options = sorted(df["race_label"].dropna().unique().tolist()) if "race_label" in df.columns else []
selected_race = st.sidebar.multiselect("Race", race_options, default=race_options)

region_options = sorted(df["region_label"].dropna().unique().tolist()) if "region_label" in df.columns else []
selected_region = st.sidebar.multiselect("Region", region_options, default=region_options)

filtered = filter_main_data(df, year_range, selected_gender, selected_race, selected_region)
lag_filtered = filter_lag_data(df_lag, year_range, selected_gender, selected_race, selected_region)

male_mean, female_mean, raw_gap_pct = build_summary_cards(filtered)

# -----------------------------
# KPI row
# -----------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Observations", f"{len(filtered):,}")
c2.metric("Individuals", f"{filtered['PUBID_1997'].nunique():,}" if "PUBID_1997" in filtered.columns else "N/A")
c3.metric("Average hourly wage", f"{filtered['HRLY_WAGE'].mean():.2f}" if len(filtered) and "HRLY_WAGE" in filtered.columns else "N/A")
c4.metric("Male average wage", f"{male_mean:.2f}" if pd.notna(male_mean) else "N/A")
c5.metric("Female average wage", f"{female_mean:.2f}" if pd.notna(female_mean) else "N/A")

st.metric("Raw female vs male wage gap (%)", f"{raw_gap_pct:.2f}%" if pd.notna(raw_gap_pct) else "N/A")

# -----------------------------
# Page: Overview
# -----------------------------
if page == "Overview":
    st.subheader("Executive Summary")

    msgs = get_top_interpretations(results_table)
    if len(msgs) == 0:
        st.warning("Model summaries are not available.")
    else:
        for msg in msgs:
            st.markdown(f"- {msg}")

    st.info(
        "Performance note: Interactive models in this dashboard are cloud-safe approximations trained on sampled data for responsiveness. Descriptive charts use the cleaned full sample."
    )

    if len(filtered) > 0 and "Year" in filtered.columns and "sex_label" in filtered.columns and "HRLY_WAGE" in filtered.columns:
        summary = (
            filtered.groupby(["Year", "sex_label"], as_index=False)["HRLY_WAGE"]
            .mean()
            .rename(columns={"HRLY_WAGE": "mean_wage"})
        )
        if len(summary) > 0:
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
    if len(results_table) > 0:
        display_cols = ["question", "model", "model_desc", "term_label", "coefficient", "std_error", "p_value", "nobs", "r_squared"]
        st.dataframe(results_table[display_cols], use_container_width=True)
    else:
        st.warning("No model results available.")

# -----------------------------
# Page: Q1
# -----------------------------
elif page == "Q1 Gender Pay Gap":
    st.subheader("Question 1")
    st.markdown("**Is there evidence of pay discrimination between male and female employees?**")

    q1_results = results_table[results_table["question"] == "Q1"].copy()

    q1_main = q1_results[(q1_results["model"] == "M_Q1") & (q1_results["term"] == "female")]
    if not q1_main.empty:
        coef = q1_main.iloc[0]["coefficient"]
        pct = safe_exp_pct(coef)
        p_val = q1_main.iloc[0]["p_value"]
        st.success(
            f"Interactive answer: the female coefficient is {coef:.4f}, which implies about {pct:.2f}% lower wages for women with similar observed characteristics. p-value = {p_val:.4g}"
        )

    a, b = st.columns(2)

    with a:
        if len(filtered) > 0 and {"HRLY_WAGE", "sex_label"}.issubset(filtered.columns):
            fig_hist = px.histogram(
                filtered,
                x="HRLY_WAGE",
                color="sex_label",
                barmode="overlay",
                nbins=50,
                opacity=0.60,
                title="Hourly Wage Distribution by Gender"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning("No data available for histogram.")

    with b:
        if len(filtered) > 0 and {"HRLY_WAGE", "sex_label"}.issubset(filtered.columns):
            trimmed = filtered[filtered["HRLY_WAGE"] <= filtered["HRLY_WAGE"].quantile(0.95)].copy()
            if len(trimmed) > 0:
                fig_box = px.box(
                    trimmed,
                    x="sex_label",
                    y="HRLY_WAGE",
                    color="sex_label",
                    title="Hourly Wage by Gender"
                )
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.warning("No trimmed data available for box plot.")
        else:
            st.warning("No data available for box plot.")

    st.subheader("Occupation-Level Context")
    if len(occupation_pivot) > 0 and "female_to_male_ratio" in occupation_pivot.columns:
        occ_plot = occupation_pivot.dropna(subset=["female_to_male_ratio"]).copy()
        occ_plot = occ_plot.sort_values("female_to_male_ratio").head(12)
        if len(occ_plot) > 0:
            fig_occ = px.bar(
                occ_plot,
                x="female_to_male_ratio",
                y="Occupation_Group2",
                orientation="h",
                title="Lowest Female-to-Male Wage Ratios by Occupation Group"
            )
            fig_occ.add_vline(x=1.0, line_dash="dash")
            st.plotly_chart(fig_occ, use_container_width=True)
        else:
            st.warning("No occupation ratio data available.")
    else:
        st.warning("Occupation summary is not available.")

    st.subheader("Q1 Model Output")
    if len(q1_results) > 0:
        st.dataframe(
            q1_results[["model", "model_desc", "term_label", "coefficient", "std_error", "p_value", "nobs", "r_squared"]],
            use_container_width=True
        )

# -----------------------------
# Page: Q2
# -----------------------------
elif page == "Q2 Prior Salary and Ban":
    st.subheader("Question 2")
    st.markdown("**Is there evidence that asking about prior salary negatively impacts female employees? Should a similar ban be implemented nationally?**")

    q2_results = results_table[results_table["question"] == "Q2"].copy()

    q2_main = q2_results[(q2_results["model"] == "M_Q2") & (q2_results["term"] == "ln_prior_wage")]
    q2_inter = q2_results[(q2_results["model"] == "M_Q2_INT") & (q2_results["term"] == "ln_prior_wage:female")]

    if not q2_main.empty:
        coef = q2_main.iloc[0]["coefficient"]
        st.success(
            f"Interactive answer: log prior wage coefficient = {coef:.4f}. Current wages remain strongly related to prior wages."
        )

    if not q2_inter.empty:
        coef = q2_inter.iloc[0]["coefficient"]
        p_val = q2_inter.iloc[0]["p_value"]
        st.warning(
            f"Gender interaction: prior wage × female = {coef:.4f}, p-value = {p_val:.4g}. The slope difference is limited in this approximation."
        )

    st.info(
        "Interpretation: Women can still be disadvantaged when current pay is highly anchored to prior pay, even if the interaction term is not strongly different by gender."
    )

    st.subheader("Prior Wage vs Current Wage")
    plot_lag = lag_filtered.copy()
    if len(plot_lag) > 0 and {"HRLY_WAGE", "prior_wage", "sex_label"}.issubset(plot_lag.columns):
        plot_lag = plot_lag[
            (plot_lag["HRLY_WAGE"] <= plot_lag["HRLY_WAGE"].quantile(0.99)) &
            (plot_lag["prior_wage"] <= plot_lag["prior_wage"].quantile(0.99))
        ].copy()
        plot_lag = maybe_sample_for_scatter(plot_lag, max_rows=MAX_SCATTER_POINTS)

        if len(plot_lag) > 0:
            fig_scatter = px.scatter(
                plot_lag,
                x="prior_wage",
                y="HRLY_WAGE",
                color="sex_label",
                opacity=0.40,
                title="Prior Wage vs Current Wage"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("No lag data available for scatter plot.")
    else:
        st.warning("Required lag variables are not available for scatter plot.")

    if len(lag_filtered) > 1 and {"prior_wage", "HRLY_WAGE"}.issubset(lag_filtered.columns):
        corr_df = lag_filtered[["prior_wage", "HRLY_WAGE"]].dropna()
        if len(corr_df) > 1:
            corr_value = corr_df.corr().iloc[0, 1]
            st.metric("Correlation: prior wage vs current wage", f"{corr_value:.3f}")

    st.subheader("Q2 Model Output")
    if len(q2_results) > 0:
        st.dataframe(
            q2_results[["model", "model_desc", "term_label", "coefficient", "std_error", "p_value", "nobs", "r_squared"]],
            use_container_width=True
        )

# -----------------------------
# Page: Q3
# -----------------------------
elif page == "Q3 Post-2018 Policy":
    st.subheader("Question 3")
    st.markdown("**Would the prior-pay and current-pay relationship weaken after the California law? Would it become zero?**")

    q3_results = results_table[results_table["question"] == "Q3"].copy()

    q3_main = q3_results[(q3_results["model"] == "M_Q3") & (q3_results["term"] == "ln_prior_wage:post_2018")]
    q3_triple = q3_results[(q3_results["model"] == "M_Q3_TRIPLE") & (q3_results["term"] == "ln_prior_wage:female:post_2018")]

    if not q3_main.empty:
        coef = q3_main.iloc[0]["coefficient"]
        p_val = q3_main.iloc[0]["p_value"]
        st.success(f"Interactive answer: log prior wage × post-2018 = {coef:.4f}, p-value = {p_val:.4g}.")

    if not q3_triple.empty:
        coef = q3_triple.iloc[0]["coefficient"]
        p_val = q3_triple.iloc[0]["p_value"]
        st.warning(f"Triple interaction: log prior wage × female × post-2018 = {coef:.4f}, p-value = {p_val:.4g}.")

    st.info(
        "Interpretation: The relationship between prior and current pay does not mechanically fall to zero after policy changes because prior pay also captures experience, job trajectory, and labor-market sorting."
    )

    if len(q3_results) > 0:
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

        st.dataframe(
            q3_results[["model", "model_desc", "term_label", "coefficient", "std_error", "p_value", "nobs", "r_squared"]],
            use_container_width=True
        )

# -----------------------------
# Page: Q4
# -----------------------------
elif page == "Q4 Hidden Factors and Limits":
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

    c, d = st.columns(2)

    with c:
        if len(occupation_pivot) > 0 and "female_to_male_ratio" in occupation_pivot.columns:
            occ_plot = occupation_pivot.dropna(subset=["female_to_male_ratio"]).copy()
            occ_plot = occ_plot.sort_values("female_to_male_ratio").head(15)
            if len(occ_plot) > 0:
                fig_occ = px.bar(
                    occ_plot,
                    x="female_to_male_ratio",
                    y="Occupation_Group2",
                    orientation="h",
                    title="Lowest Female-to-Male Wage Ratios by Occupation Group"
                )
                fig_occ.add_vline(x=1.0, line_dash="dash")
                st.plotly_chart(fig_occ, use_container_width=True)
            else:
                st.warning("No occupation ratio data available.")
        else:
            st.warning("Occupation summary is not available.")

    with d:
        if len(industry_pivot) > 0 and "female_to_male_ratio" in industry_pivot.columns:
            ind_plot = industry_pivot.dropna(subset=["female_to_male_ratio"]).copy()
            ind_plot = ind_plot.sort_values("female_to_male_ratio").head(15)
            if len(ind_plot) > 0:
                fig_ind = px.bar(
                    ind_plot,
                    x="female_to_male_ratio",
                    y="Industry_Group",
                    orientation="h",
                    title="Lowest Female-to-Male Wage Ratios by Industry Group"
                )
                fig_ind.add_vline(x=1.0, line_dash="dash")
                st.plotly_chart(fig_ind, use_container_width=True)
            else:
                st.warning("No industry ratio data available.")
        else:
            st.warning("Industry summary is not available.")

    st.warning(
        "Limitation note: The dataset does not directly observe whether employers explicitly asked for salary history. The dashboard therefore supports policy rationale and association patterns rather than strict causal identification."
    )

# -----------------------------
# Page: Simulator
# -----------------------------
elif page == "AI Wage Simulator":
    st.subheader("AI-Style Wage Projection Simulator")
    st.markdown("Choose a profile, then generate projected wages, same-profile gender comparisons, and policy comparisons.")

    year_min_allowed = int(df["Year"].min()) if "Year" in df.columns and df["Year"].notna().any() else 1997
    year_max_allowed = int(df["Year"].max()) + 10 if "Year" in df.columns and df["Year"].notna().any() else 2031

    default_year = int(df["Year"].max()) if "Year" in df.columns and df["Year"].notna().any() else 2021
    default_age = int(clamp(np.nanmedian(df["age"]) if "age" in df.columns and df["age"].notna().any() else 30, 18, 70))
    default_hgc = float(clamp(np.nanmedian(df["HGC"]) if "HGC" in df.columns and df["HGC"].notna().any() else 16.0, 0.0, 25.0))
    default_tenure = float(clamp(np.nanmedian(df["TENURE"]) if "TENURE" in df.columns and df["TENURE"].notna().any() else 2.0, 0.0, 40.0))
    default_hours = float(clamp(np.nanmedian(df["HRS_WRK"]) if "HRS_WRK" in df.columns and df["HRS_WRK"].notna().any() else 40.0, 1.0, 120.0))
    default_prior_wage = float(clamp(np.nanmedian(df_lag["prior_wage"]) if len(df_lag) > 0 and "prior_wage" in df_lag.columns else 25.0, 0.5, 500.0))

    occ_choices = sorted(df["Occupation_Group2"].dropna().unique().tolist()) if "Occupation_Group2" in df.columns else []
    ind_choices = sorted(df["Industry_Group"].dropna().unique().tolist()) if "Industry_Group" in df.columns else []
    race_choices = sorted(df["race_label"].dropna().unique().tolist()) if "race_label" in df.columns else []
    region_choices = sorted(df["region_label"].dropna().unique().tolist()) if "region_label" in df.columns else []
    marital_choices = sorted(df["marital_status"].dropna().unique().tolist()) if "marital_status" in df.columns else []

    if not occ_choices:
        occ_choices = ["Unknown"]
    if not ind_choices:
        ind_choices = ["Unknown"]
    if not race_choices:
        race_choices = ["Unknown"]
    if not region_choices:
        region_choices = ["Unknown"]
    if not marital_choices:
        marital_choices = [-1]

    with st.form("simulator_form"):
        f1, f2, f3 = st.columns(3)

        with f1:
            sim_year = st.number_input(
                "Base year",
                min_value=year_min_allowed,
                max_value=year_max_allowed,
                value=int(clamp(default_year, year_min_allowed, year_max_allowed)),
                step=1
            )
            sim_sex = st.selectbox("Gender", ["Male", "Female"], index=1)
            sim_race = safe_selectbox("Race", race_choices, default_index=0, key="sim_race")
            sim_region = safe_selectbox("Region", region_choices, default_index=0, key="sim_region")

        with f2:
            sim_marital = safe_selectbox("Marital status code", marital_choices, default_index=0, key="sim_marital")
            sim_age = st.number_input("Age", min_value=18, max_value=70, value=int(clamp(default_age, 18, 70)), step=1)
            sim_hgc = st.number_input("Education (HGC)", min_value=0.0, max_value=25.0, value=float(clamp(default_hgc, 0.0, 25.0)), step=0.5)
            sim_tenure = st.number_input("Tenure", min_value=0.0, max_value=40.0, value=float(clamp(default_tenure, 0.0, 40.0)), step=0.5)

        with f3:
            sim_hours = st.number_input("Weekly hours worked", min_value=1.0, max_value=120.0, value=float(clamp(default_hours, 1.0, 120.0)), step=1.0)
            sim_prior_wage = st.number_input("Prior wage", min_value=0.5, max_value=500.0, value=float(clamp(default_prior_wage, 0.5, 500.0)), step=0.5)
            sim_occ = safe_selectbox("Occupation group", occ_choices, default_index=0, key="sim_occ")
            sim_ind = safe_selectbox("Industry group", ind_choices, default_index=0, key="sim_ind")

        forecast_model_choice = st.selectbox(
            "Projection model",
            ["Baseline wage model", "Prior wage model", "Policy interaction model"],
            index=1
        )

        horizon_set = st.multiselect("Projection horizons", [0, 1, 3, 5], default=[0, 1, 3, 5])

        submitted = st.form_submit_button("Generate prediction")

    if submitted:
        base_inputs = {
            "year_value": int(sim_year),
            "sex_label": sim_sex,
            "race_label": sim_race if sim_race is not None else "Unknown",
            "region_label": sim_region if sim_region is not None else "Unknown",
            "marital_status": sim_marital if sim_marital is not None else -1,
            "age": float(sim_age),
            "hgc": float(sim_hgc),
            "tenure": float(sim_tenure),
            "hrs_wrk": float(sim_hours),
            "occupation_group": sim_occ if sim_occ is not None else "Unknown",
            "industry_group": sim_ind if sim_ind is not None else "Unknown",
            "prior_wage": float(sim_prior_wage)
        }

        model_map = {
            "Baseline wage model": models.get("F_BASE"),
            "Prior wage model": models.get("F_PRIOR"),
            "Policy interaction model": models.get("F_POLICY")
        }
        selected_model = model_map.get(forecast_model_choice)

        pred_path = predict_wage_path(selected_model, sorted(horizon_set), base_inputs)

        if len(pred_path) == 0:
            st.error("Prediction could not be generated.")
        else:
            p1, p2, p3, p4 = st.columns(4)
            if 0 in pred_path["horizon_years"].values:
                p1.metric("Predicted wage now", f"{pred_path.loc[pred_path['horizon_years'] == 0, 'predicted_wage'].iloc[0]:.2f}")
            if 1 in pred_path["horizon_years"].values:
                p2.metric("Predicted wage in 1 year", f"{pred_path.loc[pred_path['horizon_years'] == 1, 'predicted_wage'].iloc[0]:.2f}")
            if 3 in pred_path["horizon_years"].values:
                p3.metric("Predicted wage in 3 years", f"{pred_path.loc[pred_path['horizon_years'] == 3, 'predicted_wage'].iloc[0]:.2f}")
            if 5 in pred_path["horizon_years"].values:
                p4.metric("Predicted wage in 5 years", f"{pred_path.loc[pred_path['horizon_years'] == 5, 'predicted_wage'].iloc[0]:.2f}")

            fig_pred = px.line(
                pred_path,
                x="projected_year",
                y="predicted_wage",
                markers=True,
                title="Projected Wage Path"
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            st.subheader("Same Profile, Different Gender")
            gender_compare = same_profile_gender_comparison(models.get("F_BASE"), base_inputs)
            if len(gender_compare) > 0:
                fig_gender = px.bar(
                    gender_compare,
                    x="sex_label",
                    y="predicted_wage",
                    color="sex_label",
                    title="Predicted Wage Under the Same Observed Profile"
                )
                st.plotly_chart(fig_gender, use_container_width=True)

            st.subheader("With vs Without Prior Wage Information")
            prior_compare = prior_wage_policy_comparison(models.get("F_PRIOR"), models.get("F_BASE"), base_inputs)
            if len(prior_compare) > 0:
                fig_prior = px.bar(
                    prior_compare,
                    x="scenario",
                    y="predicted_wage",
                    color="scenario",
                    title="Predicted Wage Under Alternative Prior-Wage Information Regimes"
                )
                st.plotly_chart(fig_prior, use_container_width=True)

            st.subheader("Pre-2018 vs Post-2018 Policy Proxy")
            policy_compare = pre_post_policy_comparison(models.get("F_POLICY"), base_inputs)
            if len(policy_compare) > 0:
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
            if len(compare_df) > 0:
                st.dataframe(compare_df, use_container_width=True)

            st.info(
                "How to read the simulator: baseline compares same-profile male and female predictions. Prior wage comparison shows how salary-history reliance can preserve wage differences. Policy comparison shows that the prior-pay relationship does not automatically fall to zero."
            )

            st.warning(
                "Uncertainty note: These are model-based scenario forecasts, not guaranteed future outcomes. They rely on observed variables only and do not include unobserved productivity, caregiving burden, negotiation dynamics, employer pay-setting differences, or future labor-market shocks."
            )

# -----------------------------
# Page: Data preview
# -----------------------------
elif page == "Data Preview":
    st.subheader("Raw Data Preview")
    st.dataframe(df_raw.head(MAX_PREVIEW_ROWS), use_container_width=True)

    st.subheader("Cleaned Main Sample Preview")
    st.dataframe(filtered.head(MAX_PREVIEW_ROWS), use_container_width=True)

    st.subheader("Lag Sample Preview")
    st.dataframe(lag_filtered.head(MAX_PREVIEW_ROWS), use_container_width=True)

    st.subheader("Column Names")
    st.write(df_raw.columns.tolist())
