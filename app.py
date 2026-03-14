# =========================================================
# Streamlit App: Datathon 2026 Dashboard Team #18
# Final production-ready app.py for Streamlit Cloud
# Purpose:
# - Load graduate-full.csv.zip directly from GitHub
# - Surface the correct answers to Q1-Q4 immediately
# - Provide cloud-safe econometric models and visualizations
# - Keep performance stable on streamlit.app
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
    page_title="Datathon 2026 Dashboard Team #18",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Config
# -----------------------------
ZIP_URL = "https://raw.githubusercontent.com/Sarkis55/Datathon2026/main/graduate-full.csv.zip"

MAX_MODEL_ROWS_MAIN = 4500
MAX_MODEL_ROWS_LAG = 4000
MAX_SCATTER_POINTS = 3000
MAX_PREVIEW_ROWS = 50
REQUEST_TIMEOUTS = [60, 120, 180]

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


def maybe_sample(df, max_rows, seed=42):
    if df is None or len(df) == 0:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    if len(df) <= max_rows:
        return df.copy()
    return df.sample(max_rows, random_state=seed).copy()


def fmt_num(x, digits=3):
    if pd.isna(x):
        return "N/A"
    return f"{x:.{digits}f}"


def fmt_pct(x, digits=2):
    if pd.isna(x):
        return "N/A"
    return f"{x:.{digits}f}%"


def safe_multiselect_default(options):
    return options if len(options) > 0 else []


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
    for timeout_sec in REQUEST_TIMEOUTS:
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

    # Exact duplicate handling
    exact_duplicate_count = int(df.duplicated().sum())
    if exact_duplicate_count > 0:
        df = df.drop_duplicates().copy()

    # Key-based duplicate inspection count
    key_cols = [c for c in ["PUBID_1997", "Year", "UID", "StartDate", "StopDate"] if c in df.columns]
    if len(key_cols) > 0:
        key_duplicate_count = int(df.duplicated(subset=key_cols).sum())
    else:
        key_duplicate_count = 0

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

    if "index_col" in df.columns:
        df = df.drop(columns=["index_col"])

    if "HRLY_WAGE" not in df.columns:
        raise KeyError("Required column 'HRLY_WAGE' not found in dataset.")

    # Core cleaning
    if "Employed" in df.columns:
        df = df[df["Employed"] == 1].copy()

    df = df[df["HRLY_WAGE"].notna()].copy()
    df = df[df["HRLY_WAGE"] > 0].copy()

    if "HRS_WRK" in df.columns:
        df = df[(df["HRS_WRK"].isna()) | ((df["HRS_WRK"] > 0) & (df["HRS_WRK"] <= 120))].copy()

    if "TENURE" in df.columns:
        df = df[(df["TENURE"].isna()) | (df["TENURE"] >= 0)].copy()

    # Wage trimming
    wage_low = df["HRLY_WAGE"].quantile(0.01)
    wage_high = df["HRLY_WAGE"].quantile(0.99)
    df = df[(df["HRLY_WAGE"] >= wage_low) & (df["HRLY_WAGE"] <= wage_high)].copy()

    # Compensation cleaning
    if "HRLY_COMP" in df.columns:
        df.loc[df["HRLY_COMP"] <= 0, "HRLY_COMP"] = np.nan

    # Gender
    if "SAMPLE_SEX_1997" in df.columns:
        df["female"] = np.where(df["SAMPLE_SEX_1997"] == 2, 1, 0)
        sex_map = {1: "Male", 2: "Female"}
        df["sex_label"] = df["SAMPLE_SEX_1997"].map(sex_map).fillna("Unknown")
    else:
        df["female"] = np.nan
        df["sex_label"] = "Unknown"

    # Wage transforms
    df["ln_wage"] = np.log(df["HRLY_WAGE"])

    if "HRLY_COMP" in df.columns:
        df["ln_comp"] = np.where(df["HRLY_COMP"] > 0, np.log(df["HRLY_COMP"]), np.nan)
        df["wage_to_comp_ratio"] = np.where(df["HRLY_COMP"] > 0, df["HRLY_WAGE"] / df["HRLY_COMP"], np.nan)
    else:
        df["ln_comp"] = np.nan
        df["wage_to_comp_ratio"] = np.nan

    # Age
    if "DOB" in df.columns and "Interview_Date" in df.columns:
        df["age"] = (df["Interview_Date"] - df["DOB"]).dt.days / 365.25
        df["age_sq"] = df["age"] ** 2
    else:
        df["age"] = np.nan
        df["age_sq"] = np.nan

    # Labels
    race_map = {
        1: "Hispanic",
        2: "Black",
        3: "Non-Black/Non-Hispanic",
        4: "Mixed/Other"
    }
    region_map = {
        1: "Northeast",
        2: "North Central",
        3: "South",
        4: "West"
    }
    marital_map = {
        0: "Not married",
        1: "Married"
    }

    if "SAMPLE_RACE_1997" in df.columns:
        df["race_label"] = df["SAMPLE_RACE_1997"].map(race_map).fillna("Unknown")
    else:
        df["race_label"] = "Unknown"

    if "Region" in df.columns:
        df["region_label"] = df["Region"].map(region_map).fillna("Unknown")
    else:
        df["region_label"] = "Unknown"

    if "marital_status" not in df.columns:
        df["marital_status"] = -1
    df["marital_status"] = pd.to_numeric(df["marital_status"], errors="coerce").fillna(-1)

    df["marital_label"] = df["marital_status"].map(marital_map).fillna("Other/Unknown")

    # Text groups
    for col in ["Occupation_Group2", "Industry_Group", "Occupation", "Industry"]:
        if col not in df.columns:
            df[col] = "Unknown"
        else:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()
            df[col] = df[col].replace("", "Unknown")

    # Missing numeric placeholders
    if "HGC" not in df.columns:
        df["HGC"] = np.nan
    if "TENURE" not in df.columns:
        df["TENURE"] = np.nan
    if "HRS_WRK" not in df.columns:
        df["HRS_WRK"] = np.nan

    # Year-based features
    if "Year" in df.columns:
        df["Year_num"] = pd.to_numeric(df["Year"], errors="coerce")
    else:
        df["Year_num"] = np.nan

    df["post_2018"] = np.where(df["Year_num"] >= 2018, 1, 0)

    # Sort for lag construction
    sort_cols = []
    if "PUBID_1997" in df.columns:
        sort_cols.append("PUBID_1997")
    if "Year" in df.columns:
        sort_cols.append("Year")
    if "Interview_Date" in df.columns:
        sort_cols.append("Interview_Date")

    if len(sort_cols) > 0:
        df = df.sort_values(sort_cols).copy()

    # Prior wage
    if "PUBID_1997" in df.columns:
        df["prior_wage"] = df.groupby("PUBID_1997")["HRLY_WAGE"].shift(1)
        df["ln_prior_wage"] = np.where(df["prior_wage"] > 0, np.log(df["prior_wage"]), np.nan)
        if "Year" in df.columns:
            df["prior_year"] = df.groupby("PUBID_1997")["Year"].shift(1)
            df["year_gap"] = df["Year"] - df["prior_year"]
        else:
            df["prior_year"] = np.nan
            df["year_gap"] = np.nan
    else:
        df["prior_wage"] = np.nan
        df["ln_prior_wage"] = np.nan
        df["prior_year"] = np.nan
        df["year_gap"] = np.nan

    df_lag = df[df["prior_wage"].notna()].copy()
    df_lag = df_lag[df_lag["prior_wage"] > 0].copy()
    df_lag = df_lag[(df_lag["year_gap"].isna()) | (df_lag["year_gap"] <= 3)].copy()

    meta = {
        "raw_rows": int(len(df_raw)),
        "clean_rows": int(len(df)),
        "lag_rows": int(len(df_lag)),
        "exact_duplicate_count": int(exact_duplicate_count),
        "key_duplicate_count": int(key_duplicate_count),
        "individual_count": int(df["PUBID_1997"].nunique()) if "PUBID_1997" in df.columns else np.nan,
        "year_min": int(df["Year"].min()) if "Year" in df.columns and df["Year"].notna().any() else None,
        "year_max": int(df["Year"].max()) if "Year" in df.columns and df["Year"].notna().any() else None
    }

    return df, df_lag, meta


# -----------------------------
# Summary tables
# -----------------------------
@st.cache_data(show_spinner=True)
def build_summaries(df, df_lag):
    # Year by gender
    year_gender = pd.DataFrame()
    if "Year" in df.columns and "sex_label" in df.columns and "HRLY_WAGE" in df.columns and len(df) > 0:
        year_gender = (
            df.groupby(["Year", "sex_label"], as_index=False)
              .agg(
                  mean_wage=("HRLY_WAGE", "mean"),
                  median_wage=("HRLY_WAGE", "median"),
                  mean_ln_wage=("ln_wage", "mean"),
                  n=("HRLY_WAGE", "size")
              )
        )

    # Occupation
    occupation_pivot = pd.DataFrame(columns=["Occupation_Group2", "female_to_male_ratio"])
    if {"Occupation_Group2", "sex_label", "HRLY_WAGE"}.issubset(df.columns):
        occupation_gap = (
            df.groupby(["Occupation_Group2", "sex_label"], as_index=False)
              .agg(mean_wage=("HRLY_WAGE", "mean"), n=("HRLY_WAGE", "size"))
        )
        occupation_pivot = occupation_gap.pivot(index="Occupation_Group2", columns="sex_label", values="mean_wage")
        if "Female" in occupation_pivot.columns and "Male" in occupation_pivot.columns:
            occupation_pivot["female_minus_male"] = occupation_pivot["Female"] - occupation_pivot["Male"]
            occupation_pivot["female_to_male_ratio"] = occupation_pivot["Female"] / occupation_pivot["Male"]
        else:
            occupation_pivot["female_minus_male"] = np.nan
            occupation_pivot["female_to_male_ratio"] = np.nan
        occupation_pivot = occupation_pivot.reset_index()

    # Industry
    industry_pivot = pd.DataFrame(columns=["Industry_Group", "female_to_male_ratio"])
    if {"Industry_Group", "sex_label", "HRLY_WAGE"}.issubset(df.columns):
        industry_gap = (
            df.groupby(["Industry_Group", "sex_label"], as_index=False)
              .agg(mean_wage=("HRLY_WAGE", "mean"), n=("HRLY_WAGE", "size"))
        )
        industry_pivot = industry_gap.pivot(index="Industry_Group", columns="sex_label", values="mean_wage")
        if "Female" in industry_pivot.columns and "Male" in industry_pivot.columns:
            industry_pivot["female_minus_male"] = industry_pivot["Female"] - industry_pivot["Male"]
            industry_pivot["female_to_male_ratio"] = industry_pivot["Female"] / industry_pivot["Male"]
        else:
            industry_pivot["female_minus_male"] = np.nan
            industry_pivot["female_to_male_ratio"] = np.nan
        industry_pivot = industry_pivot.reset_index()

    # Prior scatter sample
    prior_scatter = df_lag.copy()
    if len(prior_scatter) > 0 and {"HRLY_WAGE", "prior_wage"}.issubset(prior_scatter.columns):
        prior_scatter = prior_scatter[
            (prior_scatter["HRLY_WAGE"] <= prior_scatter["HRLY_WAGE"].quantile(0.99)) &
            (prior_scatter["prior_wage"] <= prior_scatter["prior_wage"].quantile(0.99))
        ].copy()
        prior_scatter = maybe_sample(prior_scatter, MAX_SCATTER_POINTS, seed=42)

    return year_gender, occupation_pivot, industry_pivot, prior_scatter


# -----------------------------
# Modeling
# -----------------------------
def fit_ols(formula, data):
    try:
        if data is None or len(data) == 0:
            return None
        return smf.ols(formula=formula, data=data).fit(cov_type="HC3")
    except Exception:
        return None


@st.cache_resource(show_spinner=True)
def fit_models(df, df_lag):
    models = {}

    df_m = maybe_sample(df, MAX_MODEL_ROWS_MAIN, seed=42)
    df_lag_m = maybe_sample(df_lag, MAX_MODEL_ROWS_LAG, seed=42)

    for c in ["race_label", "region_label", "marital_label", "Occupation_Group2", "Industry_Group"]:
        if c in df_m.columns:
            df_m[c] = df_m[c].astype("category")
        if c in df_lag_m.columns:
            df_lag_m[c] = df_lag_m[c].astype("category")

    q1_needed = [
        "ln_wage", "female", "age", "age_sq", "HGC", "TENURE", "HRS_WRK",
        "Year_num", "race_label", "region_label", "marital_label",
        "Occupation_Group2", "Industry_Group"
    ]
    q2_needed = q1_needed + ["ln_prior_wage"]
    q3_needed = q2_needed + ["post_2018"]

    df_m_q1 = df_m.dropna(subset=[c for c in q1_needed if c in df_m.columns]).copy()
    df_lag_q2 = df_lag_m.dropna(subset=[c for c in q2_needed if c in df_lag_m.columns]).copy()
    df_lag_q3 = df_lag_m.dropna(subset=[c for c in q3_needed if c in df_lag_m.columns]).copy()

    models["M_Q1"] = None
    models["M_Q2"] = None
    models["M_Q2_INT"] = None
    models["M_Q3"] = None
    models["M_Q3_TRIPLE"] = None

    if len(df_m_q1) >= 200:
        models["M_Q1"] = fit_ols(
            """
            ln_wage ~ female + age + age_sq + HGC + TENURE + HRS_WRK + Year_num
                    + C(race_label) + C(region_label) + C(marital_label)
                    + C(Occupation_Group2) + C(Industry_Group)
            """,
            df_m_q1
        )

    if len(df_lag_q2) >= 200:
        models["M_Q2"] = fit_ols(
            """
            ln_wage ~ ln_prior_wage + female + age + age_sq + HGC + TENURE + HRS_WRK + Year_num
                    + C(race_label) + C(region_label) + C(marital_label)
                    + C(Occupation_Group2) + C(Industry_Group)
            """,
            df_lag_q2
        )

        models["M_Q2_INT"] = fit_ols(
            """
            ln_wage ~ ln_prior_wage * female + age + age_sq + HGC + TENURE + HRS_WRK + Year_num
                    + C(race_label) + C(region_label) + C(marital_label)
                    + C(Occupation_Group2) + C(Industry_Group)
            """,
            df_lag_q2
        )

    if len(df_lag_q3) >= 200:
        models["M_Q3"] = fit_ols(
            """
            ln_wage ~ ln_prior_wage * post_2018 + female + age + age_sq + HGC + TENURE + HRS_WRK + Year_num
                    + C(race_label) + C(region_label) + C(marital_label)
                    + C(Occupation_Group2) + C(Industry_Group)
            """,
            df_lag_q3
        )

        models["M_Q3_TRIPLE"] = fit_ols(
            """
            ln_wage ~ ln_prior_wage * female * post_2018 + age + age_sq + HGC + TENURE + HRS_WRK + Year_num
                    + C(race_label) + C(region_label) + C(marital_label)
                    + C(Occupation_Group2) + C(Industry_Group)
            """,
            df_lag_q3
        )

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

    for model_name, (q_label, model_desc) in model_meta.items():
        model = models.get(model_name)
        if model is None:
            continue

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
# Filters
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
# Interpretation helpers
# -----------------------------
def get_key_result(results_table, model_name, term_name):
    if results_table is None or len(results_table) == 0:
        return None
    sub = results_table[(results_table["model"] == model_name) & (results_table["term"] == term_name)]
    if len(sub) == 0:
        return None
    return sub.iloc[0].to_dict()


def get_top_interpretations(results_table):
    messages = []

    q1 = get_key_result(results_table, "M_Q1", "female")
    if q1 is not None:
        pct = safe_exp_pct(q1["coefficient"])
        messages.append(
            f"Q1. The female coefficient is {q1['coefficient']:.4f}, implying about {pct:.2f}% lower wages for women after observed controls."
        )

    q2 = get_key_result(results_table, "M_Q2", "ln_prior_wage")
    if q2 is not None:
        messages.append(
            f"Q2. Current pay remains strongly related to prior pay. The log prior wage coefficient is {q2['coefficient']:.4f}."
        )

    q2_int = get_key_result(results_table, "M_Q2_INT", "ln_prior_wage:female")
    if q2_int is not None:
        messages.append(
            f"Q2. The prior wage × female interaction is {q2_int['coefficient']:.4f} with p-value {q2_int['p_value']:.3f}. The slope difference by gender is limited."
        )

    q3 = get_key_result(results_table, "M_Q3", "ln_prior_wage:post_2018")
    if q3 is not None:
        messages.append(
            f"Q3. The post-2018 interaction is {q3['coefficient']:.4f} with p-value {q3['p_value']:.3f}. The prior-pay relationship does not fall to zero."
        )

    return messages


# -----------------------------
# Simulator helpers
# -----------------------------
def build_profile_row(
    year_value,
    sex_label,
    race_label,
    region_label,
    marital_label,
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
        "Year_num": year_value,
        "female": female,
        "race_label": race_label,
        "region_label": region_label,
        "marital_label": marital_label,
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


def predict_profile(model, row):
    if model is None:
        return np.nan
    try:
        pred_ln = float(model.predict(row)[0])
        return float(np.exp(pred_ln))
    except Exception:
        return np.nan


# -----------------------------
# Load everything
# -----------------------------
st.title("Datathon 2026 Dashboard")
st.caption("Fast policy-oriented answers to Q1-Q4 with cloud-safe models and supporting visuals")

try:
    with st.spinner("Loading data..."):
        df_raw = load_data_from_github_zip(ZIP_URL)
        df, df_lag, meta = preprocess_data(df_raw)
        year_gender, occupation_pivot, industry_pivot, prior_scatter = build_summaries(df, df_lag)

    with st.spinner("Fitting cloud-safe econometric models..."):
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
        "Policy Simulator",
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
selected_gender = st.sidebar.multiselect("Gender", gender_options, default=safe_multiselect_default(gender_options))

race_options = sorted(df["race_label"].dropna().unique().tolist()) if "race_label" in df.columns else []
selected_race = st.sidebar.multiselect("Race", race_options, default=safe_multiselect_default(race_options))

region_options = sorted(df["region_label"].dropna().unique().tolist()) if "region_label" in df.columns else []
selected_region = st.sidebar.multiselect("Region", region_options, default=safe_multiselect_default(region_options))

filtered = filter_main_data(df, year_range, selected_gender, selected_race, selected_region)
lag_filtered = filter_lag_data(df_lag, year_range, selected_gender, selected_race, selected_region)

male_mean, female_mean, raw_gap_pct = build_summary_cards(filtered)

# -----------------------------
# KPI row
# -----------------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Clean observations", f"{len(filtered):,}")
k2.metric("Individuals", f"{filtered['PUBID_1997'].nunique():,}" if "PUBID_1997" in filtered.columns else "N/A")
k3.metric("Average hourly wage", f"{filtered['HRLY_WAGE'].mean():.2f}" if len(filtered) and "HRLY_WAGE" in filtered.columns else "N/A")
k4.metric("Male average wage", f"{male_mean:.2f}" if pd.notna(male_mean) else "N/A")
k5.metric("Female average wage", f"{female_mean:.2f}" if pd.notna(female_mean) else "N/A")

st.metric("Raw female vs male wage gap", fmt_pct(raw_gap_pct))

# -----------------------------
# Page: Overview
# -----------------------------
if page == "Overview":
    st.subheader("Judge-ready answer panel")

    q1 = get_key_result(results_table, "M_Q1", "female")
    q2 = get_key_result(results_table, "M_Q2", "ln_prior_wage")
    q2_int = get_key_result(results_table, "M_Q2_INT", "ln_prior_wage:female")
    q3 = get_key_result(results_table, "M_Q3", "ln_prior_wage:post_2018")
    q3_triple = get_key_result(results_table, "M_Q3_TRIPLE", "ln_prior_wage:female:post_2018")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if q1 is not None:
            st.metric("Q1 adjusted female effect", fmt_pct(safe_exp_pct(q1["coefficient"])))
            st.caption(f"coef={q1['coefficient']:.4f}, p={q1['p_value']:.4g}")
        else:
            st.metric("Q1 adjusted female effect", "N/A")

    with c2:
        if q2 is not None:
            st.metric("Q2 prior wage coefficient", fmt_num(q2["coefficient"], 4))
            st.caption(f"p={q2['p_value']:.4g}")
        else:
            st.metric("Q2 prior wage coefficient", "N/A")

    with c3:
        if q2_int is not None:
            st.metric("Q2 prior wage × female", fmt_num(q2_int["coefficient"], 4))
            st.caption(f"p={q2_int['p_value']:.4g}")
        else:
            st.metric("Q2 prior wage × female", "N/A")

    with c4:
        if q3 is not None:
            st.metric("Q3 prior wage × post-2018", fmt_num(q3["coefficient"], 4))
            st.caption(f"p={q3['p_value']:.4g}")
        else:
            st.metric("Q3 prior wage × post-2018", "N/A")

    st.markdown("### Short answers")
    messages = get_top_interpretations(results_table)
    if len(messages) > 0:
        for msg in messages:
            st.markdown(f"- {msg}")
    else:
        st.warning("Key model outputs are not available.")

    st.markdown("- Q4. Hidden factors likely include career interruptions, caregiving burden, negotiation dynamics, employer pay-setting policy, local labor market conditions, and unobserved productivity.")

    st.info(
        "This dashboard prioritizes fast and interpretable policy answers. Descriptive charts use the cleaned full sample. Regressions use cloud-safe samples for stable deployment on streamlit.app."
    )

    if len(filtered) > 0 and {"Year", "sex_label", "HRLY_WAGE"}.issubset(filtered.columns):
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

    if len(results_table) > 0:
        st.subheader("Key model output table")
        display_cols = ["question", "model", "model_desc", "term_label", "coefficient", "std_error", "p_value", "nobs", "r_squared"]
        st.dataframe(results_table[display_cols], use_container_width=True)


# -----------------------------
# Page: Q1
# -----------------------------
elif page == "Q1 Gender Pay Gap":
    st.subheader("Question 1")
    st.markdown("**Is there evidence of pay discrimination between male and female employees?**")

    q1_results = results_table[results_table["question"] == "Q1"].copy()
    q1_main = get_key_result(results_table, "M_Q1", "female")

    if q1_main is not None:
        coef = q1_main["coefficient"]
        pct = safe_exp_pct(coef)
        p_val = q1_main["p_value"]
        st.success(
            f"Answer: Yes. The adjusted female coefficient is {coef:.4f}, implying about {pct:.2f}% lower wages for women after controls. p-value = {p_val:.4g}."
        )
    else:
        st.warning("Q1 model is not available.")

    left, right = st.columns(2)

    with left:
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

    with right:
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

    if len(q1_results) > 0:
        st.subheader("Q1 model output")
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
    q2_main = get_key_result(results_table, "M_Q2", "ln_prior_wage")
    q2_inter = get_key_result(results_table, "M_Q2_INT", "ln_prior_wage:female")

    if q2_main is not None:
        st.success(
            f"Answer: Prior pay is strongly associated with current pay. The log prior wage coefficient is {q2_main['coefficient']:.4f} with p-value {q2_main['p_value']:.4g}."
        )

    if q2_inter is not None:
        st.warning(
            f"Gender interaction: log prior wage × female = {q2_inter['coefficient']:.4f}, p-value = {q2_inter['p_value']:.4g}."
        )

    st.info(
        "Interpretation: Even if the slope difference by gender is limited, strong dependence on prior pay can still transmit earlier wage inequality into current pay. This supports the logic of a salary-history ban."
    )

    if len(lag_filtered) > 0 and {"prior_wage", "HRLY_WAGE", "sex_label"}.issubset(lag_filtered.columns):
        plot_lag = lag_filtered.copy()
        plot_lag = plot_lag[
            (plot_lag["HRLY_WAGE"] <= plot_lag["HRLY_WAGE"].quantile(0.99)) &
            (plot_lag["prior_wage"] <= plot_lag["prior_wage"].quantile(0.99))
        ].copy()
        plot_lag = maybe_sample(plot_lag, MAX_SCATTER_POINTS, seed=42)

        fig_scatter = px.scatter(
            plot_lag,
            x="prior_wage",
            y="HRLY_WAGE",
            color="sex_label",
            opacity=0.40,
            title="Prior Wage vs Current Wage"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        corr_df = plot_lag[["prior_wage", "HRLY_WAGE"]].dropna()
        if len(corr_df) > 1:
            st.metric("Correlation: prior wage vs current wage", f"{corr_df.corr().iloc[0, 1]:.3f}")

    if len(q2_results) > 0:
        st.subheader("Q2 model output")
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
    q3_main = get_key_result(results_table, "M_Q3", "ln_prior_wage:post_2018")
    q3_triple = get_key_result(results_table, "M_Q3_TRIPLE", "ln_prior_wage:female:post_2018")

    if q3_main is not None:
        st.success(
            f"Answer: The post-2018 interaction is {q3_main['coefficient']:.4f} with p-value {q3_main['p_value']:.4g}. The prior-pay relationship does not fall to zero."
        )

    if q3_triple is not None:
        st.warning(
            f"Female-specific post-2018 triple interaction = {q3_triple['coefficient']:.4f}, p-value = {q3_triple['p_value']:.4g}."
        )

    st.info(
        "Interpretation: A salary-history ban may reduce direct reliance on past wages, but prior pay still reflects education, experience, occupation, and job trajectory. It should not be expected to become mechanically irrelevant."
    )

    if len(q3_results) > 0:
        q3_plot = q3_results.copy()
        q3_plot["ci_low"] = q3_plot["coefficient"] - 1.96 * q3_plot["std_error"]
        q3_plot["ci_high"] = q3_plot["coefficient"] + 1.96 * q3_plot["std_error"]
        q3_plot["label"] = q3_plot["model"] + " | " + q3_plot["term_label"]

        fig_q3_coef = go.Figure()
        fig_q3_coef.add_trace(
            go.Scatter(
                x=q3_plot["coefficient"],
                y=q3_plot["label"],
                mode="markers",
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=q3_plot["ci_high"] - q3_plot["coefficient"],
                    arrayminus=q3_plot["coefficient"] - q3_plot["ci_low"]
                )
            )
        )
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
    **Likely latent factors**
    - Career interruptions and caregiving burden
    - Negotiation behavior and bargaining power
    - Firm compensation policy and employer fixed effects
    - Local labor market conditions
    - Promotion opportunity and job quality
    - Union coverage
    - Measurement error in wages or hours
    - Selection into employment
    - Unobserved productivity and employer match quality
    """)

    left, right = st.columns(2)

    with left:
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

    with right:
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

    st.warning(
        "Limitation: The dataset does not directly observe whether an employer explicitly asked for salary history. The analysis therefore identifies patterns consistent with policy concerns rather than strict causal proof."
    )


# -----------------------------
# Page: Policy Simulator
# -----------------------------
elif page == "Policy Simulator":
    st.subheader("Policy and Profile Simulator")
    st.markdown("Use the fitted econometric models to compare same-profile outcomes under alternative assumptions.")

    year_min_allowed = int(df["Year"].min()) if "Year" in df.columns and df["Year"].notna().any() else 1997
    year_max_allowed = int(df["Year"].max()) + 5 if "Year" in df.columns and df["Year"].notna().any() else 2026

    default_year = int(df["Year"].max()) if "Year" in df.columns and df["Year"].notna().any() else 2021
    default_age = int(clamp(np.nanmedian(df["age"]) if "age" in df.columns and df["age"].notna().any() else 30, 18, 70))
    default_hgc = float(clamp(np.nanmedian(df["HGC"]) if "HGC" in df.columns and df["HGC"].notna().any() else 16, 0, 25))
    default_tenure = float(clamp(np.nanmedian(df["TENURE"]) if "TENURE" in df.columns and df["TENURE"].notna().any() else 2, 0, 40))
    default_hours = float(clamp(np.nanmedian(df["HRS_WRK"]) if "HRS_WRK" in df.columns and df["HRS_WRK"].notna().any() else 40, 1, 120))
    default_prior = float(clamp(np.nanmedian(df_lag["prior_wage"]) if "prior_wage" in df_lag.columns and df_lag["prior_wage"].notna().any() else 15, 0.5, 500))

    occ_choices = sorted(df["Occupation_Group2"].dropna().unique().tolist()) if "Occupation_Group2" in df.columns else ["Unknown"]
    ind_choices = sorted(df["Industry_Group"].dropna().unique().tolist()) if "Industry_Group" in df.columns else ["Unknown"]
    race_choices = sorted(df["race_label"].dropna().unique().tolist()) if "race_label" in df.columns else ["Unknown"]
    region_choices = sorted(df["region_label"].dropna().unique().tolist()) if "region_label" in df.columns else ["Unknown"]
    marital_choices = sorted(df["marital_label"].dropna().unique().tolist()) if "marital_label" in df.columns else ["Other/Unknown"]

    with st.form("policy_simulator_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            sim_year = st.number_input("Base year", min_value=year_min_allowed, max_value=year_max_allowed, value=default_year, step=1)
            sim_sex = st.selectbox("Gender", ["Male", "Female"], index=1)
            sim_race = safe_selectbox("Race", race_choices, 0, key="sim_race")
            sim_region = safe_selectbox("Region", region_choices, 0, key="sim_region")

        with c2:
            sim_marital = safe_selectbox("Marital status", marital_choices, 0, key="sim_marital")
            sim_age = st.number_input("Age", min_value=18, max_value=70, value=default_age, step=1)
            sim_hgc = st.number_input("Education (HGC)", min_value=0.0, max_value=25.0, value=default_hgc, step=0.5)
            sim_tenure = st.number_input("Tenure", min_value=0.0, max_value=40.0, value=default_tenure, step=0.5)

        with c3:
            sim_hours = st.number_input("Weekly hours worked", min_value=1.0, max_value=120.0, value=default_hours, step=1.0)
            sim_prior = st.number_input("Prior wage", min_value=0.5, max_value=500.0, value=default_prior, step=0.5)
            sim_occ = safe_selectbox("Occupation group", occ_choices, 0, key="sim_occ")
            sim_ind = safe_selectbox("Industry group", ind_choices, 0, key="sim_ind")

        submitted = st.form_submit_button("Run simulator")

    if submitted:
        base_row = build_profile_row(
            year_value=int(sim_year),
            sex_label=sim_sex,
            race_label=sim_race if sim_race else "Unknown",
            region_label=sim_region if sim_region else "Unknown",
            marital_label=sim_marital if sim_marital else "Other/Unknown",
            age=float(sim_age),
            hgc=float(sim_hgc),
            tenure=float(sim_tenure),
            hrs_wrk=float(sim_hours),
            occupation_group=sim_occ if sim_occ else "Unknown",
            industry_group=sim_ind if sim_ind else "Unknown",
            prior_wage=float(sim_prior)
        )

        pred_base = predict_profile(models.get("M_Q1"), base_row)
        pred_prior = predict_profile(models.get("M_Q2_INT"), base_row)
        pred_policy = predict_profile(models.get("M_Q3_TRIPLE"), base_row)

        m1, m2, m3 = st.columns(3)
        m1.metric("Baseline model prediction", f"{pred_base:.2f}" if pd.notna(pred_base) else "N/A")
        m2.metric("Prior-pay model prediction", f"{pred_prior:.2f}" if pd.notna(pred_prior) else "N/A")
        m3.metric("Policy model prediction", f"{pred_policy:.2f}" if pd.notna(pred_policy) else "N/A")

        # Same profile by gender
        male_row = base_row.copy()
        male_row["female"] = 0
        female_row = base_row.copy()
        female_row["female"] = 1

        male_pred = predict_profile(models.get("M_Q1"), male_row)
        female_pred = predict_profile(models.get("M_Q1"), female_row)

        gender_compare = pd.DataFrame({
            "sex_label": ["Male", "Female"],
            "predicted_wage": [male_pred, female_pred]
        }).dropna()

        if len(gender_compare) > 0:
            fig_gender = px.bar(
                gender_compare,
                x="sex_label",
                y="predicted_wage",
                color="sex_label",
                title="Same Profile, Different Gender"
            )
            st.plotly_chart(fig_gender, use_container_width=True)

        # Pre vs post proxy
        pre_row = base_row.copy()
        pre_row["Year_num"] = 2017
        pre_row["post_2018"] = 0

        post_row = base_row.copy()
        post_row["Year_num"] = 2021
        post_row["post_2018"] = 1

        policy_compare = pd.DataFrame({
            "policy_period": ["Pre-2018 proxy", "Post-2018 proxy"],
            "predicted_wage": [
                predict_profile(models.get("M_Q3_TRIPLE"), pre_row),
                predict_profile(models.get("M_Q3_TRIPLE"), post_row)
            ]
        }).dropna()

        if len(policy_compare) > 0:
            fig_policy = px.bar(
                policy_compare,
                x="policy_period",
                y="predicted_wage",
                color="policy_period",
                title="Pre-2018 vs Post-2018 Proxy"
            )
            st.plotly_chart(fig_policy, use_container_width=True)

        st.info(
            "These are model-based scenario comparisons. They are useful for intuition, not causal guarantees."
        )


# -----------------------------
# Page: Data preview
# -----------------------------
elif page == "Data Preview":
    st.subheader("Data diagnostics")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Raw rows", f"{meta['raw_rows']:,}")
    d2.metric("Clean rows", f"{meta['clean_rows']:,}")
    d3.metric("Lag rows", f"{meta['lag_rows']:,}")
    d4.metric("Exact duplicates removed", f"{meta['exact_duplicate_count']:,}")
    d5.metric("Potential key duplicates", f"{meta['key_duplicate_count']:,}")

    st.subheader("Raw data preview")
    st.dataframe(df_raw.head(MAX_PREVIEW_ROWS), use_container_width=True)

    st.subheader("Cleaned main sample preview")
    st.dataframe(filtered.head(MAX_PREVIEW_ROWS), use_container_width=True)

    st.subheader("Lag sample preview")
    st.dataframe(lag_filtered.head(MAX_PREVIEW_ROWS), use_container_width=True)

    st.subheader("Column names")
    st.write(df_raw.columns.tolist())
