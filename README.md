# 📊 Gender Pay Gap Analysis — NLSY97 Dataset

## 🚀 Overview
This project investigates gender-based wage disparities using the National Longitudinal Survey of Youth 1997 (NLSY97). The goal is to understand how factors such as prior salary, education, and work experience contribute to differences in earnings, and to explore whether historical compensation influences future pay outcomes.

## 🎯 Objectives
- Analyze wage differences between genders
- Quantify the impact of prior salary on future earnings
- Identify key factors contributing to pay disparities
- Build predictive models for wage estimation
- Communicate findings through clear visualizations and an interactive dashboard

## 🧠 Key Insights
- Prior salary is a strong predictor of future earnings, reinforcing long-term income disparities
- Gender remains a statistically significant factor even after controlling for experience and education
- Certain features (e.g., job tenure, industry, education level) heavily influence wage outcomes
- Model comparisons reveal trade-offs between interpretability (linear regression) and predictive power (ML models)

## 🛠️ Tech Stack
- Programming: Python
- Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- Modeling: Linear Regression
- Visualization: Matplotlib, Seaborn
- Dashboard: Streamlit

## 📂 Project Structure
├── Datathon.ipynb        # Main analysis notebook

├── data/                 # Dataset files (if included)

├── app.py                # Streamlit dashboard (if applicable)

├── README.md             # Project documentation

## 🔍 Methodology
- Data Cleaning & Preprocessing
  - Handled missing values and filtered relevant variables
  - Standardized and transformed features
- Exploratory Data Analysis (EDA)
  - Examined wage distributions across gender
  - Visualized correlations and trends
- Feature Engineering
  - Selected meaningful predictors (education, experience, prior salary, etc.)
  - Encoded categorical variables
- Modeling and Evaluation
  - Compared performance using evaluation metrics
  - Assessed model accuracy and interpretability
  - Analyzed feature importance

## 📊 Results
- Regression models provided interpretable insights into wage determinants
- Machine learning models improved prediction accuracy
- Evidence suggests systemic factors contribute to wage gaps beyond observable variables

##💡 Applications
- Inform policy discussions around wage transparency and equity
- Help organizations evaluate compensation practices
- Provide a foundation for further research in labor economics

## ▶️ How to Run
### Clone the repository
- git clone https://github.com/yourusername/gender-pay-gap-analysis.git

### Navigate into the project
- cd gender-pay-gap-analysis

### Install dependencies
- pip install -r requirements.txt

### Run the notebook

### (Optional) Run the dashboard
- streamlit run app.py

## 📜 License
This project is for educational and research purposes.
