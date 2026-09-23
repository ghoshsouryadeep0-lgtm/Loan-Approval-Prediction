# 🏦 Loan Approval Prediction & Analytics

A production-ready, end-to-end machine learning web application built with **Python**, **Streamlit**, and **scikit-learn** that predicts loan approval outcomes and provides interactive analytics on applicant data.

---

## 📁 Project Structure

```
Loan approval/
├── app.py                                   # Main Streamlit application
├── loan_approval_prediction_500_rows.csv    # Raw dataset (500 records)
├── loan_approval_prediction_cleaned.csv     # Cleaned dataset (used by app)
├── requirements.txt                         # Python dependencies
└── README.md                                # This file
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📦 Requirements

| Package | Minimum Version |
|---|---|
| streamlit | 1.30.0 |
| pandas | 2.0.0 |
| numpy | 1.24.0 |
| matplotlib | 3.7.0 |
| seaborn | 0.12.0 |
| scikit-learn | 1.2.0 |

Install all at once:

```bash
pip install streamlit>=1.30.0 pandas>=2.0.0 numpy>=1.24.0 matplotlib>=3.7.0 seaborn>=0.12.0 scikit-learn>=1.2.0
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source file | `loan_approval_prediction_cleaned.csv` |
| Rows | 500 |
| Features | 12 input + 1 derived flag |
| Target | `Loan_Status` (Approved / Rejected) |
| Approval rate | 26% Approved, 74% Rejected |

### Features

| Column | Type | Description |
|---|---|---|
| `Age` | int | Applicant age (18–60) |
| `Annual_Income` | float | Yearly income in USD |
| `Loan_Amount` | float | Requested loan amount in USD |
| `Loan_Term_Months` | int | Loan duration: 12, 24, 36, 48, or 60 months |
| `Credit_Score` | float | FICO credit score (501–850) |
| `Employment_Years` | int | Years of employment (0–20) |
| `Existing_Debt` | float | Current outstanding debt in USD |
| `Dependents` | int | Number of dependents (0–4) |
| `Education` | categorical | Graduate / Not Graduate |
| `Self_Employed` | categorical | Yes / No |
| `Property_Area` | categorical | Urban / Semiurban / Rural |
| `High_LTI_Flag` | int | 1 if Loan-to-Income ratio > 33× (outlier flag) |
| `Loan_Status` | categorical | **Target** — Approved / Rejected |

---

## 🖥️ Application Pages

### 🏠 Overview
- KPI metric cards: total applications, approvals, approval rate, average credit score, average income
- Donut chart — overall approval split
- Bar chart — approval rate by property area
- Dataset preview table
- Descriptive statistics for all numeric features

### 📊 EDA & Insights
Four interactive tabs:

| Tab | Content |
|---|---|
| **Distributions** | Overlaid histograms for all 8 numeric features, split by loan status |
| **Correlation** | Full heatmap + correlation-with-target ranking table |
| **Feature vs Target** | Selectable feature — box plot + density overlay + summary stats table |
| **Categorical Breakdown** | Stacked bar charts for Education, Self-Employed, Property Area, Dependents, Loan Term |

### 🤖 Model Performance
- Side-by-side comparison table (Accuracy, ROC-AUC, Avg Precision, 5-fold CV AUC)
- ROC curves and Precision-Recall curves for all 3 models
- Confusion matrices (test set)
- Full classification report per model
- Random Forest feature importance bar chart + rank table

### 🔮 Predict Loan
- Input form: all 12 applicant features
- Instant prediction with approval probability
- Visual probability gauge bar
- Risk tier badge: Low / Moderate / High / Very High
- All-models agreement table
- Feature comparison table: your input vs dataset average

---

## 🤖 Machine Learning Models

Three classifiers are trained, evaluated, and compared:

| Model | Accuracy | ROC-AUC | Notes |
|---|---|---|---|
| **Random Forest** ⭐ | 85.0% | **0.9444** | Best ROC-AUC; used for predictions |
| Gradient Boosting | 86.0% | 0.9106 | Highest raw accuracy |
| Logistic Regression | 82.0% | 0.9246 | Baseline linear model |

> The **Random Forest** model is automatically selected as best by ROC-AUC and used for the prediction page.

### Training details
- **Train / Test split:** 80% / 20%, stratified by target
- **Cross-validation:** 5-fold stratified CV (ROC-AUC scoring)
- **Class imbalance handling:** `class_weight="balanced"` on Random Forest and Logistic Regression
- **Preprocessing:** `LabelEncoder` for categoricals; `StandardScaler` inside the Logistic Regression pipeline

---

## 🧹 Data Cleaning

The raw dataset was cleaned before use. Steps applied:

1. Strip leading/trailing whitespace from all cells
2. Normalise categorical columns to consistent casing
3. Remove exact duplicate rows
4. Drop rows with any missing / empty values
5. Cast numeric columns to correct Python types
6. Validate domain constraints (age range, credit score range, etc.)
7. Add `High_LTI_Flag` — marks rows where Loan-to-Income ratio exceeds mean + 3 SD (threshold ≈ 33×)

**Result:** 500 rows retained, 0 rows dropped, 14 rows flagged with `High_LTI_Flag = 1`.

---

## 🗂️ Architecture

```
app.py
 ├── load_data()          — cached CSV loader with dtype casting
 ├── train_models()       — cached model training, evaluation, and feature importance
 ├── Sidebar              — navigation + dataset/model summary
 ├── Page: Overview       — KPIs, charts, dataset preview
 ├── Page: EDA            — distributions, correlation, feature vs target, categoricals
 ├── Page: Model Perf.    — curves, confusion matrices, feature importance
 └── Page: Predict        — input form → encode → predict → display results
```

---

## 📝 Notes

- All model training is cached with `@st.cache_resource` — models train once on first load and are reused across page navigations.
- Data loading is cached with `@st.cache_data`.
- The best model is selected automatically at runtime by highest ROC-AUC.
- The `High_LTI_Flag` column is computed at both training time (from data) and prediction time (from user input) using the same threshold of **33.11×**.
