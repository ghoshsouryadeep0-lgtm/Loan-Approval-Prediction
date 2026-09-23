"""
Loan Approval Prediction & Analytics Web Application
Built with Streamlit + scikit-learn
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.pipeline import Pipeline
import warnings
import io

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700; color: #1e3a5f;
        border-bottom: 3px solid #2e86de; padding-bottom: 8px; margin-bottom: 4px;
    }
    .sub-header {
        font-size: 0.95rem; color: #57606a; margin-bottom: 24px;
    }
    .metric-card {
        background: #f0f4ff; border-radius: 10px; padding: 16px 20px;
        border-left: 5px solid #2e86de; margin-bottom: 10px;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #1e3a5f; }
    .metric-label { font-size: 0.85rem; color: #57606a; margin-top: 2px; }
    .approved-badge {
        background: #d4edda; color: #155724; padding: 4px 12px;
        border-radius: 20px; font-weight: 600; font-size: 1.1rem;
    }
    .rejected-badge {
        background: #f8d7da; color: #721c24; padding: 4px 12px;
        border-radius: 20px; font-weight: 600; font-size: 1.1rem;
    }
    .section-header {
        font-size: 1.2rem; font-weight: 600; color: #1e3a5f;
        margin-top: 20px; margin-bottom: 8px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background: #f0f4ff; border-radius: 8px 8px 0 0;
        padding: 8px 18px; font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ── Data loading ───────────────────────────────────────────────────────────
DATA_PATH = "loan_approval_prediction_cleaned.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    # Ensure correct dtypes
    int_cols   = ["Age", "Loan_Term_Months", "Dependents", "Employment_Years", "High_LTI_Flag"]
    float_cols = ["Annual_Income", "Loan_Amount", "Credit_Score", "Existing_Debt"]
    for c in int_cols:
        df[c] = df[c].astype(int)
    for c in float_cols:
        df[c] = df[c].astype(float)
    return df

@st.cache_resource
def train_models(df: pd.DataFrame):
    """Train three classifiers and return fitted pipelines + evaluation metrics."""
    feature_cols = [
        "Age", "Annual_Income", "Loan_Amount", "Loan_Term_Months",
        "Credit_Score", "Employment_Years", "Existing_Debt", "Dependents",
        "Education", "Self_Employed", "Property_Area", "High_LTI_Flag"
    ]
    target_col = "Loan_Status"

    X = df[feature_cols].copy()
    y = (df[target_col] == "Approved").astype(int)

    # Encode categoricals
    le_edu  = LabelEncoder().fit(["Graduate", "Not Graduate"])
    le_emp  = LabelEncoder().fit(["No", "Yes"])
    le_area = LabelEncoder().fit(["Rural", "Semiurban", "Urban"])

    X["Education"]     = le_edu.transform(X["Education"])
    X["Self_Employed"] = le_emp.transform(X["Self_Employed"])
    X["Property_Area"] = le_area.transform(X["Property_Area"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=4,
            random_state=42
        ),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
        ]),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

        results[name] = {
            "model":      model,
            "y_test":     y_test,
            "y_pred":     y_pred,
            "y_proba":    y_proba,
            "accuracy":   accuracy_score(y_test, y_pred),
            "roc_auc":    roc_auc_score(y_test, y_proba),
            "avg_prec":   average_precision_score(y_test, y_proba),
            "cv_auc_mean": cv_scores.mean(),
            "cv_auc_std":  cv_scores.std(),
            "report":     classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]),
            "cm":         confusion_matrix(y_test, y_pred),
        }

    # Feature importances from Random Forest
    rf = results["Random Forest"]["model"]
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    return results, importances, X_test, y_test, le_edu, le_emp, le_area, feature_cols

# ── Load ───────────────────────────────────────────────────────────────────
df = load_data()
model_results, feat_imp, X_test_global, y_test_global, le_edu, le_emp, le_area, feature_cols = train_models(df)
best_model_name = max(model_results, key=lambda k: model_results[k]["roc_auc"])
best_model      = model_results[best_model_name]["model"]

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Loan Approval App")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📊 EDA & Insights", "🤖 Model Performance", "🔮 Predict Loan"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown(f"- Rows: **{len(df):,}**")
    st.markdown(f"- Features: **{len(feature_cols)}**")
    st.markdown(f"- Approval rate: **{df['Loan_Status'].value_counts(normalize=True)['Approved']*100:.1f}%**")
    st.markdown("---")
    st.markdown("**Best Model**")
    st.markdown(f"- {best_model_name}")
    st.markdown(f"- ROC-AUC: **{model_results[best_model_name]['roc_auc']:.3f}**")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ══════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<div class="main-header">🏦 Loan Approval Prediction & Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">End-to-end machine learning dashboard for loan approval decisions</div>', unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    approved  = int((df["Loan_Status"] == "Approved").sum())
    rejected  = int((df["Loan_Status"] == "Rejected").sum())
    app_rate  = approved / len(df) * 100
    avg_score = df["Credit_Score"].mean()
    avg_inc   = df["Annual_Income"].mean()

    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df):,}</div><div class="metric-label">Total Applications</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{approved:,}</div><div class="metric-label">Approved Loans</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{app_rate:.1f}%</div><div class="metric-label">Approval Rate</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_score:.0f}</div><div class="metric-label">Avg Credit Score</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="metric-card"><div class="metric-value">${avg_inc:,.0f}</div><div class="metric-label">Avg Annual Income</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="section-header">Loan Status Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        colors = ["#2e86de", "#e74c3c"]
        sizes  = [approved, rejected]
        labels = [f"Approved\n{approved} ({app_rate:.1f}%)", f"Rejected\n{rejected} ({100-app_rate:.1f}%)"]
        wedges, texts = ax.pie(sizes, labels=labels, colors=colors, startangle=90,
                               wedgeprops=dict(width=0.55), textprops={"fontsize": 11})
        ax.set_title("Overall Approval Split", fontsize=13, fontweight="bold", pad=10)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="section-header">Approval Rate by Property Area</div>', unsafe_allow_html=True)
        area_rates = df.groupby("Property_Area")["Loan_Status"].apply(
            lambda x: (x == "Approved").sum() / len(x) * 100
        ).reset_index()
        area_rates.columns = ["Property_Area", "Approval_Rate"]
        fig, ax = plt.subplots(figsize=(5, 4))
        bar_colors = ["#2e86de", "#3cb371", "#f39c12"]
        bars = ax.bar(area_rates["Property_Area"], area_rates["Approval_Rate"],
                      color=bar_colors, edgecolor="white", linewidth=1.5, width=0.55)
        for bar, val in zip(bars, area_rates["Approval_Rate"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_xlabel("Property Area", fontsize=11)
        ax.set_ylabel("Approval Rate (%)", fontsize=11)
        ax.set_title("Approval Rate by Area", fontsize=13, fontweight="bold")
        ax.set_ylim(0, max(area_rates["Approval_Rate"]) + 10)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown('<div class="section-header">📋 Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown('<div class="section-header">📈 Descriptive Statistics</div>', unsafe_allow_html=True)
    num_cols = ["Age", "Annual_Income", "Loan_Amount", "Credit_Score",
                "Employment_Years", "Existing_Debt", "Dependents"]
    st.dataframe(df[num_cols].describe().round(2), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ══════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA & Insights":
    st.markdown('<div class="main-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deep-dive into feature distributions and their relationship with loan approval</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Correlation", "Feature vs Target", "Categorical Breakdown"])

    # ── Tab 1: Distributions ──────────────────────────────────────────────
    with tab1:
        num_features = ["Age", "Annual_Income", "Loan_Amount", "Credit_Score",
                        "Employment_Years", "Existing_Debt", "Dependents", "Loan_Term_Months"]
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        approved_df = df[df["Loan_Status"] == "Approved"]
        rejected_df = df[df["Loan_Status"] == "Rejected"]
        for i, col in enumerate(num_features):
            axes[i].hist(rejected_df[col], bins=20, alpha=0.6, color="#e74c3c", label="Rejected", density=True)
            axes[i].hist(approved_df[col], bins=20, alpha=0.6, color="#2e86de", label="Approved", density=True)
            axes[i].set_title(col.replace("_", " "), fontsize=11, fontweight="bold")
            axes[i].set_xlabel("")
            axes[i].spines[["top", "right"]].set_visible(False)
            if i == 0:
                axes[i].legend(fontsize=9)
        fig.suptitle("Feature Distributions by Loan Status", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Tab 2: Correlation ────────────────────────────────────────────────
    with tab2:
        num_df = df[["Age", "Annual_Income", "Loan_Amount", "Credit_Score",
                     "Employment_Years", "Existing_Debt", "Dependents",
                     "Loan_Term_Months", "High_LTI_Flag"]].copy()
        num_df["Approved"] = (df["Loan_Status"] == "Approved").astype(int)
        corr = num_df.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
                    center=0, square=True, linewidths=0.5, ax=ax,
                    annot_kws={"size": 9}, cbar_kws={"shrink": 0.8})
        ax.set_title("Correlation Matrix", fontsize=14, fontweight="bold", pad=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("**Correlations with Approval**")
        corr_with_target = corr["Approved"].drop("Approved").sort_values(key=abs, ascending=False)
        corr_df = corr_with_target.reset_index()
        corr_df.columns = ["Feature", "Correlation with Approval"]
        corr_df["Correlation with Approval"] = corr_df["Correlation with Approval"].round(4)
        st.dataframe(corr_df, use_container_width=True, hide_index=True)

    # ── Tab 3: Feature vs Target ──────────────────────────────────────────
    with tab3:
        sel_feature = st.selectbox("Select feature", ["Credit_Score", "Annual_Income", "Loan_Amount",
                                                        "Age", "Employment_Years", "Existing_Debt"])
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Box plot
        data_app = df[df["Loan_Status"] == "Approved"][sel_feature]
        data_rej = df[df["Loan_Status"] == "Rejected"][sel_feature]
        bp = axes[0].boxplot([data_rej, data_app], labels=["Rejected", "Approved"],
                             patch_artist=True, medianprops={"color": "black", "linewidth": 2})
        for patch, color in zip(bp["boxes"], ["#e74c3c", "#2e86de"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[0].set_title(f"{sel_feature.replace('_',' ')} — Box Plot", fontsize=12, fontweight="bold")
        axes[0].set_ylabel(sel_feature.replace("_", " "))
        axes[0].spines[["top", "right"]].set_visible(False)

        # KDE / histogram overlay
        axes[1].hist(data_rej, bins=25, alpha=0.5, color="#e74c3c", label="Rejected", density=True)
        axes[1].hist(data_app, bins=25, alpha=0.5, color="#2e86de", label="Approved", density=True)
        axes[1].set_title(f"{sel_feature.replace('_',' ')} — Density", fontsize=12, fontweight="bold")
        axes[1].set_xlabel(sel_feature.replace("_", " "))
        axes[1].set_ylabel("Density")
        axes[1].legend()
        axes[1].spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Stats table
        stats = pd.DataFrame({
            "Metric": ["Mean", "Median", "Std Dev", "Min", "Max"],
            "Approved": [f"{data_app.mean():.2f}", f"{data_app.median():.2f}",
                         f"{data_app.std():.2f}", f"{data_app.min():.2f}", f"{data_app.max():.2f}"],
            "Rejected": [f"{data_rej.mean():.2f}", f"{data_rej.median():.2f}",
                         f"{data_rej.std():.2f}", f"{data_rej.min():.2f}", f"{data_rej.max():.2f}"],
        })
        st.dataframe(stats, use_container_width=True, hide_index=True)

    # ── Tab 4: Categorical Breakdown ──────────────────────────────────────
    with tab4:
        cat_features = ["Education", "Self_Employed", "Property_Area", "Dependents", "Loan_Term_Months"]
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes = axes.flatten()
        for i, col in enumerate(cat_features):
            ct = df.groupby(col)["Loan_Status"].value_counts(normalize=True).unstack().fillna(0) * 100
            if "Approved" not in ct.columns:
                ct["Approved"] = 0
            if "Rejected" not in ct.columns:
                ct["Rejected"] = 0
            ct[["Rejected", "Approved"]].plot(
                kind="bar", stacked=True, ax=axes[i],
                color=["#e74c3c", "#2e86de"], edgecolor="white", width=0.65
            )
            axes[i].set_title(col.replace("_", " "), fontsize=11, fontweight="bold")
            axes[i].set_xlabel("")
            axes[i].set_ylabel("Percentage (%)")
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=30, ha="right")
            axes[i].legend(["Rejected", "Approved"], fontsize=9)
            axes[i].spines[["top", "right"]].set_visible(False)
            axes[i].set_ylim(0, 110)
        axes[5].axis("off")
        plt.suptitle("Approval Rate by Categorical Features", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — Model Performance
# ══════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.markdown('<div class="main-header">🤖 Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comparison of three classifiers trained on the loan dataset</div>', unsafe_allow_html=True)

    # ── Model comparison table ─────────────────────────────────────────────
    st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
    comp_rows = []
    for name, res in model_results.items():
        comp_rows.append({
            "Model":        name,
            "Accuracy":     f"{res['accuracy']*100:.2f}%",
            "ROC-AUC":      f"{res['roc_auc']:.4f}",
            "Avg Precision":f"{res['avg_prec']:.4f}",
            "CV AUC Mean":  f"{res['cv_auc_mean']:.4f}",
            "CV AUC ±Std":  f"±{res['cv_auc_std']:.4f}",
            "Best":         "⭐" if name == best_model_name else "",
        })
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["ROC & PR Curves", "Confusion Matrices", "Feature Importance"])

    # ── Tab 1: Curves ─────────────────────────────────────────────────────
    with tab1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = ["#2e86de", "#e74c3c", "#27ae60"]
        for (name, res), color in zip(model_results.items(), colors):
            fpr, tpr, _ = roc_curve(res["y_test"], res["y_proba"])
            axes[0].plot(fpr, tpr, color=color, lw=2,
                         label=f"{name} (AUC={res['roc_auc']:.3f})")
            prec, rec, _ = precision_recall_curve(res["y_test"], res["y_proba"])
            axes[1].plot(rec, prec, color=color, lw=2,
                         label=f"{name} (AP={res['avg_prec']:.3f})")

        axes[0].plot([0,1],[0,1],"k--", lw=1.2, alpha=0.5)
        axes[0].set_xlabel("False Positive Rate", fontsize=11)
        axes[0].set_ylabel("True Positive Rate", fontsize=11)
        axes[0].set_title("ROC Curves", fontsize=13, fontweight="bold")
        axes[0].legend(fontsize=9); axes[0].spines[["top","right"]].set_visible(False)

        axes[1].set_xlabel("Recall", fontsize=11)
        axes[1].set_ylabel("Precision", fontsize=11)
        axes[1].set_title("Precision-Recall Curves", fontsize=13, fontweight="bold")
        axes[1].legend(fontsize=9); axes[1].spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Tab 2: Confusion Matrices ─────────────────────────────────────────
    with tab2:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (name, res) in zip(axes, model_results.items()):
            cm = res["cm"]
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Rejected","Approved"],
                        yticklabels=["Rejected","Approved"],
                        linewidths=0.5, annot_kws={"size": 13})
            ax.set_title(name, fontsize=11, fontweight="bold")
            ax.set_xlabel("Predicted", fontsize=10)
            ax.set_ylabel("Actual", fontsize=10)
        plt.suptitle("Confusion Matrices (Test Set)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        selected = st.selectbox("Detailed classification report", list(model_results.keys()))
        st.code(model_results[selected]["report"], language="text")

    # ── Tab 3: Feature Importance ─────────────────────────────────────────
    with tab3:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors_imp = ["#2e86de" if v > feat_imp.mean() else "#a8c6e8" for v in feat_imp.values]
        bars = ax.barh(feat_imp.index[::-1], feat_imp.values[::-1],
                       color=colors_imp[::-1], edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, feat_imp.values[::-1]):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", ha="left", fontsize=9)
        ax.set_xlabel("Feature Importance (Mean Decrease Impurity)", fontsize=11)
        ax.set_title("Random Forest — Feature Importances", fontsize=13, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        ax.set_xlim(0, feat_imp.max() + 0.05)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        fi_df = feat_imp.reset_index()
        fi_df.columns = ["Feature", "Importance"]
        fi_df["Importance"] = fi_df["Importance"].round(4)
        fi_df["Rank"] = range(1, len(fi_df)+1)
        st.dataframe(fi_df[["Rank","Feature","Importance"]], use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — Predict
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Loan":
    st.markdown('<div class="main-header">🔮 Loan Approval Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter applicant details below to get an instant prediction from the best model</div>', unsafe_allow_html=True)
    st.info(f"Using **{best_model_name}** — ROC-AUC: **{model_results[best_model_name]['roc_auc']:.4f}**")

    with st.form("prediction_form"):
        st.markdown('<div class="section-header">👤 Personal Information</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            age         = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
            dependents  = st.selectbox("Number of Dependents", [0, 1, 2, 3, 4], index=1)
        with c2:
            education   = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
            self_emp    = st.selectbox("Self Employed", ["No", "Yes"])
        with c3:
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
            emp_years   = st.number_input("Employment Years", min_value=0, max_value=50, value=5, step=1)

        st.markdown('<div class="section-header">💰 Financial Information</div>', unsafe_allow_html=True)
        c4, c5, c6 = st.columns(3)
        with c4:
            annual_income = st.number_input("Annual Income ($)", min_value=0, value=75000, step=1000)
            existing_debt = st.number_input("Existing Debt ($)", min_value=0, value=50000, step=1000)
        with c5:
            loan_amount   = st.number_input("Loan Amount ($)", min_value=1000, value=200000, step=5000)
            loan_term     = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60], index=2)
        with c6:
            credit_score  = st.slider("Credit Score", min_value=300, max_value=850,
                                       value=680, step=1,
                                       help="FICO credit score between 300–850")
            st.markdown(f"Selected: **{credit_score}**")

        submitted = st.form_submit_button("🔍 Predict Approval", use_container_width=True, type="primary")

    if submitted:
        # Compute LTI flag
        lti_ratio = loan_amount / annual_income if annual_income > 0 else 0
        lti_flag  = 1 if lti_ratio > 33.11 else 0

        # Encode
        edu_enc  = le_edu.transform([education])[0]
        emp_enc  = le_emp.transform([self_emp])[0]
        area_enc = le_area.transform([property_area])[0]

        input_data = np.array([[
            age, annual_income, loan_amount, loan_term,
            credit_score, emp_years, existing_debt, dependents,
            edu_enc, emp_enc, area_enc, lti_flag
        ]])

        # Predict with all models
        st.markdown("---")
        st.markdown('<div class="section-header">🎯 Prediction Results</div>', unsafe_allow_html=True)

        # Primary prediction
        proba = best_model.predict_proba(input_data)[0][1]
        pred  = "Approved" if proba >= 0.5 else "Rejected"
        badge = "approved-badge" if pred == "Approved" else "rejected-badge"

        col_res, col_gauge = st.columns([1, 1])
        with col_res:
            st.markdown(f"**Decision:** <span class='{badge}'>{pred}</span>", unsafe_allow_html=True)
            st.markdown(f"**Approval Probability:** `{proba*100:.1f}%`")
            st.markdown(f"**Model used:** {best_model_name}")
            st.markdown(f"**Loan-to-Income Ratio:** `{lti_ratio:.1f}×`" +
                        (" ⚠️ High LTI" if lti_flag else ""))

            # Risk tier
            if proba >= 0.75:
                risk_label, risk_color = "Low Risk", "#155724"
            elif proba >= 0.5:
                risk_label, risk_color = "Moderate Risk", "#856404"
            elif proba >= 0.25:
                risk_label, risk_color = "High Risk", "#721c24"
            else:
                risk_label, risk_color = "Very High Risk", "#491217"
            st.markdown(f"**Risk Tier:** <span style='color:{risk_color}; font-weight:600'>{risk_label}</span>",
                        unsafe_allow_html=True)

        with col_gauge:
            # Probability bar
            fig, ax = plt.subplots(figsize=(5, 2.5))
            bar_color = "#2e86de" if proba >= 0.5 else "#e74c3c"
            ax.barh([""], [proba], color=bar_color, height=0.5, edgecolor="white")
            ax.barh([""], [1 - proba], left=[proba], color="#eee", height=0.5, edgecolor="white")
            ax.axvline(0.5, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
            ax.text(proba/2, 0, f"{proba*100:.1f}%", ha="center", va="center",
                    fontsize=14, fontweight="bold", color="white")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Approval Probability", fontsize=10)
            ax.set_title("Prediction Confidence", fontsize=11, fontweight="bold")
            ax.spines[["top","right","left"]].set_visible(False)
            ax.set_yticks([])
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # All models agreement
        st.markdown('<div class="section-header">📊 All Models Agreement</div>', unsafe_allow_html=True)
        agree_data = []
        for mname, res in model_results.items():
            p = res["model"].predict_proba(input_data)[0][1]
            agree_data.append({
                "Model":       mname,
                "Prediction":  "✅ Approved" if p >= 0.5 else "❌ Rejected",
                "Probability": f"{p*100:.1f}%",
            })
        st.dataframe(pd.DataFrame(agree_data), use_container_width=True, hide_index=True)

        # Key factor analysis
        st.markdown('<div class="section-header">🔍 Key Factors (Feature Importance vs Your Input)</div>', unsafe_allow_html=True)
        factor_map = {
            "Credit_Score":     credit_score,
            "Annual_Income":    annual_income,
            "Loan_Amount":      loan_amount,
            "Existing_Debt":    existing_debt,
            "Employment_Years": emp_years,
            "Age":              age,
            "Loan_Term_Months": loan_term,
            "Dependents":       dependents,
        }
        factor_df = pd.DataFrame([
            {
                "Feature":    feat,
                "Your Value": f"{val:,.0f}",
                "Dataset Avg": f"{df[feat].mean():,.0f}" if feat in df.columns else "N/A",
                "Importance": f"{feat_imp.get(feat, 0):.4f}",
            }
            for feat, val in factor_map.items()
        ])
        st.dataframe(factor_df, use_container_width=True, hide_index=True)
