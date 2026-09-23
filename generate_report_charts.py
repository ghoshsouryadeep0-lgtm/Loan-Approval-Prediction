"""
Generate all charts for the project report as PNG files.
Output directory: report_charts/
"""
import os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, classification_report
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
os.makedirs("report_charts", exist_ok=True)

PALETTE = {"Approved": "#2e86de", "Rejected": "#e74c3c"}
sns.set_style("whitegrid")
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})

# ── Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv("loan_approval_prediction_cleaned.csv")
for c in ["Age","Loan_Term_Months","Dependents","Employment_Years","High_LTI_Flag"]:
    df[c] = df[c].astype(int)
for c in ["Annual_Income","Loan_Amount","Credit_Score","Existing_Debt"]:
    df[c] = df[c].astype(float)

approved_df = df[df["Loan_Status"] == "Approved"]
rejected_df = df[df["Loan_Status"] == "Rejected"]

# ── Train models ────────────────────────────────────────────────────────────
feature_cols = ["Age","Annual_Income","Loan_Amount","Loan_Term_Months","Credit_Score",
                "Employment_Years","Existing_Debt","Dependents","Education","Self_Employed",
                "Property_Area","High_LTI_Flag"]
X = df[feature_cols].copy()
y = (df["Loan_Status"] == "Approved").astype(int)
le_edu  = LabelEncoder().fit(["Graduate", "Not Graduate"])
le_emp  = LabelEncoder().fit(["No", "Yes"])
le_area = LabelEncoder().fit(["Rural", "Semiurban", "Urban"])
X["Education"]     = le_edu.transform(X["Education"])
X["Self_Employed"] = le_emp.transform(X["Self_Employed"])
X["Property_Area"] = le_area.transform(X["Property_Area"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42),
    "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))]),
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    yp = model.predict(X_test)
    ypr = model.predict_proba(X_test)[:,1]
    results[name] = {
        "model": model, "y_pred": yp, "y_proba": ypr,
        "accuracy": accuracy_score(y_test, yp),
        "roc_auc": roc_auc_score(y_test, ypr),
        "avg_prec": average_precision_score(y_test, ypr),
        "cm": confusion_matrix(y_test, yp),
    }

rf = results["Random Forest"]["model"]
feat_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

# ════════════════════════════════════════════════════════════════════════════
# CHART 1 — Loan Status Donut
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 5))
counts = df["Loan_Status"].value_counts()
labels = [f"{k}\n{v} ({v/len(df)*100:.1f}%)" for k, v in counts.items()]
ax.pie(counts, labels=labels, colors=["#e74c3c","#2e86de"], startangle=90,
       wedgeprops=dict(width=0.55), textprops={"fontsize": 12})
ax.set_title("Loan Status Distribution", fontsize=14, fontweight="bold", pad=14)
plt.tight_layout()
plt.savefig("report_charts/chart1_loan_status.png", dpi=150, bbox_inches="tight")
plt.close()
print("chart1 done")

# ════════════════════════════════════════════════════════════════════════════
# CHART 2 — Approval Rate by Property Area
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 4.5))
area_rates = df.groupby("Property_Area")["Loan_Status"].apply(
    lambda x: (x == "Approved").sum() / len(x) * 100).reset_index()
area_rates.columns = ["Area", "Rate"]
bars = ax.bar(area_rates["Area"], area_rates["Rate"],
              color=["#2e86de","#3cb371","#f39c12"], width=0.5, edgecolor="white")
for bar, val in zip(bars, area_rates["Rate"]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.6,
            f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Approval Rate (%)", fontsize=11)
ax.set_title("Approval Rate by Property Area", fontsize=13, fontweight="bold")
ax.set_ylim(0, max(area_rates["Rate"])+10)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("report_charts/chart2_area_approval.png", dpi=150, bbox_inches="tight")
plt.close()
print("chart2 done")

# ════════════════════════════════════════════════════════════════════════════
# CHART 3 — Feature Distributions (8-panel)
# ════════════════════════════════════════════════════════════════════════════
num_features = ["Age","Annual_Income","Loan_Amount","Credit_Score",
                "Employment_Years","Existing_Debt","Dependents","Loan_Term_Months"]
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()
for i, col in enumerate(num_features):
    axes[i].hist(rejected_df[col], bins=18, alpha=0.6, color="#e74c3c", label="Rejected", density=True)
    axes[i].hist(approved_df[col], bins=18, alpha=0.6, color="#2e86de", label="Approved", density=True)
    axes[i].set_title(col.replace("_"," "), fontsize=10, fontweight="bold")
    axes[i].spines[["top","right"]].set_visible(False)
    if i == 0:
        axes[i].legend(fontsize=9)
fig.suptitle("Feature Distributions by Loan Status", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("report_charts/chart3_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("chart3 done")

# ════════════════════════════════════════════════════════════════════════════
# CHART 4 — Correlation Heatmap
# ════════════════════════════════════════════════════════════════════════════
num_df = df[["Age","Annual_Income","Loan_Amount","Credit_Score","Employment_Years",
             "Existing_Debt","Dependents","Loan_Term_Months","High_LTI_Flag"]].copy()
num_df["Approved"] = y.values
corr = num_df.corr()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
            center=0, square=True, linewidths=0.5, ax=ax,
            annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Matrix", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("report_charts/chart4_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("chart4 done")

# ════════════════════════════════════════════════════════════════════════════
# CHART 5 — Credit Score Box Plot
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 4.5))
bp = ax.boxplot([rejected_df["Credit_Score"], approved_df["Credit_Score"]],
                labels=["Rejected","Approved"], patch_artist=True,
                medianprops={"color":"black","linewidth":2})
for patch, color in zip(bp["boxes"], ["#e74c3c","#2e86de"]):
    patch.set_facecolor(color); patch.set_alpha(0.75)
ax.set_ylabel("Credit Score", fontsize=11)
ax.set_title("Credit Score Distribution by Loan Status", fontsize=12, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("report_charts/chart5_credit_score_box.png", dpi=150, bbox_inches="tight")
plt.close()
print("chart5 done")

# ════════════════════════════════════════════════════════════════════════════
# CHART 6 — Categorical Stacked Bars
# ════════════════════════════════════════════════════════════════════════════
cat_feats = ["Education","Self_Employed","Property_Area"]
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, col in zip(axes, cat_feats):
    ct = df.groupby(col)["Loan_Status"].value_counts(normalize=True).unstack().fillna(0) * 100
    for s in ["Approved","Rejected"]:
        if s not in ct.columns: ct[s] = 0
    ct[["Rejected","Approved"]].plot(kind="bar", stacked=True, ax=ax,
        color=["#e74c3c","#2e86de"], edgecolor="white", width=0.55, legend=(col=="Education"))
    ax.set_title(col.replace("_"," "), fontsize=11, fontweight="bold")
    ax.set_ylabel("Percentage (%)"); ax.set_ylim(0, 110)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.spines[["top","right"]].set_visible(False)
fig.suptitle("Approval Rate by Categorical Features", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("report_charts/chart6_categorical.png", dpi=150, bbox_inches="tight")
plt.close()
print("chart6 done")

# ════════════════════════════════════════════════════════════════════════════
# CHART 7 — ROC Curves
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5.5))
colors = ["#2e86de","#e74c3c","#27ae60"]
for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    ax.plot(fpr, tpr, color=color, lw=2.2, label=f"{name} (AUC={res['roc_auc']:.3f})")
ax.plot([0,1],[0,1],"k--",lw=1.2,alpha=0.5)
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
ax.legend(fontsize=10); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("report_charts/chart7_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("chart7 done")

# ════════════════════════════════════════════════════════════════════════════
# CHART 8 — Confusion Matrices (3 side by side)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, res) in zip(axes, results.items()):
    sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Rejected","Approved"], yticklabels=["Rejected","Approved"],
                linewidths=0.5, annot_kws={"size":13})
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=10); ax.set_ylabel("Actual", fontsize=10)
fig.suptitle("Confusion Matrices (Test Set)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("report_charts/chart8_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("chart8 done")

# ════════════════════════════════════════════════════════════════════════════
# CHART 9 — Feature Importance
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5.5))
imp_colors = ["#2e86de" if v > feat_imp.mean() else "#a8c6e8" for v in feat_imp.values]
bars = ax.barh(feat_imp.index[::-1], feat_imp.values[::-1],
               color=imp_colors[::-1], edgecolor="white")
for bar, val in zip(bars, feat_imp.values[::-1]):
    ax.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)
ax.set_xlabel("Feature Importance (Mean Decrease Impurity)", fontsize=11)
ax.set_title("Random Forest — Feature Importances", fontsize=13, fontweight="bold")
ax.set_xlim(0, feat_imp.max()+0.05)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("report_charts/chart9_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("chart9 done")

# ════════════════════════════════════════════════════════════════════════════
# CHART 10 — Model Comparison Bar
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4.5))
metrics = ["Accuracy", "ROC-AUC", "Avg Precision"]
x = np.arange(len(metrics))
width = 0.22
model_names = list(results.keys())
bar_colors = ["#2e86de","#e74c3c","#27ae60"]
for i, (name, color) in enumerate(zip(model_names, bar_colors)):
    vals = [results[name]["accuracy"], results[name]["roc_auc"], results[name]["avg_prec"]]
    bars = ax.bar(x + i*width, vals, width, label=name, color=color, alpha=0.88, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x + width)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylabel("Score", fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("report_charts/chart10_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("chart10 done")

print("\nAll 10 charts saved to report_charts/")
