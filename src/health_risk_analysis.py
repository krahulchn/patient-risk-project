"""
Real-World Data Project — Patient Cardiovascular Risk Analysis & Prediction
--------------------------------------------------------------------------------
Domain  : Health
Dataset : 1,500 simulated patient records with cardiovascular risk
          factors (see generate_data.py for how it was built)

This is an END-TO-END pipeline:
    PART A - Exploratory Data Analysis
        - Distributions of key health measurements
        - High-risk rate by lifestyle/health factors
        - Correlation analysis
    PART B - Predictive Modeling
        - Train Logistic Regression & Random Forest to predict
          patient high-risk status
        - Evaluate with Accuracy, Precision, Recall, F1, ROC-AUC
        - Visualize confusion matrices, ROC curves, feature importance

Outputs are saved to outputs/ and outputs/plots/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
)

sns.set_style("whitegrid")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "patient_health_data.csv")
PLOTS_DIR = os.path.join(ROOT, "outputs", "plots")
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(PLOTS_DIR, exist_ok=True)

SEED = 7

# ==================================================================
# PART A: EXPLORATORY DATA ANALYSIS
# ==================================================================
print("=" * 60)
print("PART A: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nStep 1: Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"  Missing values: {df.isnull().sum().sum()}")
print(f"  High-risk rate: {df['high_risk'].mean():.1%}\n")

print("Step 2: Summary statistics...")
numeric_cols = ["age", "bmi", "systolic_bp", "cholesterol", "exercise_hours_per_week"]
summary = df[numeric_cols].describe()
summary.to_csv(os.path.join(OUT_DIR, "statistical_summary.csv"))
print(summary.round(2).to_string())
print()

# --- Distributions ---
print("Step 3: Plotting feature distributions...")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.ravel(), numeric_cols):
    sns.histplot(df[col], kde=True, ax=ax, color="steelblue")
    ax.set_title(col, fontsize=11)
axes.ravel()[-1].axis("off")
plt.suptitle("Distribution of Patient Health Measurements", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "feature_distributions.png"), dpi=150)
plt.close(fig)

# --- High-risk rate by lifestyle/health factors ---
print("Step 4: Analyzing high-risk rate by key factors...")
factor_cols = ["smoker", "family_history", "diabetic"]
risk_by_factor = {c: df.groupby(c)["high_risk"].mean() for c in factor_cols}
for c, vals in risk_by_factor.items():
    print(f"  High-risk rate by {c}: {dict(vals.round(3))}")
print()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col in zip(axes, factor_cols):
    sns.barplot(data=df, x=col, y="high_risk", hue=col, palette="Set2", legend=False, ax=ax)
    ax.set_title(f"High-Risk Rate by {col.replace('_', ' ').title()}")
    ax.set_ylabel("Proportion High-Risk")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "risk_by_lifestyle_factors.png"), dpi=150)
plt.close(fig)

# --- Correlation heatmap ---
print("Step 5: Correlation analysis...")
corr_df = df.copy()
for c in ["smoker", "family_history", "diabetic"]:
    corr_df[c] = corr_df[c].astype(int)
corr_df["sex"] = (corr_df["sex"] == "Male").astype(int)
corr = corr_df.drop(columns=[]).corr(numeric_only=True)
corr.to_csv(os.path.join(OUT_DIR, "correlation_matrix.csv"))

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 8}, ax=ax)
ax.set_title("Correlation Matrix - Patient Health Factors")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=150)
plt.close(fig)

target_corr = corr["high_risk"].drop("high_risk").sort_values(key=lambda x: x.abs(), ascending=False)
print("  Correlation with high_risk (sorted by strength):")
print(target_corr.round(3).to_string())
print()

# ==================================================================
# PART B: PREDICTIVE MODELING
# ==================================================================
print("=" * 60)
print("PART B: PREDICTIVE MODELING (High-Risk Classification)")
print("=" * 60)

print("\nStep 6: Preparing features...")
model_df = df.copy()
model_df["sex"] = (model_df["sex"] == "Male").astype(int)
for c in ["smoker", "family_history", "diabetic"]:
    model_df[c] = model_df[c].astype(int)

feature_cols = [c for c in model_df.columns if c != "high_risk"]
X = model_df[feature_cols]
y = model_df["high_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"  Train: {len(X_train)} | Test: {len(X_test)}\n")

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=8, random_state=SEED),
}

results = []
roc_data = {}

for name, clf in models.items():
    print(f"Step 7: Training {name}...")
    if name == "Logistic Regression":
        clf.fit(X_train_scaled, y_train)
        preds = clf.predict(X_test_scaled)
        probs = clf.predict_proba(X_test_scaled)[:, 1]
    else:
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1, "ROC-AUC": auc})
    print(f"  Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low Risk", "High Risk"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Reds", colorbar=False)
    ax.set_title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    fname = name.lower().replace(" ", "_")
    plt.savefig(os.path.join(PLOTS_DIR, f"confusion_matrix_{fname}.png"), dpi=150)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_data[name] = (fpr, tpr, auc)
    print()

print("Step 8: Plotting combined ROC curve...")
fig, ax = plt.subplots(figsize=(7, 6))
for name, (fpr, tpr, auc) in roc_data.items():
    ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve Comparison - Patient Risk Prediction")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "roc_curve_comparison.png"), dpi=150)
plt.close(fig)

results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
results_df.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False)
print("\nFinal Model Comparison:")
print(results_df.round(4).to_string(index=False))

# Feature importance from Random Forest
rf_model = models["Random Forest"]
importance = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
importance.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"))
print("\nTop features (Random Forest importance):")
print(importance.round(3).to_string())

fig, ax = plt.subplots(figsize=(8, 6))
importance.sort_values().plot(kind="barh", ax=ax, color="darkred")
ax.set_title("Feature Importance - Random Forest (High-Risk Prediction)")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150)
plt.close(fig)

print(f"\nAll outputs saved to: {OUT_DIR}")
print("Done.")
