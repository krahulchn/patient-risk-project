# Real-World Data Project — Patient Cardiovascular Risk Analysis & Prediction

**Domain:** Health
An end-to-end project: exploratory analysis of patient health data, followed by a model that predicts which patients are at high cardiovascular risk.

## 1. About the Dataset

1,500 simulated patient records with common cardiovascular risk factors: age, sex, BMI, systolic blood pressure, cholesterol, smoking status, weekly exercise hours, family history of heart disease, and diabetic status. Target: `high_risk` (1 = high cardiovascular risk, 0 = low risk).

**A note on the data source:** real patient-level health records are protected by privacy regulations and aren't publicly available without a data-use agreement, so this project uses a **realistically simulated dataset** (see [`src/generate_data.py`](src/generate_data.py)). The risk label was generated from a weighted combination of the same factors doctors actually consider (age, blood pressure, cholesterol, smoking, family history, diabetes, exercise) plus random noise — so the dataset behaves like genuine risk data, even though the individual records are synthetic.

Raw data: [`data/patient_health_data.csv`](data/patient_health_data.csv)
Overall high-risk rate: **35%**

## 2. Exploratory Data Analysis

**No missing values** — the dataset required no cleaning.

### Distributions
[`outputs/plots/feature_distributions.png`](outputs/plots/feature_distributions.png)

Age, BMI, blood pressure, and cholesterol all follow roughly normal, realistic distributions with no extreme outliers.

### High-risk rate by lifestyle/health factors
[`outputs/plots/risk_by_lifestyle_factors.png`](outputs/plots/risk_by_lifestyle_factors.png)

| Factor | High-Risk Rate (No) | High-Risk Rate (Yes) |
|---|---|---|
| Smoker | 30.4% | 52.9% |
| Family History | 29.2% | 47.7% |
| Diabetic | 32.4% | 53.8% |

**Insight:** Each of these three factors roughly **doubles** a patient's likelihood of being high-risk on its own — smoking has the single largest swing (+22.5 percentage points).

### Correlation analysis
[`outputs/plots/correlation_heatmap.png`](outputs/plots/correlation_heatmap.png)

Correlation with `high_risk`, strongest to weakest: `age` (0.34), `systolic_bp` (0.27), `cholesterol` (0.26), `smoker` (0.19), `family_history` (0.18), `bmi` (0.17), `exercise_hours_per_week` (-0.16), `diabetic` (0.15). `sex` shows essentially no relationship (-0.003).

## 3. Predictive Modeling

**Goal:** Classify each patient as high-risk or low-risk using their health profile.

**Models compared:** Logistic Regression vs. Random Forest Classifier

### Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.793 | 0.736 | 0.638 | 0.684 | **0.870** |
| Random Forest | 0.780 | 0.760 | 0.543 | 0.633 | 0.867 |

Full table: [`outputs/model_comparison.csv`](outputs/model_comparison.csv)

**Logistic Regression edges out Random Forest overall**, particularly on recall — meaning it catches more of the truly high-risk patients, which matters more than raw accuracy in a health-screening context (missing a high-risk patient is costlier than a false alarm).

See [`outputs/plots/roc_curve_comparison.png`](outputs/plots/roc_curve_comparison.png) and per-model confusion matrices (`confusion_matrix_logistic_regression.png`, `confusion_matrix_random_forest.png`).

### What drives the predictions?
[`outputs/plots/feature_importance.png`](outputs/plots/feature_importance.png)

| Feature | Importance (Random Forest) |
|---|---|
| age | 0.234 |
| cholesterol | 0.192 |
| systolic_bp | 0.156 |
| exercise_hours_per_week | 0.133 |
| bmi | 0.128 |

**Insight:** `age` is the single strongest predictor, followed closely by `cholesterol` and blood pressure — matching real-world cardiovascular risk understanding. Exercise habits carry more predictive weight than smoking or family history in this model, suggesting lifestyle factors are picked up as a meaningful ongoing signal rather than a one-time flag.

## 4. Conclusions

1. **Age, blood pressure, and cholesterol are the dominant risk drivers**, both by correlation and by model feature importance — consistent with established cardiovascular risk factors.
2. **Smoking, family history, and diabetes each roughly double a patient's high-risk likelihood** individually, making them useful quick-screen flags even before running a full model.
3. **Logistic Regression outperformed Random Forest on this dataset**, especially on recall — a reminder that simpler, more interpretable models can beat more complex ones, particularly on data with mostly linear relationships.
4. An AUC of ~0.87 shows the model reliably separates high- and low-risk patients using only routine health-checkup data, suggesting this kind of model could support (not replace) clinical risk screening.

## Project Structure

```
patient-risk-project/
├── data/
│   └── patient_health_data.csv
├── src/
│   ├── generate_data.py
│   └── health_risk_analysis.py
├── outputs/
│   ├── statistical_summary.csv
│   ├── correlation_matrix.csv
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   └── plots/
│       ├── feature_distributions.png
│       ├── risk_by_lifestyle_factors.png
│       ├── correlation_heatmap.png
│       ├── confusion_matrix_logistic_regression.png
│       ├── confusion_matrix_random_forest.png
│       ├── roc_curve_comparison.png
│       └── feature_importance.png
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt
python src/generate_data.py            # generates data/patient_health_data.csv
python src/health_risk_analysis.py     # runs EDA + prediction, saves outputs
```

## Key Learnings

- How to structure a full health-domain data science project: data preparation → EDA → feature engineering → predictive modeling → evaluation → conclusions
- How to quantify how much individual risk factors (smoking, family history, diabetes) shift an outcome
- How to compare a linear model against an ensemble model and interpret which one — and why — performs better
- How to translate feature importance and correlation results into clinically-relevant conclusions
