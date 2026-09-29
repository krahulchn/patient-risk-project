"""
Generates a realistic synthetic patient health dataset for
cardiovascular risk prediction.

Why synthetic: real patient-level health records are protected by
privacy regulations and aren't freely available without a data-use
agreement. This script builds a stand-in with the same statistical
structure genuine cardiovascular risk data has (age/BMI/blood
pressure/cholesterol relationships, smoking and family-history
effects, realistic noise) so the analysis and modeling pipeline
downstream is genuine, even though the raw records are simulated.
"""

import numpy as np
import pandas as pd
import os

RNG = np.random.RandomState(7)
N_PATIENTS = 1500

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Generate patient attributes
# ------------------------------------------------------------------
age = RNG.randint(25, 80, N_PATIENTS)
sex = RNG.choice(["Male", "Female"], N_PATIENTS, p=[0.49, 0.51])
bmi = np.clip(RNG.normal(27, 5, N_PATIENTS), 16, 50)
systolic_bp = np.clip(RNG.normal(120, 15, N_PATIENTS) + (age - 45) * 0.35, 90, 200)
cholesterol = np.clip(RNG.normal(200, 35, N_PATIENTS) + (bmi - 27) * 1.5, 120, 340)
smoker = RNG.rand(N_PATIENTS) < 0.22
exercise_hours_per_week = np.clip(RNG.normal(3, 2, N_PATIENTS) - (bmi > 30) * 1.0, 0, 14)
family_history = RNG.rand(N_PATIENTS) < 0.30
diabetic = RNG.rand(N_PATIENTS) < (0.08 + (bmi > 30) * 0.12)

# ------------------------------------------------------------------
# Compute an underlying risk score (weighted combination + noise)
# then threshold into a binary high-risk label
# ------------------------------------------------------------------
risk_score = (
    0.04 * (age - 45)
    + 0.05 * (bmi - 27)
    + 0.03 * (systolic_bp - 120)
    + 0.015 * (cholesterol - 200)
    + 1.1 * smoker.astype(float)
    + 0.9 * family_history.astype(float)
    + 0.8 * diabetic.astype(float)
    - 0.15 * exercise_hours_per_week
    + RNG.normal(0, 1.3, N_PATIENTS)
)
# Threshold at the 60th percentile-ish to get a realistic ~35% high-risk rate
threshold = np.percentile(risk_score, 65)
high_risk = (risk_score > threshold).astype(int)

df = pd.DataFrame({
    "age": age,
    "sex": sex,
    "bmi": bmi.round(1),
    "systolic_bp": systolic_bp.round(0).astype(int),
    "cholesterol": cholesterol.round(0).astype(int),
    "smoker": smoker,
    "exercise_hours_per_week": exercise_hours_per_week.round(1),
    "family_history": family_history,
    "diabetic": diabetic,
    "high_risk": high_risk,
})

df.to_csv(os.path.join(DATA_DIR, "patient_health_data.csv"), index=False)
print(f"Generated {len(df)} patient records -> {os.path.join(DATA_DIR, 'patient_health_data.csv')}")
print(f"High-risk rate: {df['high_risk'].mean():.1%}")
print(df.head())
