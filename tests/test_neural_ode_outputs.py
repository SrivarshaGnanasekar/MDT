"""
tests/test_neural_ode_outputs.py
================================
The Neural ODE is illustrative only: it must not return a per-week GDM risk
column (that curve has no time-specific ground truth). See
reports/validated/neural_ode_glucose_validation.txt.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
torch = pytest.importorskip("torch")

from digital_twin.neural_ode import FetalMaternalODEPredictor  # noqa: E402

PROFILE = {"AgeAtStartOfSpell": 31, "Gestation at booking (Weeks)": 12, "Body Mass Index at Booking": 29,
           "Gravida": 1, "Parity": 0, "No_Of_previous_Csections": 0, "Index of Multiple Deprivation Rank": 12000,
           "SystolicBloodPressureCuff": 120, "Diastolic Blood Pressure": 80, "Ethnicity": "WEU"}


def test_no_gdm_risk_curve_returned():
    p = FetalMaternalODEPredictor()
    if not p.is_loaded:
        pytest.skip("Neural ODE checkpoint not available")
    df = p.predict_trajectory(PROFILE, anchor_probability=0.2)
    assert "gdm_risk_probability" not in df.columns
    assert {"glucose_level", "bmi", "systolic_bp"} <= set(df.columns)
