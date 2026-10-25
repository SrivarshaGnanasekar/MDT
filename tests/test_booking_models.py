"""
tests/test_booking_models.py
============================
Guards for the validated early-booking GDM and macrosomia models.

Run:  python -m pytest tests/test_booking_models.py -v
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.booking_risk import (  # noqa: E402
    CAT_FEATURES, FORBIDDEN, GDM_BUNDLE, MACROSOMIA_BUNDLE, NUM_FEATURES, BookingRiskModel,
    apply_quality_rules,
)

PATIENT = {
    "AgeAtStartOfSpell": 31, "Gestation at booking (Weeks)": 11, "Body Mass Index at Booking": 29.4,
    "Gravida": 1, "Parity": 0, "No_Of_previous_Csections": 0, "Index of Multiple Deprivation Rank": 12183,
    "SystolicBloodPressureCuff": 125, "Diastolic Blood Pressure": 82, "Ethnicity": "WEU",
}
REPORT = ROOT / "reports" / "validated" / "booking_models_report.json"


def test_exactly_ten_booking_predictors_and_no_leakage():
    assert len(NUM_FEATURES) + len(CAT_FEATURES) == 10
    assert not FORBIDDEN & set(NUM_FEATURES + CAT_FEATURES)


def test_quality_rules_blank_implausible_values():
    df = apply_quality_rules(pd.DataFrame([{**PATIENT, "Gestation at booking (Weeks)": -37,
                                            "Gravida": 7015, "SystolicBloodPressureCuff": 11,
                                            "Diastolic Blood Pressure": 0}]))
    for c in ["Gestation at booking (Weeks)", "Gravida", "SystolicBloodPressureCuff", "Diastolic Blood Pressure"]:
        assert np.isnan(df.loc[0, c])


@pytest.mark.parametrize("bundle", [GDM_BUNDLE, MACROSOMIA_BUNDLE])
def test_bundle_scores_patient_with_calibrated_probability(bundle):
    m = BookingRiskModel(bundle)
    if not m.is_loaded:
        pytest.skip(f"bundle not available: {m.load_error}")
    r = m.predict(PATIENT)
    assert 0.0 < r["probability"] < 0.6
    assert r["validated_performance"]["n_test"] > 1000


def test_ogtt_values_do_not_change_booking_gdm_risk():
    """Diagnostic OGTT glucose must have no effect on the early-booking prediction."""
    m = BookingRiskModel(GDM_BUNDLE)
    if not m.is_loaded:
        pytest.skip("GDM bundle not available")
    low = m.predict({**PATIENT, "Glucoselevel0minblood": 4.0, "Glucoselevel120minblood": 5.0})
    high = m.predict({**PATIENT, "Glucoselevel0minblood": 9.0, "Glucoselevel120minblood": 14.0})
    assert low["probability"] == pytest.approx(high["probability"])


def test_saved_report_shows_calibrated_models():
    if not REPORT.exists():
        pytest.skip("validated report not generated yet")
    rep = json.loads(REPORT.read_text())
    for key in ("gdm", "macrosomia"):
        m = rep[key]["metrics"]["Stacking Ensemble"]
        assert abs(m["Mean_predicted"] - m["Observed_rate"]) < 0.02   # calibration-in-the-large
        assert 0.7 < m["Calibration_slope"] < 1.5
        assert m["Scaled_Brier"] > 0                                   # better than predicting prevalence
