"""
explainability/maternal_shap.py
===============================
Per-patient SHAP explanations for the validated early-booking GDM model.

Uses TreeExplainer on the CatBoost base learner of the validated stacking
ensemble (models/validated/gdm_booking_model.joblib), the same explainer used
for the cohort SHAP summary in reports/validated/. SHAP values are in the
CatBoost log-odds space; they explain that base learner, which is one input to
the stacking meta-learner, not the final stacked probability itself.
"""

import sys
from pathlib import Path
from typing import Any, Dict

import shap

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.maternal_predictor import MaternalGDMPredictor  # noqa: E402


class MaternalSHAPExplainer:
    def __init__(self, outcome: str = "gdm"):
        self.outcome = outcome
        self.predictor = None
        self.explainer = None

    def initialize(self):
        self.predictor = MaternalGDMPredictor()
        model = self.predictor.gdm_model if self.outcome == "gdm" else self.predictor.macrosomia_model
        if not model.is_loaded:
            raise RuntimeError(f"Validated model not loaded: {model.load_error}. "
                               "Run: python models/train_validated_booking_models.py")
        self.explainer = shap.TreeExplainer(model.bundle["base_models"]["CatBoost"])

    def explain_patient_prediction(self, raw_patient_dict: Dict[str, Any]) -> Dict[str, Any]:
        if self.explainer is None:
            self.initialize()
        pred = (self.predictor.predict_gdm_risk(raw_patient_dict) if self.outcome == "gdm"
                else self.predictor.predict_macrosomia_risk(raw_patient_dict))
        X = self.predictor.transform_for_explanation(raw_patient_dict, self.outcome)
        sv = self.explainer.shap_values(X)
        sv = sv[1][0] if isinstance(sv, list) else (sv[0, :, 1] if sv.ndim == 3 else sv[0])
        contributions = sorted(
            ({"feature": n, "value": round(float(v), 3), "shap_value": float(s),
              "impact": "Increases Risk" if s > 0 else "Decreases Risk"}
             for n, v, s in zip(X.columns, X.values[0], sv)),
            key=lambda c: abs(c["shap_value"]), reverse=True)
        return {"prediction": pred, "top_features": contributions[:10], "all_contributions": contributions,
                "note": "SHAP of the CatBoost base learner (log-odds), one input to the stacking ensemble."}


if __name__ == "__main__":
    res = MaternalSHAPExplainer().explain_patient_prediction({
        "AgeAtStartOfSpell": 34, "Body Mass Index at Booking": 31.2, "Gestation at booking (Weeks)": 10,
        "Gravida": 2, "Parity": 1, "No_Of_previous_Csections": 0, "Index of Multiple Deprivation Rank": 9000,
        "Ethnicity": "IND"})
    print(res["prediction"]["gdm_risk_percent"], res["top_features"][:3])
