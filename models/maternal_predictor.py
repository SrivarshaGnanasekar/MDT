"""
models/maternal_predictor.py
============================
Dashboard / API facade for the VALIDATED early-booking models.

All probabilities come from the bundles written by
models/train_validated_booking_models.py (evidence: reports/validated/):

    models/validated/gdm_booking_model.joblib
    models/validated/macrosomia_booking_model.joblib

Both are calibrated (no class re-weighting). "High Risk" means the woman is in
the top 20% of booking risk, using a threshold fixed on out-of-fold training
predictions; its held-out sensitivity, specificity and PPV are returned with
every prediction.

The previous CTGAN pseudo-label augmented ensemble and the class-weighted
ensembles were retired on 2026-10-25: their probabilities were not calibrated
(mean predicted GDM risk 43.9% vs 7.9% observed) and pseudo-labelling synthetic
rows with the model's own predictions adds no outcome information.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.booking_risk import (  # noqa: E402
    GDM_BUNDLE, MACROSOMIA_BUNDLE, BookingRiskModel, apply_quality_rules,
)

logger = logging.getLogger(__name__)


class MaternalGDMPredictor:
    """Scores GDM and macrosomia risk at booking with the validated, calibrated models."""

    def __init__(self, model_name: str = "stacking"):
        self.model_name = model_name
        self.load_models()

    # ------------------------------------------------------------------
    def load_models(self) -> bool:
        self.gdm_model = BookingRiskModel(GDM_BUNDLE)
        self.macrosomia_model = BookingRiskModel(MACROSOMIA_BUNDLE)
        self.is_trained = self.gdm_model.is_loaded and self.macrosomia_model.is_loaded
        return self.is_trained

    def train_and_evaluate(self, use_ctgan_augmentation: bool = False) -> Dict[str, Any]:
        """Retrain both validated models and reload them. Returns held-out stacking metrics."""
        if use_ctgan_augmentation:
            logger.warning("CTGAN pseudo-label augmentation is retired (it adds no outcome information); "
                           "training the validated real-data models instead.")
        from models.train_validated_booking_models import main as train_main
        train_main()
        self.load_models()
        g = self.gdm_model.bundle["test_metrics"]
        m = self.macrosomia_model.bundle["test_metrics"]
        return {
            "roc_auc": g["AUROC"], "brier": g["Brier"], "calibration_slope": g["Calibration_slope"],
            "macrosomia_roc_auc": m["AUROC"], "macrosomia_brier": m["Brier"],
            "report": "reports/validated/booking_models_report.txt",
        }

    # ------------------------------------------------------------------
    def predict_gdm_risk(self, raw_patient_dict: Dict[str, Any]) -> Dict[str, Any]:
        r = self.gdm_model.predict(raw_patient_dict)
        return {
            "gdm_prediction": int(r["flagged"]),
            "gdm_probability": r["probability"],
            "gdm_risk_percent": r["risk_percent"],
            "risk_level": "High Risk" if r["flagged"] else "Low Risk",
            "risk_level_label": r["risk_level"],
            "population_rate_percent": r["population_rate_percent"],
            "relative_risk_vs_population": r["relative_risk_vs_population"],
            "flag_threshold": r["flag_threshold"],
            "validated_performance": r["validated_performance"],
            "base_probabilities": r["base_probabilities"],
        }

    def predict_macrosomia_risk(self, raw_patient_dict: Dict[str, Any],
                                actual_ac_cm: Optional[float] = None) -> Dict[str, Any]:
        """
        Stage 1 of the macrosomia pathway: validated booking probability of
        birthweight >= 4000 g. Ultrasound AC (Stage 2) is handled by
        MacrosomiaRiskMonitor.evaluate_risk; ``actual_ac_cm`` is accepted for
        backward compatibility and ignored here.
        """
        r = self.macrosomia_model.predict(raw_patient_dict)
        return {
            "macrosomia_prediction": int(r["flagged"]),
            "macrosomia_probability": r["probability"],
            "macrosomia_risk_percent": r["risk_percent"],
            "booking_flagged": r["flagged"],
            "risk_level": "High Risk" if r["flagged"] else "Low Risk",
            "risk_level_label": r["risk_level"],
            "population_rate_percent": r["population_rate_percent"],
            "relative_risk_vs_population": r["relative_risk_vs_population"],
            "flag_threshold": r["flag_threshold"],
            "validated_performance": r["validated_performance"],
            "base_probabilities": r["base_probabilities"],
            "explanation": (
                f"Booking-model risk of birthweight >= 4000 g: {r['risk_percent']:.1f}% "
                f"(population rate {r['population_rate_percent']:.1f}%)."
            ),
        }

    # ------------------------------------------------------------------
    def transform_for_explanation(self, raw_patient_dict: Dict[str, Any], outcome: str = "gdm") -> pd.DataFrame:
        """Preprocessed feature row used by the model (for SHAP explanations)."""
        model = self.gdm_model if outcome == "gdm" else self.macrosomia_model
        if not model.is_loaded:
            raise RuntimeError(model.load_error)
        df = apply_quality_rules(pd.DataFrame([raw_patient_dict]))
        return model.bundle["preprocessor"].transform(df)


def run_maternal_model_training():
    return MaternalGDMPredictor().train_and_evaluate()


if __name__ == "__main__":
    print(run_maternal_model_training())
