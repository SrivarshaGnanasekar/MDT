"""
fusion/feature_fusion.py
========================
Assembles the maternal-fetal digital-twin STATE VECTOR for display and storage.

It concatenates (late fusion, no learned model):
  * maternal booking predictors (diagnostic OGTT glucose excluded),
  * fetal ultrasound biometry from the ACOUSLIC-AI path (AC, EFW, confidence),
  * optional extra branches (e.g. validated booking risks, Neural ODE values).

No combined risk score is computed here: maternal and fetal records in this
project cannot be linked, so a learned fusion model could not be validated.
The previous derived indices (liver-to-stomach ratio, "metabolic growth
index") came from the retired multi-organ model and had no validation; they
were removed on 2026-10-25.
"""
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

EXCLUDED_MATERNAL_KEYS = {"Glucoselevel0minblood", "Glucoselevel120minblood", "Glucoselevelblood", "UID"}
EXCLUDED_FETAL_KEYS = {"sample_id", "organ_features", "fitted_ellipse", "clinical_interpretation"}


def _scalar(v):
    return isinstance(v, (int, float, str, bool, np.number))


class MultimodalFeatureFusion:
    def fuse_features(
        self,
        maternal_dict: Dict[str, Any],
        fetal_dict: Dict[str, Any],
        temporal_dict: Optional[Dict[str, Any]] = None,
        patient_id: Optional[str] = None,
        risk_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        vec: Dict[str, Any] = {}
        for k, v in (maternal_dict or {}).items():
            if k in EXCLUDED_MATERNAL_KEYS or v is None or not _scalar(v):
                continue
            vec[k if k.startswith("maternal_") else f"maternal_{k}"] = float(v) if isinstance(v, (int, float, np.number)) and not isinstance(v, bool) else v
        for k, v in (fetal_dict or {}).items():
            if k in EXCLUDED_FETAL_KEYS or v is None or not _scalar(v):
                continue
            vec[k if k.startswith("fetal_") else f"fetal_{k}"] = float(v) if isinstance(v, (int, float, np.number)) and not isinstance(v, bool) else v
        for prefix, branch in (("temporal", temporal_dict), ("risk", risk_dict)):
            for k, v in (branch or {}).items():
                if v is not None and _scalar(v):
                    vec[f"{prefix}_{k}"] = float(v) if isinstance(v, (int, float, np.number)) and not isinstance(v, bool) else v
        return {
            "patient_id": patient_id or (maternal_dict or {}).get("UID", "PATIENT_001"),
            "maternal_branch": {k: v for k, v in (maternal_dict or {}).items() if k not in EXCLUDED_MATERNAL_KEYS},
            "fetal_branch": fetal_dict or {},
            "temporal_branch": temporal_dict or {},
            "risk_branch": risk_dict or {},
            "fused_feature_vector": vec,
            "num_features": len(vec),
            "note": "State vector for display/storage only; no fused risk score (maternal and fetal data are not linked).",
        }


if __name__ == "__main__":
    r = MultimodalFeatureFusion().fuse_features(
        {"AgeAtStartOfSpell": 31, "Body Mass Index at Booking": 28.5, "Glucoselevel0minblood": 5.2},
        {"fetal_estimated_ac_cm": 31.2, "estimated_fetal_weight_g": 2900.0, "fitted_ellipse": {}},
        risk_dict={"gdm_probability": 0.08})
    print(r["fused_feature_vector"])
