import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

class MaternalFetalDigitalTwin:
    """
    Multimodal Maternal-Fetal Digital Twin State Manager.
    Stores and manages:
      - Maternal Clinical State
      - Fetal Anatomical State
      - Synthetic Patient Profile Comparison Context
      - Historical Timeline & Visits
      - Dynamic Risk Trends (GDM & Macrosomia)
      - Tailored Clinical Recommendations
    """
    def __init__(self, patient_id: str = "TWIN_DEFAULT"):
        self.patient_id = patient_id
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        
        # Core State Components
        self.maternal_state: Dict[str, Any] = {}
        self.fetal_state: Dict[str, Any] = {}
        self.fused_state: Dict[str, Any] = {}
        self.synthetic_info: Dict[str, Any] = {}
        self.prediction_history: List[Dict[str, Any]] = []
        self.monitoring_timeline: List[Dict[str, Any]] = []
        self.risk_trends: Dict[str, List[Dict[str, Any]]] = {
            "gdm_risk": [],
            "macrosomia_risk": [],
            "fetal_growth": []
        }
        self.recommendations: List[str] = []

    def update_maternal_state(self, clinical_data: Dict[str, Any], gdm_prob: Optional[float] = None):
        """Updates maternal clinical variables and GDM risk assessment."""
        self.maternal_state.update(clinical_data)
        if gdm_prob is not None:
            self.maternal_state["gdm_risk_probability"] = float(gdm_prob)
        self.updated_at = datetime.now().isoformat()

    def update_fetal_state(self, organ_biometrics: Dict[str, Any], ultrasound_file: Optional[str] = None):
        """Updates fetal ultrasound biometry (AC, EFW, model confidence) from the ACOUSLIC-AI path."""
        self.fetal_state.update(organ_biometrics)
        if ultrasound_file:
            self.fetal_state["ultrasound_filename"] = ultrasound_file
        self.updated_at = datetime.now().isoformat()

    def update_fused_state(self, fused_features: Dict[str, Any]):
        """Stores the latest maternal-fetal fused feature vector."""
        self.fused_state = dict(fused_features)
        self.updated_at = datetime.now().isoformat()

    def record_visit(
        self,
        gestational_age_weeks: float,
        gdm_risk_prob: float,
        macrosomia_tier: str,
        fetal_ac_cm: Optional[float] = None,
        notes: str = ""
    ):
        """Records a new clinical monitoring visit to the timeline and updates risk trends."""
        visit_record = {
            "timestamp": datetime.now().isoformat(),
            "gestational_age_weeks": float(gestational_age_weeks),
            "gdm_risk_probability": float(gdm_risk_prob),
            "macrosomia_risk_tier": macrosomia_tier,
            "fetal_ac_cm": float(fetal_ac_cm) if fetal_ac_cm is not None else None,
            "notes": notes
        }
        
        self.monitoring_timeline.append(visit_record)
        
        # Append to trends
        self.risk_trends["gdm_risk"].append({
            "gestational_week": gestational_age_weeks,
            "probability": gdm_risk_prob
        })
        self.risk_trends["macrosomia_risk"].append({
            "gestational_week": gestational_age_weeks,
            "risk_tier": macrosomia_tier
        })
        if fetal_ac_cm is not None:
            self.risk_trends["fetal_growth"].append({
                "gestational_week": gestational_age_weeks,
                "ac_cm": fetal_ac_cm
            })
            
        self.updated_at = datetime.now().isoformat()

    def set_recommendations(self, recs: List[str]):
        """Sets actionable clinical recommendations."""
        self.recommendations = recs

    def set_synthetic_info(self, synth_summary: Dict[str, Any]):
        """Stores reference statistics from CTGAN synthetic cohort comparisons."""
        self.synthetic_info = synth_summary

    def to_dict(self) -> Dict[str, Any]:
        """Serializes Digital Twin state to dictionary format."""
        return {
            "patient_id": self.patient_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "maternal_state": self.maternal_state,
            "fetal_state": self.fetal_state,
            "fused_state": self.fused_state,
            "synthetic_info": self.synthetic_info,
            "prediction_history": self.prediction_history,
            "monitoring_timeline": self.monitoring_timeline,
            "risk_trends": self.risk_trends,
            "recommendations": self.recommendations
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MaternalFetalDigitalTwin':
        """Instantiates Digital Twin object from dictionary."""
        twin = cls(patient_id=data.get("patient_id", "TWIN_001"))
        twin.created_at = data.get("created_at", datetime.now().isoformat())
        twin.updated_at = data.get("updated_at", datetime.now().isoformat())
        twin.maternal_state = data.get("maternal_state", {})
        twin.fetal_state = data.get("fetal_state", {})
        twin.fused_state = data.get("fused_state", {})
        twin.synthetic_info = data.get("synthetic_info", {})
        twin.prediction_history = data.get("prediction_history", [])
        twin.monitoring_timeline = data.get("monitoring_timeline", [])
        twin.risk_trends = data.get("risk_trends", {"gdm_risk": [], "macrosomia_risk": [], "fetal_growth": []})
        twin.recommendations = data.get("recommendations", [])
        return twin

if __name__ == "__main__":
    twin = MaternalFetalDigitalTwin(patient_id="PATIENT_88")
    twin.update_maternal_state({"Age": 32, "BMI": 29.1}, gdm_prob=0.72)
    twin.update_fetal_state({"fetal_estimated_ac_cm": 31.2, "estimated_fetal_weight_g": 2900.0})
    twin.record_visit(gestational_age_weeks=28, gdm_risk_prob=0.72, macrosomia_tier="Moderate", fetal_ac_cm=24.2)
    twin.set_recommendations(["Schedule 2-hour OGTT test", "Ultrasound re-scan in 2 weeks"])
    
    d = twin.to_dict()
    print("Serialized Twin Patient ID:", d["patient_id"])
    print("Timeline visits count:", len(d["monitoring_timeline"]))
