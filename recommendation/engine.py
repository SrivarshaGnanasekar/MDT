import os
import sys
from pathlib import Path
from typing import Dict, Any, List

class ClinicalRecommendationEngine:
    """
    Evidence-based recommendation engine for Maternal-Fetal Digital Twin.
    Provides actionable clinical advice for:
      - Maternal Glucose Monitoring & Dietary Management
      - Fetal Growth Ultrasound Follow-up Intervals
      - Macrosomia Prevention & Obstetric Risk Mitigation
    """

    def generate_recommendations(
        self,
        gdm_risk_level: str,
        macrosomia_tier: str,
        gestational_age_weeks: float = 28.0,
        fetal_biometrics: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generates clinical recommendations categorized for Physician and Patient.
        """
        fetal_biometrics = fetal_biometrics or {}
        physician_recs = []
        patient_recs = []
        
        # 1. Maternal GDM Risk Recommendations
        if gdm_risk_level == "High Risk":
            physician_recs.append("📋 Schedule immediate 2-hour 75g Oral Glucose Tolerance Test (OGTT).")
            physician_recs.append("🩸 Initiate 4-times daily capillary self-monitoring of blood glucose (Fastings & 1h postprandial).")
            physician_recs.append("👩‍⚕️ Refer patient to Maternal-Fetal Medicine (MFM) specialist and endocrine care.")
            patient_recs.append("🥗 Follow a low-glycemic index medical nutrition therapy (MNT) plan.")
            patient_recs.append("🏃‍♀️ Engage in 30 minutes of daily moderate post-meal walking.")
        elif gdm_risk_level == "Moderate Risk":
            physician_recs.append("🧪 Perform early HbA1c screening and repeat OGTT at 24-28 weeks gestation.")
            physician_recs.append("📊 Track maternal weight gain against IOM recommended targets.")
            patient_recs.append("🍏 Maintain a balanced carbohydrate-distributed diet.")
            patient_recs.append("📝 Keep a daily food and activity journal.")
        else:
            physician_recs.append("✅ Routine GDM screening at standard 24-28 weeks antenatal visit.")
            patient_recs.append("🥗 Continue standard healthy pregnancy nutrition and active lifestyle.")

        # 2. Fetal Growth & Macrosomia Recommendations
        if macrosomia_tier == "High Risk":
            physician_recs.append("🔍 Perform detailed serial ultrasound growth scans every 2 weeks.")
            physician_recs.append("📏 Monitor fetal abdominal circumference (AC) percentile trajectory.")
            physician_recs.append("⚠️ Evaluate delivery timing and labor onset plan to minimize shoulder dystocia risk.")
            patient_recs.append("📞 Report any sudden changes in fetal movement patterns to your obstetrician.")
        elif macrosomia_tier == "Moderate Risk":
            physician_recs.append("📈 Repeat fetal biometrics ultrasound scan in 3-4 weeks.")
            physician_recs.append("🩺 Ensure tight maternal postprandial glucose control (<120 mg/dL).")
            patient_recs.append("💧 Stay well-hydrated and attend scheduled ultrasound follow-ups.")
        else:
            physician_recs.append("🟢 Fetal growth trajectory is normal; continue routine 3rd trimester growth monitoring.")
            patient_recs.append("👶 Normal fetal organ growth detected.")

        all_recs = physician_recs + patient_recs
        return {
            "gdm_risk_level": gdm_risk_level,
            "macrosomia_tier": macrosomia_tier,
            "gestational_age_weeks": gestational_age_weeks,
            "physician_action_plan": physician_recs,
            "patient_guidance": patient_recs,
            "recommendations": all_recs,
            "summary": f"Maternal Risk: {gdm_risk_level} | Macrosomia Monitoring: {macrosomia_tier}"
        }


if __name__ == "__main__":
    engine = ClinicalRecommendationEngine()
    recs = engine.generate_recommendations(gdm_risk_level="High Risk", macrosomia_tier="Moderate Risk", gestational_age_weeks=28.0)
    print("Generated Recommendations Summary:", recs["summary"])
