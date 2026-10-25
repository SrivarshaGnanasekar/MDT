import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.maternal_predictor import MaternalGDMPredictor
from fetal.ac_predict import FetalACPredictor
from fusion.feature_fusion import MultimodalFeatureFusion
from digital_twin.maternal_fetal_twin import MaternalFetalDigitalTwin
from digital_twin.ctgan_twin import CTGANDigitalTwin
from models.macrosomia_monitor import MacrosomiaRiskMonitor
from recommendation.engine import ClinicalRecommendationEngine
from digital_twin.neural_ode import FetalMaternalODEPredictor

# Database persistence module
from database.mongo import (
    db_manager,
    register_user,
    authenticate_user,
    save_patient_twin,
    get_patient_twin,
    log_realtime_vital,
    get_vitals_history,
    list_all_patients
)

app = FastAPI(
    title="Multimodal Maternal-Fetal Digital Twin API",
    description="REST API for GDM Risk Prediction, Fetal Ultrasound Segmentation, Biometrics Extraction, Multimodal Fusion, MongoDB Atlas Digital Twin Persistence, and Real-Time Vitals Tracking.",
    version="2.1.0"
)

# Enable CORS for Streamlit / Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Component Instances
maternal_predictor = MaternalGDMPredictor()
fetal_ac_predictor = FetalACPredictor(use_new_model=True)
fusion_module = MultimodalFeatureFusion()
ctgan_twin = CTGANDigitalTwin()
macrosomia_monitor = MacrosomiaRiskMonitor()
rec_engine = ClinicalRecommendationEngine()
ode_predictor = FetalMaternalODEPredictor()


# Helper to fetch or create a twin from MongoDB Atlas
def load_or_create_twin(patient_id: str) -> MaternalFetalDigitalTwin:
    twin_data = get_patient_twin(patient_id)
    if twin_data:
        return MaternalFetalDigitalTwin.from_dict(twin_data)
    
    twin = MaternalFetalDigitalTwin(patient_id=patient_id)
    save_patient_twin(patient_id, twin.to_dict())
    return twin


# --- Pydantic Data Schemas ---
class MaternalInput(BaseModel):
    patient_id: str = "PATIENT_001"
    age: float = Field(30.0, description="Age of mother at booking")
    bmi: float = Field(27.5, description="Body Mass Index at booking")
    glucose_blood: float = Field(5.8, description="Blood glucose level")
    glucose_0min: Optional[float] = Field(5.1, description="Fasting OGTT 0 min")
    glucose_120min: Optional[float] = Field(7.8, description="OGTT 120 min")
    systolic_bp: Optional[float] = Field(120.0, description="Systolic Blood Pressure")
    diastolic_bp: Optional[float] = Field(80.0, description="Diastolic Blood Pressure")
    gestational_age_weeks: float = Field(24.0, description="Gestation at booking in weeks")
    obese: Optional[str] = Field("NO", description="Obesity status (YES/NO)")
    ethnicity: Optional[str] = Field("WEU", description="NHS Ethnicity code (e.g. WEU, GBR, IND, etc.)")

class FusionInput(BaseModel):
    patient_id: str = "PATIENT_001"
    maternal_features: Dict[str, Any]
    fetal_features: Dict[str, Any]

class MacrosomiaInput(BaseModel):
    gdm_probability: float = Field(..., ge=0.0, le=1.0)
    gestational_age_weeks: float = Field(28.0, ge=12.0, le=42.0)
    fetal_biometrics: Optional[Dict[str, Any]] = None
    measured_ac_cm: Optional[float] = Field(
        None,
        ge=0.5,
        le=44.0,
        description=(
            "Sonographer-measured fetal abdominal circumference in centimetres, "
            "obtained with a calibrated scanner using the standard cross-sectional "
            "technique.  This is a growth-context input only — it does NOT affect "
            "the macrosomia_probability output from the stacking model.  "
            "May come from a sonographer or the ACOUSLIC-AI Attention U-Net path "
            "(research measurement)."
        )
    )

class DigitalTwinForecastInput(BaseModel):
    gestational_age_weeks: float = Field(24.0, ge=12.0, le=40.0)
    current_ac_cm: float = Field(..., ge=10.0, le=45.0)
    gdm_probability: float = Field(..., ge=0.0, le=1.0)
    current_glucose: float = Field(75.0)
    current_bmi: float = Field(20.6)

class CTGANGenerateInput(BaseModel):
    num_samples: int = Field(100, ge=10, le=1000)

class UserRegisterInput(BaseModel):
    username: str
    password: str
    full_name: str
    role: str = Field("Patient", description="Role: 'Patient' or 'Clinician'")
    patient_id: Optional[str] = None
    email: Optional[str] = ""

class UserLoginInput(BaseModel):
    username: str
    password: str

class VitalLogInput(BaseModel):
    patient_id: str
    gestational_age_weeks: float
    systolic_bp: Optional[float] = None
    diastolic_bp: Optional[float] = None
    glucose_blood: Optional[float] = None
    glucose_0min: Optional[float] = None
    glucose_120min: Optional[float] = None
    weight_kg: Optional[float] = None
    bmi: Optional[float] = None
    notes: Optional[str] = ""

# --- System & DB Status Endpoints ---

@app.get("/")
def root():
    return {
        "service": "Multimodal Maternal-Fetal Digital Twin API",
        "status": "online",
        "version": "2.1.0",
        "database": db_manager.get_connection_status()
    }

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "database": db_manager.get_connection_status()
    }

@app.get("/api/db/status")
def db_status():
    return db_manager.get_connection_status()

# --- Authentication & Patient Management Endpoints ---

@app.post("/api/auth/register")
def register(data: UserRegisterInput):
    """Registers a new patient or clinician account in MongoDB Atlas."""
    try:
        user_profile = register_user(
            username=data.username,
            password=data.password,
            full_name=data.full_name,
            role=data.role,
            patient_id=data.patient_id,
            email=data.email or ""
        )
        # Initialize Digital Twin document if patient
        if data.role == "Patient":
            load_or_create_twin(user_profile["patient_id"])

        return {
            "message": "User registered successfully",
            "user": user_profile
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/auth/login")
def login(data: UserLoginInput):
    """Authenticates user against MongoDB Atlas stored password hash."""
    user_profile = authenticate_user(data.username, data.password)
    if not user_profile:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    return {
        "message": "Login successful",
        "user": user_profile
    }

@app.get("/api/patients")
def get_patients_list():
    """Lists all registered patients for clinician view."""
    return {
        "patients": list_all_patients()
    }

# --- Real-Time Vitals Logging & Digital Twin Sync Endpoints ---

@app.post("/api/patient/vitals/log")
def log_vital_observation(data: VitalLogInput):
    """Logs real-time vital observation to MongoDB Atlas and updates digital twin state."""
    vital_dict = data.dict()
    logged_entry = log_realtime_vital(data.patient_id, vital_dict)

    # Load twin and update maternal state & visit record
    twin = load_or_create_twin(data.patient_id)
    
    maternal_update = {}
    if data.systolic_bp: maternal_update["SystolicBloodPressureCuff"] = data.systolic_bp
    if data.diastolic_bp: maternal_update["Diastolic Blood Pressure"] = data.diastolic_bp
    if data.glucose_blood: maternal_update["Glucoselevelblood"] = data.glucose_blood
    if data.glucose_0min: maternal_update["Glucoselevel0minblood"] = data.glucose_0min
    if data.glucose_120min: maternal_update["Glucoselevel120minblood"] = data.glucose_120min
    if data.bmi: maternal_update["Body Mass Index at Booking"] = data.bmi
    if data.gestational_age_weeks: maternal_update["Gestation at booking (Weeks)"] = data.gestational_age_weeks

    # Evaluate prediction if glucose/bmi available
    gdm_prob = None  # no fabricated default: unknown until predicted
    gdm_level = "Low Risk"
    booking_kwargs = {}
    if "Glucoselevelblood" in maternal_update or "Body Mass Index at Booking" in maternal_update:
        pred_res = maternal_predictor.predict_gdm_risk({
            "AgeAtStartOfSpell": 30.0,
            "Body Mass Index at Booking": maternal_update.get("Body Mass Index at Booking", 27.5),
            "Glucoselevelblood": maternal_update.get("Glucoselevelblood", 5.5),
            "Glucoselevel0minblood": maternal_update.get("Glucoselevel0minblood", 5.0),
            "Glucoselevel120minblood": maternal_update.get("Glucoselevel120minblood", 7.5),
            "SystolicBloodPressureCuff": maternal_update.get("SystolicBloodPressureCuff", 120.0),
            "Diastolic Blood Pressure": maternal_update.get("Diastolic Blood Pressure", 80.0),
            "Gestation at booking (Weeks)": data.gestational_age_weeks,
            "Obese?": "YES" if maternal_update.get("Body Mass Index at Booking", 27.5) >= 30 else "NO"
        })
        gdm_prob = pred_res["gdm_probability"]
        gdm_level = pred_res["risk_level"]
        mac = maternal_predictor.predict_macrosomia_risk(maternal_update)
        booking_kwargs = {"macrosomia_prob": mac["macrosomia_probability"], "booking_flagged": mac["booking_flagged"]}

    twin.update_maternal_state(maternal_update, gdm_prob=gdm_prob)

    # Evaluate macrosomia tier
    macro_res = macrosomia_monitor.evaluate_risk(
        maternal_gdm_prob=gdm_prob,
        gestational_age_weeks=data.gestational_age_weeks,
        fetal_biometrics=twin.fetal_state,
        **booking_kwargs,
    )

    # Record visit
    twin.record_visit(
        gestational_age_weeks=data.gestational_age_weeks,
        gdm_risk_prob=gdm_prob if gdm_prob is not None else float("nan"),
        macrosomia_tier=macro_res["macrosomia_risk_tier"],
        notes=data.notes or "Real-time vital entry"
    )

    # Update recommendations
    recs = rec_engine.generate_recommendations(
        gdm_risk_level=gdm_level,
        macrosomia_tier=macro_res["risk_tier"],
        gestational_age_weeks=data.gestational_age_weeks
    )
    rec_list = recs.get("recommendations", recs.get("physician_action_plan", []) + recs.get("patient_guidance", []))
    twin.set_recommendations(rec_list)

    # Persist updated twin back to MongoDB Atlas
    save_patient_twin(data.patient_id, twin.to_dict())

    return {
        "message": "Vital observation logged and Digital Twin updated successfully",
        "vital_log": logged_entry,
        "gdm_probability": gdm_prob,
        "macrosomia_tier": macro_res["macrosomia_risk_tier"],
        "twin": twin.to_dict()
    }

@app.get("/api/patient/{patient_id}/vitals/history")
def get_patient_vitals_history(patient_id: str):
    """Retrieves time-series vitals history for a patient from MongoDB Atlas."""
    return {
        "patient_id": patient_id,
        "history": get_vitals_history(patient_id)
    }

# --- Clinical ML & Prediction Endpoints ---

@app.post("/api/maternal/predict")
def predict_maternal_gdm(data: MaternalInput):
    """Predicts maternal GDM risk probability and persists to MongoDB Atlas."""
    raw_dict = {
        "AgeAtStartOfSpell": data.age,
        "Body Mass Index at Booking": data.bmi,
        "Glucoselevelblood": data.glucose_blood,
        "Glucoselevel0minblood": data.glucose_0min,
        "Glucoselevel120minblood": data.glucose_120min,
        "SystolicBloodPressureCuff": data.systolic_bp,
        "Diastolic Blood Pressure": data.diastolic_bp,
        "Gestation at booking (Weeks)": data.gestational_age_weeks,
        "Obese?": data.obese
    }
    
    res = maternal_predictor.predict_gdm_risk(raw_dict)
    
    # Load and update twin in MongoDB Atlas
    twin = load_or_create_twin(data.patient_id)
    twin.update_maternal_state(raw_dict, gdm_prob=res["gdm_probability"])
    save_patient_twin(data.patient_id, twin.to_dict())
    
    return {
        "patient_id": data.patient_id,
        "prediction": res
    }

@app.post("/api/fetal/upload_ultrasound")
async def upload_ultrasound_file(file: UploadFile = File(...), pixel_spacing_mm: float = 0.28,
                                 gestational_age_weeks: float = 28.0):
    """
    Runs the ACOUSLIC-AI Attention U-Net on an uploaded frame (PNG/JPG, or a 2-D .npy array)
    and returns AC (ellipse fit + Ramanujan perimeter), AC-only Hadlock EFW and the AC
    screening result. pixel_spacing_mm defaults to the ACOUSLIC-AI value (0.28 mm/pixel).
    """
    filename = file.filename
    file_path = Path(tempfile.gettempdir()) / filename
    file_path.write_bytes(await file.read())
    if filename.endswith(".npy"):
        image_np = np.load(file_path, allow_pickle=False)
    else:
        import cv2
        image_np = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if image_np is None:
            raise HTTPException(status_code=400, detail="Invalid image file format")
    res = fetal_ac_predictor.predict(image_np, pixel_spacing_mm=pixel_spacing_mm,
                                     gestational_age_weeks=gestational_age_weeks)
    keep = ["actual_ac_cm", "actual_ac_mm", "expected_ac_cm", "delta_ac_relative_pct", "efw_hadlock_g",
            "screening_tier", "model_confidence", "plane_quality", "clinical_interpretation"]
    return {"filename": filename, "image_shape": list(image_np.shape),
            **{k: res.get(k) for k in keep}}

@app.post("/api/fusion/fuse")
def fuse_multimodal_features(data: FusionInput):
    """Fuses maternal clinical features + fetal biometric features."""
    fused = fusion_module.fuse_features(
        maternal_dict=data.maternal_features,
        fetal_dict=data.fetal_features,
        patient_id=data.patient_id
    )
    return fused

@app.get("/api/twin/state/{patient_id}")
def get_digital_twin_state(patient_id: str):
    """Retrieves current Digital Twin state from MongoDB Atlas for a patient."""
    twin = load_or_create_twin(patient_id)
    return twin.to_dict()

@app.post("/api/twin/update")
def update_digital_twin_state(patient_id: str, maternal_data: Optional[Dict[str, Any]] = None, fetal_data: Optional[Dict[str, Any]] = None):
    """Updates Digital Twin state in MongoDB Atlas with new maternal or fetal data."""
    twin = load_or_create_twin(patient_id)
    
    if maternal_data:
        twin.update_maternal_state(maternal_data)
    if fetal_data:
        twin.update_fetal_state(fetal_data)
        
    save_patient_twin(patient_id, twin.to_dict())
    return twin.to_dict()

@app.post("/api/twin/ctgan/generate")
def generate_synthetic_maternal_cohort(data: CTGANGenerateInput):
    """Generates synthetic maternal profiles via CTGAN Digital Twin."""
    synth_df = ctgan_twin.generate_synthetic_cohort(num_samples=data.num_samples)
    return {
        "generated_count": len(synth_df),
        "sample_records": synth_df.head(5).to_dict(orient="records")
    }

@app.post("/api/macrosomia/evaluate")
def evaluate_macrosomia_risk(data: MacrosomiaInput):
    """
    Evaluates macrosomia risk tier and AC growth context.

    Returns two logically separate outputs:
      - ``macrosomia_probability`` / ``risk_tier``: from the maternal stacking
        ensemble model (tabular features only, no AC/imaging).
      - ``ac_source``, ``ac_measured_cm``, ``ac_expected_cm``,
        ``ac_deviation_cm``, ``ac_deviation_pct``, ``ac_context_note``:
        descriptive AC growth context relative to the Choi 2016 reference
        median (DOI: 10.5468/ogs.2016.59.2.91).  These are NOT probability
        outputs and are only populated when ``measured_ac_cm`` is supplied.

    ``measured_ac_cm`` may be sonographer-measured or from the ACOUSLIC-AI
    Attention U-Net path (research measurement).
    Without booking inputs this endpoint returns Stage-2 (published AC/EFW
    criterion) information only.
    """
    biometrics = dict(data.fetal_biometrics or {})
    if data.measured_ac_cm is not None:
        # Route the explicitly supplied AC under the canonical key
        biometrics["measured_ac_cm"] = data.measured_ac_cm

    res = macrosomia_monitor.evaluate_risk(
        maternal_gdm_prob=data.gdm_probability,
        gestational_age_weeks=data.gestational_age_weeks,
        fetal_biometrics=biometrics,
    )
    return res

from models.digital_twin_forecast import run_digital_twin_screening

@app.post("/api/digital_twin/forecast")
def digital_twin_forecast(data: DigitalTwinForecastInput):
    """
    Executes the Digital Twin Neural ODE Forecast to predict fetal AC at Week 40
    using RK4 integration over maternal glucose and BMI trajectories.
    """
    # Simple trajectory simulation for the API based on current values
    # In a full production system, these might be actual Neural ODE functions.
    # Here we mock steady or increasing trajectories based on GDM prob.
    
    # If GDM prob is high, mock an aggressive glucose trajectory
    if data.gdm_probability > 0.65:
        glucose_fn = lambda t: data.current_glucose * 1.05  # slightly rising
        bmi_fn = lambda t: data.current_bmi * 1.02
    else:
        glucose_fn = lambda t: data.current_glucose * 0.95  # controlled
        bmi_fn = lambda t: data.current_bmi
        
    res = run_digital_twin_screening(
        t_c=data.gestational_age_weeks,
        ac_obs=data.current_ac_cm,
        p_gdm=data.gdm_probability,
        glucose_fn=glucose_fn,
        bmi_fn=bmi_fn,
        alpha=0.015,
        beta=0.01
    )
    return res

@app.post("/api/recommendations")
def get_clinical_recommendations(gdm_risk_level: str, macrosomia_tier: str, gestational_age_weeks: float = 28.0):
    """Generates clinical recommendations."""
    recs = rec_engine.generate_recommendations(
        gdm_risk_level=gdm_risk_level,
        macrosomia_tier=macrosomia_tier,
        gestational_age_weeks=gestational_age_weeks
    )
    return recs

@app.post("/api/twin/trajectory")
def get_twin_trajectory(data: MaternalInput):
    """Generates longitudinal patient-specific trajectories using Neural ODEs."""
    raw_dict = {
        "AgeAtStartOfSpell": data.age,
        "Body Mass Index at Booking": data.bmi,
        "Glucoselevelblood": data.glucose_blood,
        "Glucoselevel0minblood": data.glucose_0min,
        "Glucoselevel120minblood": data.glucose_120min,
        "SystolicBloodPressureCuff": data.systolic_bp,
        "Diastolic Blood Pressure": data.diastolic_bp,
        "Gestation at booking (Weeks)": data.gestational_age_weeks,
        "Obese?": data.obese,
        "Ethnicity": data.ethnicity or "WEU"
    }
    df_traj = ode_predictor.predict_trajectory(raw_dict)
    return df_traj.to_dict(orient="records")

@app.post("/api/maternal/predict_macrosomia")
def predict_maternal_macrosomia(data: MaternalInput):
    """Stage-1 validated early-booking macrosomia risk (birthweight >= 4000 g)."""
    raw_dict = {
        "AgeAtStartOfSpell": data.age,
        "Body Mass Index at Booking": data.bmi,
        "Glucoselevelblood": data.glucose_blood,
        "Glucoselevel0minblood": data.glucose_0min,
        "Glucoselevel120minblood": data.glucose_120min,
        "SystolicBloodPressureCuff": data.systolic_bp,
        "Diastolic Blood Pressure": data.diastolic_bp,
        "Gestation at booking (Weeks)": data.gestational_age_weeks,
        "Obese?": data.obese,
        "Ethnicity": data.ethnicity or "WEU"
    }
    res = maternal_predictor.predict_macrosomia_risk(raw_dict)
    return res

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
