import os
from pathlib import Path
from dotenv import load_dotenv

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "maternal" / "raw" / "GDM Dataset.xlsx"
PROCESSED_DIR = DATA_DIR / "maternal" / "processed"
SYNTHETIC_DIR = DATA_DIR / "maternal" / "synthetic"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure all key directories exist
for path in [PROCESSED_DIR, SYNTHETIC_DIR, MODELS_DIR, REPORTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Database Configuration (MongoDB Atlas with SQLite/Local fallback)
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/maternal_twin")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "maternal_twin")

# Collection names
MONGO_COLL_USERS = "users"
MONGO_COLL_PATIENT_TWINS = "patient_twins"
MONGO_COLL_VITALS_LOGS = "vitals_logs"
MONGO_COLL_VISITS = "clinical_visits"

POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "maternal_twin")

POSTGRES_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
SQLITE_URL = f"sqlite:///{BASE_DIR / 'database' / 'maternal_twin.db'}"

# Fallback check
DATABASE_URL = os.getenv("DATABASE_URL", SQLITE_URL)

# --- Feature Columns Configuration ---
TARGET_COLUMN = "Gestational Diabetes"

# Columns to drop entirely
COLS_TO_DROP = [
    "UID",  # Patient unique identifier (redundant for modeling)
    "O_Thyroidfunctionblood",  # 100% missing values
    "LOS mother after delivery.1"  # Duplicate of 'LOS mother after delivery'
]

# Columns containing laboratory errors or special strings needing conversion
LAB_ERROR_COLS = [
    "Glucoselevelblood",
    "Glucoselevel0minblood",
    "Glucoselevel120minblood",
    "VitaminDlevelblood"
]

# --- The 12 Principal Clinical Predictors (Derived from PCA Scree & Clinical Guidelines) ---
# Height & Weight are condensed into Body Mass Index (BMI = Weight / Height^2).
# Socioeconomic IMD rank is included as a booking-time deprivation indicator.
# (NICE NG3 GDM risk factors cover BMI, ethnicity, previous macrosomia/GDM and
#  family history; NG3 does NOT name deprivation/IMD, so IMD needs its own source.)
# --- Early Booking Clinical Predictors (available at <= 12 weeks) ---
# Note: Diagnostic OGTT glucose values (Glucoselevel0minblood and Glucoselevel120minblood)
# are EXCLUDED from early booking prediction to prevent target definition leakage.
STATIC_FEATURES = [
    "AgeAtStartOfSpell",
    "Gestation at booking (Weeks)",
    "Body Mass Index at Booking",
    "Gravida",
    "Parity",
    "No_Of_previous_Csections",
    "Index of Multiple Deprivation Rank",
    "Ethnicity"
]

DYNAMIC_FEATURES = [
    "SystolicBloodPressureCuff",
    "Diastolic Blood Pressure"
]

CORE_12_PREDICTORS = STATIC_FEATURES + DYNAMIC_FEATURES
EARLY_BOOKING_PREDICTORS = CORE_12_PREDICTORS

# Diagnostic OGTT features (used only in separate leakage demonstration benchmark)
DIAGNOSTIC_OGTT_FEATURES = [
    "Glucoselevel0minblood",
    "Glucoselevel120minblood"
]

# Outcome features measured at delivery / post-delivery (ONLY used in Digital Twin cohorts, NOT for early prediction)
OUTCOME_FEATURES = [
    "Delivery_Outcome",
    "OnsetofLabourMethod",
    "Contraction frequency prior to delivery",
    "PrimaryIndicationforCaesarean",
    "Category Caesarean Section",
    "Perineal care",
    "EstimatedTotalBloodLoss",
    "Gestation",
    "Severely Premature?",
    "Gestation (Days)",
    "BabyBirthWeight",
    "Presence of meconium",
    "BW Centile",
    "Shoulder Dystocia",
    "SCBU admission (Yes/NO)",
    "LOS mother after delivery",
    "Sex",
    "Still_Birth",
    "APGAR_Score_5",
    "TotalApgarScoreat10minutes",
    "Maternity_Month"
]

# --- Fetal Attention U-Net Model Configurations ---
USE_NEW_ATTUNET_MODEL = True
NEW_MODEL_PATH = MODELS_DIR / "new_model" / "best_attunet.pth"
CONF_MIN = 0.98  # provisional, based on one sweep, needs validation on more data

# --- Modeling Constants ---
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Generative Digital Twin Parameters
CTGAN_EPOCHS = 100  # Default epochs for training (can adjust in Streamlit UI)
CTGAN_BATCH_SIZE = 500

# Predictor Hyperparameters (standard defaults)
MODEL_PARAMS = {
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 12,
        "random_state": RANDOM_STATE,
        "class_weight": "balanced",
        "n_jobs": -1
    },
    "xgboost": {
        "n_estimators": 250,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": RANDOM_STATE,
        "eval_metric": "logloss",
        "n_jobs": -1
    },
    "lightgbm": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.03,
        "random_state": RANDOM_STATE,
        "class_weight": "balanced",
        "verbose": -1,
        "n_jobs": -1
    },
    "catboost": {
        "iterations": 250,
        "depth": 6,
        "learning_rate": 0.03,
        "random_state": RANDOM_STATE,
        "auto_class_weights": "Balanced",
        "verbose": 0,
        "thread_count": -1
    },
    "logistic_regression": {
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE
    }
}
