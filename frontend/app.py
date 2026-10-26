import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import PROCESSED_DIR, DATA_DIR, MODELS_DIR
from models.maternal_predictor import MaternalGDMPredictor
from fetal.ac_predict import FetalACPredictor
from fetal.visualization import create_segmentation_overlay, draw_fitted_ellipse
from fusion.feature_fusion import MultimodalFeatureFusion
from digital_twin.maternal_fetal_twin import MaternalFetalDigitalTwin
from digital_twin.ctgan_twin import CTGANDigitalTwin
from models.macrosomia_monitor import MacrosomiaRiskMonitor
from explainability.maternal_shap import MaternalSHAPExplainer
from explainability.fetal_vis import generate_fetal_explainability_figure
from recommendation.engine import ClinicalRecommendationEngine
from digital_twin.neural_ode import FetalMaternalODEPredictor

# MongoDB Atlas Integration
from database.mongo import (
    db_manager,
    register_user,
    authenticate_user,
    save_patient_twin,
    get_patient_twin,
    log_realtime_vital,
    get_vitals_history,
    list_all_patients,
    list_all_users,
    get_connection_status
)

# Page Config
st.set_page_config(
    page_title="MDT-GEN | Multimodal Maternal-Fetal Digital Twin",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    :root {
        --ink: #172033;
        --muted: #667085;
        --line: #E4E7EC;
        --surface: #FFFFFF;
        --canvas: #F6F8FB;
        --navy: #183B56;
        --teal: #147D82;
        --teal-soft: #E7F5F4;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background: var(--canvas);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: rgba(246, 248, 251, 0.92);
    }

    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 1.1rem 1rem 1.5rem;
    }

    [data-testid="stSidebar"] hr {
        margin: 1rem 0;
        border-color: var(--line);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--muted);
    }

    [data-testid="stSidebar"] .stRadio > label {
        color: var(--ink);
        font-size: 0.73rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 0.18rem;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        border-radius: 7px;
        padding: 0.48rem 0.55rem;
        transition: background 0.15s ease;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: #F2F6F8;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: var(--teal-soft);
        color: #0D6266;
        font-weight: 700;
    }

    [data-testid="stSidebar"] .stButton > button 
    {
        color: #FFFFFF !important;
        min-height: 2.35rem;
        box-shadow: none;
    }

    form#admin_create_user_form button {
        background: var(--navy) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    .app-kicker {
        color: var(--teal);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .landing-nav, .auth-brand, .landing-footer {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.6rem 0 2.8rem;
    }

    .brand-name {
        color: var(--navy);
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: 0.03em;
    }

    .brand-subtitle, .landing-footer span {
        color: var(--muted);
        font-size: 0.76rem;
        margin-top: 0.15rem;
    }

    .landing-links {
        display: flex;
        gap: 1.8rem;
        color: var(--muted);
        font-size: 0.82rem;
        font-weight: 600;
    }

    .landing-links span {
        cursor: default;
        transition: color 0.2s ease;
    }

    .landing-links span:first-child,
    .landing-links span:hover {
        color: var(--teal);
    }

    .landing-title, .auth-title {
        color: var(--navy);
        font-size: clamp(2.4rem, 4.3vw, 4.25rem);
        line-height: 1.02;
        letter-spacing: -0.045em;
        margin: 0 0 1.25rem;
        max-width: 650px;
    }

    .auth-title {
        font-size: clamp(2rem, 3vw, 3rem);
    }

    .landing-copy {
        color: var(--muted);
        font-size: 1.02rem;
        line-height: 1.75;
        max-width: 580px;
        margin-bottom: 1.7rem;
    }

    .landing-availability {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #38716E;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin: 1.35rem 0 0;
    }

    .landing-availability i {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #2EAA83;
        box-shadow: 0 0 0 4px rgba(46, 170, 131, 0.12);
    }

    .landing-image-wrapper {
        position: relative;
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(20, 125, 130, 0.25);
        box-shadow: 0 16px 40px rgba(10, 25, 41, 0.18), 0 0 20px rgba(20, 125, 130, 0.12);
        background: #0B111E;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .landing-image-wrapper:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 48px rgba(10, 25, 41, 0.28), 0 0 30px rgba(20, 125, 130, 0.2);
    }

    .landing-hero-img {
        width: 100%;
        height: auto;
        display: block;
        border-radius: 16px;
        object-fit: cover;
    }

    .landing-section-title {
        color: var(--navy);
        font-size: 1.4rem;
        font-weight: 800;
        margin: 5rem 0 1.4rem;
    }

    .feature-card {
        min-height: 185px;
        border: 1px solid #DCE8E8;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.6);
        padding: 1.15rem 1rem 0.85rem;
        box-shadow: 0 7px 18px rgba(24, 59, 86, 0.035);
    }

    .feature-card h3 {
        color: var(--ink);
        font-size: 1rem;
        margin: 0.8rem 0 0.45rem;
    }

    .feature-card p, .platform-band p {
        color: var(--muted);
        font-size: 0.86rem;
        line-height: 1.6;
    }

    .feature-mark {
        width: 28px;
        height: 28px;
        border: 2px solid var(--teal);
        border-radius: 8px;
        background: linear-gradient(135deg, #DFF4F0, #FFFFFF);
    }

    .platform-band {
        display: grid;
        grid-template-columns: 0.9fr 1.1fr;
        gap: 3rem;
        margin: 3rem 0;
        padding: 2rem 0;
        border-top: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
    }

    .platform-band h2 {
        color: var(--navy);
        font-size: 1.55rem;
        margin: 0;
    }

    .landing-footer {
        padding: 1.5rem 0;
        border-top: 1px solid var(--line);
    }

    .auth-brand {
        justify-content: flex-start;
        padding-bottom: 3rem;
    }

    .auth-card {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 8px 25px rgba(16, 24, 40, 0.06);
    }

    .auth-card h2 { color: var(--navy); margin: 0; }
    .auth-card > p { color: var(--muted); font-size: 0.88rem; margin: 0.4rem 0 1.5rem; }
    .auth-help { color: var(--muted); font-size: 0.8rem; text-align: center; margin-top: 1.2rem; }

    [data-testid="stTextInput"] label,
    [data-testid="stCheckbox"] label,
    [data-testid="stNumberInput"] label,
    [data-testid="stTextArea"] label,
    [data-testid="stSelectbox"] label,
    [data-testid="stSlider"] label,
    [data-testid="stSlider"] label p {
        color: var(--ink) !important;
        font-weight: 600;
    }

    [data-testid="stSlider"] [data-baseweb="slider"] {
        margin-top: 0.15rem;
    }

    [data-testid="stSlider"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSlider"] small {
        color: var(--muted) !important;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background: #FFFFFF !important;
        color: var(--ink) !important;
        border: 1px solid #C9D2DC !important;
        border-radius: 6px !important;
    }

    [data-testid="stTextInput"] input:focus,
    [data-testid="stNumberInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus {
        border-color: var(--teal) !important;
        box-shadow: 0 0 0 1px var(--teal) !important;
    }

    [data-testid="stCheckbox"] label span {
        color: var(--muted) !important;
    }

    .workspace-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 2rem;
        border-bottom: 1px solid var(--line);
        padding: 0.4rem 0 1.25rem;
        margin-bottom: 1.5rem;
    }

    .workspace-header h1, .page-title { color: var(--navy); font-size: 1.75rem; margin: 0; }
    .workspace-header p, .page-subtitle { color: var(--muted); font-size: 0.85rem; margin: 0.35rem 0 0; }
    .patient-identity { display: flex; flex-direction: column; align-items: flex-end; gap: 0.25rem; color: var(--ink); font-size: 0.82rem; }
    .patient-identity span { color: var(--muted); font-size: 0.76rem; }
    .system-status { display: inline-flex; align-items: center; gap: 0.35rem; color: #217A4B !important; }
    .system-status i { width: 7px; height: 7px; border-radius: 50%; background: #3AA76D; display: inline-block; }
    .kpi-card { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 1rem; min-height: 92px; }
    .kpi-card span, .kpi-card small { display: block; color: var(--muted); font-size: 0.73rem; }
    .kpi-card strong { display: inline-block; color: var(--navy); font-size: 1.55rem; margin: 0.35rem 0.3rem 0.15rem 0; }
    .page-title { margin-top: 0.5rem; }
    .page-subtitle { margin-bottom: 1.2rem; }

    @media (max-width: 760px) {
        .landing-links { display: none; }
        .landing-nav, .auth-brand { padding-bottom: 1.8rem; }
        .clinical-visual { min-height: 280px; margin-top: 1.5rem; }
        .platform-band { grid-template-columns: 1fr; gap: 0.5rem; }
        .workspace-header { flex-direction: column; gap: 0.8rem; }
        .patient-identity { align-items: flex-start; }
    }

    .hero-banner {
        background: var(--surface);
        border-radius: 10px;
        padding: 1.45rem 1.7rem;
        color: var(--ink);
        box-shadow: 0 2px 8px rgba(16, 24, 40, 0.04);
        margin: 0.3rem 0 1.5rem;
        border: 1px solid var(--line);
    }
    
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: 0;
        color: var(--navy);
        margin-bottom: 0.35rem;
    }
    
    .hero-subtitle {
        font-size: 0.9rem;
        color: var(--muted);
        font-weight: 400;
        max-width: 860px;
    }

    .glass-card {
        background: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px;
        padding: 1.2rem 1.3rem;
        box-shadow: 0 2px 8px rgba(16, 24, 40, 0.04);
        margin-bottom: 1rem;
    }
    
    .glass-card p, .glass-card span, .glass-card div {
        color: #0F172A;
    }
    
    .glass-card:hover {
        box-shadow: 0 5px 16px rgba(16, 24, 40, 0.07);
    }

    .metric-badge-high {
        background: #FEE2E2;
        color: #991B1B;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .metric-badge-moderate {
        background: #FEF3C7;
        color: #92400E;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .metric-badge-low {
        background: #D1FAE5;
        color: #065F46;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }

    .section-header {
        font-size: 1.12rem;
        font-weight: 800;
        color: #0F172A !important;
        margin-bottom: 0.85rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .stButton>button {
        background: var(--teal);
        color: white !important;
        font-weight: 600;
        border-radius: 7px;
        padding: 0.55rem 1rem;
        border: none;
        box-shadow: 0 3px 8px rgba(20, 125, 130, 0.2);
        transition: background 0.15s ease, box-shadow 0.15s ease;
    }

    .stButton>button:hover {
        background: black;
        box-shadow: 0 5px 12px rgba(20, 125, 130, 0.25);
    }

    [data-testid="stSidebar"] button,
    [data-testid="stForm"] button {
        background: var(--teal) !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] button *,
    [data-testid="stForm"] button * {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] button:hover,
    [data-testid="stForm"] button:hover {
        background: var(--navy) !important;
    }

    /* HIGH-CONTRAST METRIC CARDS */
    div[data-testid="stMetric"] {
        background: #F8FAFC !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px;
        padding: 0.85rem 1rem !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] *,
    div[data-testid="stMetric"] label {
        color: #334155 !important;
        font-weight: 700 !important;
        font-size: 0.82rem !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"] * {
        color: #0F172A !important;
        font-weight: 800 !important;
        font-size: 1.65rem !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricDelta"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] * {
        color: #0369A1 !important;
        font-weight: 700 !important;
    }

    /* HIGH-CONTRAST ALERT BOXES */
    div[data-testid="stAlert"], .stAlert {
        border-radius: 8px !important;
        padding: 0.9rem 1.1rem !important;
        background-color: #FEF3C7 !important;
        border: 1px solid #F59E0B !important;
    }

    div[data-testid="stAlert"] *, .stAlert * {
        color: #78350F !important;
        -webkit-text-fill-color: #78350F !important;
        font-size: 0.94rem !important;
        font-weight: 600 !important;
    }

    div[data-testid="stAlert"] p, .stAlert p,
    div[data-testid="stAlert"] span, .stAlert span,
    div[data-testid="stAlert"] div, .stAlert div {
        color: #78350F !important;
        -webkit-text-fill-color: #78350F !important;
        font-weight: 600 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 1.25rem;
        border-bottom: 1px solid var(--line);
    }

    .stTabs [data-baseweb="tab"] {
        color: var(--muted);
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        color: var(--teal);
    }
    
    .disclaimer-box {
        background: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        color: #78350F;
        font-size: 0.92rem;
        margin: 1rem 0;
    }

    .db-badge-online {
        background: #DCFCE7;
        color: #15803D;
        border: 1px solid #86EFAC;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.82rem;
        margin-bottom: 1rem;
    }

    .db-badge-offline {
        background: #FEF3C7;
        color: #B45309;
        border: 1px solid #FCD34D;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.82rem;
        margin-bottom: 1rem;
    }
    div[data-testid="stRadio"] > div {
        flex-direction: row !important;
        gap: 20px; /* Adjust spacing between options */
    }
    div[data-testid="stRadio"] label p {
        color: #31333F !important; /* Forces text color to be visible */
        white-space: nowrap;       /* Prevents messy text wrapping */
    }
</style>
""", unsafe_allow_html=True)

# Helper function to cache model loading
@st.cache_resource
def load_cached_components():
    mat_pred = MaternalGDMPredictor()
    fet_pred = FetalACPredictor(use_new_model=True)
    fet_ext = None  # legacy multi-organ feature extractor removed 2026-10-25
    fusion = MultimodalFeatureFusion()
    ctgan = CTGANDigitalTwin()
    macro = MacrosomiaRiskMonitor()
    rec = ClinicalRecommendationEngine()
    ode_pred = FetalMaternalODEPredictor()
    return mat_pred, fet_pred, fet_ext, fusion, ctgan, macro, rec, ode_pred

mat_pred, fet_pred, fet_ext, fusion_mod, ctgan_twin, macro_monitor, rec_engine, ode_predictor = load_cached_components()


def booking_macro_kwargs(maternal: Dict[str, Any]) -> Dict[str, Any]:
    """Stage-1 validated booking macrosomia risk, passed to MacrosomiaRiskMonitor.evaluate_risk."""
    r = mat_pred.predict_macrosomia_risk(maternal)
    return {"macrosomia_prob": r["macrosomia_probability"], "booking_flagged": r["booking_flagged"]}


def derive_booking_and_current_ga(history: List[Dict[str, Any]], fallback_booking: float, fallback_current: float) -> Tuple[float, float]:
    """Booking GA is fixed at the first (booking) visit; the current week comes from the latest visit.

    The booking models were trained on gestational age AT BOOKING, so later visits must not overwrite it.
    Older records without ``booking_ga_weeks`` fall back to the first visit's gestational week.
    """
    if not history:
        return float(fallback_booking), float(fallback_current)
    first, latest = history[0], history[-1]
    booking = first.get("gestational_age_weeks") if first.get("gestational_age_weeks") is not None else fallback_booking
    for rec in history:  # an explicitly recorded booking GA anywhere in the history wins
        if rec.get("booking_ga_weeks") is not None:
            booking = rec["booking_ga_weeks"]
            break
    current = latest.get("gestational_age_weeks") if latest.get("gestational_age_weeks") is not None else fallback_current
    booking, current = float(booking), float(current)
    return min(booking, current), current


BOOKING_VITAL_FIELDS = {  # model key -> (explicit booking field, per-visit field)
    "Body Mass Index at Booking": ("booking_bmi", "bmi"),
    "SystolicBloodPressureCuff": ("booking_sbp", "systolic_bp"),
    "Diastolic Blood Pressure": ("booking_dbp", "diastolic_bp"),
}


def derive_booking_vitals(history: List[Dict[str, Any]], fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Booking BMI and blood pressure come from the booking visit, not the latest visit.

    BMI and BP change during pregnancy; the booking models were trained on values AT BOOKING,
    so later observations must not overwrite them. Uses an explicitly recorded booking value if
    present, otherwise the first visit's value, otherwise the fallback.
    """
    out = {}
    for key, (booking_field, visit_field) in BOOKING_VITAL_FIELDS.items():
        val = None
        for rec in history:
            if rec.get(booking_field) is not None:
                val = rec[booking_field]
                break
        if val is None and history and history[0].get(visit_field) is not None:
            val = history[0][visit_field]
        out[key] = val if val is not None else fallback.get(key)
    return out


# NICE NG3 (2015, updated 2020) diagnostic thresholds for GDM on a 75 g OGTT
NICE_OGTT_FASTING_MMOL = 5.6
NICE_OGTT_2H_MMOL = 7.8


def _num(value: Any, default: Optional[float] = None) -> Optional[float]:
    """float(value), or default when the value is missing/blank."""
    try:
        return float(value) if value is not None and value != "" else default
    except (TypeError, ValueError):
        return default


def ogtt_diagnosis(fasting: Optional[float], two_hour: Optional[float]) -> Dict[str, Any]:
    """Classify an OGTT against NICE NG3 (fasting >= 5.6 or 2-hour >= 7.8 mmol/L = GDM).

    This is the diagnostic test itself, so it is reported as a diagnosis, never fed to the
    booking risk models.
    """
    if fasting is None and two_hour is None:
        return {"status": "not_done", "label": "No OGTT recorded", "diagnosed": None}
    hits = []
    if fasting is not None and fasting >= NICE_OGTT_FASTING_MMOL:
        hits.append(f"fasting {fasting:.1f} ≥ {NICE_OGTT_FASTING_MMOL}")
    if two_hour is not None and two_hour >= NICE_OGTT_2H_MMOL:
        hits.append(f"2-hour {two_hour:.1f} ≥ {NICE_OGTT_2H_MMOL}")
    if hits:
        return {"status": "diagnosed", "label": "GDM diagnosed (OGTT)", "diagnosed": True, "reason": "; ".join(hits)}
    if fasting is not None and two_hour is not None:
        return {"status": "normal", "label": "OGTT normal", "diagnosed": False}
    return {"status": "incomplete", "label": "OGTT incomplete", "diagnosed": None}


def latest_ogtt_result(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Most recent OGTT that was actually entered (records flagged ``ogtt_recorded``).

    Older records stored placeholder glucose values whether or not a test was done,
    so they are not used for diagnosis.
    """
    for rec in reversed(history or []):
        if not rec.get("ogtt_recorded"):
            continue
        f0, f120 = _num(rec.get("glucose_0min")), _num(rec.get("glucose_120min"))
        if f0 is None and f120 is None:
            continue
        res = ogtt_diagnosis(f0, f120)
        res.update({"fasting": f0, "two_hour": f120, "week": _num(rec.get("gestational_age_weeks"))})
        return res
    res = ogtt_diagnosis(None, None)
    res.update({"fasting": None, "two_hour": None, "week": None})
    return res


def render_ogtt_status(ogtt: Optional[Dict[str, Any]]) -> None:
    """Show OGTT diagnosis next to (not instead of) the booking screening risk."""
    ogtt = ogtt or latest_ogtt_result([])
    wk = f" at {ogtt['week']:.0f} wk" if ogtt.get("week") is not None else ""
    vals = []
    if ogtt.get("fasting") is not None:
        vals.append(f"fasting {ogtt['fasting']:.1f}")
    if ogtt.get("two_hour") is not None:
        vals.append(f"2-hour {ogtt['two_hour']:.1f}")
    vals_txt = f" ({', '.join(vals)} mmol/L)" if vals else ""
    rule = f"NICE NG3: fasting ≥ {NICE_OGTT_FASTING_MMOL} or 2-hour ≥ {NICE_OGTT_2H_MMOL} mmol/L"
    if ogtt["status"] == "diagnosed":
        st.error(f"**GDM diagnosed by OGTT{wk}**{vals_txt} — {ogtt.get('reason', '')}. {rule}. "
                 "The booking risk is now superseded by the diagnosis; follow the GDM care pathway.")
    elif ogtt["status"] == "normal":
        st.success(f"**OGTT normal{wk}**{vals_txt}. {rule}. The booking risk remains a screening estimate only.")
    elif ogtt["status"] == "incomplete":
        st.warning(f"**OGTT incomplete{wk}**{vals_txt}: one value is missing and the recorded value is below the threshold. {rule}.")
    else:
        st.info("**No OGTT recorded yet.** NICE offers a 75 g OGTT at 24–28 weeks to women with risk factors. "
                "The booking risk helps decide who to test early.")


def validated_caption(res: Dict[str, Any], outcome_label: str) -> str:
    """One-line evidence context shown under every booking-model probability."""
    vp = res.get("validated_performance", {})
    lo, hi = vp.get("test_auroc_95ci", (float("nan"), float("nan")))
    return (
        f"Population {outcome_label} rate {res.get('population_rate_percent', float('nan')):.1f}% · "
        f"'High Risk' = top 20% at booking (≥ {res.get('flag_threshold', float('nan'))*100:.1f}%; "
        f"held-out sensitivity {vp.get('flag_sensitivity', float('nan'))*100:.0f}%, "
        f"specificity {vp.get('flag_specificity', float('nan'))*100:.0f}%, PPV {vp.get('flag_ppv', float('nan'))*100:.0f}%) · "
        f"validated AUROC {vp.get('test_auroc', float('nan')):.2f} ({lo:.2f}–{hi:.2f}), n={vp.get('n_test', '?')} held-out women. "
        "Calibrated probability from booking data only (no OGTT)."
    )

def sync_maternal_inputs_from_latest_observation() -> None:
    """Use the latest saved observation as the shared risk-model input."""
    patient_id = st.session_state.patient_id
    if st.session_state.get("maternal_inputs_patient_id") == patient_id:
        return

    vitals_history = get_vitals_history(patient_id)
    latest_observation = vitals_history[-1] if vitals_history else {}
    st.session_state.latest_observation = latest_observation

    if latest_observation:
        st.session_state.maternal_inputs.update({
            "AgeAtStartOfSpell": int(latest_observation.get("maternal_age", st.session_state.maternal_inputs.get("AgeAtStartOfSpell", 31))),
            "Gravida": int(latest_observation.get("gravida", st.session_state.maternal_inputs.get("Gravida", 1))),
            "Parity": int(latest_observation.get("parity", st.session_state.maternal_inputs.get("Parity", 0))),
            "No_Of_previous_Csections": int(latest_observation.get("csections", st.session_state.maternal_inputs.get("No_Of_previous_Csections", 0))),
            "Index of Multiple Deprivation Rank": int(latest_observation.get("imdr", st.session_state.maternal_inputs.get("Index of Multiple Deprivation Rank", 12183))),
            "Glucoselevelblood": _num(latest_observation.get("glucose_blood"), st.session_state.maternal_inputs.get("Glucoselevelblood", 6.3)),
            "Glucoselevel0minblood": _num(latest_observation.get("glucose_0min"), st.session_state.maternal_inputs.get("Glucoselevel0minblood", 5.2)),
            "Glucoselevel120minblood": _num(latest_observation.get("glucose_120min"), st.session_state.maternal_inputs.get("Glucoselevel120minblood", 8.1)),
            "Ethnicity": str(latest_observation.get("ethnicity", st.session_state.maternal_inputs.get("Ethnicity", "WEU"))).upper(),
        })

    booking_ga, current_ga = derive_booking_and_current_ga(
        vitals_history,
        st.session_state.maternal_inputs.get("Gestation at booking (Weeks)", 12),
        st.session_state.get("current_ga_weeks", 24),
    )
    st.session_state.maternal_inputs["Gestation at booking (Weeks)"] = booking_ga
    booking_vitals = derive_booking_vitals(vitals_history, st.session_state.maternal_inputs)
    st.session_state.maternal_inputs.update({
        "Body Mass Index at Booking": float(booking_vitals["Body Mass Index at Booking"]),
        "SystolicBloodPressureCuff": int(booking_vitals["SystolicBloodPressureCuff"]),
        "Diastolic Blood Pressure": int(booking_vitals["Diastolic Blood Pressure"]),
    })
    st.session_state.maternal_inputs["Obese?"] = "YES" if st.session_state.maternal_inputs["Body Mass Index at Booking"] >= 30.0 else "NO"
    st.session_state.current_ga_weeks = current_ga
    st.session_state.ogtt_result = latest_ogtt_result(vitals_history)
    st.session_state.maternal_inputs_patient_id = patient_id

# Initialize Global Session State
if "authenticated_user" not in st.session_state:
    st.session_state.authenticated_user = None

if "app_screen" not in st.session_state:
    st.session_state.app_screen = "landing"

if "patient_id" not in st.session_state:
    st.session_state.patient_id = "PATIENT_001"

if "maternal_inputs" not in st.session_state:
    st.session_state.maternal_inputs = {
        "AgeAtStartOfSpell": 31,
        "Gestation at booking (Weeks)": 12,
        "Body Mass Index at Booking": 29.4,
        "Gravida": 1,
        "Parity": 0,
        "No_Of_previous_Csections": 0,
        "Index of Multiple Deprivation Rank": 12183,
        "SystolicBloodPressureCuff": 125,
        "Diastolic Blood Pressure": 82,
        "Glucoselevelblood": 6.3,
        "Glucoselevel0minblood": 5.2,
        "Glucoselevel120minblood": 8.1,
        "Ethnicity": "WEU",
        "Obese?": "YES"
    }

if "current_ga_weeks" not in st.session_state:
    st.session_state.current_ga_weeks = 24.0  # current gestational week (separate from booking GA)

sync_maternal_inputs_from_latest_observation()

if "gdm_result" not in st.session_state:
    st.session_state.gdm_result = mat_pred.predict_gdm_risk(st.session_state.maternal_inputs)

if "current_fetal_mask" not in st.session_state or st.session_state.current_fetal_mask is None:
    try:
        sample_img_cand = Path("tests/clean_frames/1dccf454_sw4_f478.png")
        if not sample_img_cand.exists():
            clean_list = list(Path("tests/clean_frames").glob("*.png")) if Path("tests/clean_frames").exists() else []
            if clean_list:
                sample_img_cand = clean_list[0]

        if sample_img_cand.exists():
            import cv2
            img_init = cv2.imread(str(sample_img_cand), cv2.IMREAD_GRAYSCALE)
            ga_init = float(st.session_state.get("current_ga_weeks", 28.0))
            res = fet_pred.predict(img_init, pixel_spacing_mm=0.28, gestational_age_weeks=ga_init)
            
            st.session_state.current_fetal_img = img_init
            st.session_state.current_fetal_mask = res["mask"]
            st.session_state.current_fetal_overlay = res.get("overlay")
            st.session_state.active_scan_name = sample_img_cand.name
            
            fetal_bios = {
                "fetal_estimated_ac_cm": res.get("actual_ac_cm"),
                "fetal_estimated_ac_mm": res.get("actual_ac_mm"),
                "estimated_fetal_weight_g": res.get("efw_hadlock_g"),
                "fitted_ellipse": res.get("fitted_ellipse"),
                "screening_tier": res.get("screening_tier"),
                "model_confidence": res.get("model_confidence", 0.0),
                "plane_quality": res.get("plane_quality", "low"),
                "delta_ac_relative_pct": res.get("delta_ac_relative_pct"),
                "clinical_interpretation": res.get("clinical_interpretation"),
                "fetal_abdomen_mask_area_px": float(res["mask"].sum()) if res.get("mask") is not None else 0.0
            }
            st.session_state.current_fetal_biometrics = fetal_bios
            if res.get("actual_ac_cm") is not None:
                st.session_state["macrosomia_measured_ac"] = float(round(res["actual_ac_cm"], 1))
    except Exception as e:
        print("[App Init] Fetal sample preload notice:", e)

if "current_fetal_img" not in st.session_state:
    st.session_state.current_fetal_img = None
if "current_fetal_biometrics" not in st.session_state or not st.session_state.current_fetal_biometrics:
    st.session_state.current_fetal_biometrics = {}

# Helper to sync digital twin object with MongoDB Atlas
def load_digital_twin_from_db(patient_id: str) -> MaternalFetalDigitalTwin:
    twin_data = get_patient_twin(patient_id)
    if twin_data:
        twin = MaternalFetalDigitalTwin.from_dict(twin_data)
    else:
        twin = MaternalFetalDigitalTwin(patient_id=patient_id)
        save_patient_twin(patient_id, twin.to_dict())
    return twin

# Helper to resolve active fetal biometrics from session or MongoDB Atlas
def get_active_fetal_biometrics() -> Dict[str, Any]:
    if st.session_state.current_fetal_biometrics:
        return st.session_state.current_fetal_biometrics
    
    # Check MongoDB Atlas twin state
    twin = load_digital_twin_from_db(st.session_state.patient_id)
    if twin.fetal_state and any(k for k in twin.fetal_state if "area" in k or "ac" in k):
        st.session_state.current_fetal_biometrics = twin.fetal_state
        return twin.fetal_state

    # No ultrasound yet: return an empty record (no fabricated default biometrics)
    return {}


def render_landing_page():
    st.markdown("""
    <div class="landing-nav">
        <div>
            <div class="brand-name">MDT-GEN</div>
            <div class="brand-subtitle">Maternal-Fetal Digital Twin</div>
        </div>
        <div class="landing-links"><span>Home</span><span>Platform</span><span>Features</span><span>About</span></div>
    </div>
    """, unsafe_allow_html=True)

    hero_left, hero_right = st.columns([1.05, 0.95], gap="large")
    with hero_left:
        st.markdown('<div class="app-kicker">Connected maternal care</div>', unsafe_allow_html=True)
        st.markdown('<h1 class="landing-title">Intelligent maternal-fetal monitoring</h1>', unsafe_allow_html=True)
        st.markdown('<p class="landing-copy">MDT-GEN combines real-time maternal observations, risk assessment, and digital twin technology into a unified clinical monitoring platform.</p>', unsafe_allow_html=True)
        action_col, secondary_col = st.columns([1, 1])
        with action_col:
            if st.button("Access Patient Portal", use_container_width=True, key="landing_sign_in"):
                st.session_state.app_screen = "login"
                st.rerun()
        with secondary_col:
            if st.button("Explore Platform", use_container_width=True, key="landing_explore_button"):
                st.session_state.landing_explore = True
        st.markdown('<div class="landing-availability"><i></i> Built for continuous, connected care</div>', unsafe_allow_html=True)
        if st.session_state.get("landing_explore"):
            st.info("Explore real-time observations, clinical risk indicators, digital twin state, and longitudinal trends from one workspace.")
    with hero_right:
        img_path = Path(__file__).parent / "assets" / "maternal_fetal_state.jpg"
        if img_path.exists():
            import base64
            with open(img_path, "rb") as img_file:
                b64_str = base64.b64encode(img_file.read()).decode("utf-8")
            st.markdown(f"""
            <div class="landing-image-wrapper">
                <img src="data:image/jpeg;base64,{b64_str}" alt="Maternal-Fetal State Digital Twin Diagram" class="landing-hero-img" />
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Landing diagram image not found.")

    st.markdown('<div class="landing-section-title">One connected view of maternal care</div>', unsafe_allow_html=True)
    feature_cols = st.columns(4)
    features = [
        ("Real-time monitoring", "Continuously track important maternal observations and clinical measurements."),
        ("Risk assessment", "Present model-generated risk indicators in an understandable clinical format."),
        ("Digital twin", "Maintain a continuously updated representation of the maternal-fetal state."),
        ("Clinical insights", "Organize historical observations and trends to support monitoring.")
    ]
    for feature_col, (title, description) in zip(feature_cols, features):
        with feature_col:
            st.markdown(f'<div class="feature-card"><div class="feature-mark"></div><h3>{title}</h3><p>{description}</p></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="platform-band">
        <div><div class="app-kicker">Designed for connected maternal care</div><h2>Clinical context without the clutter</h2></div>
        <p>Real-time patient observations, cloud-backed persistence, digital twin monitoring, risk visualization, and historical trend analysis are organized around the patient record.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="landing-footer"><strong>MDT-GEN</strong><span>Maternal-Fetal Digital Twin</span></div>', unsafe_allow_html=True)


def render_login_page():
    st.markdown('<div class="auth-brand"><div class="brand-name">MDT-GEN</div><div class="brand-subtitle">Maternal-Fetal Digital Twin</div></div>', unsafe_allow_html=True)
    auth_left, auth_right = st.columns([0.9, 1.1], gap="large")
    with auth_left:
        st.markdown('<div class="app-kicker">Secure clinical access</div><h1 class="auth-title">Welcome to connected maternal care</h1><p class="landing-copy">Secure access to maternal monitoring, clinical administration, and digital twin insights.</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: #EEF8F8; border: 1px solid #BCE3E1; border-radius: 10px; padding: 1rem 1.1rem; margin-top: 0.8rem;">
            <div style="color: #147D82; font-weight: 800; font-size: 0.82rem; margin-bottom: 0.35rem; text-transform: uppercase; letter-spacing: 0.05em;">🔑 Default Demo Accounts</div>
            <div style="font-size: 0.8rem; color: #183B56; line-height: 1.6;">
                🛡️ <strong>Admin:</strong> <code>admin</code> / <code>admin123</code><br/>
                🩺 <strong>Clinician:</strong> <code>doctor</code> / <code>doctor123</code><br/>
                🤰 <strong>Patient:</strong> <code>patient001</code> / <code>patient123</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Back to home", key="back_to_landing"):
            st.session_state.app_screen = "landing"
            st.rerun()
    with auth_right:
        with st.container(border=True):
            st.markdown('<h2 class="auth-card-title">Welcome back</h2><p class="auth-card-subtitle">Sign in to access your portal.</p>', unsafe_allow_html=True)
            login_id = st.text_input("Username / Email / Patient ID", key="login_id")
            login_pass = st.text_input("Password", type="password", key="login_password")
            remember = st.checkbox("Remember me", key="remember_me")
            if st.button("Sign In", use_container_width=True, key="sign_in_submit"):
                authed = authenticate_user(login_id, login_pass)
                if authed:
                    st.session_state.authenticated_user = authed
                    st.session_state.patient_id = authed.get("patient_id", "PATIENT_001")
                    st.session_state.app_screen = "dashboard"
                    st.rerun()
                else:
                    st.error("We could not verify those credentials. Check your username/ID and password.")
            st.markdown('<div class="auth-help">Need system access? <strong>Contact your System Administrator</strong></div>', unsafe_allow_html=True)


if st.session_state.authenticated_user is None:
    if st.session_state.app_screen == "login":
        render_login_page()
    else:
        render_landing_page()
    st.stop()


# Sidebar Navigation Header
st.sidebar.markdown("""
<div style="padding: 0.25rem 0 0.8rem 0;">
    <div style="color: #147D82; font-size: 0.7rem; font-weight: 800; letter-spacing: 0.14em; text-transform: uppercase;">Clinical workspace</div>
    <h2 style="margin: 0.2rem 0 0.15rem; color: #183B56; font-weight: 800; letter-spacing: 0;">MDT-GEN</h2>
    <p style="margin: 0; font-size: 0.76rem; color: #667085;">Maternal-fetal digital twin</p>
</div>
""", unsafe_allow_html=True)

user = st.session_state.authenticated_user
user_role = user.get("role", "Patient")

if user_role in ["Admin", "Clinician"]:
    all_patients = list_all_patients()
    patient_options = {f"{p.get('full_name')} ({p.get('patient_id')})": p.get('patient_id') for p in all_patients}
    if patient_options:
        selected_p_label = st.sidebar.selectbox("Active patient", options=list(patient_options.keys()), key="clinician_patient_select")
        selected_pid = patient_options[selected_p_label]
        if selected_pid != st.session_state.patient_id:
            st.session_state.patient_id = selected_pid
            st.rerun()
else:
    st.session_state.patient_id = user.get("patient_id", "PATIENT_001")

# Main Navigation
if user_role == "Admin":
    nav_options = [
        "Admin User Management",
        "Overview",
        "Synthetic Cohort Generator",
        "Risk Assessment",
        "Fetal Ultrasound",
        "Maternal-Fetal Fusion",
        "Advanced Analytics",
        "Macrosomia Monitor",
        "Explainability (SHAP)",
        "Clinical Action Plan"
    ]
else:
    nav_options = [
        "Overview",
        "Synthetic Cohort Generator",
        "Risk Assessment",
        "Fetal Ultrasound",
        "Maternal-Fetal Fusion",
        "Advanced Analytics",
        "Macrosomia Monitor",
        "Explainability (SHAP)",
        "Clinical Action Plan"
    ]

nav_option = st.sidebar.radio("Workspace", nav_options)

nav_routes = {
    "Admin User Management": "Admin User Management",
    "Overview": "Patient Portal & Real-Time Tracking",
    "Synthetic Cohort Generator": "Synthetic Cohort Generator",
    "Risk Assessment": "Maternal Dashboard",
    "Fetal Ultrasound": "Fetal Ultrasound Segmentation",
    "Maternal-Fetal Fusion": "Maternal–Fetal Twin (Fusion)",
    "Advanced Analytics": "Exploratory Simulation (Neural ODE)",
    "Macrosomia Monitor": "Macrosomia Risk Monitor",
    "Explainability (SHAP)": "Multimodal Explainability (SHAP)",
    "Clinical Action Plan": "Clinical Action Plan"
}
nav_option = nav_routes[nav_option]

st.sidebar.markdown("---")
if st.sidebar.button("Logout", use_container_width=True, key="workspace_logout"):
    st.session_state.authenticated_user = None
    st.session_state.app_screen = "landing"
    st.session_state.patient_id = "PATIENT_001"
    st.rerun()

db_status = get_connection_status()
connection_label = "Connected" if db_status.get("connected") else "Local storage"
st.markdown(f"""
<div class="workspace-header">
    <div><div class="app-kicker">Patient overview</div><h1>Maternal Monitoring Overview</h1><p>Real-time patient observations and digital twin status</p></div>
    <div class="patient-identity"><strong>{user.get('full_name', 'Active patient')}</strong><span>Patient ID: {st.session_state.patient_id}</span><span class="system-status"><i></i>{connection_label}</span></div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# MODULE -1: ADMIN USER MANAGEMENT & SYSTEM CONSOLE
# ==============================================================================
if nav_option == "Admin User Management":
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-title"> System & User Management (Admin Console)</div>
        <div class="hero-subtitle">Database Cloud Sync: <b>MongoDB Atlas</b> | Managed Accounts & Role-Based Access Control</div>
    </div>
    """, unsafe_allow_html=True)

    all_users = list_all_users()

    # KPI Metrics Header
    total_users_count = len(all_users)
    patients_count = len([u for u in all_users if u.get("role") == "Patient"])
    clinicians_count = len([u for u in all_users if u.get("role") == "Clinician"])
    admins_count = len([u for u in all_users if u.get("role") == "Admin"])

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f'<div class="kpi-card"><span>Total System Users</span><strong>{total_users_count}</strong><small>Persisted in DB</small></div>', unsafe_allow_html=True)
    with kpi2:
        st.markdown(f'<div class="kpi-card"><span>Patients</span><strong>{patients_count}</strong><small>Monitored Twins</small></div>', unsafe_allow_html=True)
    with kpi3:
        st.markdown(f'<div class="kpi-card"><span>Clinicians</span><strong>{clinicians_count}</strong><small>Care Team</small></div>', unsafe_allow_html=True)
    with kpi4:
        st.markdown(f'<div class="kpi-card"><span>Administrators</span><strong>{admins_count}</strong><small>System Admins</small></div>', unsafe_allow_html=True)

    tab_users_list, tab_create_user, tab_db_status = st.tabs([" All Registered Records", " Create New User Account", " Database Sync Status"])

    with tab_users_list:
        st.markdown('<h2 class="page-title">User Directory</h2><p class="page-subtitle">View and filter all system users stored in MongoDB Atlas and local database.</p>', unsafe_allow_html=True)

        col_search, col_filter = st.columns([2, 1])
        with col_search:
            search_query = st.text_input("Search Users", placeholder="Search by name, username, email, or patient ID...", key="admin_search_query")
        with col_filter:
            role_filter = st.selectbox("Filter by Role", options=["All Roles", "Patient", "Clinician", "Admin"], key="admin_role_filter")

        filtered_users = all_users
        if role_filter != "All Roles":
            filtered_users = [u for u in filtered_users if u.get("role") == role_filter]
        if search_query:
            sq = search_query.lower()
            filtered_users = [
                u for u in filtered_users
                if sq in u.get("username", "").lower() or sq in u.get("full_name", "").lower() or sq in u.get("email", "").lower() or sq in u.get("patient_id", "").lower()
            ]

        if filtered_users:
            df_display = pd.DataFrame(filtered_users)
            cols_to_show = [c for c in ["full_name", "username", "role", "patient_id", "email", "created_at"] if c in df_display.columns]
            df_display = df_display[cols_to_show]
            df_display.columns = [c.replace("_", " ").title() for c in cols_to_show]
            st.dataframe(df_display, use_container_width=True, hide_index=True)
        else:
            st.info("No user records match your search criteria.")

    with tab_create_user:
        st.markdown('<h2 class="page-title">Register New User</h2><p class="page-subtitle">Add a new Patient, Clinician, or Admin. Saved to MongoDB Atlas & Local DB.</p>', unsafe_allow_html=True)

        with st.form("admin_create_user_form"):
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                new_username = st.text_input("Username *", placeholder="e.g. jsmith").strip()
                new_password = st.text_input("Password *", type="password", placeholder="Set account password")
                new_fullname = st.text_input("Full Name *", placeholder="e.g. Dr. John Smith")
            with c_col2:
                new_role = st.selectbox("User Role *", options=["Patient", "Clinician", "Admin"])
                new_email = st.text_input("Email Address", placeholder="e.g. jsmith@example.com")
                new_pid = st.text_input("Custom Patient / Staff ID (Optional)", placeholder="Leave blank for auto-generated ID")

            submit_create = st.form_submit_button("Register Account", use_container_width=True)

        if submit_create:
            if not new_username or not new_password or not new_fullname:
                st.error("Username, Password, and Full Name are required fields.")
            else:
                try:
                    res_user = register_user(
                        username=new_username,
                        password=new_password,
                        full_name=new_fullname,
                        role=new_role,
                        patient_id=new_pid if new_pid else None,
                        email=new_email
                    )
                    st.success(f"User **{new_username}** ({new_fullname}) successfully created as **{new_role}**! Persisted in MongoDB Atlas & Local storage.")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Failed to register user: {ex}")

    with tab_db_status:
        st.markdown('<h2 class="page-title">Database Health & Persistence Status</h2>', unsafe_allow_html=True)
        db_stat = get_connection_status()

        st.json({
            "Connection Health": "ONLINE" if db_stat.get("connected") else "OFFLINE (Local Storage)",
            "Active Engine": db_stat.get("mode"),
            "Database Name": db_stat.get("database_name"),
            "PyMongo Driver Installed": db_stat.get("pymongo_available"),
            "Synchronized Collections": ["users", "patient_twins", "vitals_logs"],
            "Total System Records": len(all_users)
        })


# ==============================================================================
# MODULE 0: PATIENT PORTAL & REAL-TIME TRACKING (MONGODB ATLAS INTEGRATED)
# ==============================================================================
elif nav_option == "Patient Portal & Real-Time Tracking":
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-title"> Patient Portal & Real-Time Vitals Tracking</div>
        <div class="hero-subtitle">Real-Time Cloud Persistence on <b>MongoDB Atlas</b> | Active Patient: <code>{st.session_state.patient_id}</code></div>
    </div>
    """, unsafe_allow_html=True)

    twin = load_digital_twin_from_db(st.session_state.patient_id)
    vitals_hist = get_vitals_history(st.session_state.patient_id)

    current_inputs = st.session_state.maternal_inputs
    _latest_obs = st.session_state.get("latest_observation") or {}
    st.markdown('<h2 class="page-title">Patient Portal</h2><p class="page-subtitle">Record observations and review the latest clinical indicators.</p>', unsafe_allow_html=True)
    kpi_cols = st.columns(4)
    kpis = [
        ("Current gestational week", f"{st.session_state.get('current_ga_weeks', 24):.0f}", f"weeks (booked at {current_inputs.get('Gestation at booking (Weeks)', 12):.0f})"),
        ("OGTT (diagnostic)", {"diagnosed": "GDM", "normal": "Normal", "incomplete": "Incomplete"}.get((st.session_state.get("ogtt_result") or {}).get("status"), "Not done"),
         f"NICE NG3: fasting ≥ {NICE_OGTT_FASTING_MMOL} / 2-h ≥ {NICE_OGTT_2H_MMOL}"),
        ("Systolic BP (latest visit)", f"{float(_latest_obs.get('systolic_bp') or current_inputs.get('SystolicBloodPressureCuff', 124)):.0f}", f"mmHg (booking {current_inputs.get('SystolicBloodPressureCuff', 124):.0f})"),
        ("BMI (latest visit)", f"{float(_latest_obs.get('bmi') or current_inputs.get('Body Mass Index at Booking', 28.4)):.1f}", f"kg/m² (booking {current_inputs.get('Body Mass Index at Booking', 28.4):.1f})")
    ]
    for kpi_col, (label, value, unit) in zip(kpi_cols, kpis):
        with kpi_col:
            st.markdown(f'<div class="kpi-card"><span>{label}</span><strong>{value}</strong><small>{unit}</small></div>', unsafe_allow_html=True)

    col_input, col_status = st.columns([1.2, 1])

    with col_input:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Record Patient Observation</div>', unsafe_allow_html=True)

        with st.form("realtime_vital_form"):
            st.markdown("##### 12 Clinical Predictors & Vitals")
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                st.caption(" Demographic & Obstetric History")
                v_age = st.number_input("Maternal Age (years)", min_value=15, max_value=55, value=int(current_inputs.get("AgeAtStartOfSpell", 31)), step=1)
                v_booking_ga = st.number_input("Gestational age at booking (weeks)", min_value=4.0, max_value=42.0,
                                               value=float(current_inputs.get("Gestation at booking (Weeks)", 12)), step=1.0,
                                               help="Week of the first antenatal (booking) visit. Used by the booking risk models; it should not change at later visits.")
                v_ga = st.number_input("Current gestational week", min_value=4.0, max_value=42.0,
                                       value=float(st.session_state.get("current_ga_weeks", 24)), step=1.0,
                                       help="Week of this observation. Used for the visit timeline and the ultrasound stage, not by the booking models.")
                v_gravida = st.number_input("Gravida (Total Pregnancies)", min_value=1, max_value=15, value=int(current_inputs.get("Gravida", 1)), step=1)
                v_parity = st.number_input("Parity (Prior Live Births)", min_value=0, max_value=15, value=int(current_inputs.get("Parity", 0)), step=1)
                v_csections = st.number_input("Previous C-Sections", min_value=0, max_value=10, value=int(current_inputs.get("No_Of_previous_Csections", 0)), step=1)
                st.caption(" Booking measurements (used by the risk models; keep the values from the booking visit)")
                v_booking_bmi = st.number_input("BMI at booking (kg/m²)", min_value=15.0, max_value=55.0, value=float(current_inputs.get("Body Mass Index at Booking", 28.4)), step=0.1)
                v_booking_sys = st.number_input("Systolic BP at booking (mmHg)", min_value=70, max_value=220, value=int(current_inputs.get("SystolicBloodPressureCuff", 124)), step=1)
                v_booking_dia = st.number_input("Diastolic BP at booking (mmHg)", min_value=40, max_value=140, value=int(current_inputs.get("Diastolic Blood Pressure", 82)), step=1)
                v_imdr = st.number_input("Index of Multiple Deprivation Rank (IMDR)", min_value=1, max_value=35000, value=int(current_inputs.get("Index of Multiple Deprivation Rank", 12183)), step=100)
                eth_cohort_list = ["WEU", "GBR", "OTH", "NAF", "MEA", "IND", "CAR", "PAK", "IRL", "BGD", "CAE", "AFE", "ASE"]
                curr_eth_val = str(current_inputs.get("Ethnicity", "WEU")).upper()
                v_ethnicity = st.selectbox("Ethnicity (NHS Cohort)", eth_cohort_list, index=eth_cohort_list.index(curr_eth_val) if curr_eth_val in eth_cohort_list else 0)

            with f_col2:
                st.caption(" Today's vitals (tracked over time; not used by the booking models)")
                v_bmi = st.number_input("BMI today (kg/m²)", min_value=15.0, max_value=55.0, value=float(_latest_obs.get("bmi") or current_inputs.get("Body Mass Index at Booking", 28.4)), step=0.1)
                v_weight = st.number_input("Current Weight (kg)", min_value=40.0, max_value=150.0, value=float(_num((st.session_state.get("latest_observation") or {}).get("weight_kg"), 72.5)), step=0.1)
                v_sys = st.number_input("Systolic BP today (mmHg)", min_value=70, max_value=220, value=int(_latest_obs.get("systolic_bp") or current_inputs.get("SystolicBloodPressureCuff", 124)), step=1)
                v_dia = st.number_input("Diastolic BP today (mmHg)", min_value=40, max_value=140, value=int(_latest_obs.get("diastolic_bp") or current_inputs.get("Diastolic Blood Pressure", 82)), step=1)

            st.markdown("##### Lab results (optional)")
            st.caption("Recorded for the clinical record and the OGTT diagnosis. **Not used by the risk models**, because the OGTT is the "
                       "test that diagnoses GDM. Leave blank if not measured at this visit.")
            l_col1, l_col2, l_col3 = st.columns(3)
            with l_col1:
                v_glu = st.number_input("Random blood glucose (mmol/L)", min_value=2.0, max_value=25.0, value=None, step=0.1, placeholder="not measured")
            with l_col2:
                v_ogtt0 = st.number_input("OGTT fasting, 0 min (mmol/L)", min_value=2.0, max_value=15.0, value=None, step=0.1, placeholder="not done")
            with l_col3:
                v_ogtt120 = st.number_input("OGTT 2-hour, 120 min (mmol/L)", min_value=2.0, max_value=25.0, value=None, step=0.1, placeholder="not done")

            v_notes = st.text_area("Clinical Observations & Patient Symptoms", placeholder="e.g. Patient reports mild fatigue; fasting glucose controlled.")
            
            submit_vital = st.form_submit_button("Save Observation", use_container_width=True)

        if submit_vital:
            if v_booking_ga > v_ga:
                st.warning(f"Booking GA ({v_booking_ga:.0f} wk) was later than the current week ({v_ga:.0f} wk); booking GA set to {v_ga:.0f} wk.")
                v_booking_ga = v_ga
            vital_payload = {
                "maternal_age": v_age,
                "gestational_age_weeks": v_ga,
                "booking_ga_weeks": v_booking_ga,
                "booking_bmi": v_booking_bmi,
                "booking_sbp": v_booking_sys,
                "booking_dbp": v_booking_dia,
                "ogtt_recorded": bool(v_ogtt0 is not None or v_ogtt120 is not None),
                "gravida": v_gravida,
                "parity": v_parity,
                "csections": v_csections,
                "imdr": v_imdr,
                "ethnicity": v_ethnicity,
                "systolic_bp": v_sys,
                "diastolic_bp": v_dia,
                "glucose_blood": v_glu,
                "glucose_0min": v_ogtt0,
                "glucose_120min": v_ogtt120,
                "weight_kg": v_weight,
                "bmi": v_bmi,
                "notes": v_notes
            }

            # 1. Log observation in MongoDB Atlas vitals collection
            log_realtime_vital(st.session_state.patient_id, vital_payload)

            # 2. Update Maternal State with all 12 Clinical Predictors & Re-predict GDM Risk
            st.session_state.maternal_inputs.update({
                "AgeAtStartOfSpell": v_age,
                "Gestation at booking (Weeks)": v_booking_ga,
                "Body Mass Index at Booking": v_booking_bmi,
                "Gravida": v_gravida,
                "Parity": v_parity,
                "No_Of_previous_Csections": v_csections,
                "Index of Multiple Deprivation Rank": v_imdr,
                "Ethnicity": v_ethnicity,
                "SystolicBloodPressureCuff": v_booking_sys,
                "Diastolic Blood Pressure": v_booking_dia,
                "Obese?": "YES" if v_booking_bmi >= 30.0 else "NO"
            })
            # Lab values only update the record when actually measured
            for _key, _val in (("Glucoselevelblood", v_glu), ("Glucoselevel0minblood", v_ogtt0), ("Glucoselevel120minblood", v_ogtt120)):
                if _val is not None:
                    st.session_state.maternal_inputs[_key] = float(_val)
            if vital_payload["ogtt_recorded"]:
                _o = ogtt_diagnosis(v_ogtt0, v_ogtt120)
                _o.update({"fasting": v_ogtt0, "two_hour": v_ogtt120, "week": float(v_ga)})
                st.session_state.ogtt_result = _o
            st.session_state.latest_observation = dict(vital_payload)

            st.session_state.current_ga_weeks = float(v_ga)
            new_gdm_res = mat_pred.predict_gdm_risk(st.session_state.maternal_inputs)
            st.session_state.gdm_result = new_gdm_res
            gdm_prob = new_gdm_res["gdm_probability"]

            # 3. Evaluate Macrosomia & Recommendations
            macro_res = macro_monitor.evaluate_risk(maternal_gdm_prob=gdm_prob, gestational_age_weeks=v_ga,
                                                    **booking_macro_kwargs(st.session_state.maternal_inputs))
            recs = rec_engine.generate_recommendations(
                gdm_risk_level=new_gdm_res["risk_level"],
                macrosomia_tier=macro_res["risk_tier"],
                gestational_age_weeks=v_ga
            )

            # 4. Save updated Digital Twin state to MongoDB Atlas
            twin.update_maternal_state(st.session_state.maternal_inputs, gdm_prob=gdm_prob)
            twin.record_visit(
                gestational_age_weeks=v_ga,
                gdm_risk_prob=gdm_prob,
                macrosomia_tier=macro_res["risk_tier"],
                notes=v_notes
            )
            rec_list = recs.get("recommendations", recs.get("physician_action_plan", []) + recs.get("patient_guidance", []))
            twin.set_recommendations(rec_list)
            save_patient_twin(st.session_state.patient_id, twin.to_dict())

            st.success("Observation saved and digital twin updated.")
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with col_status:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header"> Digital Twin Live Overview</div>', unsafe_allow_html=True)

        gdm_prob = st.session_state.gdm_result["gdm_probability"]
        risk_lvl = st.session_state.gdm_result["risk_level"]

        badge_class = "metric-badge-high" if risk_lvl == "High Risk" else "metric-badge-low"

        st.markdown(f"""
        <div style="text-align: center; padding: 1rem 0;">
            <p style="margin: 0; color: #64748B; font-size: 0.9rem;">GDM Risk Probability</p>
            <h1 style="margin: 0; font-size: 2.8rem; font-weight: 800; color: #0F172A;">{gdm_prob*100:.1f}%</h1>
            <span class="{badge_class}">{risk_lvl}</span>
        </div>
        """, unsafe_allow_html=True)
        st.caption(validated_caption(st.session_state.gdm_result, "GDM"))
        render_ogtt_status(st.session_state.get("ogtt_result"))

        st.markdown("---")
        st.markdown("####  Monitoring History Summary")
        st.write(f"- **Patient ID**: `{st.session_state.patient_id}`")
        st.write(f"- **Total Vital Log Entries**: `{len(vitals_hist)}` visits")
        st.write(f"- **Last Updated**: `{twin.updated_at[:19]}`")

        if twin.recommendations:
            st.markdown("####  Key Action Item")
            st.info(twin.recommendations[0])

        st.markdown('</div>', unsafe_allow_html=True)

    # --- Interactive Longitudinal Trend Graphs ---
    st.markdown("---")
    st.subheader(" Longitudinal Vitals & Risk Trends (Real-Time Cloud Analytics)")

    if vitals_hist:
        df_vitals = pd.DataFrame(vitals_hist)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("####  Blood Pressure Trajectory")
            fig_bp = go.Figure()
            fig_bp.add_trace(go.Scatter(
                x=df_vitals["gestational_age_weeks"], y=df_vitals["systolic_bp"],
                mode="lines+markers", name="Systolic BP (mmHg)", line=dict(color="#EF4444", width=3)
            ))
            fig_bp.add_trace(go.Scatter(
                x=df_vitals["gestational_age_weeks"], y=df_vitals["diastolic_bp"],
                mode="lines+markers", name="Diastolic BP (mmHg)", line=dict(color="#3B82F6", width=3)
            ))
            fig_bp.update_layout(
                xaxis_title="Gestational Week",
                yaxis_title="BP (mmHg)",
                height=320,
                template="plotly_white",
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#F8FAFC",
                font=dict(color="#1E293B", family="Plus Jakarta Sans, sans-serif"),
                legend=dict(font=dict(color="#1E293B", size=12), bgcolor="#FFFFFF"),
                xaxis=dict(
                    color="#1E293B",
                    title_font=dict(color="#1E293B", size=13),
                    tickfont=dict(color="#334155", size=11),
                    gridcolor="#CBD5E1",
                    zerolinecolor="#94A3B8"
                ),
                yaxis=dict(
                    color="#1E293B",
                    title_font=dict(color="#1E293B", size=13),
                    tickfont=dict(color="#334155", size=11),
                    gridcolor="#CBD5E1",
                    zerolinecolor="#94A3B8"
                )
            )
            st.plotly_chart(fig_bp, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("####  Blood Glucose & OGTT 120-min Trajectory")
            fig_glu = go.Figure()
            if "glucose_blood" in df_vitals:
                fig_glu.add_trace(go.Scatter(
                    x=df_vitals["gestational_age_weeks"], y=df_vitals["glucose_blood"],
                    mode="lines+markers", name="Blood Glucose", line=dict(color="#10B981", width=3)
                ))
            if "glucose_120min" in df_vitals:
                fig_glu.add_trace(go.Scatter(
                    x=df_vitals["gestational_age_weeks"], y=df_vitals["glucose_120min"],
                    mode="lines+markers", name="OGTT 120-min", line=dict(color="#8B5CF6", width=3)
                ))
            fig_glu.update_layout(
                xaxis_title="Gestational Week",
                yaxis_title="mmol/L",
                height=320,
                template="plotly_white",
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#F8FAFC",
                font=dict(color="#1E293B", family="Plus Jakarta Sans, sans-serif"),
                legend=dict(font=dict(color="#1E293B", size=12), bgcolor="#FFFFFF"),
                xaxis=dict(
                    color="#1E293B",
                    title_font=dict(color="#1E293B", size=13),
                    tickfont=dict(color="#334155", size=11),
                    gridcolor="#CBD5E1",
                    zerolinecolor="#94A3B8"
                ),
                yaxis=dict(
                    color="#1E293B",
                    title_font=dict(color="#1E293B", size=13),
                    tickfont=dict(color="#334155", size=11),
                    gridcolor="#CBD5E1",
                    zerolinecolor="#94A3B8"
                )
            )
            st.plotly_chart(fig_glu, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("####  Historical Vital Observation Logs (MongoDB Atlas)")
        st.dataframe(
            df_vitals[["timestamp", "gestational_age_weeks", "systolic_bp", "diastolic_bp", "glucose_blood", "glucose_120min", "weight_kg", "notes"]],
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("ℹ No vital observations logged yet for this patient. Submit your first observation above to begin real-time cloud tracking!")


# ==============================================================================
# MODULE 1: MATERNAL DASHBOARD
# ==============================================================================
elif nav_option == "Maternal Dashboard":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title"> Maternal Clinical Dashboard</div>
        <div class="hero-subtitle">Personalized Early Gestational Diabetes Mellitus (GDM) Risk Prediction powered by Stacking Ensemble ML</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.25, 1])

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header"> Patient Clinical Inputs</div>', unsafe_allow_html=True)

        m_inputs = st.session_state.maternal_inputs
        
        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown("#####  Demographic & Obstetric History")
            age = st.slider("Maternal Age (years)", 18, 50, int(m_inputs.get("AgeAtStartOfSpell", 31)))
            gest_wks = st.slider("Gestation at Booking (Weeks)", 4, 42, int(m_inputs.get("Gestation at booking (Weeks)", 12)))
            gravida = st.slider("Gravida (Total Pregnancies)", 1, 12, int(m_inputs.get("Gravida", 1)))
            parity = st.slider("Parity (Prior Live Births)", 0, 10, int(m_inputs.get("Parity", 0)))
            csections = st.slider("Previous C-Sections", 0, 8, int(m_inputs.get("No_Of_previous_Csections", 0)))
            imdr = st.number_input("Index of Multiple Deprivation Rank (IMDR)", 1, 35000, int(m_inputs.get("Index of Multiple Deprivation Rank", 12183)), step=100)
            eth_cohort_list = ["WEU", "GBR", "OTH", "NAF", "MEA", "IND", "CAR", "PAK", "IRL", "BGD", "CAE", "AFE", "ASE"]
            curr_eth_p = str(m_inputs.get("Ethnicity", "WEU")).upper()
            m_ethnicity = st.selectbox("Ethnicity (NHS Cohort)", eth_cohort_list, index=eth_cohort_list.index(curr_eth_p) if curr_eth_p in eth_cohort_list else 0, key="mod3_ethnicity")

        with c_b:
            st.markdown("#####  Booking BMI & Blood Pressure")
            bmi = st.slider("BMI at Booking (kg/m²)", 15.0, 45.0, float(m_inputs.get("Body Mass Index at Booking", 29.4)), step=0.1)
            sys_bp = st.slider("Systolic BP (mmHg)", 80, 200, int(m_inputs.get("SystolicBloodPressureCuff", 125)))
            dia_bp = st.slider("Diastolic BP (mmHg)", 50, 120, int(m_inputs.get("Diastolic Blood Pressure", 82)))
            with st.expander("Lab results (not used by the risk model)"):
                st.caption("Recorded values only. OGTT results are judged against NICE NG3 in the panel on the right; "
                           "enter new results through Patient Portal → Record Observation.")
                glu = st.slider("Random blood glucose (mmol/L)", 3.0, 15.0, float(_num(m_inputs.get("Glucoselevelblood"), 6.3)), step=0.1)
                glu0 = st.slider("OGTT fasting, 0 min (mmol/L)", 3.0, 12.0, float(_num(m_inputs.get("Glucoselevel0minblood"), 5.2)), step=0.1)
                glu120 = st.slider("OGTT 2-hour, 120 min (mmol/L)", 4.0, 20.0, float(_num(m_inputs.get("Glucoselevel120minblood"), 8.1)), step=0.1)

        obese_str = "YES" if bmi >= 30.0 else "NO"

        new_inputs = {
            "AgeAtStartOfSpell": age,
            "Gestation at booking (Weeks)": gest_wks,
            "Body Mass Index at Booking": bmi,
            "Gravida": gravida,
            "Parity": parity,
            "No_Of_previous_Csections": csections,
            "Index of Multiple Deprivation Rank": imdr,
            "Ethnicity": m_ethnicity,
            "SystolicBloodPressureCuff": sys_bp,
            "Diastolic Blood Pressure": dia_bp,
            "Glucoselevelblood": glu,
            "Glucoselevel0minblood": glu0,
            "Glucoselevel120minblood": glu120,
            "Obese?": obese_str
        }

        if st.button(" Evaluate GDM Risk & Persist to Atlas", use_container_width=True):
            st.session_state.maternal_inputs = new_inputs
            res = mat_pred.predict_gdm_risk(new_inputs)
            st.session_state.gdm_result = res
            
            # Save twin update to MongoDB Atlas
            twin = load_digital_twin_from_db(st.session_state.patient_id)
            twin.update_maternal_state(new_inputs, gdm_prob=res["gdm_probability"])
            save_patient_twin(st.session_state.patient_id, twin.to_dict())

            st.success("GDM Assessment updated & saved to MongoDB Atlas!")
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header"> GDM Assessment Output</div>', unsafe_allow_html=True)

        res = st.session_state.gdm_result
        prob = res["gdm_probability"]
        lvl = res["risk_level"]

        badge_cls = "metric-badge-high" if lvl == "High Risk" else "metric-badge-low"

        st.markdown(f"""
        <div style="text-align: center; padding: 1.5rem 0;">
            <p style="margin: 0; color: #64748B; font-weight: 600;">Gestational Diabetes Risk Probability</p>
            <h1 style="font-size: 3.5rem; margin: 0.2rem 0; color: #0F172A; font-weight: 800;">{prob*100:.1f}%</h1>
            <span class="{badge_cls}">{lvl}</span>
        </div>
        """, unsafe_allow_html=True)
        st.caption(validated_caption(res, "GDM"))
        render_ogtt_status(st.session_state.get("ogtt_result"))

        st.markdown("#### Base Model Consensus Breakdown")
        model_probs = res.get("base_probabilities", res.get("model_probabilities", {}))
        df_models = pd.DataFrame(list(model_probs.items()), columns=["Model", "Probability"])
        df_models["Probability (%)"] = (df_models["Probability"] * 100).round(1)

        fig_m = px.bar(df_models, x="Model", y="Probability (%)", color="Probability (%)",
                       color_continuous_scale="Blues", text="Probability (%)")
        fig_m.update_layout(
            height=280,
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC",
            font=dict(color="#1E293B", family="Plus Jakarta Sans, sans-serif"),
            coloraxis_colorbar=dict(
                title=dict(text="Probability (%)", font=dict(color="#1E293B")),
                tickfont=dict(color="#334155")
            ),
            xaxis=dict(
                title=dict(text="Model", font=dict(color="#1E293B", size=13)),
                tickfont=dict(color="#334155", size=11),
                gridcolor="#CBD5E1"
            ),
            yaxis=dict(
                title=dict(text="Probability (%)", font=dict(color="#1E293B", size=13)),
                tickfont=dict(color="#334155", size=11),
                gridcolor="#CBD5E1",
                range=[0, 110]
            )
        )
        st.plotly_chart(fig_m, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
elif nav_option == "Monitoring History":
    st.markdown('<h2 class="page-title">Monitoring History</h2><p class="page-subtitle">Review and export the patient\'s longitudinal observations.</p>', unsafe_allow_html=True)
    history = get_vitals_history(st.session_state.patient_id)
    if history:
        history_df = pd.DataFrame(history)
        filter_cols = st.columns([1, 1, 1, 1])
        with filter_cols[0]:
            date_filter = st.date_input("Date", value=None, key="history_date")
        with filter_cols[1]:
            risk_filter = st.selectbox("Risk level", ["All", "High Risk", "Moderate Risk", "Low Risk"], key="history_risk")
        with filter_cols[2]:
            observation_filter = st.selectbox("Observation type", ["All observations", "Vitals", "Clinical notes"], key="history_type")
        with filter_cols[3]:
            st.markdown('<div style="height: 1.7rem"></div>', unsafe_allow_html=True)
            st.download_button("Export Report", history_df.to_csv(index=False), "mdt-gen-monitoring-history.csv", "text/csv", use_container_width=True)

        if date_filter is not None and "timestamp" in history_df:
            history_dates = pd.to_datetime(history_df["timestamp"], errors="coerce").dt.date
            history_df = history_df[history_dates == date_filter]
        if observation_filter == "Clinical notes" and "notes" in history_df:
            history_df = history_df[history_df["notes"].fillna("").astype(str).str.len() > 0]

        display_columns = [column for column in ["timestamp", "gestational_age_weeks", "glucose_blood", "systolic_bp", "diastolic_bp", "glucose_120min", "bmi", "notes"] if column in history_df.columns]
        st.dataframe(history_df[display_columns], use_container_width=True, hide_index=True)
    else:
        st.info("No monitoring observations are available for this patient yet.")


# MODULE 2: SYNTHETIC COHORT GENERATOR (CTGAN)
# ==============================================================================
elif nav_option == "Synthetic Cohort Generator":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title"> CTGAN Synthetic Cohort Generator</div>
        <div class="hero-subtitle">Generative Adversarial Modeling for Clinical Synthetic Data & Population Counterfactual Scenarios</div>
    </div>
    """, unsafe_allow_html=True)

    col_ctrl, col_viz = st.columns([1, 1.4])

    with col_ctrl:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header"> Synthetic Cohort Parameters</div>', unsafe_allow_html=True)
        
        num_samples = st.slider("Number of Synthetic Patients to Generate", 50, 1000, 200, step=50)
        
        if st.button(" Generate Synthetic Cohort", use_container_width=True):
            with st.spinner("Executing CTGAN Generator..."):
                synth_df = ctgan_twin.generate_synthetic_cohort(num_samples=num_samples)
                st.session_state.synth_cohort = synth_df
                st.success(f"Generated {len(synth_df)} synthetic patient profiles!")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_viz:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header"> Synthetic vs Real Clinical Feature Distribution</div>', unsafe_allow_html=True)

        if "synth_cohort" in st.session_state and not st.session_state.synth_cohort.empty:
            synth_df = st.session_state.synth_cohort
            st.dataframe(synth_df.head(6), use_container_width=True)

            train_path = PROCESSED_DIR / "train_clean.csv"
            if train_path.exists():
                real_df = pd.read_csv(train_path)
                feat_col = st.selectbox("Select Feature for Distribution Comparison:", ["Body Mass Index at Booking", "Glucoselevelblood", "AgeAtStartOfSpell"])
                
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(x=real_df[feat_col], name="Real Patient Data", opacity=0.6, marker_color="#2563EB"))
                fig_dist.add_trace(go.Histogram(x=synth_df[feat_col], name="CTGAN Synthetic Twin", opacity=0.6, marker_color="#EC4899"))
                fig_dist.update_layout(barmode="overlay", title=f"Fidelity Assessment: {feat_col}", height=280, paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("ℹ Click 'Generate Synthetic Cohort' to synthesize patient digital twin profiles.")
        st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# MODULE 3: FETAL ULTRASOUND SEGMENTATION
# ==============================================================================
elif nav_option == "Fetal Ultrasound Segmentation":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🔬 Fetal Ultrasound Attention U-Net Segmentation</div>
        <div class="hero-subtitle">Production Deep Learning Biometry Engine (<code>best_attunet.pth</code>) &amp; Ramanujan Perimeter Ellipse Fitting</div>
    </div>
    """, unsafe_allow_html=True)

    col_up, col_res = st.columns([1, 1.25])

    clean_frames_dir = Path("tests/clean_frames")
    available_clean_frames = sorted([f.name for f in clean_frames_dir.glob("*.png")]) if clean_frames_dir.exists() else []
    if not available_clean_frames:
        available_clean_frames = ["1dccf454_sw4_f478.png"]

    with col_up:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📷 Select Ultrasound Scan</div>', unsafe_allow_html=True)

        input_mode = st.radio(
            "Select Scan Source:", 
            ["🎯 Pre-loaded Acouslic-AI Ultrasound Frame", "📤 Upload Custom Ultrasound Scan (.png, .jpg, .npy)"], 
            horizontal=True
        )

        img_np = None
        active_sample_id = available_clean_frames[0]
        uploaded_file = None

        if input_mode == "🎯 Pre-loaded Acouslic-AI Ultrasound Frame":
            selected_sample = st.selectbox(
                "Choose Acouslic-AI Ultrasound Frame:", 
                available_clean_frames,
                index=0
            )
            active_sample_id = selected_sample
            sample_file_path = clean_frames_dir / selected_sample
            if sample_file_path.exists():
                import cv2
                img_np = cv2.imread(str(sample_file_path), cv2.IMREAD_GRAYSCALE)
                st.image(img_np, caption=f"Selected Frame: {active_sample_id}", use_container_width=True)
        else:
            uploaded_file = st.file_uploader("Upload Ultrasound Image (.png, .jpg, .jpeg, .npy):", type=["png", "jpg", "jpeg", "npy"])
            if uploaded_file is not None:
                st.session_state.custom_file_name = uploaded_file.name
                st.session_state.custom_file_bytes = uploaded_file.getvalue()

            if "custom_file_bytes" in st.session_state and st.session_state.custom_file_bytes:
                active_sample_id = st.session_state.get("custom_file_name", "custom_scan.png")
                import io
                file_bytes = st.session_state.custom_file_bytes
                
                if active_sample_id.endswith(".npy"):
                    loaded_data = np.load(io.BytesIO(file_bytes), allow_pickle=True)
                    if hasattr(loaded_data, "item") and isinstance(loaded_data.item(), dict):
                        data_dict = loaded_data.item()
                        img_np = data_dict.get("image", data_dict.get("img", list(data_dict.values())[0]))
                    elif isinstance(loaded_data, np.ndarray):
                        img_np = loaded_data
                    else:
                        img_np = np.zeros((320, 416), dtype=np.uint8)
                else:
                    from PIL import Image
                    img_np = np.array(Image.open(io.BytesIO(file_bytes)).convert("L"))

                st.image(img_np, caption=f"Uploaded Ultrasound Scan: {active_sample_id}", use_container_width=True)

        col_cal1, col_cal2 = st.columns([1, 1])
        with col_cal1:
            px_spacing_input = st.number_input(
                "Pixel Spacing (mm/px):", 
                min_value=0.05, 
                max_value=2.00, 
                value=0.28, 
                step=0.01,
                help="Official Acouslic-AI resolution is 0.28 mm/pixel."
            )
        with col_cal2:
            current_ga = float(min(max(st.session_state.get("current_ga_weeks", 28.0), 14.0), 42.0))
            ga_input = st.number_input(
                "Gestational Age (weeks):", 
                min_value=14.0, 
                max_value=42.0, 
                value=current_ga, 
                step=0.5,
                help="Used to compute expected reference AC and relative growth deviation."
            )

        if st.button("🚀 Run Attention U-Net Biometry Pipeline", use_container_width=True):
            if img_np is None:
                st.error("Please select or upload a valid ultrasound scan.")
            else:
                with st.spinner("Executing Attention U-Net (best_attunet.pth) & Fitting Obstetric Ellipse..."):
                    res = fet_pred.predict(
                        img_np,
                        pixel_spacing_mm=px_spacing_input,
                        gestational_age_weeks=ga_input
                    )

                    st.session_state.current_fetal_img = img_np
                    st.session_state.current_fetal_mask = res["mask"]
                    st.session_state.current_fetal_overlay = res.get("overlay")
                    st.session_state.active_scan_name = active_sample_id

                    fetal_bios = {
                        "fetal_estimated_ac_cm": res.get("actual_ac_cm"),
                        "fetal_estimated_ac_mm": res.get("actual_ac_mm"),
                        "expected_ac_cm": res.get("expected_ac_cm"),
                        "delta_ac_relative_pct": res.get("delta_ac_relative_pct"),
                        "estimated_fetal_weight_g": res.get("efw_hadlock_g"),
                        "fitted_ellipse": res.get("fitted_ellipse"),
                        "screening_tier": res.get("screening_tier"),
                        "model_confidence": res.get("model_confidence", 0.0),
                        "plane_quality": res.get("plane_quality", "low"),
                        "clinical_interpretation": res.get("clinical_interpretation"),
                        "fetal_abdomen_mask_area_px": float(res["mask"].sum()) if res.get("mask") is not None else 0.0,
                        "pixel_spacing_mm": px_spacing_input,
                        "gestational_age_weeks": ga_input
                    }
                    st.session_state.current_fetal_biometrics = fetal_bios

                    if res.get("actual_ac_cm") is not None:
                        st.session_state["macrosomia_measured_ac"] = float(round(res["actual_ac_cm"], 1))

                    # Update Twin in MongoDB Atlas
                    try:
                        twin = load_digital_twin_from_db(st.session_state.patient_id)
                        twin.update_fetal_state(fetal_bios, ultrasound_file=active_sample_id)
                        save_patient_twin(st.session_state.patient_id, twin.to_dict())
                    except Exception as e:
                        logger.warning(f"Failed to persist fetal biometrics to DB: {e}")

                    st.success(f"Segmentation & Biometry complete for {active_sample_id}!")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_res:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📊 Segmentation Overlay & Clinical Biometrics</div>', unsafe_allow_html=True)

        if st.session_state.current_fetal_img is not None and st.session_state.current_fetal_mask is not None:
            active_name = st.session_state.get("active_scan_name", "Ultrasound Scan")
            st.info(f"Active Scan Being Displayed: **{active_name}** | Model: `models/new_model/best_attunet.pth`")

            # Display Biometry Overlay
            overlay_disp = st.session_state.get("current_fetal_overlay")
            if overlay_disp is None:
                overlay_disp = create_segmentation_overlay(
                    st.session_state.current_fetal_img,
                    st.session_state.current_fetal_mask,
                    alpha=0.45
                )
                ellipse_info = st.session_state.current_fetal_biometrics.get("fitted_ellipse", None) if isinstance(st.session_state.current_fetal_biometrics, dict) else None
                overlay_disp = draw_fitted_ellipse(
                    overlay_disp,
                    ellipse_data=ellipse_info,
                    mask_np=st.session_state.current_fetal_mask
                )

            st.image(
                overlay_disp, 
                caption="Fetal Abdominal Circumference Attention U-Net Segmentation with Fitted Ellipse Calipers & Ramanujan Perimeter", 
                use_container_width=True
            )

            # Key Clinical Metrics Cards
            bios = st.session_state.current_fetal_biometrics or {}
            ac_cm = bios.get("fetal_estimated_ac_cm")
            ac_mm = bios.get("fetal_estimated_ac_mm")
            exp_ac = bios.get("expected_ac_cm")
            delta_pct = bios.get("delta_ac_relative_pct")
            efw_g = bios.get("estimated_fetal_weight_g")
            conf_val = bios.get("model_confidence", 0.0)
            plane_q = bios.get("plane_quality", "low")
            screening_tier = bios.get("screening_tier", "Unavailable")
            clin_note = bios.get("clinical_interpretation", "")

            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                st.metric(
                    "Measured AC", 
                    f"{ac_cm:.1f} cm" if ac_cm is not None else "N/A",
                    f"{ac_mm:.1f} mm" if ac_mm is not None else None
                )
            with mcol2:
                st.metric(
                    "GA Expected AC", 
                    f"{exp_ac:.1f} cm" if exp_ac is not None else "N/A"
                )
            with mcol3:
                st.metric(
                    "Growth Deviation", 
                    f"{delta_pct:+.1f}%" if delta_pct is not None else "N/A",
                    delta_color="inverse" if (delta_pct and delta_pct > 10.0) else "normal"
                )
            with mcol4:
                st.metric(
                    "Hadlock EFW", 
                    f"{efw_g:,.0f} g" if efw_g is not None else "N/A"
                )

            # Screening Alert Banner
            if "High Risk" in str(screening_tier):
                st.error(f"🚨 **{screening_tier}**: {clin_note}")
            elif "Moderate Risk" in str(screening_tier):
                st.warning(f"⚠️ **{screening_tier}**: {clin_note}")
            elif "Low Risk" in str(screening_tier):
                st.success(f"✅ **{screening_tier}**: {clin_note}")
            else:
                st.info(f"ℹ️ **Status: {screening_tier}**: {clin_note}")

            # Confidence & Plane Quality Tag
            q_color = "#166534" if plane_q == "ok" else "#991B1B"
            st.markdown(
                f"""
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 14px; margin-top: 10px; margin-bottom: 12px;">
                    <span style="font-weight: 600; color: #1E293B;">Confidence Score:</span> <code>{conf_val * 100.0:.1f}%</code> &nbsp;|&nbsp;
                    <span style="font-weight: 600; color: #1E293B;">Plane Quality:</span> <span style="font-weight: bold; color: {q_color}; text-transform: uppercase;">{plane_q}</span> &nbsp;|&nbsp;
                    <span style="font-weight: 600; color: #1E293B;">Model Weights:</span> <code>best_attunet.pth</code>
                </div>
                """, 
                unsafe_allow_html=True
            )

            # Detailed Parameter Table
            ell = bios.get("fitted_ellipse") or {}
            rows = [
                {"Biometric Parameter": "Measured Abdominal Circumference (AC)", "Value": f"{ac_cm:.2f} cm ({ac_mm:.1f} mm)" if ac_cm is not None else "N/A"},
                {"Biometric Parameter": "Expected Reference Median AC (Choi 2016: GA − 5)", "Value": f"{exp_ac:.2f} cm" if exp_ac is not None else "N/A"},
                {"Biometric Parameter": "Relative Growth Deviation (ΔAC_rel)", "Value": f"{delta_pct:+.2f}%" if delta_pct is not None else "N/A"},
                {"Biometric Parameter": "Estimated Fetal Weight (Hadlock 1984 AC-only)", "Value": f"{efw_g:,.1f} grams" if efw_g is not None else "N/A"},
                {"Biometric Parameter": "Fitted Semi-Major Axis (a)", "Value": f"{ell.get('a_cm'):.2f} cm" if ell.get('a_cm') is not None else "N/A"},
                {"Biometric Parameter": "Fitted Semi-Minor Axis (b)", "Value": f"{ell.get('b_cm'):.2f} cm" if ell.get('b_cm') is not None else "N/A"},
                {"Biometric Parameter": "Roundness / Aspect Ratio", "Value": f"{ell.get('roundness_ratio'):.2f}" if ell.get('roundness_ratio') is not None else "N/A"},
                {"Biometric Parameter": "Fetal Abdominal Mask Area", "Value": f"{bios.get('fetal_abdomen_mask_area_px', 0):,.0f} pixels"},
                {"Biometric Parameter": "Calibrated Physical Spacing", "Value": f"{bios.get('pixel_spacing_mm', 0.28):.2f} mm/pixel"}
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        else:
            st.info("ℹ️ Click 'Run Attention U-Net Biometry Pipeline' to visualize the segmentation boundary and extract clinical biometrics.")
        st.markdown('</div>', unsafe_allow_html=True)





# ==============================================================================
# MODULE 5: MATERNAL-FETAL FUSION
# ==============================================================================
elif nav_option == "Maternal–Fetal Twin (Fusion)":
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-title"> Multimodal Feature Fusion Engine</div>
        <div class="hero-subtitle">Twin state vector: maternal booking data + fetal AC biometry + validated risks (no fused score) | Patient: <code>{st.session_state.patient_id}</code></div>
    </div>
    """, unsafe_allow_html=True)

    bios = get_active_fetal_biometrics()
    
    _gdm_r = mat_pred.predict_gdm_risk(st.session_state.maternal_inputs)
    _mac_r = mat_pred.predict_macrosomia_risk(st.session_state.maternal_inputs)
    fused = fusion_mod.fuse_features(
        maternal_dict=st.session_state.maternal_inputs,
        fetal_dict=bios,
        patient_id=st.session_state.patient_id,
        risk_dict={"gdm_booking_probability": _gdm_r["gdm_probability"],
                   "macrosomia_booking_probability": _mac_r["macrosomia_probability"]},
    )

    vector_data = fused.get("fused_feature_vector", fused.get("fused_vector", {}))

    col_f1, col_f2 = st.columns([1, 1.2])

    with col_f1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header"> Multimodal Fusion Summary</div>', unsafe_allow_html=True)
        st.write(f"- **Patient ID**: `{fused['patient_id']}`")
        st.write(f"- **Total Integrated Features**: `{fused.get('num_features', len(vector_data))}`")
        
        st.metric("Booking GDM risk (validated)", f"{_gdm_r['gdm_probability']*100:.1f}%")
        st.metric("Booking macrosomia risk (validated)", f"{_mac_r['macrosomia_probability']*100:.1f}%")
        _ac = bios.get("fetal_estimated_ac_cm") if isinstance(bios, dict) else None
        st.metric("Fetal AC (ACOUSLIC-AI U-Net)", f"{_ac:.1f} cm" if _ac else "—")
        st.caption(fused.get("note", ""))

        st.markdown('</div>', unsafe_allow_html=True)

    with col_f2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header"> Unified Multimodal Vector (JSON)</div>', unsafe_allow_html=True)
        st.json(vector_data)
        if st.button("Save Fused Twin State to MongoDB Atlas", use_container_width=True, key="save_fused_twin_state"):
            twin = load_digital_twin_from_db(st.session_state.patient_id)
            twin.update_maternal_state(st.session_state.maternal_inputs)
            twin.update_fetal_state(bios)
            twin.update_fused_state(vector_data)
            if save_patient_twin(st.session_state.patient_id, twin.to_dict()):
                st.success("Fused maternal-fetal state saved to MongoDB Atlas.")
            else:
                st.warning("Fused state was retained in local fallback storage; MongoDB Atlas was unavailable.")
        st.markdown('</div>', unsafe_allow_html=True)



# ==============================================================================
# MODULE 6: TRAJECTORY FORECASTING (NEURAL ODE)
# ==============================================================================
elif nav_option == "Exploratory Simulation (Neural ODE)":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title"> Exploratory Maternal Simulation (Neural ODE)</div>
        <div class="hero-subtitle">Continuous-Time Numerical Simulation from Booking Baseline | Not a Validated Patient Forecast</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.warning("⚠️ **Illustrative simulation, not a forecast.** This cohort has one glucose value per woman (the OGTT) and no repeated measurements, so the Neural ODE cannot learn how glucose, BMI or blood pressure change over pregnancy. In the held-out check (reports/validated/neural_ode_glucose_validation.txt) the simulated fasting glucose at the OGTT window (week 26) was worse than predicting the cohort average (R² −2.2). The curves below are anchored to the values you enter; risk shown elsewhere comes only from the validated booking models.")
    st.markdown("#### ⏳ Simulated Maternal Trajectories (Weeks 12 to 40 - Illustrative)")

    live_gdm = st.session_state.gdm_result
    vitals_hist = get_vitals_history(st.session_state.patient_id)
    df_traj = ode_predictor.predict_trajectory(
        st.session_state.maternal_inputs,
        vitals_history=vitals_hist
    )
    x_col = "gestational_age_weeks" if "gestational_age_weeks" in df_traj.columns else "gestational_week"
    obs_df = df_traj[df_traj["is_observed_visit"] == True] if "is_observed_visit" in df_traj.columns else df_traj.iloc[0:0]

    st.metric("Validated booking GDM risk", f"{live_gdm['gdm_probability']*100:.1f}%",
              help="From the validated early-booking model. A single probability for the pregnancy; the Neural ODE does not produce a validated risk curve.")
    st.caption(validated_caption(live_gdm, "GDM"))

    # Decomposed Simulated Biomarkers Trajectories (Glucose, BMI, SBP)
    with st.expander("📊 Simulated Biomarkers (Fasting Glucose, BMI, Systolic BP) - illustrative", expanded=True):
        b_c1, b_c2, b_c3 = st.columns(3)
        with b_c1:
            fig_g = go.Figure()
            fig_g.add_trace(go.Scatter(x=df_traj[x_col], y=df_traj["glucose_level"], mode="lines", name="Simulated", line=dict(color="#059669", width=2.5)))
            if not obs_df.empty:
                fig_g.add_trace(go.Scatter(x=obs_df[x_col], y=obs_df["glucose_level"], mode="markers", name="Recorded visit",
                                           marker=dict(size=10, color="#2563EB", symbol="diamond")))
            fig_g.update_layout(title="Simulated Fasting Glucose (mmol/L)", xaxis_title="Week", yaxis_title="mmol/L", height=280, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_g, use_container_width=True)
        with b_c2:
            fig_b = go.Figure()
            fig_b.add_trace(go.Scatter(x=df_traj[x_col], y=df_traj["bmi"], mode="lines", name="Maternal BMI", line=dict(color="#D97706", width=2.5)))
            fig_b.update_layout(title="Simulated BMI Progression", xaxis_title="Week", yaxis_title="kg/m²", height=280, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_b, use_container_width=True)
        with b_c3:
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(x=df_traj[x_col], y=df_traj["systolic_bp"], mode="lines", name="Systolic BP", line=dict(color="#2563EB", width=2.5)))
            fig_s.update_layout(title="Simulated Systolic BP (mmHg)", xaxis_title="Week", yaxis_title="mmHg", height=280, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_s, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"> Trajectory Simulator</div>', unsafe_allow_html=True)
    st.caption("Change a clinical feature to generate a second forecast beside the original trajectory.")

    base_inputs = dict(st.session_state.maternal_inputs)
    sim_col, comparison_col = st.columns([0.85, 1.15])

    with sim_col:
        st.markdown("#### Adjust forecast features")
        sim_age = st.slider(
            "Maternal Age (years)",
            18, 50,
            int(base_inputs.get("AgeAtStartOfSpell", 31)),
            key="forecast_sim_age"
        )
        sim_gravida = st.slider(
            "Gravida (Total Pregnancies)",
            1, 15,
            int(base_inputs.get("Gravida", 1)),
            key="forecast_sim_gravida"
        )
        sim_parity = st.slider(
            "Parity (Prior Live Births)",
            0, 15,
            int(base_inputs.get("Parity", 0)),
            key="forecast_sim_parity"
        )
        sim_csections = st.slider(
            "Previous C-Sections",
            0, 8,
            int(base_inputs.get("No_Of_previous_Csections", 0)),
            key="forecast_sim_csections"
        )
        sim_imdr = st.number_input(
            "Index of Multiple Deprivation Rank (IMDR)",
            1, 35000,
            int(base_inputs.get("Index of Multiple Deprivation Rank", 12183)),
            step=100,
            key="forecast_sim_imdr"
        )
        eth_cohort_list = ["WEU", "GBR", "OTH", "NAF", "MEA", "IND", "CAR", "PAK", "IRL", "BGD", "CAE", "AFE", "ASE"]
        sim_curr_eth = str(base_inputs.get("Ethnicity", "WEU")).upper()
        sim_ethnicity = st.selectbox(
            "Ethnicity (NHS Cohort)",
            eth_cohort_list,
            index=eth_cohort_list.index(sim_curr_eth) if sim_curr_eth in eth_cohort_list else 0,
            key="forecast_sim_ethnicity"
        )
        sim_weight = st.slider(
            "Current Weight (kg)",
            40.0, 150.0,
            float(_num((st.session_state.get("latest_observation") or {}).get("weight_kg"), _num(base_inputs.get("WeightMeasured"), 72.5))),
            step=0.1,
            key="forecast_sim_weight"
        )
        sim_bmi = st.slider(
            "BMI at Booking (kg/m²)",
            15.0, 55.0,
            float(base_inputs.get("Body Mass Index at Booking", 29.4)),
            step=0.1,
            key="forecast_sim_bmi"
        )
        sim_glucose = st.slider(
            "Blood Glucose Level (mmol/L)",
            2.0, 25.0,
            float(base_inputs.get("Glucoselevelblood", 6.3)),
            step=0.1,
            key="forecast_sim_glucose"
        )
        sim_ogtt0 = st.slider(
            "Fasting OGTT 0 min (mmol/L)",
            2.0, 15.0,
            float(base_inputs.get("Glucoselevel0minblood", 5.2)),
            step=0.1,
            key="forecast_sim_ogtt0"
        )
        sim_ogtt120 = st.slider(
            "120-min OGTT (mmol/L)",
            3.0, 25.0,
            float(base_inputs.get("Glucoselevel120minblood", 8.1)),
            step=0.1,
            key="forecast_sim_ogtt120"
        )
        sim_sys_bp = st.slider(
            "Systolic BP (mmHg)",
            70, 220,
            int(base_inputs.get("SystolicBloodPressureCuff", 125)),
            key="forecast_sim_sys_bp"
        )
        sim_dia_bp = st.slider(
            "Diastolic BP (mmHg)",
            40, 140,
            int(base_inputs.get("Diastolic Blood Pressure", 82)),
            key="forecast_sim_dia_bp"
        )
        sim_gestation = st.slider(
            "Gestational age at booking (weeks)",
            4, 42,
            min(max(int(base_inputs.get("Gestation at booking (Weeks)", 12)), 4), 42),
            key="forecast_sim_gestation"
        )
        sim_inputs = dict(base_inputs)
        sim_inputs.update({
            "AgeAtStartOfSpell": sim_age,
            "Ethnicity": sim_ethnicity,
            "Gestation at booking (Weeks)": sim_gestation,
            "Body Mass Index at Booking": sim_bmi,
            "Gravida": sim_gravida,
            "Parity": sim_parity,
            "No_Of_previous_Csections": sim_csections,
            "Index of Multiple Deprivation Rank": sim_imdr,
            "WeightMeasured": sim_weight,
            "Glucoselevel0minblood": sim_ogtt0,
            "Glucoselevelblood": sim_glucose,
            "Glucoselevel120minblood": sim_ogtt120,
            "SystolicBloodPressureCuff": sim_sys_bp,
            "Diastolic Blood Pressure": sim_dia_bp,
            "Obese?": "YES" if sim_bmi >= 30.0 else "NO"
        })
        st.caption("All patient inputs are editable. This Continuous-Time Neural ODE encodes 9 baseline booking clinical vitals + NHS Ethnicity (with diagnostic OGTT glucose removed from inputs to eliminate target leakage). It simulates continuous fasting glucose, BMI progression, and systolic BP via RK4 numerical integration across Weeks 12 to 40.")
        sim_gdm_probability = mat_pred.predict_gdm_risk(sim_inputs)["gdm_probability"]
        sim_traj = ode_predictor.predict_trajectory(sim_inputs)
        sim_x_col = "gestational_age_weeks" if "gestational_age_weeks" in sim_traj.columns else "gestational_week"

    with comparison_col:
        st.markdown("#### Simulated fasting glucose: baseline vs what-if (illustrative)")
        c_r1, c_r2 = st.columns(2)
        c_r1.metric("Booking GDM risk (baseline)", f"{live_gdm['gdm_probability']*100:.1f}%")
        c_r2.metric("Booking GDM risk (what-if)", f"{sim_gdm_probability*100:.1f}%",
                    delta=f"{(sim_gdm_probability - live_gdm['gdm_probability'])*100:+.1f} pts", delta_color="inverse")
        st.caption("Risk comes from the validated booking model; OGTT glucose sliders do not change it (diagnostic values cannot be used to predict GDM). "
                   "The curves are Neural ODE simulations anchored to the entered glucose and are not validated forecasts.")
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(
            x=df_traj[x_col], y=df_traj["glucose_level"],
            mode="lines", name="Baseline (simulated)",
            line=dict(color="#64748B", width=3)
        ))
        fig_sim.add_trace(go.Scatter(
            x=sim_traj[sim_x_col], y=sim_traj["glucose_level"],
            mode="lines+markers", name="What-if (simulated)",
            line=dict(color="#E11D48", width=3), marker=dict(size=5)
        ))
        fig_sim.update_layout(
            height=420,
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC",
            font=dict(color="#1E293B", family="Plus Jakarta Sans, sans-serif"),
            legend=dict(font=dict(color="#1E293B", size=11), bgcolor="#FFFFFF"),
            xaxis=dict(
                title="Gestational Week",
                color="#1E293B",
                title_font=dict(color="#1E293B", size=13),
                tickfont=dict(color="#334155", size=11),
                gridcolor="#CBD5E1"
            ),
            yaxis=dict(
                title="Simulated fasting glucose (mmol/L)",
                color="#1E293B",
                title_font=dict(color="#1E293B", size=13),
                tickfont=dict(color="#334155", size=11),
                gridcolor="#CBD5E1"
            )
        )
        st.plotly_chart(fig_sim, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# MODULE 7: MACROSOMIA RISK MONITOR
# ==============================================================================
elif nav_option == "Macrosomia Risk Monitor":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title"> Dynamic Fetal Macrosomia Risk Monitor</div>
        <div class="hero-subtitle">Multi-Factor Assessment for Fetal Overgrowth (>4000g Birth Weight Risk)</div>
    </div>
    """, unsafe_allow_html=True)

    gdm_prob = st.session_state.gdm_result["gdm_probability"]
    ga_wks = st.number_input(
        "Gestational age at the ultrasound scan (weeks)", min_value=12.0, max_value=42.0,
        value=float(st.session_state.get("macrosomia_scan_ga", min(max(float(st.session_state.get("current_ga_weeks", 28.0)), 12.0), 42.0))),
        step=0.5, key="macrosomia_scan_ga",
        help="Current gestational age at the scan (not the booking GA). The published AC criterion is applied at >= 36 weeks.")

    auto_ac_cm = 0.0
    active_bio = st.session_state.current_fetal_biometrics
    if isinstance(active_bio, dict):
        if "fitted_ellipse" in active_bio and isinstance(active_bio["fitted_ellipse"], dict):
            auto_ac_cm = active_bio["fitted_ellipse"].get("ac_cm", 0.0)
        if auto_ac_cm == 0.0 and "fetal_estimated_ac_cm" in active_bio and active_bio["fetal_estimated_ac_cm"]:
            auto_ac_cm = active_bio["fetal_estimated_ac_cm"]

    if auto_ac_cm > 0 and ("macrosomia_measured_ac" not in st.session_state or st.session_state["macrosomia_measured_ac"] == 0.0):
        st.session_state["macrosomia_measured_ac"] = float(round(auto_ac_cm, 1))

    # --- Optional clinician-supplied measured AC ---
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📐 Fetal Abdominal Circumference (AC)</div>', unsafe_allow_html=True)
    st.info(
        "AC is taken from Module 3 (ACOUSLIC-AI Attention U-Net, ellipse fit + Ramanujan perimeter, 0.28 mm/pixel). "
        "It is a research measurement; you can override it with a sonographer-measured AC below."
    )
    col_ac_in, col_ac_ref = st.columns([1, 1])
    with col_ac_in:
        measured_ac_input = st.number_input(
            "Auto-Extracted AC (cm; modify if needed)",
            min_value=0.0,
            max_value=44.0,
            value=float(st.session_state.get("macrosomia_measured_ac", auto_ac_cm)),
            step=0.2,
            key="macrosomia_measured_ac",
            help="This value was automatically computed from the U-Net mask using your new AC Biometry Pipeline."
        )
    from models.macrosomia_monitor import expected_fetal_ac_cm
    with col_ac_ref:
        ref_ac = expected_fetal_ac_cm(ga_wks)
        st.markdown(f"""
        <div style='padding: 0.8rem; background: #F0FDF4; border: 1px solid #86EFAC; border-radius: 8px; margin-top: 1.8rem;'>
          <div style='font-size: 0.78rem; color: #15803D; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;'>Reference Median AC</div>
          <div style='font-size: 1.4rem; font-weight: 800; color: #166534;'>{ref_ac:.1f} cm</div>
          <div style='font-size: 0.78rem; color: #4B5563;'>At {ga_wks:.0f} wk (Choi 2016, DOI: 10.5468/ogs.2016.59.2.91)</div>
          <div style='font-size: 0.72rem; color: #6B7280; margin-top: 0.3rem;'>Population-level median · NOT a probability</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Build biometrics dict with optional measured AC
    active_biometrics = dict(st.session_state.current_fetal_biometrics)
    effective_ac = float(measured_ac_input) if measured_ac_input > 0 else (float(auto_ac_cm) if auto_ac_cm > 0 else 0.0)
    if effective_ac > 0:
        active_biometrics["measured_ac_cm"] = effective_ac

    res_macro = macro_monitor.evaluate_risk(
        maternal_gdm_prob=gdm_prob,
        gestational_age_weeks=ga_wks,
        fetal_biometrics=active_biometrics,
        **booking_macro_kwargs(st.session_state.maternal_inputs)
    )

    # --- Two-stage macrosomia pathway ---
    booking_macro = mat_pred.predict_macrosomia_risk(st.session_state.maternal_inputs)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🎯 Two-Stage Macrosomia Pathway (birthweight ≥ 4000 g)</div>', unsafe_allow_html=True)
    st.caption(
        "Stage 1: validated early-booking model (held-out evidence in reports/validated/). "
        "Stage 2: published ultrasound criterion — AC ≥ 36 cm (LR+ 6.9, LR− 0.37) or Hadlock EFW ≥ 4000 g "
        "(LR+ 5.7, LR− 0.48); Coomarasamy et al., BJOG 2005;112:1461-6."
    )

    tier = res_macro.get("macrosomia_risk_tier", "Awaiting Data")
    risk_tier = res_macro.get("risk_tier", "")
    stage2 = res_macro.get("stage2", {})
    ac_measured = res_macro.get("ac_measured_cm")
    ac_expected = res_macro.get("ac_expected_cm")
    ac_dev_cm = res_macro.get("ac_deviation_cm")
    ac_dev_pct = res_macro.get("ac_deviation_pct")
    efw = res_macro.get("estimated_fetal_weight_g")
    badge_cls = "metric-badge-high" if risk_tier == "High Risk" else ("metric-badge-low" if risk_tier == "Low Risk" else "metric-badge-mid")

    c_s1, c_s2, c_s3 = st.columns(3)
    with c_s1:
        st.metric("Stage 1 · Booking risk", f"{booking_macro['macrosomia_risk_percent']:.1f}%",
                  help="Calibrated probability of birthweight >= 4000 g from the 10 booking predictors.")
        st.caption("Flagged (top 20% at booking)" if booking_macro["booking_flagged"] else "Not flagged at booking")
    with c_s2:
        if stage2.get("applied"):
            st.metric(f"Stage 2 · {stage2['criterion']}", "Positive" if stage2["positive"] else "Negative",
                      help=f"Published likelihood ratio applied: {stage2['likelihood_ratio']}")
        else:
            st.metric("Stage 2 · Ultrasound criterion", "Not applied")
            st.caption(stage2.get("reason") or "")
    with c_s3:
        post = res_macro.get("posttest_probability")
        st.metric("Combined risk (exploratory)", f"{post*100:.1f}%" if post is not None else "—",
                  help="Booking odds × published LR. Assumes the published LR transfers to this population; "
                       "not validated on linked booking + scan + birthweight data.")

    st.markdown(f"<div style='text-align:center;margin:0.6rem 0;'><span class='{badge_cls}'>{tier}</span></div>", unsafe_allow_html=True)
    st.caption(validated_caption(booking_macro, "macrosomia"))

    c_m1, c_m2 = st.columns([1.1, 1.2])
    with c_m1:
        st.markdown("#### 📏 AC growth context (descriptive)")
        if ac_measured is not None:
            col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
            with col_kpi1:
                st.metric("Measured AC", f"{ac_measured:.1f} cm")
            with col_kpi2:
                st.metric("Median AC (Choi)", f"{ac_expected:.1f} cm", help=f"Choi 2016: GA − 5 at {ga_wks:.0f} weeks")
            with col_kpi3:
                st.metric("Deviation", f"{ac_dev_cm:+.1f} cm" if ac_dev_cm is not None else "—",
                          delta=f"{ac_dev_pct:+.1f}%" if ac_dev_pct is not None else None, delta_color="off")
            if efw:
                st.info(f"Hadlock 4-parameter EFW: **{efw:,.0f} g**")
        else:
            st.info("ℹ️ No AC loaded. Run Module 3 (Fetal Ultrasound) or enter a measured AC above.")

    with c_m2:
        st.markdown("#### 🔬 How this result was reached")
        for rat in res_macro.get("clinical_rationales", []):
            st.markdown(f"- {rat}")
        st.markdown("""
        <div style="font-size: 0.82rem; color: #475569; background: #F1F5F9; border-radius: 6px; padding: 8px 12px; margin-top: 10px;">
            <b>Stage 2 criterion:</b> Coomarasamy A et al. <i>BJOG</i> 2005;112(11):1461-6 (systematic review, 63 accuracy studies).<br/>
            <b>AC median (context only):</b> Choi HY et al. <i>Obstet Gynecol Sci</i> 2016;59(2):91-96 — median AC (cm) = GA − 5, 20–40 wk.<br/>
            <b>Applicability:</b> the published criterion is applied only to scans at ≥ 36 weeks (project rule).
        </div>
        """, unsafe_allow_html=True)

    for w in res_macro.get("ga_warnings", []):
        st.warning(w)

    st.markdown('</div>', unsafe_allow_html=True)

    # --- 3. Disclaimer ---
    st.markdown(f'<div class="disclaimer-box">{res_macro["disclaimer"]}</div>', unsafe_allow_html=True)




# ==============================================================================
# MODULE 7.5: WHAT-IF RISK SIMULATOR
# ==============================================================================
elif nav_option == "What-If Risk Simulator":
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-title"> Interactive What-If Counterfactual & Borderline Tipping Risk Simulator</div>
        <div class="hero-subtitle">Change booking inputs and see how the validated models respond, against the <b>unchanged original baseline</b> (associational, not causal) | Patient: <code>{st.session_state.patient_id}</code></div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Original patient baseline (unchanged reference)
    orig_maternal = dict(st.session_state.maternal_inputs)
    orig_fetal_bio = dict(get_active_fetal_biometrics())
    orig_booking_ga = float(orig_maternal.get("Gestation at booking (Weeks)", 12.0))
    # Current (scan) week is separate from booking GA: it only controls the ultrasound stage.
    orig_scan_ga = float(st.session_state.get("macrosomia_scan_ga", max(float(st.session_state.get("current_ga_weeks", orig_booking_ga)), 12.0)))

    def _whatif_eval(maternal: Dict[str, Any], scan_ga: float, fetal_bio: Dict[str, Any]) -> Dict[str, Any]:
        """Run the validated booking models + two-stage macrosomia pathway on one input set."""
        g = mat_pred.predict_gdm_risk(maternal)
        m = macro_monitor.evaluate_risk(
            maternal_gdm_prob=g["gdm_probability"],
            gestational_age_weeks=scan_ga,
            fetal_biometrics=fetal_bio,
            **booking_macro_kwargs(maternal)
        )
        return {"gdm": g, "gdm_pct": g["gdm_probability"] * 100.0,
                "macro": m, "macro_pct": float(m.get("risk_score", 0.0))}

    orig = _whatif_eval(orig_maternal, orig_scan_ga, orig_fetal_bio)
    orig_traj_df = ode_predictor.predict_trajectory(orig_maternal)

    # 2. Controls
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"> Counterfactual Parameter Adjuster</div>', unsafe_allow_html=True)
    st.info(
        "Adjust the sliders to see how the **validated booking models** respond. The original baseline stays unchanged.\n\n"
        "**How to read this:** the models describe *associations* in 23,490 past pregnancies. A lower predicted risk "
        "after changing BMI or blood pressure shows how risk differs between women with those values; it does **not** "
        "promise that changing this woman's BMI or blood pressure will lower her risk by that amount."
    )

    col_sim1, col_sim2 = st.columns(2)

    with col_sim1:
        st.markdown("####  Booking-model inputs (change the risk)")
        sim_age = st.slider("Maternal age (years)", 18, 50, int(orig_maternal.get("AgeAtStartOfSpell", 31)), key="sim_age")
        sim_gravida = st.slider("Gravida (total pregnancies)", 1, 12, int(orig_maternal.get("Gravida", 1)), key="sim_gravida")
        sim_parity = st.slider("Parity (prior births)", 0, 10, int(orig_maternal.get("Parity", 0)), key="sim_parity")
        sim_csections = st.slider("Previous C-sections", 0, 8, int(orig_maternal.get("No_Of_previous_Csections", 0)), key="sim_csections")
        sim_imdr = st.number_input("Deprivation rank (IMD)", 1, 35000, int(orig_maternal.get("Index of Multiple Deprivation Rank", 12183)), step=100, key="sim_imdr")
        sim_booking_ga = st.slider("Gestational age at booking (weeks)", 4.0, 42.0, float(orig_booking_ga), step=1.0, key="sim_booking_ga")
        st.caption("Booking GA is a model input. It is **not** the current week; use the scan-week slider for that.")

        sim_bmi = st.slider("Body mass index at booking (kg/m²)", 15.0, 50.0, float(orig_maternal.get("Body Mass Index at Booking", 29.4)), step=0.1, key="sim_bmi")
        b_bmi = "Normal (<25.0)" if sim_bmi < 25.0 else "Overweight (25.0-29.9)" if sim_bmi < 30.0 else "Obese (≥30.0)"
        st.caption(f"Category: **{b_bmi}**")

        sim_sys = st.slider("Systolic BP at booking (mmHg)", 80, 200, int(orig_maternal.get("SystolicBloodPressureCuff", 125)), step=1, key="sim_sys")
        sim_dia = st.slider("Diastolic BP at booking (mmHg)", 50, 120, int(orig_maternal.get("Diastolic Blood Pressure", 82)), step=1, key="sim_dia")
        b_bp = "Normal (<120/80)" if (sim_sys < 120 and sim_dia < 80) else "Hypertensive (≥140/90)" if (sim_sys >= 140 or sim_dia >= 90) else "Elevated (120-139 / 80-89)"
        st.caption(f"Category: **{b_bp}**")

    with col_sim2:
        st.markdown("####  Ultrasound stage (macrosomia, Stage 2)")
        sim_scan_ga = st.slider("Current gestational week (scan)", 12.0, 42.0, float(min(max(orig_scan_ga, 12.0), 42.0)), step=1.0, key="sim_scan_ga")
        sim_ac = st.number_input(
            "Measured AC (cm; 0 = unavailable)",
            min_value=0.0, max_value=42.0,
            value=float(orig_fetal_bio.get("measured_ac_cm") or orig_fetal_bio.get("fetal_estimated_ac_cm") or 0.0),
            step=0.2, key="sim_ac"
        )
        st.caption("The published criterion (AC ≥ 36 cm, Coomarasamy 2005) is applied only at ≥ 36 weeks. "
                   "Below 36 weeks the macrosomia risk is the booking risk.")

        st.markdown("####  Illustrative only (do NOT change the risk)")
        st.caption("The validated models use booking data only, because OGTT glucose is the GDM diagnostic test itself. "
                   "These sliders only move the illustrative Neural ODE glucose curve below.")
        sim_glu = st.slider("Random blood glucose (mmol/L)", 3.0, 15.0, float(orig_maternal.get("Glucoselevelblood", 6.3)), step=0.1, key="sim_glu")
        sim_glu0 = st.slider("OGTT fasting (0-min) glucose (mmol/L)", 3.0, 12.0, float(orig_maternal.get("Glucoselevel0minblood", 5.2)), step=0.1, key="sim_glu0")
        sim_glu120 = st.slider("OGTT 2-hour (120-min) glucose (mmol/L)", 4.0, 20.0, float(orig_maternal.get("Glucoselevel120minblood", 8.1)), step=0.1, key="sim_glu120")
        st.caption("NICE NG3 diagnostic values: fasting ≥ 5.6 or 2-hour ≥ 7.8 mmol/L.")

    st.markdown('</div>', unsafe_allow_html=True)

    # 3. What-if state
    sim_maternal = dict(orig_maternal)
    sim_maternal.update({
        "AgeAtStartOfSpell": sim_age,
        "Gravida": sim_gravida,
        "Parity": sim_parity,
        "No_Of_previous_Csections": sim_csections,
        "Index of Multiple Deprivation Rank": sim_imdr,
        "Gestation at booking (Weeks)": sim_booking_ga,
        "Glucoselevelblood": sim_glu,
        "Glucoselevel0minblood": sim_glu0,
        "Glucoselevel120minblood": sim_glu120,
        "Body Mass Index at Booking": sim_bmi,
        "SystolicBloodPressureCuff": sim_sys,
        "Diastolic Blood Pressure": sim_dia,
        "Obese?": "YES" if sim_bmi >= 30.0 else "NO"
    })
    sim_fetal_bio = dict(orig_fetal_bio)
    sim_fetal_bio["measured_ac_cm"] = sim_ac if sim_ac > 0 else None
    sim_fetal_bio["fetal_estimated_ac_cm"] = sim_ac if sim_ac > 0 else None
    sim = _whatif_eval(sim_maternal, sim_scan_ga, sim_fetal_bio)
    sim_traj_df = ode_predictor.predict_trajectory(sim_maternal)

    # 4. Before / after cards (differences in percentage points)
    st.markdown("---")
    st.markdown("###  Risk shift: original baseline vs what-if")
    gdm_delta_pp = sim["gdm_pct"] - orig["gdm_pct"]
    macro_delta_pp = sim["macro_pct"] - orig["macro_pct"]

    def _card(col, title, value_pct, badge_level, color, delta_pp=None):
        with col:
            st.markdown('<div class="glass-card" style="text-align: center;">', unsafe_allow_html=True)
            st.markdown(f"<p style='color: #64748B; margin: 0; font-size: 0.85rem;'>{title}</p>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='margin: 0.3rem 0; font-weight: 800; color: {color};'>{value_pct:.1f}%</h2>", unsafe_allow_html=True)
            badge = "metric-badge-high" if badge_level == "High Risk" else "metric-badge-low"
            st.markdown(f"<span class='{badge}'>{badge_level}</span>", unsafe_allow_html=True)
            if delta_pp is not None:
                d_col = "#DC2626" if delta_pp > 0.05 else "#16A34A" if delta_pp < -0.05 else "#64748B"
                st.markdown(f"<p style='margin: 0.4rem 0 0 0; font-size: 0.85rem; font-weight: 700; color: {d_col};'>"
                            f"Change: {delta_pp:+.1f} percentage points</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    _card(c_m1, "Original GDM risk", orig["gdm_pct"], orig["gdm"]["risk_level"], "#1E293B")
    _card(c_m2, "What-if GDM risk", sim["gdm_pct"], sim["gdm"]["risk_level"], "#2563EB", gdm_delta_pp)
    _card(c_m3, "Original macrosomia risk", orig["macro_pct"], orig["macro"]["risk_tier"], "#1E293B")
    _card(c_m4, "What-if macrosomia risk", sim["macro_pct"], sim["macro"]["risk_tier"], "#7C3AED", macro_delta_pp)
    st.caption(sim["macro"].get("explanation", "") or "")

    # 5. One-factor-at-a-time sensitivity (each row changes ONE factor from the what-if state)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("####  One-factor sensitivity (from the what-if state)")
    st.caption("Each row changes a single factor and keeps every other what-if value fixed, so the effect of each factor is shown separately.")

    def _variant(**updates):
        mat = dict(sim_maternal)
        mat.update(updates)
        if "Body Mass Index at Booking" in updates:
            mat["Obese?"] = "YES" if updates["Body Mass Index at Booking"] >= 30.0 else "NO"
        return _whatif_eval(mat, sim_scan_ga, sim_fetal_bio)

    scenarios = [
        ("BMI 23.5 (normal)", _variant(**{"Body Mass Index at Booking": 23.5})),
        ("BMI 34.0 (obese)", _variant(**{"Body Mass Index at Booking": 34.0})),
        ("BP 115/75 (normal)", _variant(**{"SystolicBloodPressureCuff": 115, "Diastolic Blood Pressure": 75})),
        ("BP 145/95 (hypertensive)", _variant(**{"SystolicBloodPressureCuff": 145, "Diastolic Blood Pressure": 95})),
        ("Age 25", _variant(**{"AgeAtStartOfSpell": 25})),
        ("Age 38", _variant(**{"AgeAtStartOfSpell": 38})),
    ]
    rows = []
    for name, r in scenarios:
        rows.append({
            "Scenario (one change)": name,
            "GDM risk (%)": round(r["gdm_pct"], 1),
            "GDM change (pp)": round(r["gdm_pct"] - sim["gdm_pct"], 1),
            "GDM flag": r["gdm"]["risk_level"],
            "Macrosomia risk (%)": round(r["macro_pct"], 1),
            "Macrosomia change (pp)": round(r["macro_pct"] - sim["macro_pct"], 1),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("pp = percentage points relative to the current what-if values. Associational, not causal.")
    st.markdown('</div>', unsafe_allow_html=True)

    # 6. Illustrative glucose trajectories
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("####  Simulated fasting glucose (illustrative Neural ODE, not validated)")
    st.caption("The Neural ODE failed its held-out glucose check (R² = −2.20 at week 26), so these curves show model behaviour, not a forecast.")
    x_col = "gestational_age_weeks" if "gestational_age_weeks" in orig_traj_df.columns else "gestational_week"
    fig_traj = go.Figure()
    fig_traj.add_trace(go.Scatter(x=orig_traj_df[x_col], y=orig_traj_df["glucose_level"], mode="lines+markers",
                                  name="Original baseline", line=dict(color="#2563EB", width=3)))
    fig_traj.add_trace(go.Scatter(x=sim_traj_df[x_col], y=sim_traj_df["glucose_level"], mode="lines+markers",
                                  name="What-if", line=dict(color="#EC4899", width=3, dash="dash")))
    fig_traj.add_hline(y=5.6, line_dash="dot", line_color="#EF4444",
                       annotation_text="Fasting 5.6 mmol/L (NICE NG3 diagnostic value)", annotation_position="top right")
    fig_traj.update_layout(xaxis_title="Gestational week", yaxis_title="Simulated fasting glucose (mmol/L)",
                           height=420, paper_bgcolor="rgba(0,0,0,0)",
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_traj, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 7. Summary
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("####  Summary")
    st.write(f"- **What-if GDM risk:** {sim['gdm_pct']:.1f}% ({gdm_delta_pp:+.1f} pp vs baseline)")
    st.write(f"- **What-if macrosomia risk:** {sim['macro_pct']:.1f}% ({macro_delta_pp:+.1f} pp vs baseline)")
    if sim["gdm"]["risk_level"] == "High Risk":
        st.error(f"**Flagged at booking** (top 20% risk, ≥ {sim['gdm']['flag_threshold']*100:.1f}%): offer early glucose testing per local protocol.")
    else:
        st.success(f"**Not flagged at booking**: below the validated top-20% threshold ({sim['gdm']['flag_threshold']*100:.1f}%). Routine OGTT at 24–28 weeks still applies.")
    st.markdown('</div>', unsafe_allow_html=True)



# ==============================================================================
# MODULE 8: EXPLAINABILITY (SHAP & VIS)
# ==============================================================================
elif nav_option == "Multimodal Explainability (SHAP)":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title"> Multimodal Explainable AI (XAI)</div>
        <div class="hero-subtitle">SHAP Feature Attribution & Fetal Segmentation Visual Rationales</div>
    </div>
    """, unsafe_allow_html=True)

    tab_shap, tab_vis = st.tabs(["Maternal SHAP Attribution", "Fetal Visual Rationales"])

    with tab_shap:
        st.subheader("Top Clinical Drivers for GDM Prediction")
        try:
            shap_exp = MaternalSHAPExplainer()
            shap_res = shap_exp.explain_patient_prediction(st.session_state.maternal_inputs)
            df_shap = pd.DataFrame(shap_res["top_features"])

            df_shap["color"] = df_shap["shap_value"].apply(lambda x: "#EF4444" if x > 0 else "#10B981")
            fig_shap = px.bar(df_shap, x="shap_value", y="feature", orientation="h", color="color",
                              color_discrete_map={"#EF4444": "#EF4444", "#10B981": "#10B981"},
                              title="SHAP Feature Importance (+ Increases Risk, - Decreases Risk)")
            fig_shap.update_layout(
                height=420,
                showlegend=False,
                template="plotly_white",
                paper_bgcolor="#F6F8FB",
                plot_bgcolor="#FFFFFF",
                font=dict(color="#172033", family="Plus Jakarta Sans, sans-serif"),
                title=dict(font=dict(color="#172033", size=16)),
                xaxis=dict(
                    title="SHAP value",
                    color="#344054",
                    title_font=dict(color="#172033", size=13),
                    tickfont=dict(color="#344054", size=11),
                    gridcolor="#E4E7EC",
                    zerolinecolor="#98A2B3"
                ),
                yaxis=dict(
                    title="Feature",
                    color="#344054",
                    title_font=dict(color="#172033", size=13),
                    tickfont=dict(color="#344054", size=11),
                    gridcolor="#E4E7EC"
                )
            )
            st.plotly_chart(fig_shap, use_container_width=True)
        except Exception as e:
            st.info(f"SHAP explanation note: {e}")

    with tab_vis:
        st.subheader("Fetal Ultrasound Segmentation Visual Rationales")
        if st.session_state.current_fetal_img is not None and st.session_state.current_fetal_mask is not None:
            fig_exp = generate_fetal_explainability_figure(
                st.session_state.current_fetal_img,
                st.session_state.current_fetal_mask,
                st.session_state.current_fetal_biometrics
            )
            st.pyplot(fig_exp)
        else:
            st.warning(" Please run Fetal Ultrasound Segmentation on Module 3 first.")


# ==============================================================================
# MODULE 9: CLINICAL ACTION PLAN
# ==============================================================================
elif nav_option == "Clinical Action Plan":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title"> Actionable Clinical Recommendations</div>
        <div class="hero-subtitle">Tailored Guidance for Obstetricians & Patients based on Multimodal Digital Twin Risk</div>
    </div>
    """, unsafe_allow_html=True)

    gdm_lvl = st.session_state.gdm_result["risk_level"]
    gdm_prob = st.session_state.gdm_result["gdm_probability"]
    ga_wks = float(st.session_state.get("current_ga_weeks", st.session_state.maternal_inputs["Gestation at booking (Weeks)"]))

    macro_res = macro_monitor.evaluate_risk(maternal_gdm_prob=gdm_prob, gestational_age_weeks=ga_wks, **booking_macro_kwargs(st.session_state.maternal_inputs))
    macro_tier = macro_res["risk_tier"]
    render_ogtt_status(st.session_state.get("ogtt_result"))

    recs = rec_engine.generate_recommendations(
        gdm_risk_level=gdm_lvl,
        macrosomia_tier=macro_tier,
        gestational_age_weeks=ga_wks
    )

    col_phys, col_pat = st.columns(2)
    with col_phys:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader(" Obstetrician Clinical Protocol")
        for r in recs["physician_action_plan"]:
            st.markdown(f"- <b>{r}</b>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_pat:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader(" Patient Lifestyle & Dietary Guidance")
        for r in recs["patient_guidance"]:
            st.markdown(f"- <b>{r}</b>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
