"""
PHASE 3: DIGITAL TWIN EARLY SCREENING ENGINE
============================================
Implements the exact logic from Slides 61-65:
- Slide 61: Baseline AC Ramanujan (Alternate Form)
- Slide 62: Neural ODE RK4 Fetal Growth Forecasting
- Slide 63: Choi Reference vs Forecasted Growth (Week 40)
- Slide 64: Future Deviation & Dynamic Risk Tiering
- Slide 65: Macrosomia JSON Output Formatting
"""

import math
import json
from typing import Callable

# ==========================================
# SLIDE 61: BASELINE FETAL STATE EXTRACTION
# ==========================================
def extract_baseline_ac_slide61(a: float, b: float) -> float:
    """
    Calculates AC using the specific Ramanujan formulation from Slide 61.
    AC_obs(t_c) ≈ π * [3(a + b) - sqrt((3a + b)(a + 3b))]
    
    Args:
        a: Semi-major axis in cm
        b: Semi-minor axis in cm
    """
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


# ==========================================
# SLIDE 62: FETAL GROWTH FORECASTING (ODE)
# ==========================================
def growth_derivative(t: float, alpha: float, glucose_fn: Callable, beta: float, bmi_fn: Callable) -> float:
    """
    Fetal Growth Differential Equation: d(AC)/dt = α * Glucose_pred(t) + β * BMI_pred(t)
    """
    return alpha * glucose_fn(t) + beta * bmi_fn(t)

def rk4_integrate_growth(
    t_c: float, 
    ac_obs: float, 
    t_d: float, 
    alpha: float, 
    glucose_fn: Callable, 
    beta: float, 
    bmi_fn: Callable, 
    dt: float = 0.5
) -> float:
    """
    Neural ODE RK4 Solver to forecast Birth AC (t_d = 40).
    """
    t = t_c
    ac = ac_obs
    
    while t < t_d:
        step = min(dt, t_d - t)
        # RK4 Steps
        k1 = step * growth_derivative(t, alpha, glucose_fn, beta, bmi_fn)
        k2 = step * growth_derivative(t + step/2, alpha, glucose_fn, beta, bmi_fn)
        k3 = step * growth_derivative(t + step/2, alpha, glucose_fn, beta, bmi_fn)
        k4 = step * growth_derivative(t + step, alpha, glucose_fn, beta, bmi_fn)
        
        ac += (k1 + 2*k2 + 2*k3 + k4) / 6.0
        t += step
        
    return ac


# ==========================================
# SLIDE 63: CHOI REFERENCE
# ==========================================
def choi_expected_ac(t: float) -> float:
    """
    Choi AC_exp(t) = 1.013 * t - 4.634
    """
    return 1.013 * t - 4.634


# ==========================================
# SLIDE 64 & 65: RISK TIERS & OUTPUT
# ==========================================
def run_digital_twin_screening(
    t_c: float,
    ac_obs: float,
    p_gdm: float,
    glucose_fn: Callable,
    bmi_fn: Callable,
    alpha: float = 0.015,
    beta: float = 0.01
) -> dict:
    
    # Current state
    ac_exp_current = choi_expected_ac(t_c)
    current_deviation = ((ac_obs - ac_exp_current) / max(ac_exp_current, 1e-5)) * 100.0
    
    # Future forecast (Slide 62 & 63)
    t_d = 40.0
    ac_pred_40 = rk4_integrate_growth(t_c, ac_obs, t_d, alpha, glucose_fn, beta, bmi_fn)
    ac_exp_40 = choi_expected_ac(t_d)
    
    # Forecasted Relative Deviation (Slide 64)
    future_deviation = ((ac_pred_40 - ac_exp_40) / max(ac_exp_40, 1e-5)) * 100.0
    
    # Dynamic Risk Tiering (Slide 64)
    if current_deviation >= 10.0:
        tier = "Tier 3: Late/Biometric Escalation"
        status = "Critical Escalation"
        rec = "Immediate clinical intervention and close fetal monitoring required."
        exp = f"Current ultrasound AC already exceeds expected 90th percentile ({current_deviation:+.1f}%). High risk of macrosomia established."
    elif future_deviation >= 10.0:
        tier = "Tier 2: Trajectory Escalation"
        status = "Early Warning Generated"
        rec = "Initiate maternal glycemic control immediately to alter fetal growth trajectory."
        # Use the exact string from slide 65
        exp = f"Current ultrasound is normal, but high GDM probability ({int(p_gdm*100)}%) and Neural ODE glucose forecast indicate high risk of future macrosomia."
    else:
        tier = "Tier 1: Normal"
        status = "Routine Monitoring"
        rec = "Continue standard prenatal care."
        exp = "Current ultrasound and forecasted growth trajectories are within normal physiological bounds."
        
    # JSON Formatting (Slide 65)
    return {
        "screening_status": status,
        "risk_tier": tier,
        "forecasted_deviation": f"{future_deviation:+.1f}% by Week 40",
        "explanation": exp,
        "clinical_recommendation": rec
    }


if __name__ == "__main__":
    print("Executing Digital Twin Early Screening Engine (Slides 61-65)...\n")
    
    # ---------------------------------------------------------
    # Replicating Slide 65 Example
    # Inputs: P_GDM = 0.82, Current GA = 24, Current AC = Normal
    # ---------------------------------------------------------
    t_c = 24.0
    p_gdm = 0.82
    
    # Establish a "Normal" current AC using Slide 61 Ramanujan
    # Expected at 24 weeks is ~19.678 cm. 
    # Let's mock semi-axes that yield exactly normal AC
    # If a=3.4, b=2.864...
    a_mock, b_mock = 3.4, 2.8642
    ac_obs = extract_baseline_ac_slide61(a_mock, b_mock)
    
    # To hit exactly "+14.2% by Week 40" as shown in Slide 65:
    # ac_exp_40 = 35.886
    # Target ac_pred_40 = 35.886 * 1.142 = 40.9818
    # Delta AC needed = 40.9818 - 19.678 = 21.3038 over 16 weeks (~1.331 cm/wk)
    
    # Mocking aggressive maternal trajectories that average 1.331 d(AC)/dt
    alpha_test = 0.015
    beta_test = 0.01
    
    def aggressive_glucose(t): return 75.0  # Hyperglycemic trajectory
    def aggressive_bmi(t): return 20.6      # High steady BMI metric
    # (0.015 * 75.0) + (0.01 * 20.6) = 1.125 + 0.206 = 1.331 cm/week!

    slide_65_output = run_digital_twin_screening(
        t_c=t_c,
        ac_obs=ac_obs,
        p_gdm=p_gdm,
        glucose_fn=aggressive_glucose,
        bmi_fn=aggressive_bmi,
        alpha=alpha_test,
        beta=beta_test
    )
    
    print("========================================")
    print(" SLIDE 65 OUTPUT (JSON REPLICATION)     ")
    print("========================================")
    print(json.dumps(slide_65_output, indent=2))
