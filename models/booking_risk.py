"""
models/booking_risk.py
======================
Shared, validated early-booking risk models for MDT-GEN (GDM and macrosomia).

This module holds everything that must be identical between training and the
dashboard, so the probabilities shown to users are exactly the ones evaluated
in reports/validated/:

* the 10 booking predictors and the data-quality rules,
* the fold-safe preprocessor,
* the stacking ensemble (LR, RF, XGBoost, CatBoost -> LR meta-learner),
* ``BookingRiskModel``: loads a saved bundle and scores one patient.

No class re-weighting is used, so outputs are calibrated probabilities on the
real risk scale. Training lives in models/train_validated_booking_models.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
VALIDATED_MODELS_DIR = BASE_DIR / "models" / "validated"
GDM_BUNDLE = VALIDATED_MODELS_DIR / "gdm_booking_model.joblib"
MACROSOMIA_BUNDLE = VALIDATED_MODELS_DIR / "macrosomia_booking_model.joblib"

RANDOM_STATE = 42

NUM_FEATURES = [
    "AgeAtStartOfSpell",
    "Gestation at booking (Weeks)",
    "Body Mass Index at Booking",
    "Gravida",
    "Parity",
    "No_Of_previous_Csections",
    "Index of Multiple Deprivation Rank",
    "SystolicBloodPressureCuff",
    "Diastolic Blood Pressure",
]
CAT_FEATURES = ["Ethnicity"]
MISSING_FLAG_FEATURES = ["SystolicBloodPressureCuff", "Diastolic Blood Pressure"]

# Never allowed as predictors (post-booking, diagnostic or outcome-derived).
FORBIDDEN = {
    "BabyBirthWeight", "BW Centile", "Gestational Diabetes", "Glucoselevel0minblood",
    "Glucoselevel120minblood", "GlucoseToleranceTest", "Glucoselevelblood", "Gestation",
    "Gestation (Days)", "Delivery_Outcome", "Shoulder Dystocia", "Sex", "Still_Birth",
}
assert not FORBIDDEN & set(NUM_FEATURES + CAT_FEATURES), "Leakage: forbidden column in predictors"

QUALITY_RULES_TEXT = (
    "booking GA outside 4-42 weeks -> missing; Gravida > 20 -> missing; "
    "systolic BP outside 70-250 mmHg -> missing; diastolic BP outside 30-150 mmHg -> missing. "
    "Missing values are median-imputed inside each fold, with missingness indicators for the "
    "two BP columns."
)


def apply_quality_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Set physiologically implausible booking values to missing (no fitting involved)."""
    df = df.copy()
    for c in NUM_FEATURES:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Ethnicity" not in df.columns:
        df["Ethnicity"] = "Unknown"
    df["Ethnicity"] = df["Ethnicity"].fillna("Unknown").astype(str).str.strip().str.upper()
    ga = df["Gestation at booking (Weeks)"]
    df.loc[(ga < 4) | (ga > 42), "Gestation at booking (Weeks)"] = np.nan
    df.loc[df["Gravida"] > 20, "Gravida"] = np.nan
    sbp = df["SystolicBloodPressureCuff"]
    df.loc[(sbp < 70) | (sbp > 250), "SystolicBloodPressureCuff"] = np.nan
    dbp = df["Diastolic Blood Pressure"]
    df.loc[(dbp < 30) | (dbp > 150), "Diastolic Blood Pressure"] = np.nan
    return df


def _clean(c: str) -> str:
    return c.replace("[", "_").replace("]", "_").replace("<", "_").replace(" ", "_")


class BookingPreprocessor:
    """Median imputation + BP missingness flags + scaling + one-hot. Fit on training data only."""

    def fit(self, df: pd.DataFrame) -> "BookingPreprocessor":
        self.imp = SimpleImputer(strategy="median").fit(df[NUM_FEATURES])
        num = self.imp.transform(df[NUM_FEATURES])
        flags = df[MISSING_FLAG_FEATURES].isna().astype(float).values
        self.scaler = StandardScaler().fit(np.hstack([num, flags]))
        self.enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df[CAT_FEATURES].astype(str))
        cat_names = [f"Ethnicity_{c}" for c in self.enc.categories_[0]]
        self.columns = [_clean(c) for c in
                        NUM_FEATURES + [f"{c}_missing" for c in MISSING_FLAG_FEATURES] + cat_names]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        num = self.imp.transform(df[NUM_FEATURES])
        flags = df[MISSING_FLAG_FEATURES].isna().astype(float).values
        X = np.hstack([self.scaler.transform(np.hstack([num, flags])),
                       self.enc.transform(df[CAT_FEATURES].astype(str))])
        return pd.DataFrame(X, columns=self.columns)


def make_base_models(class_weight: Optional[str] = None) -> Dict[str, Any]:
    """Base learners. class_weight='balanced' reproduces the legacy (uncalibrated) setup."""
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier

    xgb_kw, cat_kw = {}, {}
    if class_weight == "balanced":
        xgb_kw = {"scale_pos_weight": 11.0}
        cat_kw = {"auto_class_weights": "Balanced"}
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight=class_weight,
                                                  random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20,
                                                class_weight=class_weight,
                                                random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                                 colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1,
                                 eval_metric="logloss", **xgb_kw),
        "CatBoost": CatBoostClassifier(iterations=400, depth=4, learning_rate=0.03,
                                       random_state=RANDOM_STATE, verbose=0, thread_count=-1, **cat_kw),
    }


def fit_stack(train_df: pd.DataFrame, y: np.ndarray, class_weight: Optional[str] = None, n_splits: int = 5):
    """5-fold out-of-fold stacking. Returns (preprocessor, base models, meta-learner, oof base probs, oof stack probs)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    names = list(make_base_models(class_weight))
    oof = np.zeros((len(train_df), len(names)))
    for ti, vi in skf.split(train_df, y):
        pre = BookingPreprocessor().fit(train_df.iloc[ti])
        Xtr, Xva = pre.transform(train_df.iloc[ti]), pre.transform(train_df.iloc[vi])
        for j, m in enumerate(make_base_models(class_weight).values()):
            m.fit(Xtr, y[ti])
            oof[vi, j] = m.predict_proba(Xva)[:, 1]
    meta = LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=RANDOM_STATE).fit(oof, y)
    oof_stack = meta.predict_proba(oof)[:, 1]

    pre = BookingPreprocessor().fit(train_df)
    Xfull = pre.transform(train_df)
    bases = {name: m.fit(Xfull, y) for name, m in make_base_models(class_weight).items()}
    return pre, bases, meta, oof, oof_stack


def predict_stack(pre, bases, meta, df: pd.DataFrame):
    X = pre.transform(df)
    base_p = np.column_stack([m.predict_proba(X)[:, 1] for m in bases.values()])
    return base_p, meta.predict_proba(base_p)[:, 1]


# ---------------------------------------------------------------------------
# Runtime scoring
# ---------------------------------------------------------------------------

class BookingRiskModel:
    """Loads a validated bundle and scores a single patient dictionary."""

    def __init__(self, bundle_path: Path):
        self.bundle_path = Path(bundle_path)
        self.bundle: Optional[dict] = None
        self.load_error: Optional[str] = None
        try:
            self.bundle = joblib.load(self.bundle_path)
        except Exception as e:  # missing file or library-version mismatch
            self.load_error = f"{type(e).__name__}: {e}"
            logger.warning("Could not load %s (%s). Run models/train_validated_booking_models.py.",
                           self.bundle_path, self.load_error)

    @property
    def is_loaded(self) -> bool:
        return self.bundle is not None

    def predict(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError(
                f"Validated model not available at {self.bundle_path} ({self.load_error}). "
                "Run: python models/train_validated_booking_models.py")
        b = self.bundle
        df = apply_quality_rules(pd.DataFrame([raw]))
        base_p, p = predict_stack(b["preprocessor"], b["base_models"], b["meta_learner"], df)
        prob = float(p[0])
        prev = float(b["prevalence_train"])
        t = float(b["threshold_top20pct_oof"])
        op = b["test_operating_point_top20"]
        flagged = prob >= t
        return {
            "outcome": b["outcome"],
            "probability": prob,
            "risk_percent": round(prob * 100.0, 1),
            "population_rate_percent": round(prev * 100.0, 1),
            "relative_risk_vs_population": round(prob / prev, 2) if prev > 0 else None,
            "flagged": bool(flagged),
            "risk_level": "Higher risk (top 20% at booking)" if flagged else "Not flagged at booking",
            "flag_threshold": t,
            "validated_performance": {
                "test_auroc": b["test_metrics"]["AUROC"],
                "test_auroc_95ci": b["test_metrics"]["AUROC_95CI"],
                "test_brier": b["test_metrics"]["Brier"],
                "calibration_slope": b["test_metrics"]["Calibration_slope"],
                "flag_sensitivity": op["sensitivity"],
                "flag_specificity": op["specificity"],
                "flag_ppv": op["ppv"],
                "n_test": b["n_test"],
            },
            "base_probabilities": {k: float(v) for k, v in zip(b["base_models"].keys(), base_p[0])},
        }
