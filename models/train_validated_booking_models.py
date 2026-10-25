"""
models/train_validated_booking_models.py
========================================
Trains and validates the two EARLY-BOOKING risk models used by the MDT-GEN
dashboard, and writes the evidence to reports/validated/.

  1. GDM            : outcome = "Gestational Diabetes" (Yes/No)
  2. Macrosomia     : outcome = BabyBirthWeight >= 4000 g
                      (sensitivity outcome: LGA = BW Centile >= 90)

Both use only the 10 booking predictors, the same leakage-safe 80/20 split
(data/maternal/processed/train_clean.csv / test_clean.csv), 5-fold
out-of-fold stacking, and NO class re-weighting, so predictions are
calibrated probabilities. For GDM the legacy class-weighted ensemble is
retrained for comparison, to document the effect of recalibration.

Evidence written per model: held-out AUROC / AUPRC / Brier with 1,000x
bootstrap 95% CIs, scaled Brier, calibration intercept and slope, reliability
plot, decision curve, operating points chosen on OOF training data only,
SHAP (CatBoost base model), and a booking GA <= 12 weeks subgroup.

Run:  python models/train_validated_booking_models.py      (~3-4 minutes)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.booking_risk import (  # noqa: E402
    BASE_DIR, GDM_BUNDLE, MACROSOMIA_BUNDLE, VALIDATED_MODELS_DIR, NUM_FEATURES, CAT_FEATURES,
    QUALITY_RULES_TEXT, apply_quality_rules, fit_stack, predict_stack,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("validated_booking_models")

PROCESSED_DIR = BASE_DIR / "data" / "maternal" / "processed"
REPORTS_DIR = BASE_DIR / "reports" / "validated"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
VALIDATED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_BOOT = 1000


def calibration_intercept_slope(y, p):
    """Logistic recalibration: logit(P(y=1)) = a + b*logit(p). Ideal a=0 (calibration-in-the-large via offset), b=1."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    slope = LogisticRegression(C=1e6, max_iter=1000).fit(lp, y).coef_[0][0]
    # Intercept with slope fixed at 1 (offset model) via 1-D Newton iterations.
    a = 0.0
    for _ in range(50):
        q = 1 / (1 + np.exp(-(a + lp[:, 0])))
        g = np.sum(y - q)
        h = -np.sum(q * (1 - q))
        a -= g / h
    return float(a), float(slope)


def bootstrap_ci(y, p, fn, n=N_BOOT, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if y[b].min() == y[b].max():
            continue
        vals.append(fn(y[b], p[b]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def scaled_brier(y, p):
    ref = brier_score_loss(y, np.full_like(p, y.mean(), dtype=float))
    return 1 - brier_score_loss(y, p) / ref


def threshold_metrics(y, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(t),
        "flagged_pct": float(pred.mean() * 100),
        "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "ppv": tp / (tp + fp) if tp + fp else 0.0,
        "npv": tn / (tn + fn) if tn + fn else 0.0,
    }


def net_benefit(y, p, thresholds):
    n = len(y)
    out = []
    for t in thresholds:
        pred = p >= t
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        out.append(tp / n - fp / n * (t / (1 - t)))
    return np.array(out)


def full_metrics(y, p):
    a, b = calibration_intercept_slope(y, p)
    return {
        "AUROC": roc_auc_score(y, p),
        "AUROC_95CI": bootstrap_ci(y, p, roc_auc_score),
        "AUPRC": average_precision_score(y, p),
        "AUPRC_95CI": bootstrap_ci(y, p, average_precision_score),
        "Brier": brier_score_loss(y, p),
        "Brier_95CI": bootstrap_ci(y, p, brier_score_loss),
        "Scaled_Brier": scaled_brier(y, p),
        "Calibration_intercept": a,
        "Calibration_slope": b,
        "Mean_predicted": float(p.mean()),
        "Observed_rate": float(y.mean()),
    }


def fmt_ci(m, k):
    lo, hi = m[f"{k}_95CI"]
    return f"{m[k]:.3f} ({lo:.3f}-{hi:.3f})"


def results_table(res):
    rows = []
    for name, m in res["metrics"].items():
        rows.append({
            "Model": name,
            "AUROC (95% CI)": fmt_ci(m, "AUROC"),
            "AUPRC (95% CI)": fmt_ci(m, "AUPRC"),
            "Brier (95% CI)": fmt_ci(m, "Brier"),
            "Scaled Brier": f"{m['Scaled_Brier']:.3f}",
            "Cal. intercept": f"{m['Calibration_intercept']:.2f}",
            "Cal. slope": f"{m['Calibration_slope']:.2f}",
            "OOF AUROC": f"{m['OOF_AUROC']:.3f}",
        })
    return pd.DataFrame(rows).set_index("Model")


def make_plots(y_test, stack_test, base_test, base_names, prefix):
    """ROC, PR, reliability and decision-curve plots for one outcome."""
    cols = ["#7f7f7f", "#2ca02c", "#ff7f0e", "#1f77b4", "#d62728"]
    probs = dict(zip(base_names, base_test.T))
    probs["Stacking Ensemble"] = stack_test

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    for (n, p), c in zip(probs.items(), cols):
        f, t, _ = roc_curve(y_test, p)
        ax[0].plot(f, t, color=c, lw=2.4 if "Stack" in n else 1.5, label=f"{n} ({roc_auc_score(y_test, p):.3f})")
        pr, rc, _ = precision_recall_curve(y_test, p)
        ax[1].plot(rc, pr, color=c, lw=2.4 if "Stack" in n else 1.5, label=f"{n} ({average_precision_score(y_test, p):.3f})")
        pt, pp = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
        ax[2].plot(pp, pt, "o-", color=c, lw=2.4 if "Stack" in n else 1.5, label=n)
    ax[0].plot([0, 1], [0, 1], "k--", alpha=.5); ax[0].set(xlabel="1 - Specificity", ylabel="Sensitivity", title="ROC (held-out test)")
    ax[1].axhline(y_test.mean(), color="k", ls="--", alpha=.5); ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    lim = max(0.3, float(np.max(stack_test)) * 1.1)
    ax[2].plot([0, lim], [0, lim], "k--", alpha=.5); ax[2].set(xlim=(0, lim), ylim=(0, lim), xlabel="Mean predicted risk", ylabel="Observed rate", title="Calibration (deciles)")
    for a in ax:
        a.legend(fontsize=8, loc="best")
    fig.tight_layout(); fig.savefig(REPORTS_DIR / f"{prefix}_roc_pr_calibration.png", dpi=200); plt.close(fig)

    ts = np.linspace(0.01, 0.40, 40)
    prev = y_test.mean()
    fig, a = plt.subplots(figsize=(7, 5))
    a.plot(ts, net_benefit(y_test, stack_test, ts), color="#d62728", lw=2.4, label="Stacking model")
    a.plot(ts, prev - (1 - prev) * ts / (1 - ts), color="#7f7f7f", label="Treat all")
    a.axhline(0, color="k", lw=1, label="Treat none")
    a.set(ylim=(-0.02, prev * 1.2), xlabel="Threshold probability", ylabel="Net benefit", title="Decision curve (held-out test)")
    a.legend(); fig.tight_layout(); fig.savefig(REPORTS_DIR / f"{prefix}_decision_curve.png", dpi=200); plt.close(fig)


def shap_plot(pre, bases, test_df, prefix):
    try:
        import shap
        X = pre.transform(test_df)
        sv = shap.TreeExplainer(bases["CatBoost"]).shap_values(X)
        plt.figure()
        shap.summary_plot(sv, X, show=False, max_display=12)
        plt.title("SHAP - CatBoost base model (held-out test)")
        plt.tight_layout(); plt.savefig(REPORTS_DIR / f"{prefix}_shap_summary.png", dpi=200); plt.close("all")
        imp = pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False)
        return imp.round(4).head(12).to_dict()
    except Exception as e:  # SHAP is optional for the evidence table
        logger.warning("SHAP skipped: %s", e)
        return {}



def evaluate(train_df, test_df, y_tr, y_te, class_weight=None):
    pre, bases, meta, oof, oof_stack = fit_stack(train_df, y_tr, class_weight=class_weight)
    base_test, stack_test = predict_stack(pre, bases, meta, test_df)
    metrics = {}
    for j, name in enumerate(bases):
        metrics[name] = full_metrics(y_te, base_test[:, j])
        metrics[name]["OOF_AUROC"] = roc_auc_score(y_tr, oof[:, j])
    metrics["Stacking Ensemble"] = full_metrics(y_te, stack_test)
    metrics["Stacking Ensemble"]["OOF_AUROC"] = roc_auc_score(y_tr, oof_stack)
    fpr, tpr, thr = roc_curve(y_tr, oof_stack)
    youden_t = float(thr[np.argmax(tpr - fpr)])
    top20_t = float(np.quantile(oof_stack, 0.80))
    ops = {"youden_oof": threshold_metrics(y_te, stack_test, youden_t),
           "top20pct_oof": threshold_metrics(y_te, stack_test, top20_t)}
    res = {"n_train": int(len(y_tr)), "n_test": int(len(y_te)),
           "prevalence_train": float(y_tr.mean()), "prevalence_test": float(y_te.mean()),
           "metrics": metrics, "operating_points": ops}
    fitted = dict(pre=pre, bases=bases, meta=meta, base_test=base_test, stack_test=stack_test,
                  youden_t=youden_t, top20_t=top20_t)
    return res, fitted


def subgroup_le12(test_df, y_te, p):
    m = (test_df["Gestation at booking (Weeks)"] <= 12).values
    return {"n": int(m.sum()), "prevalence": float(y_te[m].mean()),
            "AUROC": roc_auc_score(y_te[m], p[m]),
            "AUROC_95CI": bootstrap_ci(y_te[m], p[m], roc_auc_score),
            "Brier": brier_score_loss(y_te[m], p[m])}


def save_bundle(path, outcome, res, fitted):
    joblib.dump({
        "outcome": outcome,
        "preprocessor": fitted["pre"], "base_models": fitted["bases"], "meta_learner": fitted["meta"],
        "num_features": NUM_FEATURES, "cat_features": CAT_FEATURES,
        "quality_rules": QUALITY_RULES_TEXT,
        "prevalence_train": res["prevalence_train"], "n_test": res["n_test"],
        "threshold_youden_oof": fitted["youden_t"], "threshold_top20pct_oof": fitted["top20_t"],
        "test_metrics": res["metrics"]["Stacking Ensemble"],
        "test_operating_point_top20": res["operating_points"]["top20pct_oof"],
        "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "library_versions": library_versions(),
    }, path)
    logger.info("Saved %s", path)


def library_versions():
    import xgboost, catboost
    return {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__, "catboost": catboost.__version__,
            "numpy": np.__version__, "pandas": pd.__version__}


def ops_text(ops):
    out = []
    for key, label in [("youden_oof", "Youden"), ("top20pct_oof", "Top 20% risk (dashboard flag)")]:
        o = ops[key]
        out.append(f"- {label} (t={o['threshold']:.3f}): flagged {o['flagged_pct']:.1f}%, sens {o['sensitivity']:.3f}, "
                   f"spec {o['specificity']:.3f}, PPV {o['ppv']:.3f}, NPV {o['npv']:.3f}")
    return "\n".join(out)


def jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def main():
    train_df = apply_quality_rules(pd.read_csv(PROCESSED_DIR / "train_clean.csv", low_memory=False))
    test_df = apply_quality_rules(pd.read_csv(PROCESSED_DIR / "test_clean.csv", low_memory=False))
    report = {"quality_rules": QUALITY_RULES_TEXT, "library_versions": library_versions(),
              "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    # ======================= GDM =======================
    gdm_map = {"YES": 1, "NO": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}
    for d in (train_df, test_df):
        d["_gdm"] = d["Gestational Diabetes"].astype(str).str.strip().str.upper().map(gdm_map)
    tr_g = train_df.dropna(subset=["_gdm"]).reset_index(drop=True)
    te_g = test_df.dropna(subset=["_gdm"]).reset_index(drop=True)
    yg_tr, yg_te = tr_g["_gdm"].astype(int).values, te_g["_gdm"].astype(int).values

    logger.info("GDM: calibrated ensemble (no class weights)")
    gdm, gf = evaluate(tr_g, te_g, yg_tr, yg_te)
    logger.info("GDM: legacy class-weighted ensemble (comparison only)")
    gdm_legacy, glf = evaluate(tr_g, te_g, yg_tr, yg_te, class_weight="balanced")
    gdm["subgroup_booking_le_12wk"] = subgroup_le12(te_g, yg_te, gf["stack_test"])
    make_plots(yg_te, gf["stack_test"], gf["base_test"], list(gf["bases"]), "gdm_booking")
    gdm["shap_mean_abs_top12"] = shap_plot(gf["pre"], gf["bases"], te_g, "gdm_booking")
    save_bundle(GDM_BUNDLE, "Gestational diabetes", gdm, gf)

    # before/after calibration figure
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for p, lab, c in [(glf["stack_test"], "Legacy class-weighted", "#7f7f7f"), (gf["stack_test"], "Recalibrated (unweighted)", "#d62728")]:
        pt, pp = calibration_curve(yg_te, p, n_bins=10, strategy="quantile")
        ax.plot(pp, pt, "o-", color=c, label=f"{lab} (Brier {brier_score_loss(yg_te, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=.5)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed GDM rate", title="GDM calibration before vs after (held-out test)")
    ax.legend(); fig.tight_layout(); fig.savefig(REPORTS_DIR / "gdm_calibration_before_after.png", dpi=200); plt.close(fig)
    report["gdm"] = gdm
    report["gdm_legacy_class_weighted"] = {"metrics": {"Stacking Ensemble": gdm_legacy["metrics"]["Stacking Ensemble"]}}

    # ======================= Macrosomia =======================
    for d in (train_df, test_df):
        d["BabyBirthWeight"] = pd.to_numeric(d["BabyBirthWeight"], errors="coerce")
        d["BW Centile"] = pd.to_numeric(d["BW Centile"], errors="coerce")
    tr_m = train_df.dropna(subset=["BabyBirthWeight"]).reset_index(drop=True)
    te_m = test_df.dropna(subset=["BabyBirthWeight"]).reset_index(drop=True)
    ym_tr = (tr_m["BabyBirthWeight"] >= 4000).astype(int).values
    ym_te = (te_m["BabyBirthWeight"] >= 4000).astype(int).values
    logger.info("Macrosomia: birthweight >= 4000 g")
    mac, mf = evaluate(tr_m, te_m, ym_tr, ym_te)
    mac["subgroup_booking_le_12wk"] = subgroup_le12(te_m, ym_te, mf["stack_test"])
    make_plots(ym_te, mf["stack_test"], mf["base_test"], list(mf["bases"]), "macrosomia_booking")
    mac["shap_mean_abs_top12"] = shap_plot(mf["pre"], mf["bases"], te_m, "macrosomia_booking")
    save_bundle(MACROSOMIA_BUNDLE, "Macrosomia (birthweight >= 4000 g)", mac, mf)
    report["macrosomia"] = mac

    tr_l = train_df.dropna(subset=["BW Centile"]).reset_index(drop=True)
    te_l = test_df.dropna(subset=["BW Centile"]).reset_index(drop=True)
    logger.info("Sensitivity: LGA (BW centile >= 90)")
    lga, _ = evaluate(tr_l, te_l, (tr_l["BW Centile"] >= 90).astype(int).values,
                      (te_l["BW Centile"] >= 90).astype(int).values)
    report["macrosomia_sensitivity_LGA"] = lga

    (REPORTS_DIR / "booking_models_report.json").write_text(json.dumps(report, indent=2, default=jsonable))

    gl = gdm_legacy["metrics"]["Stacking Ensemble"]; gn = gdm["metrics"]["Stacking Ensemble"]
    sg, sm = gdm["subgroup_booking_le_12wk"], mac["subgroup_booking_le_12wk"]
    txt = f"""=== MDT-GEN VALIDATED EARLY-BOOKING MODELS - HELD-OUT EVIDENCE ===
Generated {report['generated_utc']} | libraries {json.dumps(report['library_versions'])}
Predictors: 10 booking features only (no OGTT, no GDM diagnosis for macrosomia, no delivery variables).
No class re-weighting. Thresholds chosen on out-of-fold TRAINING predictions; test set used once.
Data-quality rules: {QUALITY_RULES_TEXT}

######## A. GESTATIONAL DIABETES ########
Train n={gdm['n_train']} (prevalence {gdm['prevalence_train']*100:.2f}%), test n={gdm['n_test']} (prevalence {gdm['prevalence_test']*100:.2f}%)

A1. Held-out test performance (1,000x bootstrap 95% CI)
{results_table(gdm).to_string()}

A2. Recalibration effect (stacking ensemble, held-out test)
                       AUROC   Brier   Scaled Brier  Cal. intercept  Cal. slope  Mean predicted vs observed
Legacy class-weighted  {gl['AUROC']:.3f}   {gl['Brier']:.4f}  {gl['Scaled_Brier']:+.3f}        {gl['Calibration_intercept']:+.2f}           {gl['Calibration_slope']:.2f}        {gl['Mean_predicted']*100:.1f}% vs {gl['Observed_rate']*100:.1f}%
Recalibrated           {gn['AUROC']:.3f}   {gn['Brier']:.4f}  {gn['Scaled_Brier']:+.3f}        {gn['Calibration_intercept']:+.2f}           {gn['Calibration_slope']:.2f}        {gn['Mean_predicted']*100:.1f}% vs {gn['Observed_rate']*100:.1f}%

A3. Operating points
{ops_text(gdm['operating_points'])}

A4. Booking GA <= 12 weeks subgroup: n={sg['n']}, AUROC {sg['AUROC']:.3f} ({sg['AUROC_95CI'][0]:.3f}-{sg['AUROC_95CI'][1]:.3f}), Brier {sg['Brier']:.4f}

######## B. MACROSOMIA (birthweight >= 4000 g) ########
Train n={mac['n_train']} (prevalence {mac['prevalence_train']*100:.2f}%), test n={mac['n_test']} (prevalence {mac['prevalence_test']*100:.2f}%)

B1. Held-out test performance (1,000x bootstrap 95% CI)
{results_table(mac).to_string()}

B2. Operating points
{ops_text(mac['operating_points'])}

B3. Booking GA <= 12 weeks subgroup: n={sm['n']}, AUROC {sm['AUROC']:.3f} ({sm['AUROC_95CI'][0]:.3f}-{sm['AUROC_95CI'][1]:.3f}), Brier {sm['Brier']:.4f}

B4. Sensitivity outcome LGA (BW centile >= 90), test prevalence {lga['prevalence_test']*100:.2f}%
{results_table(lga).to_string()}

######## C. SHAP mean |value| (CatBoost base model, test set) ########
GDM: {json.dumps(gdm['shap_mean_abs_top12'])}
Macrosomia: {json.dumps(mac['shap_mean_abs_top12'])}

Notes
- Calibration slope < 1: predictions too extreme; > 1: too modest. Intercept ~0: mean risk matches observed rate.
- Internal split-sample validation in a single dataset; external validation is still required before clinical use.
- Saved bundles: models/validated/gdm_booking_model.joblib, models/validated/macrosomia_booking_model.joblib
"""
    (REPORTS_DIR / "booking_models_report.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
