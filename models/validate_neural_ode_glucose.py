"""
models/validate_neural_ode_glucose.py
=====================================
Held-out check of the ONE quantity in the maternal Neural ODE that has real
ground truth in this dataset: fasting glucose at the time of the OGTT.

Question: from booking data alone, how well does the Neural ODE's simulated
fasting-glucose curve predict the fasting OGTT glucose later measured?

* Cohort      : data/maternal/processed/test_clean.csv (never used to train the
                Neural ODE, which was fitted on train_clean.csv only), women with
                a recorded fasting OGTT glucose (3.0-15.0 mmol/L).
* Ground truth: Glucoselevel0minblood (fasting OGTT glucose).
* Timing      : the OGTT gestational week is NOT recorded in the dataset. NICE NG3
                offers the OGTT at 24-28 weeks, so the ODE curve is read at
                weeks 24, 26 (primary) and 28.
* Model input : exactly what the dashboard passes - booking predictors only, no
                glucose (so the curve is not shifted to an observed value).
* Baselines   : (1) training-set mean; (2) linear regression on the same booking
                predictors fitted on train_clean women with an OGTT.
* Metrics     : MAE, RMSE, R^2, mean bias, Spearman rho (1,000x bootstrap 95% CI);
                plus the ODE GDM risk head at week 39 vs the GDM outcome
                (AUROC, Brier) to document why that curve was removed.

Caveats written into the report: women without an OGTT are excluded (selective
testing); GDM women were usually treated after diagnosis, which does not affect
the fasting value used here because it is measured at the diagnostic test.

Run: python models/validate_neural_ode_glucose.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from digital_twin.neural_ode import ETHNICITY_CATEGORIES, ODE_NUMERIC_FEATURES, FetalMaternalODEPredictor  # noqa: E402
from models.booking_risk import BookingPreprocessor, apply_quality_rules  # noqa: E402

PROCESSED = ROOT / "data" / "maternal" / "processed"
OUT = ROOT / "reports" / "validated"
OUT.mkdir(parents=True, exist_ok=True)
READ_WEEKS = [24.0, 26.0, 28.0]
PRIMARY_WEEK = 26.0
N_BOOT = 1000
RNG = np.random.default_rng(42)


def ode_features(df: pd.DataFrame, medians: pd.Series, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    """Same feature construction as the ODE (raw values, median fill, z-score, ethnicity one-hot)."""
    num = df[ODE_NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(medians).to_numpy(np.float32)
    num = (num - mean) / std
    eth = df["Ethnicity"].astype(str).str.strip().str.upper()
    oh = np.zeros((len(df), len(ETHNICITY_CATEGORIES)), np.float32)
    for i, c in enumerate(ETHNICITY_CATEGORIES):
        oh[:, i] = (eth == c).values
    oh[oh.sum(1) == 0, ETHNICITY_CATEGORIES.index("WEU")] = 1.0  # dashboard fallback for unknown
    return torch.tensor(np.hstack([num, oh]), dtype=torch.float32)


def boot(y, p, fn):
    idx = np.arange(len(y))
    vals = [fn(y[b], p[b]) for b in (RNG.choice(idx, len(idx)) for _ in range(N_BOOT))]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def reg_metrics(y, p):
    rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
    rho = lambda a, b: float(spearmanr(a, b).statistic)
    return {
        "MAE": float(mean_absolute_error(y, p)), "MAE_95CI": boot(y, p, mean_absolute_error),
        "RMSE": rmse(y, p), "R2": float(r2_score(y, p)), "R2_95CI": boot(y, p, r2_score),
        "Bias_pred_minus_obs": float(np.mean(p - y)), "Spearman": rho(y, p), "Spearman_95CI": boot(y, p, rho),
        "Pred_SD": float(np.std(p)), "Obs_SD": float(np.std(y)),
    }


def main():
    train = pd.read_csv(PROCESSED / "train_clean.csv", low_memory=False)
    test = pd.read_csv(PROCESSED / "test_clean.csv", low_memory=False)
    for d in (train, test):
        d["_fpg"] = pd.to_numeric(d["Glucoselevel0minblood"], errors="coerce")
        d.loc[(d["_fpg"] < 3.0) | (d["_fpg"] > 15.0), "_fpg"] = np.nan
    te = test.dropna(subset=["_fpg"]).reset_index(drop=True)
    tr = train.dropna(subset=["_fpg"]).reset_index(drop=True)
    y = te["_fpg"].to_numpy()

    # ---- Neural ODE (as deployed) ----
    pred = FetalMaternalODEPredictor()
    if not pred.is_loaded:
        raise RuntimeError("Neural ODE checkpoint not found (models/neural_ode_maternal_gdm.pth).")
    medians = train[ODE_NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce").median()
    X = ode_features(te, medians, pred.mean, pred.std)
    visits = sorted(set([12.0] + READ_WEEKS + [39.0]))
    z_hist, r_hist, t_hist = pred.model.solve_trajectory(X, irregular_visits=visits)
    t_list = [float(t) for t in t_hist]
    glucose_at = {w: z_hist[t_list.index(w), :, 0].numpy() for w in visits}

    results = {"n_test_with_ogtt": int(len(te)), "n_test_total": int(len(test)),
               "observed_mean": float(y.mean()), "observed_sd": float(y.std()),
               "ogtt_week_assumption": "OGTT week not recorded; ODE read at 24/26/28 wk (NICE NG3 window)"}
    results["neural_ode"] = {f"week_{int(w)}": reg_metrics(y, glucose_at[w]) for w in READ_WEEKS}
    results["neural_ode"]["week_12_initial_state"] = reg_metrics(y, glucose_at[12.0])
    results["neural_ode"]["mean_change_wk12_to_wk26"] = float(np.mean(glucose_at[26.0] - glucose_at[12.0]))

    # ---- Baselines ----
    results["baseline_train_mean"] = reg_metrics(y, np.full_like(y, tr["_fpg"].mean()))
    trq, teq = apply_quality_rules(tr), apply_quality_rules(te)
    pre = BookingPreprocessor().fit(trq)
    lr = LinearRegression().fit(pre.transform(trq), tr["_fpg"].to_numpy())
    p_lr = lr.predict(pre.transform(teq))
    results["baseline_linear_regression_booking"] = reg_metrics(y, p_lr)

    # ---- ODE GDM risk head (documenting removal) ----
    g = test.copy()
    g["_gdm"] = pd.to_numeric(g["Gestational Diabetes"], errors="coerce")
    g = g.dropna(subset=["_gdm"]).reset_index(drop=True)
    Xg = ode_features(g, medians, pred.mean, pred.std)
    _, r_all, t_all = pred.model.solve_trajectory(Xg, irregular_visits=[12.0, 26.0, 39.0])
    yg = g["_gdm"].astype(int).to_numpy()
    risk = {}
    for i, w in enumerate([12, 26, 39]):
        p = r_all[i, :, 0].numpy()
        risk[f"week_{w}"] = {"AUROC": float(roc_auc_score(yg, p)), "Brier": float(brier_score_loss(yg, p)),
                             "mean_predicted": float(p.mean()), "observed_rate": float(yg.mean())}
    results["ode_gdm_risk_head"] = risk

    # ---- Plot ----
    p26 = glucose_at[PRIMARY_WEEK]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    lim = (3.0, max(8.0, float(np.percentile(y, 99.5))))
    ax[0].scatter(y, p26, s=6, alpha=.25, color="#2563EB", label="Neural ODE (wk 26)")
    ax[0].scatter(y, p_lr, s=6, alpha=.25, color="#F59E0B", label="Linear regression (booking)")
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set(xlim=lim, ylim=lim, xlabel="Observed fasting OGTT glucose (mmol/L)",
              ylabel="Predicted (mmol/L)", title=f"Held-out test, n={len(y)}")
    ax[0].legend(fontsize=8)
    q = pd.qcut(p26, 10, labels=False, duplicates="drop")
    dec = pd.DataFrame({"q": q, "p": p26, "y": y}).groupby("q").mean()
    ql = pd.qcut(p_lr, 10, labels=False, duplicates="drop")
    decl = pd.DataFrame({"q": ql, "p": p_lr, "y": y}).groupby("q").mean()
    ax[1].plot(dec["p"], dec["y"], "o-", color="#2563EB", label="Neural ODE (wk 26)")
    ax[1].plot(decl["p"], decl["y"], "o-", color="#F59E0B", label="Linear regression")
    lo = min(dec[["p", "y"]].min().min(), decl[["p", "y"]].min().min()) - 0.1
    hi = max(dec[["p", "y"]].max().max(), decl[["p", "y"]].max().max()) + 0.1
    ax[1].plot([lo, hi], [lo, hi], "k--", lw=1)
    ax[1].set(xlabel="Mean predicted (decile)", ylabel="Mean observed", title="Calibration by decile")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "neural_ode_glucose_validation.png", dpi=200); plt.close(fig)

    (OUT / "neural_ode_glucose_validation.json").write_text(json.dumps(results, indent=2))

    def row(name, m):
        return (f"{name:<38} {m['MAE']:.3f} ({m['MAE_95CI'][0]:.3f}-{m['MAE_95CI'][1]:.3f})  {m['RMSE']:.3f}  "
                f"{m['R2']:+.3f} ({m['R2_95CI'][0]:+.3f} to {m['R2_95CI'][1]:+.3f})  {m['Bias_pred_minus_obs']:+.3f}  "
                f"{m['Spearman']:.3f}   {m['Pred_SD']:.3f}")
    ode = results["neural_ode"]
    txt = f"""=== NEURAL ODE GLUCOSE - HELD-OUT VALIDATION ===
Target: fasting OGTT glucose (mmol/L) in held-out test women with an OGTT: n={len(y)} of {len(test)}
Observed mean {y.mean():.2f} (SD {y.std():.2f}). OGTT week not recorded -> ODE curve read at 24/26/28 weeks.
Input: booking predictors only (as on the dashboard). 1,000x bootstrap 95% CIs.

Model                                  MAE (95% CI)          RMSE   R2 (95% CI)               Bias    Spearman Pred SD
{row('Baseline: training mean', results['baseline_train_mean'])}
{row('Baseline: linear regression (booking)', results['baseline_linear_regression_booking'])}
{row('Neural ODE, week 12 (initial state)', ode['week_12_initial_state'])}
{row('Neural ODE, week 24', ode['week_24'])}
{row('Neural ODE, week 26 (primary)', ode['week_26'])}
{row('Neural ODE, week 28', ode['week_28'])}

Mean simulated change in glucose from week 12 to week 26: {ode['mean_change_wk12_to_wk26']:+.3f} mmol/L
(no repeated glucose measurements exist, so this slope is not learned from data)

Neural ODE GDM risk head vs GDM outcome (held-out, n={len(yg)}, observed {yg.mean()*100:.1f}%):
""" + "\n".join(f"- {k}: AUROC {v['AUROC']:.3f}, Brier {v['Brier']:.4f}, mean predicted {v['mean_predicted']*100:.1f}%"
                for k, v in risk.items()) + """

Interpretation guide
- R2 <= 0 means no better than predicting the average. Compare the Neural ODE rows with the baselines.
- Caveats: OGTT offered selectively (women without one excluded); OGTT week assumed.
"""
    (OUT / "neural_ode_glucose_validation.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
