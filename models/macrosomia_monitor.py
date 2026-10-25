"""
Macrosomia Risk Monitor - two-stage pathway
===========================================

Stage 1 (booking, <= 12 weeks): validated early-booking macrosomia model
    Calibrated probability of birthweight >= 4000 g from the 10 booking
    predictors (models/booking_risk.py, evidence in reports/validated/).
    Passed in as ``macrosomia_prob``.

Stage 2 (late third trimester): ultrasound biometry against a PUBLISHED criterion
    Coomarasamy A, Connock M, Thornton JG, Khan KS. Accuracy of ultrasound
    biometry in the prediction of macrosomia: a systematic quantitative review.
    BJOG. 2005;112(11):1461-1466. DOI: 10.1111/j.1471-0528.2005.00702.x
      * AC >= 36 cm  : LR+ 6.9 (95% CI 5.2-9.0), LR- 0.37 (0.30-0.45)
      * EFW >= 4000 g (Hadlock): LR+ 5.7 (4.3-7.6), LR- 0.48 (0.38-0.60)
    The 4-parameter Hadlock EFW criterion is used when HC, BPD and FL are
    supplied; otherwise the AC criterion is used. Only ONE criterion is applied
    (AC and EFW are strongly correlated, so their LRs cannot be multiplied).

    Applicability rule (project-defined, conservative): the published criterion
    is applied only to scans at >= 36 weeks, because the reviewed studies
    evaluated biometry for predicting birthweight close to delivery. Earlier
    AC values are reported descriptively only.

Combining the stages
    post-test odds = pre-test odds (Stage 1) x LR (Stage 2).
    This is standard Bayesian updating with PUBLISHED likelihood ratios. It
    assumes the published LRs transfer to this population and are independent
    of the booking predictors. It has NOT been validated on linked
    booking + scan + birthweight data, which this project does not have, and is
    labelled as such in every output.

AC GROWTH CONTEXT (descriptive only)
    Expected AC = Choi 2016 median (GA - 5 cm; models/growth_reference.py).
    Deviation from the median is shown for context; it carries no cutoff.

Tiers used by the recommendation engine
    "High Risk"     : Stage-2 published criterion positive.
    "Moderate Risk" : booking model flagged (top 20% booking risk) and no
                      positive Stage-2 result -> offer third-trimester growth scan.
    "Low Risk"      : neither.
"""

import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Published Stage-2 criteria (Coomarasamy et al., BJOG 2005)
# ---------------------------------------------------------------------------

COOMARASAMY_2005 = (
    "Coomarasamy A, Connock M, Thornton JG, Khan KS. BJOG. 2005;112(11):1461-1466. "
    "DOI: 10.1111/j.1471-0528.2005.00702.x"
)
AC_CUTOFF_CM = 36.0
AC_LR_POS, AC_LR_NEG = 6.9, 0.37
EFW_CUTOFF_G = 4000.0
EFW_LR_POS, EFW_LR_NEG = 5.7, 0.48
STAGE2_MIN_GA_WEEKS = 36.0


def expected_fetal_ac_cm(gestational_age_weeks: float) -> float:
    """
    Reference median fetal AC (cm): GA (weeks) - 5  (Choi et al. 2016,
    Obstet Gynecol Sci 59(2):91-96, DOI 10.5468/ogs.2016.59.2.91).
    Supported range 20-40 weeks. A population median, not a centile or cutoff.
    Delegates to models/growth_reference.py (single source of truth).
    """
    from models.growth_reference import expected_fetal_ac_cm as _choi_median
    return _choi_median(gestational_age_weeks)


def hadlock_efw(hc_cm: float, ac_cm: float, fl_cm: float, bpd_cm: float) -> float:
    """
    Estimated fetal weight (grams), Hadlock 4-parameter model (HC, AC, FL, BPD in cm).
    Hadlock FP et al. Am J Obstet Gynecol. 1985;151(3):333-337.
    DOI: 10.1016/0002-9378(85)90298-4
    """
    log10_efw = (
        1.3596
        + 0.0064 * hc_cm
        + 0.0424 * ac_cm
        + 0.174 * fl_cm
        + 0.00061 * bpd_cm * ac_cm
        - 0.00386 * ac_cm * fl_cm
    )
    return 10 ** log10_efw


def bayes_update(pretest_prob: float, lr: float) -> float:
    """Post-test probability from a pre-test probability and a likelihood ratio."""
    p = min(max(float(pretest_prob), 1e-6), 1 - 1e-6)
    odds = p / (1 - p) * lr
    return odds / (1 + odds)


class MacrosomiaRiskMonitor:
    """Two-stage macrosomia pathway: validated booking risk + published ultrasound criterion."""

    DISCLAIMER = (
        "Research decision-support tool. Stage 1 is an early-booking model validated on held-out "
        "data (internal split-sample validation; see reports/validated/). Stage 2 applies a "
        "published ultrasound criterion (AC >= 36 cm or Hadlock EFW >= 4000 g; Coomarasamy et al., "
        "BJOG 2005). The combined post-test probability uses published likelihood ratios and has "
        "NOT been validated on linked data in this project. Image-derived AC is a research "
        "measurement whose accuracy must be confirmed against reference AC. All outputs require "
        "review by a qualified clinician."
    )

    _GA_MIN = 20.0   # Choi 2016 supported range for the AC median
    _GA_MAX = 40.0
    _AC_MIN = 0.5
    _AC_MAX = 45.0

    def evaluate_risk(
        self,
        maternal_gdm_prob: float,
        gestational_age_weeks: Optional[float] = 28.0,
        fetal_biometrics: Optional[Dict[str, Any]] = None,
        macrosomia_prob: Optional[float] = None,
        booking_flagged: Optional[bool] = None,
        classification_threshold: float = 0.5,  # kept for backward compatibility; unused
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        maternal_gdm_prob : calibrated GDM probability (context only).
        gestational_age_weeks : CURRENT gestational age at the scan.
        fetal_biometrics : may contain ``measured_ac_cm`` (or legacy
            ``fetal_estimated_ac_cm``) and optionally ``hc_cm``, ``fl_cm``, ``bpd_cm``.
            ``fetal_organ_envelope_perimeter_px`` is ignored (not an AC).
        macrosomia_prob : Stage-1 validated booking probability (pre-test).
        booking_flagged : whether Stage 1 flagged the woman (top 20% booking risk).
        """
        fetal_biometrics = fetal_biometrics or {}
        ga = float(gestational_age_weeks) if gestational_age_weeks else 28.0
        ga_warnings = []
        if ga < self._GA_MIN or ga > self._GA_MAX:
            ga_warnings.append(
                f"Gestational age {ga:.1f} wk is outside the 20-40 wk range of the Choi 2016 AC median; "
                "the expected AC is an extrapolation.")

        pretest = float(macrosomia_prob) if macrosomia_prob is not None else None
        expected_ac = expected_fetal_ac_cm(ga)

        # ---- Resolve measured AC ----
        measured_ac = fetal_biometrics.get("measured_ac_cm") or fetal_biometrics.get("fetal_estimated_ac_cm")
        ac_source, ac_measured_cm = "unavailable", None
        ac_ratio = ac_deviation_cm = ac_deviation_pct = None
        ac_context_note = "No measured AC supplied. Run Module 3 (fetal ultrasound) or enter a measured AC."
        if measured_ac is not None and float(measured_ac) > 0:
            ac_val = float(measured_ac)
            if self._AC_MIN <= ac_val <= self._AC_MAX:
                ac_source, ac_measured_cm = "measured", ac_val
                ac_ratio = ac_val / max(expected_ac, 1e-5)
                ac_deviation_cm = ac_val - expected_ac
                ac_deviation_pct = ac_deviation_cm / max(expected_ac, 1e-5) * 100.0
                ac_context_note = (
                    f"Measured AC {ac_val:.1f} cm vs Choi (2016) median {expected_ac:.1f} cm at {ga:.0f} wk "
                    f"({ac_deviation_pct:+.1f}%). Descriptive growth context only.")
            else:
                ac_context_note = (f"Supplied AC {ac_val:.2f} cm is outside the plausible range "
                                   f"({self._AC_MIN}-{self._AC_MAX} cm) and was rejected.")

        # ---- 4-parameter Hadlock EFW (only when all inputs present) ----
        efw_g = None
        if ac_measured_cm is not None:
            vals = [fetal_biometrics.get(k) for k in ("hc_cm", "fl_cm", "bpd_cm")]
            if all(v is not None and math.isfinite(float(v)) and float(v) > 0 for v in vals):
                hc, fl, bpd = (float(v) for v in vals)
                efw_g = round(hadlock_efw(hc, ac_measured_cm, fl, bpd), 1)

        # ---- Stage 2: published criterion ----
        stage2 = {"applied": False, "criterion": None, "positive": None, "likelihood_ratio": None,
                  "reason": None}
        if ac_measured_cm is None:
            stage2["reason"] = "No measured AC."
        elif ga < STAGE2_MIN_GA_WEEKS:
            stage2["reason"] = (f"Scan at {ga:.0f} wk; the published criterion is applied only at "
                                f">= {STAGE2_MIN_GA_WEEKS:.0f} wk (project applicability rule).")
        elif efw_g is not None:
            pos = efw_g >= EFW_CUTOFF_G
            stage2.update(applied=True, criterion=f"Hadlock EFW >= {EFW_CUTOFF_G:.0f} g",
                          positive=bool(pos), likelihood_ratio=EFW_LR_POS if pos else EFW_LR_NEG)
        else:
            pos = ac_measured_cm >= AC_CUTOFF_CM
            stage2.update(applied=True, criterion=f"AC >= {AC_CUTOFF_CM:.0f} cm",
                          positive=bool(pos), likelihood_ratio=AC_LR_POS if pos else AC_LR_NEG)

        posttest = None
        if stage2["applied"] and pretest is not None:
            posttest = bayes_update(pretest, stage2["likelihood_ratio"])

        # ---- Tier ----
        if stage2["applied"] and stage2["positive"]:
            risk_tier, label, color = "High Risk", "High Risk - published ultrasound criterion positive", "#E53E3E"
        elif booking_flagged:
            risk_tier, label, color = "Moderate Risk", "Moderate Risk - higher booking risk, growth scan advised", "#DD6B20"
        elif pretest is None and ac_source == "unavailable":
            risk_tier, label, color = "Awaiting Data", "Awaiting booking risk or fetal biometry", "#718096"
        else:
            risk_tier, label, color = "Low Risk", "Low Risk - not flagged", "#38A169"

        display_prob = posttest if posttest is not None else pretest
        rationales = []
        if pretest is not None:
            rationales.append(f"Stage 1 (validated booking model): {pretest*100:.1f}% risk of birthweight >= 4000 g"
                              + (" - flagged (top 20% at booking)." if booking_flagged else "."))
        if stage2["applied"]:
            rationales.append(
                f"Stage 2 ({stage2['criterion']}, Coomarasamy 2005): "
                f"{'POSITIVE' if stage2['positive'] else 'negative'}, LR {stage2['likelihood_ratio']}.")
            if posttest is not None:
                rationales.append(
                    f"Combined post-test probability {posttest*100:.1f}% (Bayesian update with published LR; "
                    "not validated on linked data).")
        elif stage2["reason"]:
            rationales.append(f"Stage 2 not applied: {stage2['reason']}")
        rationales.append(ac_context_note)
        rationales += ga_warnings

        return {
            "risk_tier": risk_tier,
            "macrosomia_risk_tier": label,
            "risk_color": color,
            "risk_score": round(display_prob * 100.0, 1) if display_prob is not None else 0.0,
            "composite_risk_score": round(display_prob * 100.0, 1) if display_prob is not None else 0.0,
            "macrosomia_probability": round(display_prob, 4) if display_prob is not None else None,
            "booking_macrosomia_probability": pretest,
            "booking_flagged": bool(booking_flagged) if booking_flagged is not None else None,
            "posttest_probability": round(posttest, 4) if posttest is not None else None,
            "stage2": stage2,
            "maternal_gdm_probability": float(maternal_gdm_prob) if maternal_gdm_prob is not None else None,
            "gestational_age_weeks": ga,
            "ac_source": ac_source,
            "ac_measured_cm": round(ac_measured_cm, 2) if ac_measured_cm is not None else None,
            "ac_expected_cm": round(expected_ac, 2),
            "ac_ratio": round(ac_ratio, 3) if ac_ratio is not None else None,
            "ac_deviation_cm": round(ac_deviation_cm, 2) if ac_deviation_cm is not None else None,
            "ac_deviation_pct": round(ac_deviation_pct, 1) if ac_deviation_pct is not None else None,
            "ac_relative_deviation_percent": round(ac_deviation_pct, 1) if ac_deviation_pct is not None else None,
            "ac_context_note": ac_context_note,
            "estimated_fetal_weight_g": efw_g,
            "explanation": " ".join(rationales),
            "clinical_rationales": rationales,
            "ga_warnings": ga_warnings,
            "references": {"stage2": COOMARASAMY_2005,
                           "ac_median": "Choi HY et al. Obstet Gynecol Sci. 2016;59(2):91-96. DOI: 10.5468/ogs.2016.59.2.91"},
            "disclaimer": self.DISCLAIMER,
        }


if __name__ == "__main__":
    m = MacrosomiaRiskMonitor()
    for kw in [dict(gestational_age_weeks=37, fetal_biometrics={"measured_ac_cm": 37.0}, macrosomia_prob=0.10, booking_flagged=True),
               dict(gestational_age_weeks=37, fetal_biometrics={"measured_ac_cm": 33.0}, macrosomia_prob=0.10, booking_flagged=True),
               dict(gestational_age_weeks=28, fetal_biometrics={"measured_ac_cm": 25.0}, macrosomia_prob=0.05, booking_flagged=False)]:
        r = m.evaluate_risk(maternal_gdm_prob=0.08, **kw)
        print(r["risk_tier"], r["macrosomia_probability"], r["clinical_rationales"])
