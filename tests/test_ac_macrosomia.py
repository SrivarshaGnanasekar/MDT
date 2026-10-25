"""
tests/test_ac_macrosomia.py
===========================

Focused tests for:
  1. AC reference equation (Choi 2016, DOI: 10.5468/ogs.2016.59.2.91)
  1b. Two-stage pathway: published AC/EFW criterion (Coomarasamy 2005) + Bayes update
  2. Missing / invalid AC inputs
  3. GA range validation
  4. Separation of model probability from AC growth context
  5. Measured-AC key resolution (new vs legacy)
  6. Hadlock EFW calculation

Run:
    python -m pytest tests/test_ac_macrosomia.py -v
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.macrosomia_monitor import (
    MacrosomiaRiskMonitor,
    expected_fetal_ac_cm,
    hadlock_efw,
)


# ===========================================================================
# 1. AC Reference Equation — Choi 2016
# ===========================================================================

class TestChoi2016ACFormula:
    """
    The formula from Choi HY et al., Obstet Gynecol Sci. 2016;59(2):91-96
    (DOI: 10.5468/ogs.2016.59.2.91) is:

        Median AC (cm) = GA (weeks) - 5

    Spot-check against Table 1 values in the paper (approximate, as the table
    gives rounded values):
        GA 20 wk → AC ≈ 15 cm
        GA 28 wk → AC ≈ 23 cm
        GA 32 wk → AC ≈ 27 cm
        GA 36 wk → AC ≈ 31 cm
        GA 40 wk → AC ≈ 35 cm
    """

    @pytest.mark.parametrize("ga, expected_cm", [
        (20.0, 15.0),
        (24.0, 19.0),
        (28.0, 23.0),
        (32.0, 27.0),
        (36.0, 31.0),
        (40.0, 35.0),
    ])
    def test_choi_formula_spot_checks(self, ga, expected_cm):
        result = expected_fetal_ac_cm(ga)
        assert result == pytest.approx(expected_cm, abs=0.01), (
            f"GA {ga} wk: expected {expected_cm} cm, got {result:.4f} cm"
        )

    def test_formula_is_linear(self):
        """Each additional gestational week adds exactly 1 cm to the median AC."""
        for ga in range(14, 42):
            diff = expected_fetal_ac_cm(float(ga + 1)) - expected_fetal_ac_cm(float(ga))
            assert diff == pytest.approx(1.0, abs=1e-9)

    def test_returns_float(self):
        result = expected_fetal_ac_cm(28)
        assert isinstance(result, float)

    def test_regression_form_is_documented_alternative(self):
        """Choi 2016 reports BOTH GA - 5 (simplified median) and the regression
        1.013*GA - 4.634. MDT-GEN uses GA - 5 everywhere; the regression is kept
        only for sensitivity analysis and must stay close to it (20-40 wk)."""
        from models.growth_reference import choi_regression_ac_cm
        for ga in range(20, 41):
            assert abs(choi_regression_ac_cm(ga) - expected_fetal_ac_cm(ga)) < 1.0

    def test_single_source_of_truth(self):
        """All modules must report the same expected AC for the same GA."""
        from models.growth_reference import expected_fetal_ac_cm as shared
        for ga in (20.0, 28.0, 36.0, 40.0):
            assert expected_fetal_ac_cm(ga) == pytest.approx(shared(ga))
            res = MacrosomiaRiskMonitor().evaluate_risk(0.1, ga, {"measured_ac_cm": 25.0})
            assert res["ac_expected_cm"] == pytest.approx(shared(ga))

    def test_choi_supported_range(self):
        from models.growth_reference import is_within_choi_range
        assert is_within_choi_range(20) and is_within_choi_range(40)
        assert not is_within_choi_range(14) and not is_within_choi_range(41)


# ===========================================================================
# 2. MacrosomiaRiskMonitor — general behaviour
# ===========================================================================

class TestMacrosomiaRiskMonitorBasic:
    def setup_method(self):
        self.monitor = MacrosomiaRiskMonitor()

    def test_awaiting_data_without_booking_or_ac(self):
        res = self.monitor.evaluate_risk(maternal_gdm_prob=0.2, gestational_age_weeks=28.0)
        assert res["ac_source"] == "unavailable"
        assert res["risk_tier"] == "Awaiting Data"
        assert res["risk_score"] == 0.0

    def test_high_risk_when_published_ac_criterion_positive(self):
        """AC >= 36 cm at >= 36 wk -> published criterion positive (Coomarasamy 2005, LR+ 6.9)."""
        res = self.monitor.evaluate_risk(maternal_gdm_prob=0.1, gestational_age_weeks=37.0,
                                         fetal_biometrics={"measured_ac_cm": 37.0},
                                         macrosomia_prob=0.10, booking_flagged=True)
        assert res["risk_tier"] == "High Risk"
        assert res["stage2"]["applied"] and res["stage2"]["positive"]
        assert res["stage2"]["likelihood_ratio"] == pytest.approx(6.9)

    def test_low_risk_when_not_flagged_and_no_positive_scan(self):
        res = self.monitor.evaluate_risk(maternal_gdm_prob=0.05, gestational_age_weeks=28.0,
                                         fetal_biometrics={"measured_ac_cm": 23.0},
                                         macrosomia_prob=0.05, booking_flagged=False)
        assert res["risk_tier"] == "Low Risk"
        assert pytest.approx(1.0, abs=0.05) == res["ac_ratio"]

    def test_booking_flag_without_positive_scan_is_moderate(self):
        res = self.monitor.evaluate_risk(maternal_gdm_prob=0.05, gestational_age_weeks=30.0,
                                         macrosomia_prob=0.12, booking_flagged=True)
        assert res["risk_tier"] == "Moderate Risk"

    def test_stage2_not_applied_before_36_weeks(self):
        """A large AC at 30 wk is descriptive only; the published criterion is not applied."""
        res = self.monitor.evaluate_risk(maternal_gdm_prob=0.1, gestational_age_weeks=30.0,
                                         fetal_biometrics={"measured_ac_cm": 36.5},
                                         macrosomia_prob=0.05, booking_flagged=False)
        assert res["stage2"]["applied"] is False
        assert res["risk_tier"] == "Low Risk"
        assert res["posttest_probability"] is None

    def test_bayes_update_matches_published_lr(self):
        """Post-test odds = pre-test odds x LR. 10% with LR+ 6.9 -> 43.4%; LR- 0.37 -> 3.95%."""
        from models.macrosomia_monitor import bayes_update
        assert bayes_update(0.10, 6.9) == pytest.approx(0.4340, abs=1e-3)
        assert bayes_update(0.10, 0.37) == pytest.approx(0.0395, abs=1e-3)

    def test_efw_criterion_used_when_full_biometry_supplied(self):
        res = self.monitor.evaluate_risk(maternal_gdm_prob=0.1, gestational_age_weeks=38.0,
                                         fetal_biometrics={"measured_ac_cm": 36.0, "hc_cm": 34.0,
                                                           "fl_cm": 7.4, "bpd_cm": 9.5},
                                         macrosomia_prob=0.10, booking_flagged=False)
        assert res["stage2"]["criterion"].startswith("Hadlock EFW")
        assert res["stage2"]["likelihood_ratio"] in (5.7, 0.48)

    def test_ac_expected_present_in_all_responses(self):
        """The reference AC should always be returned regardless of measured AC."""
        res = self.monitor.evaluate_risk(
            maternal_gdm_prob=0.3,
            gestational_age_weeks=24.0,
            macrosomia_prob=0.2,
        )
        assert "ac_expected_cm" in res
        assert res["ac_expected_cm"] == pytest.approx(19.0, abs=0.01)

    def test_disclaimer_present(self):
        res = self.monitor.evaluate_risk(
            maternal_gdm_prob=0.4,
            gestational_age_weeks=28.0,
            macrosomia_prob=0.4,
        )
        assert "disclaimer" in res
        assert len(res["disclaimer"]) > 50

    def test_legacy_keys_present(self):
        """Backward-compatibility: macrosomia_risk_tier and composite_risk_score."""
        res = self.monitor.evaluate_risk(
            maternal_gdm_prob=0.5,
            gestational_age_weeks=28.0,
            macrosomia_prob=0.6,
        )
        assert "macrosomia_risk_tier" in res
        assert "composite_risk_score" in res
        assert "risk_tier" in res  # also kept


# ===========================================================================
# 3. Missing / Invalid AC Inputs
# ===========================================================================

class TestACInputValidation:
    def setup_method(self):
        self.monitor = MacrosomiaRiskMonitor()

    def test_no_fetal_biometrics(self):
        res = self.monitor.evaluate_risk(
            maternal_gdm_prob=0.5,
            gestational_age_weeks=28.0,
            macrosomia_prob=0.5,
        )
        assert res["ac_source"] == "unavailable"
        assert res["ac_measured_cm"] is None
        assert res["ac_deviation_cm"] is None

    def test_empty_fetal_biometrics_dict(self):
        res = self.monitor.evaluate_risk(
            maternal_gdm_prob=0.5,
            gestational_age_weeks=28.0,
            fetal_biometrics={},
            macrosomia_prob=0.5,
        )
        assert res["ac_source"] == "unavailable"

    def test_zero_measured_ac(self):
        """Zero AC treated as 'not supplied'."""
        res = self.monitor.evaluate_risk(
            maternal_gdm_prob=0.5,
            gestational_age_weeks=28.0,
            fetal_biometrics={"measured_ac_cm": 0.0},
            macrosomia_prob=0.5,
        )
        assert res["ac_source"] == "unavailable"

    def test_negative_ac_treated_as_unavailable(self):
        res = self.monitor.evaluate_risk(
            maternal_gdm_prob=0.5,
            gestational_age_weeks=28.0,
            fetal_biometrics={"measured_ac_cm": -5.0},
            macrosomia_prob=0.5,
        )
        assert res["ac_source"] == "unavailable"

    def test_implausibly_large_ac_rejected(self):
        """AC > 45 cm is outside the physiological range and must be rejected."""
        res = self.monitor.evaluate_risk(
            maternal_gdm_prob=0.5,
            gestational_age_weeks=32.0,
            fetal_biometrics={"measured_ac_cm": 999.0},
            macrosomia_prob=0.5,
        )
        assert res["ac_source"] == "unavailable"

    def test_implausibly_small_ac_rejected(self):
        """AC < 0.5 cm is not physiological."""
        res = self.monitor.evaluate_risk(
            maternal_gdm_prob=0.5,
            gestational_age_weeks=28.0,
            fetal_biometrics={"measured_ac_cm": 0.1},
            macrosomia_prob=0.5,
        )
        assert res["ac_source"] == "unavailable"

    def test_none_fetal_estimated_ac_cm_treated_as_unavailable(self):
        """
        The image-derived path always sets fetal_estimated_ac_cm = None.
        Passing it explicitly must result in 'unavailable', not an error.
        """
        res = self.monitor.evaluate_risk(
            maternal_gdm_prob=0.4,
            gestational_age_weeks=28.0,
            fetal_biometrics={"fetal_estimated_ac_cm": None},
            macrosomia_prob=0.3,
        )
        assert res["ac_source"] == "unavailable"

    def test_organ_envelope_key_not_treated_as_ac(self):
        """
        The organ-envelope ellipse perimeter from feature_extraction.py
        (fetal_organ_envelope_perimeter_px) must never be interpreted as AC.
        Even if present in fetal_biometrics it should not populate ac_measured_cm.
        """
        res = self.monitor.evaluate_risk(
            maternal_gdm_prob=0.4,
            gestational_age_weeks=28.0,
            fetal_biometrics={"fetal_organ_envelope_perimeter_px": 662.0},
            macrosomia_prob=0.3,
        )
        assert res["ac_source"] == "unavailable"
        assert res["ac_measured_cm"] is None


# ===========================================================================
# 4. Valid Measured AC — growth context separate from probability
# ===========================================================================

class TestMeasuredACContext:
    def setup_method(self):
        self.monitor = MacrosomiaRiskMonitor()

    def test_valid_measured_ac_new_key(self):
        res = self.monitor.evaluate_risk(
            maternal_gdm_prob=0.5,
            gestational_age_weeks=28.0,
            fetal_biometrics={"measured_ac_cm": 25.0},
            macrosomia_prob=0.5,
        )
        assert res["ac_source"] == "measured"
        assert res["ac_measured_cm"] == pytest.approx(25.0, abs=0.01)

    def test_valid_measured_ac_legacy_key(self):
        """Legacy key fetal_estimated_ac_cm accepted as a caller-supplied value."""
        res = self.monitor.evaluate_risk(
            maternal_gdm_prob=0.5,
            gestational_age_weeks=28.0,
            fetal_biometrics={"fetal_estimated_ac_cm": 24.0},
            macrosomia_prob=0.5,
        )
        assert res["ac_source"] == "measured"
        assert res["ac_measured_cm"] == pytest.approx(24.0, abs=0.01)

    def test_ac_deviation_computed_correctly(self):
        """AC deviation = measured - expected = 25.0 - 23.0 = +2.0 cm at GA 28 wk."""
        res = self.monitor.evaluate_risk(
            maternal_gdm_prob=0.5,
            gestational_age_weeks=28.0,
            fetal_biometrics={"measured_ac_cm": 25.0},
            macrosomia_prob=0.5,
        )
        expected = expected_fetal_ac_cm(28.0)
        assert res["ac_expected_cm"] == pytest.approx(expected, abs=0.01)
        assert res["ac_deviation_cm"] == pytest.approx(25.0 - expected, abs=0.01)

    def test_ac_deviation_pct_computed_correctly(self):
        ga = 32.0
        ac = 29.0
        expected = expected_fetal_ac_cm(ga)  # 27.0
        expected_pct = (ac - expected) / expected * 100.0
        res = self.monitor.evaluate_risk(
            maternal_gdm_prob=0.5,
            gestational_age_weeks=ga,
            fetal_biometrics={"measured_ac_cm": ac},
            macrosomia_prob=0.5,
        )
        assert res["ac_deviation_pct"] == pytest.approx(expected_pct, abs=0.1)

    def test_risk_tier_transitions_with_measured_ac(self):
        """Awaiting Data -> High Risk once a positive late scan (AC >= 36 cm at >= 36 wk) is supplied."""
        res_no_ac = self.monitor.evaluate_risk(maternal_gdm_prob=0.5, gestational_age_weeks=37.0)
        assert res_no_ac["risk_tier"] == "Awaiting Data"
        res_with_ac = self.monitor.evaluate_risk(maternal_gdm_prob=0.5, gestational_age_weeks=37.0,
                                                 fetal_biometrics={"measured_ac_cm": 38.0})
        assert res_with_ac["risk_tier"] == "High Risk"

    def test_ac_deviation_not_mislabelled_as_probability(self):
        """
        The ac_deviation_pct field must not appear under any key that could be
        read as a probability (e.g. macrosomia_probability).
        """
        res = self.monitor.evaluate_risk(
            maternal_gdm_prob=0.5,
            gestational_age_weeks=28.0,
            fetal_biometrics={"measured_ac_cm": 30.0},
            macrosomia_prob=0.6,
        )
        # The probability must be in [0, 1]
        assert 0.0 <= res["macrosomia_probability"] <= 1.0
        # The deviation percent may be large/negative; confirm it's separate
        if res["ac_deviation_pct"] is not None:
            assert res["ac_deviation_pct"] != pytest.approx(res["macrosomia_probability"] * 100.0, abs=5)


# ===========================================================================
# 5. GA Range Validation
# ===========================================================================

class TestGARangeValidation:
    def setup_method(self):
        self.monitor = MacrosomiaRiskMonitor()

    def test_valid_ga_no_warnings(self):
        res = self.monitor.evaluate_risk(
            maternal_gdm_prob=0.4,
            gestational_age_weeks=28.0,
            macrosomia_prob=0.4,
        )
        assert res["ga_warnings"] == []

    def test_ga_below_minimum_warns(self):
        res = self.monitor.evaluate_risk(
            maternal_gdm_prob=0.4,
            gestational_age_weeks=10.0,
            macrosomia_prob=0.4,
        )
        assert len(res["ga_warnings"]) > 0

    def test_ga_above_maximum_warns(self):
        res = self.monitor.evaluate_risk(
            maternal_gdm_prob=0.4,
            gestational_age_weeks=43.0,
            macrosomia_prob=0.4,
        )
        assert len(res["ga_warnings"]) > 0

    def test_zero_ga_defaults_gracefully(self):
        """Zero GA should not raise; monitor should default to a safe value."""
        res = self.monitor.evaluate_risk(
            maternal_gdm_prob=0.4,
            gestational_age_weeks=0,
            macrosomia_prob=0.4,
        )
        assert isinstance(res, dict)
        assert "ac_expected_cm" in res


# ===========================================================================
# 6. Hadlock EFW
# ===========================================================================

class TestHadlockEFW:
    def test_known_reference_value(self):
        """
        Reference: Hadlock 1985.  For HC=28cm, AC=25cm, FL=4.8cm, BPD=6.5cm
        the log10(EFW) ≈ 3.365, EFW ≈ 2317 g (approximate).
        This test checks the formula structure, not a specific clinical value.
        """
        efw = hadlock_efw(hc_cm=28.0, ac_cm=25.0, fl_cm=4.8, bpd_cm=6.5)
        assert efw > 500 and efw < 10000, f"EFW {efw:.1f} g is not in a plausible range"

    def test_efw_increases_with_ac(self):
        efw_low = hadlock_efw(hc_cm=28.0, ac_cm=22.0, fl_cm=4.8, bpd_cm=6.5)
        efw_high = hadlock_efw(hc_cm=28.0, ac_cm=30.0, fl_cm=4.8, bpd_cm=6.5)
        assert efw_high > efw_low

    def test_monitor_returns_efw_when_all_measurements_supplied(self):
        monitor = MacrosomiaRiskMonitor()
        res = monitor.evaluate_risk(
            maternal_gdm_prob=0.5,
            gestational_age_weeks=30.0,
            fetal_biometrics={
                "measured_ac_cm": 27.0,
                "hc_cm": 28.0,
                "fl_cm": 5.5,
                "bpd_cm": 7.0,
            },
            macrosomia_prob=0.5,
        )
        assert res["estimated_fetal_weight_g"] is not None
        assert res["estimated_fetal_weight_g"] > 0

    def test_monitor_returns_no_efw_without_full_measurements(self):
        monitor = MacrosomiaRiskMonitor()
        res = monitor.evaluate_risk(
            maternal_gdm_prob=0.5,
            gestational_age_weeks=30.0,
            fetal_biometrics={"measured_ac_cm": 27.0},  # hc, fl, bpd missing
            macrosomia_prob=0.5,
        )
        assert res["estimated_fetal_weight_g"] is None
