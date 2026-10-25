"""
models/growth_reference.py
==========================
Single source of truth for the gestational-age reference median fetal
abdominal circumference (AC) used throughout MDT-GEN.

Source
------
Choi HY, Kim JH, Park JY, Jung EY, No JH, Oh KJ, Hong JS.
"Simple mathematical formulae for estimation of median values of fetal
biometry at each gestational age."
Obstet Gynecol Sci. 2016;59(2):91-96. DOI: 10.5468/ogs.2016.59.2.91

The paper reports two AC expressions:
  * Simplified median formula (the paper's proposed rule):
        median AC (cm) = GA (weeks) - 5
  * Linear regression against reference values after 20 weeks:
        AC (cm) = 1.013 x GA (weeks) - 4.634

MDT-GEN uses the simplified median formula everywhere so that every module
and dashboard page reports the same expected AC. The regression is kept here
only for documentation and sensitivity analysis.

Scope and limitations
---------------------
* Derived from reference values after 20 weeks and evaluated in a single-centre
  Korean cohort (194 singleton pregnancies, 20-40 weeks).
* It is a rough population median. It is NOT a centile curve and NOT a
  universally validated growth standard. Values outside 20-40 weeks are
  extrapolations and are flagged by ``is_within_choi_range``.
"""

from __future__ import annotations

CHOI_2016_DOI = "10.5468/ogs.2016.59.2.91"
CHOI_2016_CITATION = (
    "Choi HY, Kim JH, Park JY, Jung EY, No JH, Oh KJ, Hong JS. "
    "Obstet Gynecol Sci. 2016;59(2):91-96. DOI: 10.5468/ogs.2016.59.2.91"
)
CHOI_FORMULA_TEXT = "Median AC (cm) = GA (weeks) - 5"
CHOI_VALID_GA_MIN = 20.0
CHOI_VALID_GA_MAX = 40.0


def expected_fetal_ac_cm(gestational_age_weeks: float) -> float:
    """Reference median AC (cm) at a gestational age: GA - 5 (Choi 2016)."""
    return float(gestational_age_weeks) - 5.0


def choi_regression_ac_cm(gestational_age_weeks: float) -> float:
    """Choi 2016 regression form, AC = 1.013*GA - 4.634 (sensitivity analysis only)."""
    return 1.013 * float(gestational_age_weeks) - 4.634


def is_within_choi_range(gestational_age_weeks: float) -> bool:
    """True if GA lies within the 20-40 week range supported by Choi 2016."""
    ga = float(gestational_age_weeks)
    return CHOI_VALID_GA_MIN <= ga <= CHOI_VALID_GA_MAX
