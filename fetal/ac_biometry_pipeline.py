"""
fetal/ac_biometry_pipeline.py
==============================

End-to-end pipeline for physical Abdominal Circumference (AC) measurement and
Estimated Fetal Weight (EFW) calculation from Attention U-Net segmentation masks.

Pipeline stages
---------------
1. **Dynamic Scale Discovery** — recovers mm/pixel from ultrasound depth-tick marks
   embedded in the left/right frame margins via autocorrelation peak detection.
2. **Obstetric Ellipse Fitting** — fits a standard clinical ellipse to the largest
   foreground contour of the binary segmentation mask.
3. **Geometric Conversion** — converts pixel axes to mm and applies Ramanujan's
   second ellipse perimeter approximation (1914) to compute AC in cm.
4. **Weight Modelling** — two independent AC-only clinical regression formulae:
     • Hadlock (1984)  : ln(EFW) = 2.695 + 0.253·AC − 0.00275·AC²
     • Campbell & Wilkin (1975): ln(EFW) = −4.564 + 0.282·AC − 0.00331·AC²
5. **Macrosomia Classification** — EFW_Hadlock ≥ 4 000 g → Is_Macrosomia = 1.
6. **Batch Export** — processes a directory of image/mask pairs and writes a
   summary CSV.

References
----------
Hadlock FP et al. (1984) "Sonographic estimation of fetal weight. The value of
femur length in addition to head and abdomen measurements."
Radiology 150(2):535-540.  DOI: 10.1148/radiology.150.2.6691115

Campbell S, Wilkin D. (1975) "Ultrasonic measurement of fetal abdomen
circumference in the estimation of fetal weight." BJOG 82(9):689-697.
DOI: 10.1111/j.1471-0528.1975.tb00708.x

Ramanujan S. (1914) "Modular equations and approximations to π."
Quarterly Journal of Pure and Applied Mathematics 45:350-372.

Usage
-----
    python -m fetal.ac_biometry_pipeline \
        --images  data/fetal/test_images \
        --masks   data/fetal/test_masks \
        --output  results/ac_biometry.csv \
        --fallback-mm-per-pixel 0.25

    # Or import and call programmatically:
    from fetal.ac_biometry_pipeline import BatchACProcessor
    processor = BatchACProcessor(images_dir="...", masks_dir="...")
    df = processor.run()
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ac_biometry")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Physical interval represented by each major depth-tick pair on the scanner
#: display.  For the Da Correggio dataset (Siemens/Mindray convention) each
#: major tick = 10 mm.  Adjust if your scanner uses a different convention.
MM_PER_MAJOR_TICK_INTERVAL: float = 10.0

#: Fallback scale used when automatic tick detection fails.
DEFAULT_FALLBACK_MM_PER_PIXEL: float = 0.25  # typical for clinical OB probes

#: Fraction of image width treated as the "margin strip" for tick scanning.
MARGIN_FRACTION: float = 0.10  # outer 10 % of width on each side

#: Minimum plausible tick spacing in pixels (prevents detecting noise peaks).
MIN_TICK_SPACING_PX: int = 12

#: Maximum plausible tick spacing in pixels (~100 px at 0.25 mm/px = 25 mm).
MAX_TICK_SPACING_PX: int = 120

#: Macrosomia threshold in grams.
MACROSOMIA_THRESHOLD_G: float = 4_000.0


# ===========================================================================
# Data containers
# ===========================================================================

@dataclass
class ScaleResult:
    """Output from the dynamic scale-discovery stage."""
    mm_per_pixel: float
    source: str          # 'auto_left', 'auto_right', 'auto_mean', or 'fallback'
    tick_spacing_px: Optional[float] = None
    n_peaks_found: int = 0
    margin_tried: str = ""


@dataclass
class EllipseResult:
    """Fitted obstetric ellipse in pixel units."""
    center_px: Tuple[float, float]
    semi_major_px: float   # always >= semi_minor_px
    semi_minor_px: float
    angle_deg: float
    contour_area_px2: float
    n_contour_points: int


@dataclass
class ACResult:
    """Physical measurements derived from ellipse + scale."""
    semi_major_mm: float
    semi_minor_mm: float
    ac_mm: float
    ac_cm: float
    # Ramanujan intermediate
    h_value: float


@dataclass
class EFWResult:
    """Estimated fetal weight from both regression models."""
    efw_hadlock_g: float     # Hadlock 1984 AC-only
    efw_campbell_g: float    # Campbell & Wilkin 1975
    is_macrosomia: int       # 1 if EFW_Hadlock >= 4 000 g, else 0


@dataclass
class FrameRecord:
    """Complete result for one image / mask pair."""
    image_id: str
    scale: ScaleResult
    ellipse: Optional[EllipseResult]
    ac: Optional[ACResult]
    efw: Optional[EFWResult]
    error: Optional[str] = None


# ===========================================================================
# Stage 1 — Dynamic Scale Discovery
# ===========================================================================

class TickScaleDetector:
    """
    Recovers the physical mm/pixel scale factor from depth-ruler tick marks
    embedded in the ultrasound frame margins.

    Algorithm
    ---------
    1. Extract the left margin strip (columns 0 … W*margin_frac) and the
       right margin strip (columns W*(1-margin_frac) … W).
    2. Compute the *mean row intensity* of each strip → a 1-D signal of
       length H.  Tick marks appear as periodic bright (high-intensity)
       spikes in grayscale, or as periodic dark spikes on a bright background.
       We try both a direct peak search and an inverted-signal peak search,
       keeping whichever yields more consistent spacing.
    3. Use ``scipy.signal.find_peaks`` to locate the positions of major ticks.
    4. Compute the median inter-peak distance (tick_spacing_px).
    5. mm_per_pixel = MM_PER_MAJOR_TICK_INTERVAL / tick_spacing_px.
    6. Validate: spacing must fall within [MIN_TICK_SPACING_PX,
       MAX_TICK_SPACING_PX].  If validation fails, return the fallback scale.
    """

    def __init__(
        self,
        margin_fraction: float = MARGIN_FRACTION,
        mm_per_major_tick: float = MM_PER_MAJOR_TICK_INTERVAL,
        fallback_mm_per_pixel: float = DEFAULT_FALLBACK_MM_PER_PIXEL,
        min_tick_spacing_px: int = MIN_TICK_SPACING_PX,
        max_tick_spacing_px: int = MAX_TICK_SPACING_PX,
    ):
        self.margin_fraction = margin_fraction
        self.mm_per_major_tick = mm_per_major_tick
        self.fallback_mm_per_pixel = fallback_mm_per_pixel
        self.min_tick_spacing_px = min_tick_spacing_px
        self.max_tick_spacing_px = max_tick_spacing_px

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect(self, image_gray: np.ndarray) -> ScaleResult:
        """
        Run tick detection on *image_gray* (H×W uint8 grayscale).

        Tries left and right margins independently.  Returns the average
        scale if both succeed; the successful one if only one does; or the
        fallback if neither does.

        Parameters
        ----------
        image_gray : np.ndarray, shape (H, W), dtype uint8
            Grayscale ultrasound frame.

        Returns
        -------
        ScaleResult
        """
        H, W = image_gray.shape[:2]
        margin_w = max(1, int(W * self.margin_fraction))

        left_strip  = image_gray[:, :margin_w]
        right_strip = image_gray[:, W - margin_w:]

        left_res  = self._detect_in_strip(left_strip,  label="left")
        right_res = self._detect_in_strip(right_strip, label="right")

        both_valid = (left_res is not None) and (right_res is not None)
        if both_valid:
            mean_mpp = (left_res[0] + right_res[0]) / 2.0
            tick_sp  = (left_res[1] + right_res[1]) / 2.0
            n_peaks  = left_res[2] + right_res[2]
            return ScaleResult(
                mm_per_pixel=mean_mpp,
                source="auto_mean",
                tick_spacing_px=tick_sp,
                n_peaks_found=n_peaks,
                margin_tried="left+right",
            )

        if left_res is not None:
            return ScaleResult(
                mm_per_pixel=left_res[0],
                source="auto_left",
                tick_spacing_px=left_res[1],
                n_peaks_found=left_res[2],
                margin_tried="left",
            )

        if right_res is not None:
            return ScaleResult(
                mm_per_pixel=right_res[0],
                source="auto_right",
                tick_spacing_px=right_res[1],
                n_peaks_found=right_res[2],
                margin_tried="right",
            )

        logger.warning(
            "Tick detection failed on both margins — using fallback "
            "%.4f mm/pixel.", self.fallback_mm_per_pixel
        )
        return ScaleResult(
            mm_per_pixel=self.fallback_mm_per_pixel,
            source="fallback",
            n_peaks_found=0,
            margin_tried="none",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_in_strip(
        self, strip: np.ndarray, label: str
    ) -> Optional[Tuple[float, float, int]]:
        """
        Attempt tick detection in one margin strip.

        Returns (mm_per_pixel, median_spacing_px, n_peaks) or None.
        """
        # 1. Compute mean row intensity → 1-D projection profile.
        profile = strip.mean(axis=1).astype(np.float32)

        # 2. Try to find peaks on the raw profile AND the inverted profile;
        #    use whichever gives more consistent spacing.
        best = self._best_peak_detection(profile, label)
        return best

    def _best_peak_detection(
        self, profile: np.ndarray, label: str
    ) -> Optional[Tuple[float, float, int]]:
        """Try direct and inverted peak detection; return the better result."""
        candidates = []

        for inverted in (False, True):
            sig = -profile if inverted else profile
            # Normalise to [0, 1] for stable prominence threshold.
            sig_norm = (sig - sig.min()) / (np.ptp(sig) + 1e-9)

            peaks, props = find_peaks(
                sig_norm,
                height=0.15,        # ignore sub-15 % peaks
                prominence=0.08,    # must stand out from surrounding baseline
                distance=self.min_tick_spacing_px,
            )

            if len(peaks) < 2:
                continue

            spacings = np.diff(peaks).astype(float)
            median_sp = float(np.median(spacings))

            # Validate spacing is physically plausible.
            if not (self.min_tick_spacing_px <= median_sp <= self.max_tick_spacing_px):
                logger.debug(
                    "%s margin (%s): median spacing %.1f px outside valid range "
                    "[%d, %d] px — skipping.",
                    label, "inv" if inverted else "fwd",
                    median_sp, self.min_tick_spacing_px, self.max_tick_spacing_px,
                )
                continue

            # Coefficient of variation of spacings — reject irregular patterns.
            cv = float(spacings.std() / (median_sp + 1e-9))
            if cv > 0.30:
                logger.debug(
                    "%s margin: spacing CV %.2f > 0.30 — inconsistent ticks, skipping.",
                    label, cv,
                )
                continue

            mm_per_pixel = self.mm_per_major_tick / median_sp
            candidates.append((mm_per_pixel, median_sp, len(peaks), cv))

        if not candidates:
            return None

        # Pick candidate with lowest coefficient of variation (most regular).
        best = min(candidates, key=lambda x: x[3])
        mpp, spacing, n_peaks, cv = best
        logger.info(
            "%s margin: detected %d ticks, spacing %.1f px → %.4f mm/pixel (CV=%.3f).",
            label, n_peaks, spacing, mpp, cv,
        )
        return mpp, spacing, n_peaks


# ===========================================================================
# Stage 2 — Obstetric Ellipse Fitting
# ===========================================================================

class ObstetricEllipseFitter:
    """
    Fits a clinical obstetric ellipse to the primary foreground contour of a
    binary segmentation mask.

    The U-Net output may be:
      • A probability map  (float32, values in [0, 1])
      • A hard binary mask (uint8,  values 0 or 255 / 0 or 1)

    Both are normalised internally before contour extraction.
    """

    def __init__(self, prob_threshold: float = 0.5, min_contour_area_px: int = 200):
        """
        Parameters
        ----------
        prob_threshold : float
            Binarisation threshold when the mask is a probability map.
        min_contour_area_px : int
            Contours smaller than this (in pixels²) are discarded as noise.
        """
        self.prob_threshold = prob_threshold
        self.min_contour_area_px = min_contour_area_px

    def fit(self, mask: np.ndarray) -> EllipseResult:
        """
        Fit an ellipse to the largest contour in *mask*.

        Parameters
        ----------
        mask : np.ndarray, shape (H, W)
            Binary or probability segmentation mask.

        Returns
        -------
        EllipseResult

        Raises
        ------
        ValueError
            If no valid contour with ≥ 5 points is found (OpenCV requirement
            for ``fitEllipse``).
        """
        binary = self._to_binary(mask)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            raise ValueError("No foreground contours found in segmentation mask.")

        # Select the largest contour by enclosed area.
        valid = [c for c in contours
                 if cv2.contourArea(c) >= self.min_contour_area_px]
        if not valid:
            raise ValueError(
                f"No contours ≥ {self.min_contour_area_px} px² found; "
                "mask may be empty or heavily fragmented."
            )

        largest = max(valid, key=cv2.contourArea)

        if len(largest) < 5:
            raise ValueError(
                f"Largest contour has only {len(largest)} points; "
                "cv2.fitEllipse requires ≥ 5."
            )

        # cv2.fitEllipse returns ((cx, cy), (width, height), angle)
        # where width and height are the full axis lengths (diameters).
        (cx, cy), (width_px, height_px), angle_deg = cv2.fitEllipse(largest)

        # Semi-axes (radii), ensuring semi_major ≥ semi_minor.
        semi_a_px = max(width_px, height_px) / 2.0   # semi-major
        semi_b_px = min(width_px, height_px) / 2.0   # semi-minor

        contour_area = float(cv2.contourArea(largest))
        logger.debug(
            "Ellipse fit: centre=(%.1f, %.1f), semi-major=%.1f px, "
            "semi-minor=%.1f px, angle=%.1f°, contour area=%.0f px².",
            cx, cy, semi_a_px, semi_b_px, angle_deg, contour_area,
        )

        return EllipseResult(
            center_px=(float(cx), float(cy)),
            semi_major_px=semi_a_px,
            semi_minor_px=semi_b_px,
            angle_deg=float(angle_deg),
            contour_area_px2=contour_area,
            n_contour_points=len(largest),
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _to_binary(self, mask: np.ndarray) -> np.ndarray:
        """Normalise any mask format to a uint8 binary image (0 / 255)."""
        if mask.ndim == 3:
            # Take the first channel if multi-channel (e.g., RGB export of mask).
            mask = mask[:, :, 0]

        if mask.dtype in (np.float32, np.float64):
            # Probability map → binarise.
            binary = (mask >= self.prob_threshold).astype(np.uint8) * 255
        elif mask.max() <= 1:
            # uint8 with values 0/1.
            binary = (mask > 0).astype(np.uint8) * 255
        else:
            # uint8 with values 0/255 (or other positive values).
            binary = (mask > 0).astype(np.uint8) * 255

        return binary


# ===========================================================================
# Stage 3 — Geometric Conversion (Physical AC)
# ===========================================================================

def compute_ac(
    ellipse: EllipseResult, mm_per_pixel: float
) -> ACResult:
    """
    Convert fitted pixel ellipse axes to physical AC via Ramanujan's second
    perimeter approximation (Ramanujan 1914, Q J Math 45:350-372).

    Mathematics
    -----------
    Let a and b be the semi-major and semi-minor axes in millimetres.

        h = (a − b)² / (a + b)²

        P = π · (a + b) · [1 + (3h) / (10 + √(4 − 3h))]    [mm]

        AC = P / 10    [cm]

    Ramanujan's second approximation is far more accurate than needed here:
    its relative error is below 1e-6 for b/a >= 0.4 (typical of fetal
    abdominal cross-sections), so it contributes negligibly compared with
    segmentation and inter-observer AC variability (Sarris et al. 2012).
    Note: other modules (ac_ellipse_fitter.py, new_model/inference.py,
    kaggle_acouslic_attention_unet.py) use Ramanujan's FIRST approximation,
    pi*[3(a+b) - sqrt((3a+b)(a+3b))]; the two agree to <0.01 % here.

    Parameters
    ----------
    ellipse : EllipseResult
        Fitted ellipse in pixel units.
    mm_per_pixel : float
        Physical scale recovered by TickScaleDetector.

    Returns
    -------
    ACResult
    """
    a_mm = ellipse.semi_major_px * mm_per_pixel
    b_mm = ellipse.semi_minor_px * mm_per_pixel

    if a_mm <= 0 or b_mm <= 0:
        raise ValueError(f"Degenerate ellipse axes: a={a_mm:.3f} mm, b={b_mm:.3f} mm.")

    # Ramanujan's intermediate variable.
    h = (a_mm - b_mm) ** 2 / (a_mm + b_mm) ** 2

    # Discriminant term: (4 - 3h) must be non-negative; clamp for safety.
    disc = max(4.0 - 3.0 * h, 0.0)

    perimeter_mm = math.pi * (a_mm + b_mm) * (1.0 + (3.0 * h) / (10.0 + math.sqrt(disc)))
    ac_cm = perimeter_mm / 10.0

    logger.debug(
        "AC: a=%.2f mm, b=%.2f mm, h=%.5f, P=%.2f mm → AC=%.2f cm.",
        a_mm, b_mm, h, perimeter_mm, ac_cm,
    )
    return ACResult(
        semi_major_mm=a_mm,
        semi_minor_mm=b_mm,
        ac_mm=perimeter_mm,
        ac_cm=ac_cm,
        h_value=h,
    )


# ===========================================================================
# Stage 4 — Weight Modelling (Hadlock & Campbell)
# ===========================================================================

def compute_efw(ac_cm: float) -> EFWResult:
    """
    Estimate fetal weight from AC alone using two published regression models.

    Models (AC in cm, EFW in grams)
    --------------------------------
    **Hadlock (1984) AC-only:**

        ln(EFW) = 2.695 + 0.253·AC − 0.00275·AC²
        EFW = exp(ln(EFW))

    Reference: Hadlock FP et al., Radiology 1984;150(2):535–40.
    DOI: 10.1148/radiology.150.2.6691115

    **Campbell & Wilkin (1975):**

        ln(EFW) = −4.564 + 0.282·AC − 0.00331·AC²
        EFW = exp(ln(EFW))

    Reference: Campbell S, Wilkin D. BJOG 1975;82(9):689–97.
    DOI: 10.1111/j.1471-0528.1975.tb00708.x

    **Macrosomia criterion:** EFW_Hadlock ≥ 4 000 g  (ACOG / ISUOG convention).

    Parameters
    ----------
    ac_cm : float
        Abdominal circumference in centimetres.

    Returns
    -------
    EFWResult
    """
    if ac_cm <= 0:
        raise ValueError(f"AC must be positive; got {ac_cm:.3f} cm.")

    # --- Hadlock 1984 AC-only ---
    ln_efw_hadlock = (
        2.695
        + 0.253 * ac_cm
        - 0.00275 * ac_cm ** 2
    )
    efw_hadlock_g = math.exp(ln_efw_hadlock)

    # --- Campbell & Wilkin 1975 ---
    # Note: The original publication yields EFW in kilograms.
    ln_efw_campbell = (
        -4.564
        + 0.282 * ac_cm
        - 0.00331 * ac_cm ** 2
    )
    efw_campbell_g = math.exp(ln_efw_campbell) * 1000.0

    is_macrosomia = int(efw_hadlock_g >= MACROSOMIA_THRESHOLD_G)

    logger.debug(
        "EFW: Hadlock=%.0f g, Campbell=%.0f g, Macrosomia=%d.",
        efw_hadlock_g, efw_campbell_g, is_macrosomia,
    )
    return EFWResult(
        efw_hadlock_g=round(efw_hadlock_g, 1),
        efw_campbell_g=round(efw_campbell_g, 1),
        is_macrosomia=is_macrosomia,
    )


# ===========================================================================
# Stage 5 — Single-frame processor
# ===========================================================================

class FrameProcessor:
    """
    Orchestrates all pipeline stages for a single image / mask pair.
    """

    def __init__(
        self,
        fallback_mm_per_pixel: float = DEFAULT_FALLBACK_MM_PER_PIXEL,
        prob_threshold: float = 0.5,
        min_contour_area_px: int = 200,
    ):
        self.scale_detector = TickScaleDetector(
            fallback_mm_per_pixel=fallback_mm_per_pixel
        )
        self.ellipse_fitter = ObstetricEllipseFitter(
            prob_threshold=prob_threshold,
            min_contour_area_px=min_contour_area_px,
        )

    def process(
        self,
        image_path: Path,
        mask_path: Path,
    ) -> FrameRecord:
        """
        Run the full pipeline for one frame.

        Parameters
        ----------
        image_path : Path
            Path to the original grayscale ultrasound PNG.
        mask_path : Path
            Path to the binary (or probability float) segmentation mask.
            Accepted formats: PNG, NPY (numpy array).

        Returns
        -------
        FrameRecord
        """
        image_id = image_path.stem

        # ---- Load image ------------------------------------------------
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            return FrameRecord(
                image_id=image_id,
                scale=ScaleResult(
                    mm_per_pixel=DEFAULT_FALLBACK_MM_PER_PIXEL,
                    source="fallback",
                ),
                ellipse=None, ac=None, efw=None,
                error=f"Cannot read image: {image_path}",
            )
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # ---- Load mask -------------------------------------------------
        mask = self._load_mask(mask_path)
        if mask is None:
            return FrameRecord(
                image_id=image_id,
                scale=ScaleResult(
                    mm_per_pixel=DEFAULT_FALLBACK_MM_PER_PIXEL,
                    source="fallback",
                ),
                ellipse=None, ac=None, efw=None,
                error=f"Cannot read mask: {mask_path}",
            )

        # ---- Stage 1: Scale detection ----------------------------------
        scale = self.scale_detector.detect(image_gray)

        # ---- Stage 2: Ellipse fitting ----------------------------------
        try:
            ellipse = self.ellipse_fitter.fit(mask)
        except ValueError as exc:
            logger.warning("[%s] Ellipse fitting failed: %s", image_id, exc)
            return FrameRecord(
                image_id=image_id, scale=scale,
                ellipse=None, ac=None, efw=None,
                error=f"Ellipse: {exc}",
            )

        # ---- Stage 3: Physical AC --------------------------------------
        try:
            ac = compute_ac(ellipse, scale.mm_per_pixel)
        except ValueError as exc:
            logger.warning("[%s] AC computation failed: %s", image_id, exc)
            return FrameRecord(
                image_id=image_id, scale=scale,
                ellipse=ellipse, ac=None, efw=None,
                error=f"AC: {exc}",
            )

        # ---- Stage 4: EFW ----------------------------------------------
        try:
            efw = compute_efw(ac.ac_cm)
        except ValueError as exc:
            logger.warning("[%s] EFW computation failed: %s", image_id, exc)
            return FrameRecord(
                image_id=image_id, scale=scale,
                ellipse=ellipse, ac=ac, efw=None,
                error=f"EFW: {exc}",
            )

        logger.info(
            "[%s] scale=%s(%.4f mm/px) AC=%.2f cm  Hadlock=%.0f g  "
            "Campbell=%.0f g  Macro=%d",
            image_id,
            scale.source, scale.mm_per_pixel,
            ac.ac_cm,
            efw.efw_hadlock_g,
            efw.efw_campbell_g,
            efw.is_macrosomia,
        )
        return FrameRecord(
            image_id=image_id, scale=scale,
            ellipse=ellipse, ac=ac, efw=efw,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _load_mask(mask_path: Path) -> Optional[np.ndarray]:
        """Load a mask from PNG or NPY file."""
        ext = mask_path.suffix.lower()
        if ext == ".npy":
            try:
                arr = np.load(str(mask_path), allow_pickle=False)
                return arr.astype(np.float32) if arr.dtype.kind == "f" else arr
            except Exception as exc:
                logger.error("Failed to load NPY mask %s: %s", mask_path, exc)
                return None
        else:
            # PNG / any OpenCV-readable format.
            img = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                logger.error("cv2.imread returned None for mask: %s", mask_path)
            return img


# ===========================================================================
# Stage 6 — Batch processing
# ===========================================================================

class BatchACProcessor:
    """
    Discovers image / mask pairs in two directories, runs the full pipeline
    on each pair, and exports a tidy summary CSV.

    Directory conventions
    ---------------------
    Images and masks are matched by **stem** (filename without extension):
      images/P001_frame_01.png  ↔  masks/P001_frame_01.png
      images/P001_frame_01.png  ↔  masks/P001_frame_01.npy

    The first mask file whose stem matches the image stem is used.

    Output CSV columns
    ------------------
    | Column               | Description                                   |
    |----------------------|-----------------------------------------------|
    | Image_ID             | Image filename stem                           |
    | Auto_Detected_Scale  | mm/pixel scale + source tag                   |
    | mm_per_pixel         | Raw numeric scale                             |
    | Scale_Source         | 'auto_left', 'auto_right', 'auto_mean', or    |
    |                      | 'fallback'                                    |
    | Tick_Spacing_px      | Median inter-tick spacing (pixels)            |
    | Semi_Major_px        | Ellipse semi-major axis (pixels)              |
    | Semi_Minor_px        | Ellipse semi-minor axis (pixels)              |
    | Semi_Major_mm        | Ellipse semi-major axis (mm)                  |
    | Semi_Minor_mm        | Ellipse semi-minor axis (mm)                  |
    | AC_mm                | Abdominal circumference (mm)                  |
    | AC_cm                | Abdominal circumference (cm)                  |
    | EFW_Hadlock_g        | Estimated fetal weight — Hadlock 1984 (g)     |
    | EFW_Campbell_g       | Estimated fetal weight — Campbell 1975 (g)    |
    | Is_Macrosomia        | 1 if EFW_Hadlock ≥ 4 000 g, else 0           |
    | Error                | Non-empty if any pipeline stage failed        |
    """

    #: Supported image extensions.
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    #: Supported mask extensions (checked in priority order).
    MASK_EXTS  = [".npy", ".png", ".jpg", ".bmp", ".tif", ".tiff"]

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir:  str | Path,
        output_csv: Optional[str | Path] = None,
        fallback_mm_per_pixel: float = DEFAULT_FALLBACK_MM_PER_PIXEL,
        prob_threshold: float = 0.5,
        min_contour_area_px: int = 200,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir  = Path(masks_dir)
        self.output_csv = Path(output_csv) if output_csv else None
        self.processor  = FrameProcessor(
            fallback_mm_per_pixel=fallback_mm_per_pixel,
            prob_threshold=prob_threshold,
            min_contour_area_px=min_contour_area_px,
        )

    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Execute the batch pipeline.

        Returns
        -------
        pd.DataFrame
            One row per image; see class docstring for column definitions.
        """
        image_paths = sorted(
            p for p in self.images_dir.iterdir()
            if p.suffix.lower() in self.IMAGE_EXTS
        )
        if not image_paths:
            raise FileNotFoundError(
                f"No images found in {self.images_dir} "
                f"(extensions: {self.IMAGE_EXTS})."
            )

        logger.info(
            "Batch processing %d images from %s …",
            len(image_paths), self.images_dir,
        )

        rows: List[Dict] = []
        n_success = n_macro = n_fail = 0

        for img_path in image_paths:
            mask_path = self._find_mask(img_path)
            if mask_path is None:
                logger.warning(
                    "[%s] No matching mask found — skipping.", img_path.stem
                )
                rows.append({"Image_ID": img_path.stem, "Error": "no_mask_found"})
                n_fail += 1
                continue

            record = self.processor.process(img_path, mask_path)
            row = self._record_to_row(record)
            rows.append(row)

            if record.error:
                n_fail += 1
            else:
                n_success += 1
                if record.efw and record.efw.is_macrosomia:
                    n_macro += 1

        df = pd.DataFrame(rows)
        # Enforce column order; fill missing cols with NaN.
        df = df.reindex(columns=self._column_order(), fill_value=float("nan"))

        logger.info(
            "Batch complete — %d processed, %d failed, %d macrosomic (%.1f %%).",
            n_success, n_fail, n_macro,
            100.0 * n_macro / max(n_success, 1),
        )

        if self.output_csv:
            self.output_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.output_csv, index=False, float_format="%.4f")
            logger.info("Results written to %s.", self.output_csv)

        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_mask(self, image_path: Path) -> Optional[Path]:
        """Return the first mask file whose stem matches *image_path*."""
        stem = image_path.stem
        for ext in self.MASK_EXTS:
            candidate = self.masks_dir / (stem + ext)
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _record_to_row(rec: FrameRecord) -> Dict:
        """Flatten a FrameRecord into a dict for the summary DataFrame."""
        row: Dict = {"Image_ID": rec.image_id}

        # Scale
        row["Auto_Detected_Scale"] = (
            f"{rec.scale.mm_per_pixel:.4f} mm/px [{rec.scale.source}]"
        )
        row["mm_per_pixel"]   = rec.scale.mm_per_pixel
        row["Scale_Source"]   = rec.scale.source
        row["Tick_Spacing_px"] = rec.scale.tick_spacing_px

        # Ellipse (pixels)
        if rec.ellipse:
            row["Semi_Major_px"] = round(rec.ellipse.semi_major_px, 2)
            row["Semi_Minor_px"] = round(rec.ellipse.semi_minor_px, 2)
            row["Ellipse_Angle_deg"] = round(rec.ellipse.angle_deg, 2)
        else:
            row.update(Semi_Major_px=None, Semi_Minor_px=None,
                       Ellipse_Angle_deg=None)

        # AC (physical)
        if rec.ac:
            row["Semi_Major_mm"] = round(rec.ac.semi_major_mm, 3)
            row["Semi_Minor_mm"] = round(rec.ac.semi_minor_mm, 3)
            row["AC_mm"]         = round(rec.ac.ac_mm, 2)
            row["AC_cm"]         = round(rec.ac.ac_cm, 3)
        else:
            row.update(Semi_Major_mm=None, Semi_Minor_mm=None,
                       AC_mm=None, AC_cm=None)

        # EFW
        if rec.efw:
            row["EFW_Hadlock_g"]  = rec.efw.efw_hadlock_g
            row["EFW_Campbell_g"] = rec.efw.efw_campbell_g
            row["Is_Macrosomia"]  = rec.efw.is_macrosomia
        else:
            row.update(EFW_Hadlock_g=None, EFW_Campbell_g=None,
                       Is_Macrosomia=None)

        row["Error"] = rec.error or ""
        return row

    @staticmethod
    def _column_order() -> List[str]:
        return [
            "Image_ID",
            "Auto_Detected_Scale", "mm_per_pixel", "Scale_Source",
            "Tick_Spacing_px",
            "Semi_Major_px", "Semi_Minor_px", "Ellipse_Angle_deg",
            "Semi_Major_mm", "Semi_Minor_mm",
            "AC_mm", "AC_cm",
            "EFW_Hadlock_g", "EFW_Campbell_g", "Is_Macrosomia",
            "Error",
        ]


# ===========================================================================
# CLI entry point
# ===========================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Fetal AC Biometry Pipeline — "
            "Physical AC measurement and EFW estimation from U-Net masks."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--images",  required=True, help="Directory of ultrasound PNG images.")
    p.add_argument("--masks",   required=True, help="Directory of binary/prob segmentation masks.")
    p.add_argument("--output",  default="results/ac_biometry.csv",
                   help="Output CSV path.")
    p.add_argument("--fallback-mm-per-pixel", type=float,
                   default=DEFAULT_FALLBACK_MM_PER_PIXEL,
                   help="mm/pixel scale used when tick detection fails.")
    p.add_argument("--prob-threshold", type=float, default=0.5,
                   help="Binarisation threshold for probability masks.")
    p.add_argument("--min-contour-area", type=int, default=200,
                   help="Minimum contour area (px²) to consider for ellipse fitting.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable DEBUG-level logging.")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    processor = BatchACProcessor(
        images_dir=args.images,
        masks_dir=args.masks,
        output_csv=args.output,
        fallback_mm_per_pixel=args.fallback_mm_per_pixel,
        prob_threshold=args.prob_threshold,
        min_contour_area_px=args.min_contour_area,
    )
    df = processor.run()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
