"""
tests/test_ac_biometry_pipeline.py
====================================

Unit and integration tests for fetal/ac_biometry_pipeline.py.

Tests cover:
  - TickScaleDetector: synthetic tick signals, fallback, edge cases
  - ObstetricEllipseFitter: various mask formats, degenerate cases
  - compute_ac: Ramanujan geometry, unit correctness
  - compute_efw: Hadlock & Campbell formula values, macrosomia threshold
  - BatchACProcessor: end-to-end with synthetic temp files

Run:
    python -m pytest tests/test_ac_biometry_pipeline.py -v
"""

import math
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fetal.ac_biometry_pipeline import (
    BatchACProcessor,
    EllipseResult,
    FrameProcessor,
    ObstetricEllipseFitter,
    TickScaleDetector,
    compute_ac,
    compute_efw,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_tick_image(
    height: int = 768,
    width:  int = 1024,
    tick_spacing_px: int = 40,
    margin_w: int = 50,
    tick_brightness: int = 220,
) -> np.ndarray:
    """
    Synthesise a grayscale image with periodic bright tick marks
    on the right margin, simulating a scanner depth ruler.
    """
    img = np.zeros((height, width), dtype=np.uint8)
    img[:, :] = 30  # dim background
    for y in range(0, height, tick_spacing_px):
        img[y, width - margin_w:] = tick_brightness
    return img


def _make_circular_mask(
    height: int = 256,
    width:  int = 256,
    radius: int = 80,
) -> np.ndarray:
    """Return a uint8 binary mask (0/255) with a filled circle."""
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (width // 2, height // 2), radius, 255, -1)
    return mask


def _make_ellipse_mask(
    height: int = 256,
    width:  int = 256,
    semi_a: int = 90,   # semi-major (horizontal)
    semi_b: int = 60,   # semi-minor (vertical)
) -> np.ndarray:
    """Return a uint8 binary mask with a filled ellipse."""
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(
        mask,
        (width // 2, height // 2),
        (semi_a, semi_b),
        0, 0, 360, 255, -1
    )
    return mask


# ===========================================================================
# 1. TickScaleDetector
# ===========================================================================

class TestTickScaleDetector:

    def _detector(self, **kwargs):
        return TickScaleDetector(
            margin_fraction=0.10,
            mm_per_major_tick=10.0,
            fallback_mm_per_pixel=0.25,
            **kwargs,
        )

    def test_detects_correct_scale_from_synthetic_ticks(self):
        """
        With 40 px tick spacing and 10 mm/tick the expected scale is
        10 / 40 = 0.25 mm/px.  The detector should recover this within 5 %.
        """
        tick_spacing = 40
        img = _make_tick_image(tick_spacing_px=tick_spacing)
        det = self._detector()
        result = det.detect(img)
        expected = 10.0 / tick_spacing   # 0.25 mm/px
        assert result.mm_per_pixel == pytest.approx(expected, rel=0.05), (
            f"Expected ~{expected:.4f} mm/px, got {result.mm_per_pixel:.4f}"
        )
        assert result.source != "fallback"

    def test_detects_scale_at_different_spacings(self):
        for spacing in (20, 30, 50, 60):
            img = _make_tick_image(tick_spacing_px=spacing)
            det = self._detector()
            result = det.detect(img)
            expected = 10.0 / spacing
            assert result.mm_per_pixel == pytest.approx(expected, rel=0.10), (
                f"spacing={spacing}: expected {expected:.4f}, got {result.mm_per_pixel:.4f}"
            )

    def test_fallback_on_blank_image(self):
        """Fully black image has no ticks → must fall back."""
        img = np.zeros((768, 1024), dtype=np.uint8)
        det = self._detector()
        result = det.detect(img)
        assert result.source == "fallback"
        assert result.mm_per_pixel == pytest.approx(0.25, abs=1e-9)

    def test_fallback_on_uniform_image(self):
        """Uniform grey image has no peaks → must fall back."""
        img = np.full((768, 1024), 128, dtype=np.uint8)
        det = self._detector()
        result = det.detect(img)
        assert result.source == "fallback"

    def test_custom_fallback_value_used(self):
        img = np.zeros((768, 1024), dtype=np.uint8)
        det = TickScaleDetector(fallback_mm_per_pixel=0.35)
        result = det.detect(img)
        assert result.source == "fallback"
        assert result.mm_per_pixel == pytest.approx(0.35, abs=1e-9)

    def test_scale_result_has_required_fields(self):
        img = _make_tick_image(tick_spacing_px=40)
        det = self._detector()
        result = det.detect(img)
        assert hasattr(result, "mm_per_pixel")
        assert hasattr(result, "source")
        assert hasattr(result, "n_peaks_found")
        assert result.mm_per_pixel > 0

    def test_source_label_is_valid_string(self):
        valid_sources = {"auto_left", "auto_right", "auto_mean", "fallback"}
        img = _make_tick_image(tick_spacing_px=40)
        det = self._detector()
        result = det.detect(img)
        assert result.source in valid_sources


# ===========================================================================
# 2. ObstetricEllipseFitter
# ===========================================================================

class TestObstetricEllipseFitter:

    def _fitter(self, **kwargs):
        return ObstetricEllipseFitter(**kwargs)

    def test_circle_mask_gives_nearly_equal_axes(self):
        """A perfect circle should have semi_major ≈ semi_minor."""
        mask = _make_circular_mask(radius=80)
        fitter = self._fitter()
        result = fitter.fit(mask)
        ratio = result.semi_minor_px / result.semi_major_px
        assert ratio == pytest.approx(1.0, abs=0.08), (
            f"Circle axes ratio {ratio:.3f} deviates too much from 1.0"
        )

    def test_ellipse_mask_axes_approximate_input(self):
        """Fitted axes should approximate the drawn semi-axes within 10 %."""
        semi_a, semi_b = 90, 55
        mask = _make_ellipse_mask(semi_a=semi_a, semi_b=semi_b)
        fitter = self._fitter()
        result = fitter.fit(mask)
        assert result.semi_major_px == pytest.approx(semi_a, rel=0.10)
        assert result.semi_minor_px == pytest.approx(semi_b, rel=0.10)

    def test_semi_major_always_ge_semi_minor(self):
        mask = _make_ellipse_mask(semi_a=40, semi_b=90)  # deliberately "wide"
        fitter = self._fitter()
        result = fitter.fit(mask)
        assert result.semi_major_px >= result.semi_minor_px

    def test_probability_mask_binarised_correctly(self):
        """Float probability mask in [0, 1] should be handled the same as binary."""
        mask_bin = _make_circular_mask(radius=70)
        mask_prob = (mask_bin / 255.0).astype(np.float32)
        fitter = self._fitter()
        r_bin  = fitter.fit(mask_bin)
        r_prob = fitter.fit(mask_prob)
        assert r_bin.semi_major_px == pytest.approx(r_prob.semi_major_px, rel=0.02)

    def test_zero_one_mask_accepted(self):
        """uint8 mask with values 0/1 (not 0/255) must work."""
        mask = (_make_circular_mask(radius=60) // 255).astype(np.uint8)
        fitter = self._fitter()
        result = fitter.fit(mask)
        assert result.semi_major_px > 0

    def test_empty_mask_raises_value_error(self):
        mask = np.zeros((256, 256), dtype=np.uint8)
        fitter = self._fitter()
        with pytest.raises(ValueError, match="No foreground"):
            fitter.fit(mask)

    def test_tiny_contour_below_min_area_raises(self):
        """A single bright pixel (area = 1 px²) should be rejected."""
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[128, 128] = 255
        fitter = ObstetricEllipseFitter(min_contour_area_px=50)
        with pytest.raises(ValueError, match="No contours"):
            fitter.fit(mask)

    def test_result_fields_present(self):
        mask = _make_ellipse_mask(semi_a=80, semi_b=50)
        fitter = self._fitter()
        result = fitter.fit(mask)
        assert hasattr(result, "center_px")
        assert hasattr(result, "semi_major_px")
        assert hasattr(result, "semi_minor_px")
        assert hasattr(result, "angle_deg")
        assert hasattr(result, "contour_area_px2")


# ===========================================================================
# 3. compute_ac
# ===========================================================================

class TestComputeAC:

    def _ellipse(self, semi_major_px, semi_minor_px):
        return EllipseResult(
            center_px=(128.0, 128.0),
            semi_major_px=float(semi_major_px),
            semi_minor_px=float(semi_minor_px),
            angle_deg=0.0,
            contour_area_px2=math.pi * semi_major_px * semi_minor_px,
            n_contour_points=100,
        )

    def test_circle_ac_equals_circumference(self):
        """
        For a perfect circle (a == b) Ramanujan's formula degenerates to
        the standard circumference C = 2πr.  Use 1.0 mm/px and r=100 px:
        expected AC = 2π × 100 mm = 628.3 mm = 62.83 cm.
        """
        r_px = 100
        ellipse = self._ellipse(r_px, r_px)
        result = compute_ac(ellipse, mm_per_pixel=1.0)
        expected_mm = 2.0 * math.pi * r_px
        assert result.ac_mm == pytest.approx(expected_mm, rel=0.001)

    def test_scale_factor_proportional(self):
        """Doubling mm_per_pixel should double AC."""
        ellipse = self._ellipse(100, 70)
        r1 = compute_ac(ellipse, mm_per_pixel=0.25)
        r2 = compute_ac(ellipse, mm_per_pixel=0.50)
        assert r2.ac_mm == pytest.approx(2.0 * r1.ac_mm, rel=1e-6)

    def test_ac_cm_is_ac_mm_divided_by_10(self):
        ellipse = self._ellipse(120, 80)
        result = compute_ac(ellipse, mm_per_pixel=0.25)
        assert result.ac_cm == pytest.approx(result.ac_mm / 10.0, rel=1e-9)

    def test_h_value_is_zero_for_circle(self):
        """h = (a−b)²/(a+b)² = 0 when a == b."""
        ellipse = self._ellipse(80, 80)
        result = compute_ac(ellipse, mm_per_pixel=0.25)
        assert result.h_value == pytest.approx(0.0, abs=1e-10)

    def test_h_value_bounded_in_zero_one(self):
        """h ∈ [0, 1) for all valid ellipses."""
        for a, b in [(50, 50), (100, 10), (80, 60), (200, 1)]:
            ellipse = self._ellipse(a, b)
            result = compute_ac(ellipse, mm_per_pixel=0.25)
            assert 0.0 <= result.h_value < 1.0

    def test_degenerate_zero_axes_raises(self):
        ellipse = self._ellipse(0, 50)
        with pytest.raises(ValueError, match="Degenerate"):
            compute_ac(ellipse, mm_per_pixel=0.25)

    def test_typical_fetal_ac_range(self):
        """
        At 32 weeks, typical AC ≈ 27 cm.  With 0.25 mm/px, the ellipse
        needs axes of roughly 86 × 86 mm ÷ 0.25 mm/px = 344 × 344 px.
        Verify the result is within a plausible clinical range (20–40 cm).
        """
        # AC ≈ 27 cm → r ≈ 43 mm → 172 px at 0.25 mm/px
        ellipse = self._ellipse(172, 140)
        result = compute_ac(ellipse, mm_per_pixel=0.25)
        assert 20.0 <= result.ac_cm <= 40.0


# ===========================================================================
# 4. compute_efw
# ===========================================================================

class TestComputeEFW:

    def test_hadlock_formula_known_value(self):
        """
        At AC = 30 cm (typical ~34 wk):
        ln(EFW) = 2.695 + 0.253*30 - 0.00275*900
        EFW = exp(7.81) ≈ 2465 g
        """
        ac = 30.0
        expected_ln = 2.695 + 0.253 * ac - 0.00275 * ac ** 2
        expected_g  = math.exp(expected_ln)
        result = compute_efw(ac)
        assert result.efw_hadlock_g == pytest.approx(expected_g, rel=1e-3)

    def test_campbell_formula_known_value(self):
        ac = 30.0
        expected_ln = -4.564 + 0.282 * ac - 0.00331 * ac ** 2
        expected_g  = math.exp(expected_ln) * 1000.0
        result = compute_efw(ac)
        assert result.efw_campbell_g == pytest.approx(expected_g, rel=1e-3)

    def test_macrosomia_flag_set_above_4000g(self):
        """
        Solve for AC such that Hadlock EFW = 4 000 g:
        ln(4000) = 2.695 + 0.253*AC - 0.00275*AC²
        8.294 ≈ 2.695 + 0.253*AC - 0.00275*AC²
        Numerically, AC ≈ 37.5 cm gives EFW > 4 000 g.
        """
        result = compute_efw(37.5)
        assert result.efw_hadlock_g >= 4000.0
        assert result.is_macrosomia == 1

    def test_macrosomia_flag_not_set_below_4000g(self):
        result = compute_efw(25.0)
        assert result.efw_hadlock_g < 4000.0
        assert result.is_macrosomia == 0

    def test_macrosomia_threshold_is_exactly_4000g(self):
        """is_macrosomia uses ≥ 4 000, not > 4 000."""
        # Find the AC where Hadlock EFW ≈ exactly 4 000 g and nudge slightly.
        from scipy.optimize import brentq
        target = 4000.0
        def fn(ac): return math.exp(2.695 + 0.253*ac - 0.00275*ac**2) - target
        ac_boundary = brentq(fn, 30, 45)

        above = compute_efw(ac_boundary + 0.1)
        below = compute_efw(ac_boundary - 0.1)
        assert above.is_macrosomia == 1
        assert below.is_macrosomia == 0

    def test_efw_positive(self):
        for ac in (15.0, 25.0, 35.0):
            result = compute_efw(ac)
            assert result.efw_hadlock_g  > 0
            assert result.efw_campbell_g > 0

    def test_zero_ac_raises(self):
        with pytest.raises(ValueError, match="positive"):
            compute_efw(0.0)

    def test_negative_ac_raises(self):
        with pytest.raises(ValueError, match="positive"):
            compute_efw(-5.0)

    def test_is_macrosomia_is_binary(self):
        for ac in (20.0, 25.0, 30.0, 35.0, 40.0):
            r = compute_efw(ac)
            assert r.is_macrosomia in (0, 1)


# ===========================================================================
# 5. FrameProcessor — integration
# ===========================================================================

class TestFrameProcessor:

    def test_succeeds_on_synthetic_image_and_mask(self, tmp_path):
        # Create a synthetic ultrasound image with tick marks
        img = _make_tick_image(tick_spacing_px=40)
        img_path = tmp_path / "frame_001.png"
        cv2.imwrite(str(img_path), img)

        # Create a synthetic elliptical mask
        mask = _make_ellipse_mask(height=768, width=1024, semi_a=150, semi_b=100)
        mask_path = tmp_path / "frame_001_mask.png"
        cv2.imwrite(str(mask_path), mask)

        processor = FrameProcessor()
        record = processor.process(img_path, mask_path)

        assert record.error is None
        assert record.ac is not None
        assert record.efw is not None
        assert record.ac.ac_cm > 0
        assert record.efw.efw_hadlock_g > 0
        assert record.efw.is_macrosomia in (0, 1)

    def test_missing_image_returns_error_record(self, tmp_path):
        processor = FrameProcessor()
        record = processor.process(
            tmp_path / "does_not_exist.png",
            tmp_path / "also_missing.png",
        )
        assert record.error is not None

    def test_empty_mask_returns_error_record(self, tmp_path):
        img = _make_tick_image()
        img_path = tmp_path / "frame.png"
        cv2.imwrite(str(img_path), img)

        blank_mask = np.zeros((768, 1024), dtype=np.uint8)
        mask_path = tmp_path / "frame_mask.png"
        cv2.imwrite(str(mask_path), blank_mask)

        processor = FrameProcessor()
        record = processor.process(img_path, mask_path)
        assert record.error is not None
        assert record.ellipse is None


# ===========================================================================
# 6. BatchACProcessor — end-to-end
# ===========================================================================

class TestBatchACProcessor:

    def _make_dataset(self, tmp_path, n_frames: int = 3):
        """Create n_frames synthetic image/mask pairs in tmp dirs."""
        img_dir  = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        img_dir.mkdir(); mask_dir.mkdir()

        for i in range(n_frames):
            stem = f"frame_{i:03d}"
            img  = _make_tick_image(tick_spacing_px=40)
            mask = _make_ellipse_mask(
                height=768, width=1024,
                semi_a=150 + i * 5,
                semi_b=100 + i * 3,
            )
            cv2.imwrite(str(img_dir  / f"{stem}.png"), img)
            cv2.imwrite(str(mask_dir / f"{stem}.png"), mask)

        return img_dir, mask_dir

    def test_returns_dataframe_with_correct_shape(self, tmp_path):
        img_dir, mask_dir = self._make_dataset(tmp_path, n_frames=3)
        proc = BatchACProcessor(images_dir=img_dir, masks_dir=mask_dir)
        df = proc.run()
        assert len(df) == 3
        assert "AC_cm" in df.columns
        assert "EFW_Hadlock_g" in df.columns
        assert "Is_Macrosomia" in df.columns

    def test_csv_written_to_disk(self, tmp_path):
        img_dir, mask_dir = self._make_dataset(tmp_path, n_frames=2)
        csv_path = tmp_path / "results" / "out.csv"
        proc = BatchACProcessor(
            images_dir=img_dir, masks_dir=mask_dir, output_csv=csv_path
        )
        proc.run()
        assert csv_path.exists()
        import pandas as pd
        df_read = pd.read_csv(csv_path)
        assert len(df_read) == 2

    def test_image_id_matches_filename_stem(self, tmp_path):
        img_dir, mask_dir = self._make_dataset(tmp_path, n_frames=2)
        proc = BatchACProcessor(images_dir=img_dir, masks_dir=mask_dir)
        df = proc.run()
        assert set(df["Image_ID"]) == {"frame_000", "frame_001"}

    def test_missing_mask_dir_raises(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        # Write one image but mask dir is the images dir (no matching masks).
        img = _make_tick_image()
        cv2.imwrite(str(img_dir / "frame_000.png"), img)
        proc = BatchACProcessor(
            images_dir=img_dir,
            masks_dir=tmp_path / "nonexistent_masks",
        )
        # The processor should not crash; it records "no_mask_found" errors.
        df = proc.run()
        assert df["Error"].iloc[0] != ""

    def test_no_images_raises_file_not_found(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        mask_dir = tmp_path / "masks"
        mask_dir.mkdir()
        proc = BatchACProcessor(images_dir=empty_dir, masks_dir=mask_dir)
        with pytest.raises(FileNotFoundError):
            proc.run()

    def test_all_required_columns_present(self, tmp_path):
        img_dir, mask_dir = self._make_dataset(tmp_path, n_frames=1)
        proc = BatchACProcessor(images_dir=img_dir, masks_dir=mask_dir)
        df = proc.run()
        required = [
            "Image_ID", "Auto_Detected_Scale", "mm_per_pixel",
            "Scale_Source", "AC_cm", "EFW_Hadlock_g",
            "EFW_Campbell_g", "Is_Macrosomia",
        ]
        for col in required:
            assert col in df.columns, f"Column '{col}' missing from output."


if __name__ == "__main__":
    import subprocess
    subprocess.run(["python", "-m", "pytest", __file__, "-v"])
