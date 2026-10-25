"""
fetal/ac_predict.py
===================
Production Inference Engine for Fetal Abdominal Circumference (AC) Biometry.

Accepts:
  - 2D ultrasound image (file path or numpy array)
  - Pixel spacing (mm/pixel)
  - Gestational age in weeks (GA)

Computes:
  1. Segmentation of the fetal abdomen using FetalACAttentionUNet or new AttUNet.
  2. Obstetric ellipse fitting (ISUOG standard) and Ramanujan perimeter.
  3. Calibrated AC in cm and mm.
  4. Relative AC deviation vs. the Choi (2016) gestational-age median AC (context only)
     and the published macrosomia criterion AC >= 36 cm (Coomarasamy 2005, >= 36 wk)
     (shared implementation in models/growth_reference.py):
       Delta_AC_rel = ((AC_actual - AC_expected) / AC_expected) * 100%
  5. Estimated Fetal Weight (Hadlock 1984 AC-only model).
  6. Visual overlay with biometry calipers.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List
import numpy as np
import cv2
import torch
import torch.nn.functional as F

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from config import MODELS_DIR, USE_NEW_ATTUNET_MODEL, CONF_MIN
from fetal.ac_model import FetalACAttentionUNet
from fetal.ac_ellipse_fitter import ObstetricEllipseFitter, ObstetricEllipse, MeasuredAC, render_biometry_overlay

logger = logging.getLogger("ac_predict")


class FetalACPredictor:
    """
    Dedicated predictor for Fetal Abdominal Circumference and Gestational Growth Biometry.
    """
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        img_size: Tuple[int, int] = (256, 320),
        use_new_model: Optional[bool] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.use_new_model = USE_NEW_ATTUNET_MODEL if use_new_model is None else use_new_model

        # Default model search paths:
        # 1. models/acouslic_attention_unet.pth
        # 2. models/fetal_ac_attention_unet.pth
        # 3. Provided custom path
        if model_path is not None:
            self.model_path = Path(model_path)
        else:
            cand1 = MODELS_DIR / "acouslic_attention_unet.pth"
            cand2 = MODELS_DIR / "fetal_ac_attention_unet.pth"
            self.model_path = cand1 if cand1.exists() else cand2

        self.ellipse_fitter = ObstetricEllipseFitter()
        self.new_model = None
        self.new_device = None

        if self.use_new_model:
            self.model = None
            self.load_new_model()
        else:
            self.model = FetalACAttentionUNet(in_channels=1, num_classes=1, base_filters=32).to(self.device)
            self.load_weights()

    def load_new_model(self) -> bool:
        """Loads new AttUNet model from models/new_model/best_attunet.pth."""
        try:
            import models.new_model.inference as inference
            self.new_model, self.new_device = inference.load_model(str(config.NEW_MODEL_PATH), device=self.device)
            self.is_loaded = True
            logger.info(f"[FetalACPredictor] Loaded new AttUNet model from {config.NEW_MODEL_PATH}")
            return True
        except Exception as e:
            logger.warning(f"[FetalACPredictor] Could not load new AttUNet model {config.NEW_MODEL_PATH}: {e}")
            self.is_loaded = False
            return False

    def load_weights(self) -> bool:
        """Loads trained model weights from disk checkpoint."""
        if self.model_path.exists():
            try:
                ckpt = torch.load(self.model_path, map_location=self.device)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    self.model.load_state_dict(ckpt["model_state_dict"])
                    logger.info(f"[FetalACPredictor] Loaded weights from {self.model_path} (Epoch: {ckpt.get('epoch')})")
                elif isinstance(ckpt, dict) and "state_dict" in ckpt:
                    self.model.load_state_dict(ckpt["state_dict"])
                    logger.info(f"[FetalACPredictor] Loaded state_dict from {self.model_path}")
                else:
                    self.model.load_state_dict(ckpt)
                    logger.info(f"[FetalACPredictor] Loaded raw checkpoint from {self.model_path}")
                self.model.eval()
                self.is_loaded = True
                return True
            except Exception as e:
                logger.warning(f"[FetalACPredictor] Could not load checkpoint {self.model_path}: {e}")
                self.model.eval()
                self.is_loaded = False
                return False
        else:
            logger.info(f"[FetalACPredictor] Note: Model weights not found at {self.model_path}. Predictor in unweighted fallback mode.")
            self.model.eval()
            self.is_loaded = False
            return False

    @staticmethod
    def calculate_expected_ac_cm(ga_weeks: float) -> float:
        """
        Expected population median AC (cm) for a gestational age.
        Reference: Choi et al. (2016), Obstet Gynecol Sci 59(2):91-96,
        DOI 10.5468/ogs.2016.59.2.91 -- Median AC (cm) = GA (weeks) - 5.
        Supported range: GA 20-40 weeks (Korean single-centre cohort). Values
        outside that range are extrapolations. Not a centile curve.
        """
        from models.growth_reference import expected_fetal_ac_cm
        return float(max(5.0, round(expected_fetal_ac_cm(ga_weeks), 2)))

    @staticmethod
    def screen_ac(ac_cm: float, ga_weeks: float, expected_ac_cm: float, delta_pct: float):
        """
        Published macrosomia criterion: AC >= 36 cm predicts birthweight > 4000 g
        (LR+ 6.9, LR- 0.37; Coomarasamy et al., BJOG 2005;112:1461-6,
        DOI 10.1111/j.1471-0528.2005.00702.x). Applied only to scans at >= 36 weeks
        (project applicability rule); earlier scans are reported descriptively.
        The deviation from the Choi (2016) median is context only, with no cutoff.
        """
        context = f"AC {ac_cm:.1f} cm vs Choi (2016) median {expected_ac_cm:.1f} cm ({delta_pct:+.1f}%)."
        if float(ga_weeks) < 36.0:
            return ("Descriptive only (scan before 36 weeks)",
                    context + " The published macrosomia criterion (AC >= 36 cm) is applied only at >= 36 weeks.")
        if ac_cm >= 36.0:
            return ("High Risk (AC >= 36 cm, published criterion)",
                    context + " AC >= 36 cm: LR+ 6.9 for birthweight > 4000 g (Coomarasamy 2005).")
        return ("Low Risk (AC < 36 cm, published criterion)",
                context + " AC < 36 cm: LR- 0.37 for birthweight > 4000 g (Coomarasamy 2005).")

    @staticmethod
    def calculate_hadlock_efw_g(ac_cm: float) -> float:
        """
        Hadlock (1984) AC-only Estimated Fetal Weight (EFW) formula:
            ln(EFW) = 2.695 + 0.253 * AC - 0.00275 * AC^2
        """
        ln_efw = 2.695 + (0.253 * ac_cm) - (0.00275 * (ac_cm ** 2))
        return float(round(np.exp(ln_efw), 1))

    def predict(
        self,
        image_input: Union[str, Path, np.ndarray],
        pixel_spacing_mm: Optional[float] = None,
        gestational_age_weeks: float = 28.0,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Runs end-to-end segmentation and clinical biometry on an ultrasound image.
        """
        # 1. Load and format image
        if isinstance(image_input, (str, Path)):
            img_path = str(image_input)
            if img_path.endswith(".npy"):
                d = np.load(img_path, allow_pickle=True)
                if isinstance(d, np.ndarray) and d.dtype == object:
                    d = d.item()
                img_np = d.get("image", d) if isinstance(d, dict) else d
            else:
                img_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            img_np = np.array(image_input)

        if img_np is None:
            raise ValueError("Input image could not be resolved.")

        orig_h, orig_w = img_np.shape[:2]
        expected_ac_cm = self.calculate_expected_ac_cm(gestational_age_weeks)

        # Base RGB image for overlays
        if img_np.ndim == 2:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_rgb = img_np[:, :, :3].copy()
        else:
            img_rgb = img_np.copy()

        research_notice = (
            "Single-frame estimate. Accuracy depends on the image showing a good abdominal plane. Research use only."
            if self.use_new_model else None
        )

        if self.use_new_model:
            if self.new_model is None:
                success = self.load_new_model()
                if not success or self.new_model is None:
                    return {
                        "actual_ac_cm": None,
                        "actual_ac_mm": None,
                        "expected_ac_cm": expected_ac_cm,
                        "delta_ac_relative_pct": None,
                        "efw_hadlock_g": None,
                        "is_calibrated": False,
                        "screening_tier": "Model Unavailable",
                        "clinical_interpretation": f"Error: Could not load new AttUNet model weights from {config.NEW_MODEL_PATH}.",
                        "mask": np.zeros((orig_h, orig_w), dtype=np.uint8),
                        "overlay": img_rgb,
                        "fitted_ellipse": None,
                        "is_model_loaded": False,
                        "model_confidence": 0.0,
                        "plane_quality": "low",
                        "research_notice": research_notice,
                        "error": "Model Unavailable"
                    }

            import models.new_model.inference as inference

            px_input = pixel_spacing_mm if pixel_spacing_mm is not None else 1.0
            res = inference.image_ac(self.new_model, self.new_device, img_np, px=px_input)

            ac_mm_val = res["ac_mm"]
            conf = float(res["conf"])
            mask_orig = res["mask"]

            model_confidence = conf
            plane_quality = "ok" if conf >= CONF_MIN else "low"

            is_calibrated = (pixel_spacing_mm is not None) and (not math.isnan(ac_mm_val))

            # Fit obstetric ellipse & biometry overlay for the new model's mask
            ellipse, ac_biometry = self.ellipse_fitter.fit_mask(mask_orig, pixel_spacing_mm=pixel_spacing_mm)

            if ellipse is not None and ac_biometry is not None:
                overlay_img = render_biometry_overlay(img_rgb, mask_orig, ellipse, ac_biometry)
                ellipse_dict = {
                    "center": (ellipse.center_x_px, ellipse.center_y_px),
                    "axes_px": (ellipse.semi_major_px * 2.0, ellipse.semi_minor_px * 2.0),
                    "angle": ellipse.angle_deg,
                    "a_cm": (ac_biometry.semi_major_mm / 10.0) if ac_biometry.is_calibrated else None,
                    "b_cm": (ac_biometry.semi_minor_mm / 10.0) if ac_biometry.is_calibrated else None,
                    "ac_cm": ac_biometry.ac_cm if ac_biometry.is_calibrated else None,
                    "ac_mm": ac_biometry.ac_mm if ac_biometry.is_calibrated else None,
                    "scale_mm_per_px": pixel_spacing_mm,
                    "center_x_px": ellipse.center_x_px,
                    "center_y_px": ellipse.center_y_px,
                    "semi_major_px": ellipse.semi_major_px,
                    "semi_minor_px": ellipse.semi_minor_px,
                    "angle_deg": ellipse.angle_deg,
                    "roundness_ratio": ellipse.aspect_ratio
                }
            else:
                overlay_img = img_rgb
                ellipse_dict = None

            # New Model path policy: when plane_quality == "low" or actual_ac_mm is None:
            # set efw_hadlock_g = None, delta_ac_relative_pct = None, screening_tier = "Unavailable (low plane quality)"
            if plane_quality == "low" or not is_calibrated or math.isnan(ac_mm_val):
                actual_ac_mm = float(ac_mm_val) if (is_calibrated and not math.isnan(ac_mm_val)) else None
                actual_ac_cm = float(ac_mm_val / 10.0) if (is_calibrated and not math.isnan(ac_mm_val)) else None
                delta_ac_rel = None
                efw_g = None
                screening_tier = "Unavailable (low plane quality)"

                if pixel_spacing_mm is None:
                    rec = "Image spacing (mm/pixel) is missing. Physical AC and relative growth deviation cannot be reported in cm without calibration."
                elif math.isnan(ac_mm_val) or not mask_orig.any():
                    rec = "No fetal abdomen detected in the image."
                else:
                    rec = f"Frame confidence ({conf:.4f}) is below required quality threshold ({CONF_MIN}). Growth percentiles, EFW, and screening tier are suppressed. Multi-frame sweep verification recommended."
            else:
                actual_ac_mm = float(ac_mm_val)
                actual_ac_cm = float(ac_mm_val / 10.0)
                delta_ac_rel = ((actual_ac_cm - expected_ac_cm) / expected_ac_cm) * 100.0
                efw_g = self.calculate_hadlock_efw_g(actual_ac_cm)

                screening_tier, rec = self.screen_ac(actual_ac_cm, gestational_age_weeks, expected_ac_cm, delta_ac_rel)

        else:
            # Legacy model path (untouched)
            if img_np.ndim == 3:
                if img_np.shape[2] == 3:
                    img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
                else:
                    img_gray = img_np[:, :, 0]
            else:
                img_gray = img_np.copy()

            target_h, target_w = self.img_size
            img_resized = cv2.resize(img_gray, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            img_float = img_resized.astype(np.float32)
            if img_float.max() > 1.0:
                img_float = img_float / 255.0

            img_tensor = torch.from_numpy(img_float).unsqueeze(0).unsqueeze(0).to(self.device)

            if self.is_loaded:
                with torch.no_grad():
                    logits = self.model(img_tensor)
                    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                    mask_low = (probs >= threshold).astype(np.uint8)
            else:
                blur = cv2.GaussianBlur(img_resized, (7, 7), 0)
                _, mask_low = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            mask_orig = cv2.resize(mask_low, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            ellipse, ac_biometry = self.ellipse_fitter.fit_mask(mask_orig, pixel_spacing_mm=pixel_spacing_mm)

            actual_ac_cm = ac_biometry.ac_cm if (ac_biometry and ac_biometry.is_calibrated) else None
            actual_ac_mm = ac_biometry.ac_mm if (ac_biometry and ac_biometry.is_calibrated) else None
            is_calibrated = ac_biometry.is_calibrated if ac_biometry else False
            model_confidence = 0.0
            plane_quality = "low"

            if actual_ac_cm is not None:
                delta_ac_rel = ((actual_ac_cm - expected_ac_cm) / expected_ac_cm) * 100.0
                efw_g = self.calculate_hadlock_efw_g(actual_ac_cm)

                screening_tier, rec = self.screen_ac(actual_ac_cm, gestational_age_weeks, expected_ac_cm, delta_ac_rel)
            else:
                delta_ac_rel = None
                efw_g = None
                screening_tier = "Uncalibrated - Physical Spacing Required"
                rec = "Image spacing (mm/pixel) is missing. Physical AC and Relative deviation cannot be reported in cm without calibration."

            if ellipse is not None and ac_biometry is not None:
                overlay_img = render_biometry_overlay(img_rgb, mask_orig, ellipse, ac_biometry)
                ellipse_dict = {
                    "center_x_px": ellipse.center_x_px,
                    "center_y_px": ellipse.center_y_px,
                    "semi_major_px": ellipse.semi_major_px,
                    "semi_minor_px": ellipse.semi_minor_px,
                    "angle_deg": ellipse.angle_deg,
                    "roundness_ratio": ellipse.aspect_ratio
                }
            else:
                overlay_img = img_rgb
                ellipse_dict = None

        return {
            "actual_ac_cm": actual_ac_cm,
            "actual_ac_mm": actual_ac_mm,
            "expected_ac_cm": expected_ac_cm,
            "delta_ac_relative_pct": round(delta_ac_rel, 2) if delta_ac_rel is not None else None,
            "efw_hadlock_g": efw_g,
            "is_calibrated": is_calibrated,
            "screening_tier": screening_tier,
            "clinical_interpretation": rec,
            "mask": mask_orig,
            "overlay": overlay_img,
            "fitted_ellipse": ellipse_dict,
            "is_model_loaded": self.is_loaded,
            "model_confidence": model_confidence,
            "plane_quality": plane_quality,
            "research_notice": research_notice
        }

    def predict_mha(self, mha_path: Union[str, Path], px: float = 0.28) -> List[Optional[float]]:
        """
        Runs multi-sweep AC estimation across 6 sweeps of a 3D .mha volume using the new model.
        Returns list of 6 ACs in mm (or None if NaN / no abdomen detected).
        """
        if self.new_model is None:
            self.load_new_model()

        if self.new_model is None:
            raise RuntimeError(f"Could not load new AttUNet model from {config.NEW_MODEL_PATH}")

        import models.new_model.inference as inference
        raw_acs = inference.case_acs(self.new_model, self.new_device, str(mha_path), px=px)
        return [None if (ac is None or math.isnan(ac)) else float(ac) for ac in raw_acs]
