"""
fetal/visualization.py
======================
Overlay helpers for the ACOUSLIC-AI fetal-abdomen Attention U-Net
(binary mask: 0 = background, >0 = fetal abdomen).

The legacy 5-class organ model (artery / vein / liver / stomach) and its
dataset were removed on 2026-10-25; organ colours and bounding boxes are gone.
"""
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

ABDOMEN_RGB = (0, 200, 120)        # green fill for the segmented abdomen
ELLIPSE_RGB = (0, 255, 255)        # cyan fitted ellipse
COLOR_PALETTE = {0: (0, 0, 0), 1: ABDOMEN_RGB}
CLASS_NAMES = {0: "Background", 1: "Fetal abdomen"}


def _to_rgb(image_np: np.ndarray) -> np.ndarray:
    if image_np.ndim == 2:
        return np.stack([image_np] * 3, axis=-1)
    if image_np.shape[2] == 4:
        return image_np[:, :, :3]
    return image_np


def create_segmentation_overlay(image_np: np.ndarray, mask_np: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend the abdomen mask (mask > 0) onto the ultrasound image."""
    img = _to_rgb(image_np).astype(np.float32)
    fg = (mask_np > 0)[:, :, None]
    color = np.zeros_like(img)
    color[:] = ABDOMEN_RGB
    out = np.where(fg, (1.0 - alpha) * img + alpha * color, img)
    return np.clip(out, 0, 255).astype(np.uint8)


def _fit_from_mask(mask_np: np.ndarray) -> Optional[Dict[str, Any]]:
    cnts, _ = cv2.findContours((mask_np > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5:
        return None
    (cx, cy), (w, h), ang = cv2.fitEllipse(c)
    return {"center": (cx, cy), "axes_px": (w, h), "angle": ang, "ac_cm": None}


def draw_fitted_ellipse(
    image_np: np.ndarray,
    ellipse_data: Optional[Dict[str, Any]] = None,
    mask_np: Optional[np.ndarray] = None,
    color_rgb: Tuple[int, int, int] = ELLIPSE_RGB,
    thickness: int = 2,
) -> np.ndarray:
    """Draw the fitted abdominal ellipse (axes_px are FULL axis lengths) and the AC label."""
    img_out = _to_rgb(image_np).copy()
    if ellipse_data is None and mask_np is not None:
        ellipse_data = _fit_from_mask(mask_np)
    if not ellipse_data:
        return img_out
    cx, cy = ellipse_data.get("center", (0.0, 0.0))
    w_px, h_px = ellipse_data.get("axes_px", (0.0, 0.0))
    if w_px <= 0 or h_px <= 0:
        return img_out
    center = (int(round(cx)), int(round(cy)))
    axes = (int(round(w_px / 2.0)), int(round(h_px / 2.0)))
    cv2.ellipse(img_out, center, axes, float(ellipse_data.get("angle", 0.0)), 0, 360, color_rgb, thickness, cv2.LINE_AA)
    cv2.drawMarker(img_out, center, (255, 0, 255), cv2.MARKER_CROSS, 10, 2)
    ac_cm = ellipse_data.get("ac_cm")
    label = f"AC: {ac_cm:.1f} cm" if ac_cm else "AC: uncalibrated"
    pos = (max(center[0] - 60, 10), max(center[1] - axes[1] - 10, 25))
    cv2.putText(img_out, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img_out, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_rgb, 1, cv2.LINE_AA)
    return img_out
