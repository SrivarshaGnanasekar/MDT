"""
explainability/fetal_vis.py
===========================
Visual rationale for the ACOUSLIC-AI fetal-abdomen Attention U-Net:
raw frame, predicted abdomen mask, and overlay with the fitted ellipse used
for AC. No synthetic fallback masks are drawn: if the model finds no abdomen,
the figure says so.
"""
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

from fetal.visualization import create_segmentation_overlay, draw_fitted_ellipse


def generate_fetal_explainability_figure(image_np: np.ndarray, pred_mask: np.ndarray,
                                         biometrics: Dict[str, Any]) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_np, cmap="gray")
    axes[0].set_title("1. Ultrasound frame", fontweight="bold")
    mask = (np.asarray(pred_mask) > 0).astype(np.uint8)
    axes[1].imshow(mask, cmap="Greens")
    axes[1].set_title("2. Predicted fetal-abdomen mask", fontweight="bold")
    if mask.sum() == 0:
        axes[2].imshow(image_np, cmap="gray")
        axes[2].set_title("3. No abdomen detected", fontweight="bold")
    else:
        ell = biometrics.get("fitted_ellipse") if isinstance(biometrics, dict) else None
        overlay = draw_fitted_ellipse(create_segmentation_overlay(image_np, mask), ellipse_data=ell, mask_np=mask)
        axes[2].imshow(overlay)
        ac = biometrics.get("fetal_estimated_ac_cm") if isinstance(biometrics, dict) else None
        conf = biometrics.get("model_confidence") if isinstance(biometrics, dict) else None
        sub = f"AC {ac:.1f} cm" if ac else "AC unavailable"
        if conf is not None:
            sub += f" | confidence {conf:.3f}"
        axes[2].set_title(f"3. Fitted ellipse ({sub})", fontweight="bold")
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    return fig
