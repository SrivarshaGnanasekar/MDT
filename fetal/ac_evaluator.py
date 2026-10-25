"""
fetal/ac_evaluator.py
=====================
Clinical Biometric Evaluation and Bland-Altman Agreement Analysis for Fetal AC.

Computes:
  1. Segmentation Overlap: Dice Similarity Coefficient (DSC) and Jaccard / IoU.
  2. Biometric Accuracy:
     - Mean Absolute Error (MAE in mm) vs. Expert Sonographer ground truth.
     - Mean Percentage Error (MPE in %).
  3. Bland-Altman Agreement:
     - Systematic Bias (Mean Difference: Pred - Ref).
     - Standard Deviation of Differences (SD).
     - 95% Limits of Agreement (LoA: Bias ± 1.96 * SD).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class BlandAltmanResult:
    mean_bias_mm: float
    sd_diff_mm: float
    lower_loa_mm: float
    upper_loa_mm: float
    n_samples: int


@dataclass
class EvaluationSummary:
    mean_dice: float
    median_dice: float
    mean_iou: float
    mae_mm: float
    rmse_mm: float
    mpe_pct: float
    bland_altman: Optional[BlandAltmanResult]
    total_scans: int
    calibrated_scans: int


def compute_dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-6) -> float:
    """Computes Dice Similarity Coefficient between two binary masks."""
    p = (pred_mask > 0).astype(np.float32)
    g = (gt_mask > 0).astype(np.float32)
    intersection = float(np.sum(p * g))
    total = float(np.sum(p) + np.sum(g))
    if total == 0.0:
        return 1.0  # Both empty
    return (2.0 * intersection + eps) / (total + eps)


def compute_iou_score(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-6) -> float:
    """Computes Jaccard / Intersection over Union between two binary masks."""
    p = (pred_mask > 0).astype(np.float32)
    g = (gt_mask > 0).astype(np.float32)
    intersection = float(np.sum(p * g))
    union = float(np.sum(p) + np.sum(g) - intersection)
    if union == 0.0:
        return 1.0
    return (intersection + eps) / (union + eps)


def compute_bland_altman(
    pred_ac_mm: List[float],
    ref_ac_mm: List[float]
) -> Optional[BlandAltmanResult]:
    """
    Computes Bland-Altman bias and 95% Limits of Agreement.
    diff = pred - ref
    """
    if len(pred_ac_mm) < 3:
        return None

    p = np.array(pred_ac_mm, dtype=np.float64)
    r = np.array(ref_ac_mm, dtype=np.float64)
    diff = p - r

    mean_bias = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))
    lower_loa = float(mean_bias - 1.96 * sd_diff)
    upper_loa = float(mean_bias + 1.96 * sd_diff)

    return BlandAltmanResult(
        mean_bias_mm=mean_bias,
        sd_diff_mm=sd_diff,
        lower_loa_mm=lower_loa,
        upper_loa_mm=upper_loa,
        n_samples=len(diff)
    )


class FetalACEvaluator:
    """Accumulates predictions and reference biometrics across scans to compute evaluation metrics."""
    def __init__(self):
        self.dice_scores: List[float] = []
        self.iou_scores: List[float] = []
        self.pred_acs_mm: List[float] = []
        self.ref_acs_mm: List[float] = []
        self.patient_ids: List[str] = []

    def add_sample(
        self,
        pred_mask: np.ndarray,
        gt_mask: Optional[np.ndarray],
        pred_ac_mm: Optional[float],
        ref_ac_mm: Optional[float],
        patient_id: str = ""
    ):
        if gt_mask is not None:
            self.dice_scores.append(compute_dice_score(pred_mask, gt_mask))
            self.iou_scores.append(compute_iou_score(pred_mask, gt_mask))

        if pred_ac_mm is not None and ref_ac_mm is not None and not math.isnan(ref_ac_mm):
            self.pred_acs_mm.append(float(pred_ac_mm))
            self.ref_acs_mm.append(float(ref_ac_mm))
            self.patient_ids.append(patient_id)

    def summarize(self) -> EvaluationSummary:
        mean_dice = float(np.mean(self.dice_scores)) if self.dice_scores else 0.0
        median_dice = float(np.median(self.dice_scores)) if self.dice_scores else 0.0
        mean_iou = float(np.mean(self.iou_scores)) if self.iou_scores else 0.0

        if self.pred_acs_mm and self.ref_acs_mm:
            p = np.array(self.pred_acs_mm)
            r = np.array(self.ref_acs_mm)
            abs_diff = np.abs(p - r)
            mae_mm = float(np.mean(abs_diff))
            rmse_mm = float(np.sqrt(np.mean((p - r) ** 2)))
            mpe_pct = float(np.mean(abs_diff / np.maximum(r, 1.0)) * 100.0)
            ba = compute_bland_altman(self.pred_acs_mm, self.ref_acs_mm)
        else:
            mae_mm = 0.0
            rmse_mm = 0.0
            mpe_pct = 0.0
            ba = None

        return EvaluationSummary(
            mean_dice=mean_dice,
            median_dice=median_dice,
            mean_iou=mean_iou,
            mae_mm=mae_mm,
            rmse_mm=rmse_mm,
            mpe_pct=mpe_pct,
            bland_altman=ba,
            total_scans=len(self.dice_scores),
            calibrated_scans=len(self.pred_acs_mm)
        )
