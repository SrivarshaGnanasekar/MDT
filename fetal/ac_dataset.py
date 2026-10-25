"""
fetal/ac_dataset.py
===================
Dataset loaders and patient-level splitting for Fetal Abdominal Circumference (AC) segmentation.

Supports:
  1. ACOUSLIC-AI Challenge v1.1 dataset (Zenodo / MICCAI challenge structure).
  2. Standard annotated folder layouts (images/ + masks/ + metadata.csv).
  3. Strict patient/pregnancy-level stratification to ensure no frames from the same subject
     leak across train, validation, and test splits.
"""

from __future__ import annotations

import os
import glob
import json
import random
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any, Union

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


class UltrasoundACAugmentation:
    """
    Synchronized Ultrasound Augmentation pipeline for Image & Binary Abdominal Mask.
    Applies spatial transformations simultaneously to both image and mask,
    and ultrasound-specific intensity changes to image only.
    """
    def __init__(
        self,
        flip_prob: float = 0.5,
        rotation_prob: float = 0.5,
        max_rotation_deg: float = 15.0,
        jitter_prob: float = 0.4
    ):
        self.flip_prob = flip_prob
        self.rotation_prob = rotation_prob
        self.max_rotation_deg = max_rotation_deg
        self.jitter_prob = jitter_prob

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            img: Tensor of shape (1, H, W) or (3, H, W) in [0.0, 1.0]
            mask: Tensor of shape (1, H, W) in {0, 1}
        """
        # 1. Random Horizontal Flip (probe orientation invariance)
        if random.random() < self.flip_prob:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # 2. Subtle Rotation (simulating hand-held probe angle variance)
        if random.random() < self.rotation_prob:
            angle = random.uniform(-self.max_rotation_deg, self.max_rotation_deg)
            img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        # 3. Brightness & Contrast Jitter (acoustic gain variance)
        if random.random() < self.jitter_prob:
            brightness_factor = random.uniform(0.85, 1.15)
            contrast_factor = random.uniform(0.85, 1.15)
            img = TF.adjust_brightness(img, brightness_factor)
            img = TF.adjust_contrast(img, contrast_factor)
            img = torch.clamp(img, 0.0, 1.0)

        return img, mask


class FetalACItem:
    """Holds metadata and paths for a single ultrasound AC frame."""
    def __init__(
        self,
        image_path: Path,
        mask_path: Optional[Path],
        patient_id: str,
        pixel_spacing_mm: Optional[float] = None,
        reference_ac_mm: Optional[float] = None,
        frame_id: Optional[str] = None
    ):
        self.image_path = Path(image_path)
        self.mask_path = Path(mask_path) if mask_path is not None else None
        self.patient_id = patient_id
        self.pixel_spacing_mm = pixel_spacing_mm
        self.reference_ac_mm = reference_ac_mm
        self.frame_id = frame_id or self.image_path.stem


class FetalACDataset(Dataset):
    """
    PyTorch Dataset for Fetal Abdominal Circumference (AC) segmentation.
    Produces:
      - 'image': Tensor (1, H, W) normalized to [0, 1]
      - 'mask': Tensor (1, H, W) binary {0, 1}
      - 'pixel_spacing_mm': Float or NaN
      - 'reference_ac_mm': Float or NaN
      - 'patient_id': String
    """
    def __init__(
        self,
        items: List[FetalACItem],
        img_size: Tuple[int, int] = (256, 320),
        transform: Optional[UltrasoundACAugmentation] = None
    ):
        self.items = items
        self.img_size = img_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]

        # Load image
        img_path = str(item.image_path)
        if img_path.endswith(".npy"):
            data = np.load(img_path, allow_pickle=True)
            if isinstance(data, np.ndarray) and data.dtype == object:
                data = data.item()
            if isinstance(data, dict):
                img_np = data.get("image", data.get("img"))
            else:
                img_np = data
        else:
            img_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img_np is None:
            raise ValueError(f"Could not load image at {img_path}")

        # If 3-channel, convert to 1-channel grayscale
        if img_np.ndim == 3:
            if img_np.shape[2] == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
            else:
                img_np = img_np[:, :, 0]

        # Load mask
        if item.mask_path is not None and item.mask_path.exists():
            mask_path = str(item.mask_path)
            if mask_path.endswith(".npy"):
                m_data = np.load(mask_path, allow_pickle=True)
                if isinstance(m_data, np.ndarray) and m_data.dtype == object:
                    m_data = m_data.item()
                if isinstance(m_data, dict):
                    mask_np = m_data.get("mask", m_data.get("abdomen"))
                else:
                    mask_np = m_data
            else:
                mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_np = (mask_np > 0).astype(np.float32)
        else:
            mask_np = np.zeros_like(img_np, dtype=np.float32)

        # Resize to fixed input resolution
        target_h, target_w = self.img_size
        img_resized = cv2.resize(img_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask_np, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # Normalize image to [0.0, 1.0]
        img_float = img_resized.astype(np.float32)
        if img_float.max() > 1.0:
            img_float = img_float / 255.0

        # To PyTorch tensors (1, H, W)
        img_tensor = torch.from_numpy(img_float).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0)

        # Apply synchronized spatial/intensity augmentations
        if self.transform is not None:
            img_tensor, mask_tensor = self.transform(img_tensor, mask_tensor)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "patient_id": item.patient_id,
            "pixel_spacing_mm": item.pixel_spacing_mm if item.pixel_spacing_mm is not None else float("nan"),
            "reference_ac_mm": item.reference_ac_mm if item.reference_ac_mm is not None else float("nan"),
            "frame_id": item.frame_id,
            "orig_size": img_np.shape[:2]
        }


def split_items_by_patient(
    items: List[FetalACItem],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[FetalACItem], List[FetalACItem], List[FetalACItem]]:
    """
    Splits items strictly by patient/pregnancy ID to prevent data leakage.
    No patient's frames will ever cross multiple splits.
    """
    random.seed(seed)
    patient_ids = sorted(list(set(item.patient_id for item in items)))
    random.shuffle(patient_ids)

    n_patients = len(patient_ids)
    n_train = max(1, int(round(train_ratio * n_patients)))
    n_val = max(1, int(round(val_ratio * n_patients)))

    train_pids = set(patient_ids[:n_train])
    val_pids = set(patient_ids[n_train:n_train + n_val])
    test_pids = set(patient_ids[n_train + n_val:])

    train_items = [item for item in items if item.patient_id in train_pids]
    val_items = [item for item in items if item.patient_id in val_pids]
    test_items = [item for item in items if item.patient_id in test_pids]

    return train_items, val_items, test_items
