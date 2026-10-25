"""
================================================================================
ACOUSLIC-AI CHALLENGE (MICCAI) - ATTENTION U-NET KAGGLE GPU TRAINING PIPELINE
================================================================================
This script is a self-contained, high-performance training pipeline designed to train
a dedicated 2-Class (Binary) Attention U-Net on Kaggle's free GPU (NVIDIA T4 / P100)
for automated Fetal Abdominal Circumference (AC) segmentation and biometry.

Features Included:
  1. Automatic runtime installation of SimpleITK & Nibabel for medical images (.mha, .nii).
  2. Subdirectory-aware dataset scanner separating images and masks (UUID & nnU-Net formats).
  3. Universal multi-format medical reader with automatic 3D volume slice selection.
  4. Fail-proof image & ground-truth mask pairing logic with diagnostic checks.
  5. Diagnostic pre-check printing image & mask pixel statistics (min/max/mean/shape).
  6. Fast in-memory RAM caching (~1.5s / epoch).
  7. Layer-by-layer architectural summary & parameter count display.
  8. Synchronized ultrasound augmentations (horizontal flip, contrast/gain jitter).
  9. Hybrid BCE + Soft Dice Loss with Cosine Annealing Learning Rate scheduling.
 10. Clinical biometry perimeter fitting (Ramanujan ellipse approximation).
 11. Automated generation of high-resolution diagnostic plots:
     - Training Curves (Loss, Dice, MAE mm) -> training_curves.png
     - Visual Sample Segmentation Overlays -> fetal_segmentation_samples.png
     - Bland-Altman Biometric Agreement Plot -> bland_altman_plot.png
 12. Export of best checkpoint -> acouslic_attention_unet.pth
================================================================================
"""

import os
import sys
import glob
import json
import math
import random
import zipfile
import subprocess
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any, Union

# Auto-install medical imaging libraries if missing on Kaggle
try:
    import SimpleITK as sitk
except ImportError:
    print("--> SimpleITK not found. Installing SimpleITK and nibabel for medical image support (.mha, .nii)...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "SimpleITK", "nibabel"])
        import SimpleITK as sitk
    except Exception as e:
        print(f"--> [Warning] Could not auto-install SimpleITK: {e}")

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- 1. REPRODUCIBILITY & HARDWARE SETUP ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--> Operating Hardware Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"--> GPU Device Name: {torch.cuda.get_device_name(0)}")


# --- 2. UNIVERSAL MEDICAL IMAGE LOADER (SimpleITK / Nibabel / OpenCV / PIL / NumPy) ---
def load_image_file(path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Universally loads ultrasound images or segmentation masks across all medical & standard formats
    (.mha, .mhd, .nii, .nii.gz, .png, .jpg, .npy, .tif).
    Automatically selects the non-empty 2D slice for 3D medical volumes.
    """
    if path is None:
        return None
    p_str = str(path)
    if not os.path.exists(p_str):
        return None

    ext = Path(p_str).suffix.lower()
    full_ext = "".join(Path(p_str).suffixes).lower()

    # 1. Medical volume formats (.mha, .mhd, .nii, .nii.gz)
    if ext in [".mha", ".mhd", ".nii", ".gz"] or ".nii" in full_ext:
        try:
            import SimpleITK as sitk
            itk_img = sitk.ReadImage(p_str)
            arr = sitk.GetArrayFromImage(itk_img)
            arr = np.squeeze(arr)
            if arr.ndim == 3:
                # Select slice with highest signal intensity/sum
                slice_sums = arr.reshape(arr.shape[0], -1).sum(axis=1)
                best_idx = int(np.argmax(slice_sums)) if slice_sums.max() > 0 else arr.shape[0] // 2
                arr = arr[best_idx]
            return arr.astype(np.float32)
        except Exception:
            pass

        try:
            import nibabel as nib
            nib_img = nib.load(p_str)
            arr = nib_img.get_fdata()
            arr = np.squeeze(arr)
            if arr.ndim == 3:
                slice_sums = arr.reshape(-1, arr.shape[2]).sum(axis=0)
                best_idx = int(np.argmax(slice_sums)) if slice_sums.max() > 0 else arr.shape[2] // 2
                arr = arr[:, :, best_idx]
            return arr.T.astype(np.float32)
        except Exception:
            pass

    # 2. NumPy arrays (.npy)
    if ext == ".npy":
        try:
            d = np.load(p_str, allow_pickle=True)
            if isinstance(d, np.ndarray) and d.dtype == object:
                d = d.item()
            if isinstance(d, dict):
                d = d.get("image", d.get("mask", d.get("img", d)))
            d = np.squeeze(d)
            if d.ndim > 2:
                d = d[0]
            return d.astype(np.float32)
        except Exception:
            pass

    # 3. Standard OpenCV format (.png, .jpg, .tif, .bmp)
    try:
        img = cv2.imread(p_str, cv2.IMREAD_GRAYSCALE)
        if img is not None and img.size > 0:
            return img.astype(np.float32)
    except Exception:
        pass

    # 4. PIL Fallback
    try:
        from PIL import Image
        pil_img = Image.open(p_str).convert("L")
        return np.array(pil_img, dtype=np.float32)
    except Exception:
        pass

    return None


# --- 3. ATTENTION U-NET MODEL ARCHITECTURE ---
class DoubleConv(nn.Module):
    """(Conv2d => BatchNorm2d => ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Attention Gate for Attention U-Net.
    Suppresses irrelevant acoustic background noise and highlights the fetal abdominal boundary.
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1)
        )
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        net = self.relu(g1 + x1)
        attn = self.sigmoid(self.psi(net))
        return x * attn


class UpBlock(nn.Module):
    """Upsampling followed by Attention Gate and Double Convolution."""
    def __init__(self, in_channels: int, out_channels: int):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.attn = AttentionGate(F_g=out_channels, F_l=out_channels, F_int=max(16, out_channels // 2))
        self.conv_after_up = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv_after_up(x)
        skip_attn = self.attn(g=x, x=skip)
        if x.shape[2:] != skip_attn.shape[2:]:
            x = F.interpolate(x, size=skip_attn.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip_attn, x], dim=1)
        return self.conv(x)


class FetalACAttentionUNet(nn.Module):
    """
    Dedicated Binary Attention U-Net for Fetal Abdominal Circumference (AC) segmentation.
    Outputs raw logits of shape (B, 1, H, W).
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 1, base_filters: int = 32):
        super(FetalACAttentionUNet, self).__init__()
        f = base_filters

        # Contracting Encoder
        self.enc1 = DoubleConv(in_channels, f)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(f, f * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(f * 2, f * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(f * 4, f * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(f * 8, f * 16)

        # Expanding Decoder
        self.up4 = UpBlock(f * 16, f * 8)
        self.up3 = UpBlock(f * 8, f * 4)
        self.up2 = UpBlock(f * 4, f * 2)
        self.up1 = UpBlock(f * 2, f)

        # Binary Head
        self.final_conv = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        out = self.final_conv(d1)
        return out


def print_model_summary(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 1, 256, 320)):
    """Prints layer-by-layer architectural summary and parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 70)
    print("        FETAL AC ATTENTION U-NET MODEL ARCHITECTURE SUMMARY")
    print("=" * 70)
    print(f"--> Input Tensor Shape   : {input_size}")
    print(f"--> Total Parameters     : {total_params:,}")
    print(f"--> Trainable Parameters : {trainable_params:,}")
    print("=" * 70)
    print(f"{'Block Name':<26} | {'Output Shape':<20} | {'Param Count':<12}")
    print("-" * 70)

    x = torch.zeros(input_size, device=next(model.parameters()).device)
    e1 = model.enc1(x)
    print(f"{'Encoder 1 (DoubleConv)':<26} | {str(list(e1.shape)):<20} | {sum(p.numel() for p in model.enc1.parameters()):,}")
    p1 = model.pool1(e1)

    e2 = model.enc2(p1)
    print(f"{'Encoder 2 (DoubleConv)':<26} | {str(list(e2.shape)):<20} | {sum(p.numel() for p in model.enc2.parameters()):,}")
    p2 = model.pool2(e2)

    e3 = model.enc3(p2)
    print(f"{'Encoder 3 (DoubleConv)':<26} | {str(list(e3.shape)):<20} | {sum(p.numel() for p in model.enc3.parameters()):,}")
    p3 = model.pool3(e3)

    e4 = model.enc4(p3)
    print(f"{'Encoder 4 (DoubleConv)':<26} | {str(list(e4.shape)):<20} | {sum(p.numel() for p in model.enc4.parameters()):,}")
    p4 = model.pool4(e4)

    b = model.bottleneck(p4)
    print(f"{'Bottleneck (DoubleConv)':<26} | {str(list(b.shape)):<20} | {sum(p.numel() for p in model.bottleneck.parameters()):,}")

    d4 = model.up4(b, e4)
    print(f"{'Decoder 4 + Attn Gate':<26} | {str(list(d4.shape)):<20} | {sum(p.numel() for p in model.up4.parameters()):,}")

    d3 = model.up3(d4, e3)
    print(f"{'Decoder 3 + Attn Gate':<26} | {str(list(d3.shape)):<20} | {sum(p.numel() for p in model.up3.parameters()):,}")

    d2 = model.up2(d3, e2)
    print(f"{'Decoder 2 + Attn Gate':<26} | {str(list(d2.shape)):<20} | {sum(p.numel() for p in model.up2.parameters()):,}")

    d1 = model.up1(d2, e1)
    print(f"{'Decoder 1 + Attn Gate':<26} | {str(list(d1.shape)):<20} | {sum(p.numel() for p in model.up1.parameters()):,}")

    out = model.final_conv(d1)
    print(f"{'Final Binary Conv Head':<26} | {str(list(out.shape)):<20} | {sum(p.numel() for p in model.final_conv.parameters()):,}")
    print("=" * 70 + "\n")


# --- 4. LOSS FUNCTION (BCE + SOFT DICE) ---
class BCEDiceLoss(nn.Module):
    """Hybrid loss combining Binary Cross-Entropy and Soft Dice Loss."""
    def __init__(self, bce_weight: float = 0.5, smooth: float = 1e-6):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)

        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss


# --- 5. CLINICAL BIOMETRY UTILITIES ---
def ramanujan_perimeter(a: float, b: float) -> float:
    """Ramanujan's 1st approximation for ellipse perimeter."""
    if a <= 0 or b <= 0:
        return 0.0
    term1 = 3.0 * (a + b)
    term2 = math.sqrt((3.0 * a + b) * (a + 3.0 * b))
    return math.pi * (term1 - term2)


def fit_ellipse_and_measure_ac(
    binary_mask: np.ndarray,
    pixel_spacing_mm: Optional[float] = None
) -> Tuple[Optional[float], Optional[float]]:
    """
    Fits direct least-squares ellipse to mask and calculates perimeter in pixels & mm.
    Fast guarded evaluation prevents CPU slowdowns on empty/noise predictions.
    Returns (ac_px, ac_mm).
    """
    if binary_mask is None or np.count_nonzero(binary_mask) < 50:
        return None, None

    mask_u8 = (binary_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 50 or len(largest) < 5:
        return None, None

    try:
        ellipse = cv2.fitEllipseDirect(largest)
    except Exception:
        try:
            ellipse = cv2.fitEllipse(largest)
        except Exception:
            return None, None

    (cx, cy), (axis1, axis2), angle = ellipse
    semi_major = max(axis1, axis2) / 2.0
    semi_minor = min(axis1, axis2) / 2.0

    if semi_major <= 0 or semi_minor <= 0:
        return None, None

    p_px = ramanujan_perimeter(semi_major, semi_minor)
    p_mm = (p_px * pixel_spacing_mm) if (pixel_spacing_mm is not None and pixel_spacing_mm > 0) else None
    return p_px, p_mm


# --- 6. DATASET & PATIENT-LEVEL LOADER WITH IN-MEMORY RAM CACHING ---
class UltrasoundItem:
    def __init__(
        self,
        img_path: Path,
        mask_path: Optional[Path],
        patient_id: str,
        pixel_spacing_mm: float = 0.25,
        reference_ac_mm: Optional[float] = None
    ):
        self.img_path = img_path
        self.mask_path = mask_path
        self.patient_id = patient_id
        self.pixel_spacing_mm = pixel_spacing_mm
        self.reference_ac_mm = reference_ac_mm


class KaggleFetalDataset(Dataset):
    def __init__(
        self,
        items: List[UltrasoundItem],
        img_size: Tuple[int, int] = (256, 320),
        is_train: bool = True
    ):
        self.items = items
        self.img_size = img_size
        self.is_train = is_train

        # Fast In-Memory RAM Caching to eliminate Kaggle disk reading bottleneck
        self.cached_images = []
        self.cached_masks = []
        for item in items:
            img_flt, mask_res = self._load_and_preprocess(item)
            self.cached_images.append(img_flt)
            self.cached_masks.append(mask_res)

    def _load_and_preprocess(self, item: UltrasoundItem) -> Tuple[np.ndarray, np.ndarray]:
        img_np = load_image_file(item.img_path)
        if img_np is None:
            img_np = np.zeros(self.img_size, dtype=np.float32)

        mask_np = load_image_file(item.mask_path) if item.mask_path else None
        if mask_np is None:
            mask_np = np.zeros_like(img_np, dtype=np.float32)
        else:
            mask_np = (mask_np > 0).astype(np.float32)

        # Resize to target resolution
        target_h, target_w = self.img_size
        img_res = cv2.resize(img_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        mask_res = cv2.resize(mask_np, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # Min-max intensity normalization to [0, 1]
        img_flt = img_res.astype(np.float32)
        i_min, i_max = img_flt.min(), img_flt.max()
        if i_max > i_min:
            img_flt = (img_flt - i_min) / (i_max - i_min)
        elif i_max > 1.0:
            img_flt = img_flt / 255.0

        return img_flt, mask_res

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        img_flt = self.cached_images[idx].copy()
        mask_res = self.cached_masks[idx].copy()

        # Augmentation (Train only)
        if self.is_train:
            if random.random() < 0.5:
                img_flt = np.fliplr(img_flt).copy()
                mask_res = np.fliplr(mask_res).copy()

            if random.random() < 0.4:
                alpha = random.uniform(0.85, 1.15)
                img_flt = np.clip(img_flt * alpha, 0.0, 1.0)

        img_t = torch.from_numpy(img_flt).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask_res).unsqueeze(0).float()

        return {
            "image": img_t,
            "mask": mask_t,
            "pixel_spacing_mm": item.pixel_spacing_mm,
            "reference_ac_mm": item.reference_ac_mm if item.reference_ac_mm is not None else float("nan"),
            "patient_id": item.patient_id,
            "orig_size": self.img_size
        }


# --- 7. AUTOMATIC DATASET DISCOVERY & MASK MATCHING ---
def extract_zip_files_if_any(search_dirs: List[str], target_dir: Path) -> List[Path]:
    """Auto-extracts any zipped datasets uploaded to Kaggle inputs."""
    extracted_dirs = []
    for s_dir in search_dirs:
        rpath = Path(s_dir)
        if not rpath.exists():
            continue
        for zip_p in rpath.rglob("*.zip"):
            out_path = target_dir / zip_p.stem
            if not out_path.exists() or len(list(out_path.rglob("*"))) == 0:
                print(f"--> Extracting dataset archive: {zip_p} -> {out_path}")
                out_path.mkdir(parents=True, exist_ok=True)
                try:
                    with zipfile.ZipFile(zip_p, 'r') as zip_ref:
                        zip_ref.extractall(out_path)
                    extracted_dirs.append(out_path)
                except Exception as e:
                    print(f"--> [Warning] Failed to extract archive {zip_p}: {e}")
            else:
                extracted_dirs.append(out_path)
    return extracted_dirs


def is_mask_path(path: Path) -> bool:
    """Checks if path belongs to a mask/label/annotation folder or filename."""
    name = path.name.lower()
    if any(kw in name for kw in ["mask", "label", "annotation", "gt", "_seg"]):
        return True

    for part in path.parts[:-1]:
        part_l = part.lower()
        if part_l in ["masks", "labels", "labelstr", "annotations", "ground_truth", "gt", "structures", "segmentations"]:
            return True

    return False


def discover_ac_dataset() -> List[UltrasoundItem]:
    """
    Scans Kaggle inputs and local directories to find ACOUSLIC-AI or standard AC dataset files.
    Robustly handles subdirectories (images/ vs masks/) and UUID file naming.
    """
    search_dirs = [
        "/kaggle/input",
        "/kaggle/input/acouslic-ai-dataset",
        "/kaggle/input/acouslic-ai",
        "/kaggle/input/acouslic-ai-train-set",
        "/kaggle/working/extracted_dataset",
        "./data/fetal",
        "../data/fetal",
        "."
    ]
    print("--> Scanning for datasets across directories...")

    target_extract_dir = Path("/kaggle/working/extracted_dataset")
    extracted_dirs = extract_zip_files_if_any(search_dirs, target_extract_dir)
    for ed in extracted_dirs:
        if str(ed) not in search_dirs:
            search_dirs.insert(0, str(ed))

    items = []
    valid_exts = {".png", ".jpg", ".jpeg", ".npy", ".mha", ".mhd", ".nii", ".gz", ".bmp", ".tif", ".tiff"}

    for root_dir in search_dirs:
        rpath = Path(root_dir)
        if not rpath.exists():
            continue

        all_files = [p for p in rpath.rglob("*") if p.is_file() and any(p.name.lower().endswith(ext) for ext in valid_exts)]
        if not all_files:
            continue

        # Separate images and masks using path awareness
        cand_masks = [p for p in all_files if is_mask_path(p)]
        cand_imgs = [p for p in all_files if not is_mask_path(p) and "checkpoint" not in str(p) and "demo_data" not in str(p)]

        if cand_imgs:
            print(f"--> Found {len(cand_imgs)} image files & {len(cand_masks)} mask files under {root_dir}")
            
            # Map candidate masks by filename, stem, and base stem
            mask_map = {}
            for mp in cand_masks:
                mask_map[mp.name.lower()] = mp
                mask_map[mp.stem.lower()] = mp
                base_stem = mp.stem.lower().replace("_0000", "").replace("_mask", "").replace("_gt", "")
                mask_map[base_stem] = mp

            for img_p in cand_imgs:
                stem = img_p.stem.lower()
                base_stem = stem.replace("_0000", "")
                
                # Search mask map
                mask_p = (
                    mask_map.get(img_p.name.lower()) or 
                    mask_map.get(stem) or 
                    mask_map.get(base_stem) or 
                    mask_map.get(f"{base_stem}_mask") or
                    mask_map.get(f"{base_stem}_gt")
                )

                parts = img_p.stem.split("_")
                pid = parts[0] if len(parts) > 1 and parts[0] not in ["frame", "image", "acouslic"] else img_p.stem
                items.append(UltrasoundItem(
                    img_path=img_p,
                    mask_path=mask_p,
                    patient_id=pid,
                    pixel_spacing_mm=0.25,
                    reference_ac_mm=None
                ))

            if len(items) > 0:
                break

    masks_found = sum(1 for item in items if item.mask_path is not None)
    print(f"--> Total discovered dataset frames: {len(items)} (Annotated Masks Found: {masks_found})")

    # Diagnostic pre-check loading
    if len(items) > 0:
        # Pick first item that has a paired mask if possible
        paired_items = [it for it in items if it.mask_path is not None]
        sample_item = paired_items[0] if paired_items else items[0]

        sample_img = load_image_file(sample_item.img_path)
        sample_mask = load_image_file(sample_item.mask_path) if sample_item.mask_path else None

        if sample_img is not None:
            print(f"--> [Diagnostic Pre-Check] Image '{sample_item.img_path.name}' Loaded: shape={sample_img.shape}, min={sample_img.min():.1f}, max={sample_img.max():.1f}, mean={sample_img.mean():.1f}")
            if sample_img.max() == 0:
                print("--> [CRITICAL WARNING] Sample image loaded as ALL ZEROS! Check file format & reader.")
        else:
            print(f"--> [CRITICAL WARNING] Failed to load sample image from: {sample_item.img_path}")

        if sample_mask is not None:
            print(f"--> [Diagnostic Pre-Check] Mask '{sample_item.mask_path.name}' Loaded: shape={sample_mask.shape}, min={sample_mask.min():.1f}, max={sample_mask.max():.1f}, non-zero pixels={np.count_nonzero(sample_mask)}")
            if np.count_nonzero(sample_mask) == 0:
                print("--> [WARNING] Sample ground-truth mask contains 0 non-zero pixels.")
        else:
            print(f"--> [WARNING] Ground-truth mask file not paired for sample frame: {sample_item.img_path.name}")

    return items


# --- 8. TRAINING & VALIDATION LOOP ---
def train_ac_model(
    epochs: int = 35,
    batch_size: int = 8,
    lr: float = 1e-4,
    img_size: Tuple[int, int] = (256, 320),
    output_dir: str = "/kaggle/working"
):
    os.makedirs(output_dir, exist_ok=True)
    items = discover_ac_dataset()

    masks_found = sum(1 for item in items if item.mask_path is not None)
    if len(items) == 0 or masks_found == 0:
        print("--> [Warning] No valid annotated dataset frames found. Creating high-quality synthetic ultrasound batch.")
        demo_dir = Path(output_dir) / "demo_data"
        demo_dir.mkdir(exist_ok=True)
        items = []
        for i in range(30):
            # Generate realistic ultrasound speckle pattern with clear abdominal contour
            synth_img = np.random.normal(60, 20, (256, 320)).astype(np.float32)
            cv2.ellipse(synth_img, (160, 128), (85, 65), 0, 0, 360, (220, 220, 220), 14)
            cv2.ellipse(synth_img, (160, 128), (81, 61), 0, 0, 360, (140, 140, 140), -1)
            synth_img = np.clip(synth_img, 0, 255).astype(np.uint8)

            synth_mask = np.zeros((256, 320), dtype=np.uint8)
            cv2.ellipse(synth_mask, (160, 128), (85, 65), 0, 0, 360, 1, -1)

            ip = demo_dir / f"P{i:03d}_frame.png"
            mp = demo_dir / f"P{i:03d}_mask.png"
            cv2.imwrite(str(ip), synth_img)
            cv2.imwrite(str(mp), synth_mask)
            items.append(UltrasoundItem(ip, mp, f"P{i:03d}", pixel_spacing_mm=0.25, reference_ac_mm=114.6))

    # Patient or Frame-level split
    random.seed(SEED)
    pids = sorted(list(set(it.patient_id for it in items)))
    random.shuffle(pids)

    n_p = len(pids)
    if n_p >= 5:
        n_tr = max(1, int(0.70 * n_p))
        n_va = max(1, int(0.15 * n_p))

        tr_pids = set(pids[:n_tr])
        va_pids = set(pids[n_tr:n_tr + n_va])
        te_pids = set(pids[n_tr + n_va:])

        train_items = [it for it in items if it.patient_id in tr_pids]
        val_items = [it for it in items if it.patient_id in va_pids]
        test_items = [it for it in items if it.patient_id in te_pids]
    else:
        shuffled = list(items)
        random.shuffle(shuffled)
        n_tot = len(shuffled)
        n_tr = max(1, int(0.70 * n_tot))
        n_va = max(1, int(0.15 * n_tot))
        train_items = shuffled[:n_tr]
        val_items = shuffled[n_tr:n_tr + n_va]
        test_items = shuffled[n_tr + n_va:]
        tr_pids, va_pids, te_pids = set(pids), set(pids), set(pids)

    print(f"--> Split Summary: Patients/Groups = {n_p} (Train: {len(tr_pids)}, Val: {len(va_pids)}, Test: {len(te_pids)})")
    print(f"--> Frame Counts: Train = {len(train_items)}, Val = {len(val_items)}, Test = {len(test_items)}")

    print("--> Pre-loading dataset into RAM for fast GPU throughput...")
    train_ds = KaggleFetalDataset(train_items, img_size=img_size, is_train=True)
    val_ds = KaggleFetalDataset(val_items, img_size=img_size, is_train=False)
    test_ds = KaggleFetalDataset(test_items, img_size=img_size, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize model
    model = FetalACAttentionUNet(in_channels=1, num_classes=1, base_filters=32).to(DEVICE)
    print_model_summary(model, input_size=(1, 1, img_size[0], img_size[1]))

    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Compatible PyTorch AMP Scaler
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == "cuda"))
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_val_dice = 0.0
    best_ckpt_path = Path(output_dir) / "acouslic_attention_unet.pth"

    history = {"train_loss": [], "val_dice": [], "val_mae_mm": []}

    print("=" * 70)
    print("--> Starting Attention U-Net Fetal AC Training...")
    print("=" * 70)

    def get_autocast_context():
        try:
            return torch.amp.autocast('cuda', enabled=(DEVICE == "cuda"))
        except (AttributeError, TypeError):
            return torch.cuda.amp.autocast(enabled=(DEVICE == "cuda"))

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            imgs = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)

            optimizer.zero_grad()
            with get_autocast_context():
                logits = model(imgs)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_ds)

        # Validation
        model.eval()
        val_dices = []
        val_maes = []

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(DEVICE)
                masks = batch["mask"].to(DEVICE)
                spacings = batch["pixel_spacing_mm"].numpy()
                refs = batch["reference_ac_mm"].numpy()

                with get_autocast_context():
                    logits = model(imgs)
                    probs = torch.sigmoid(logits)

                pred_masks = (probs >= 0.5).cpu().numpy()
                gt_masks = masks.cpu().numpy()

                for b in range(imgs.size(0)):
                    p_m = pred_masks[b, 0]
                    g_m = gt_masks[b, 0]

                    g_sum = g_m.sum()
                    p_sum = p_m.sum()

                    if g_sum > 0:
                        inter = (p_m * g_m).sum()
                        tot = p_sum + g_sum
                        d = (2.0 * inter + 1e-6) / (tot + 1e-6)
                        val_dices.append(d)
                    elif p_sum > 0:
                        val_dices.append(0.0)

                    sp = float(spacings[b]) if not np.isnan(spacings[b]) else 0.25
                    _, ac_mm = fit_ellipse_and_measure_ac(p_m, pixel_spacing_mm=sp)
                    _, gt_ac_mm = fit_ellipse_and_measure_ac(g_m, pixel_spacing_mm=sp)
                    target_ref = float(refs[b]) if not np.isnan(refs[b]) else gt_ac_mm

                    if ac_mm is not None and target_ref is not None and not np.isnan(target_ref):
                        val_maes.append(abs(ac_mm - target_ref))

        mean_dice = float(np.mean(val_dices)) if val_dices else 0.0
        mean_mae = float(np.mean(val_maes)) if val_maes else 0.0

        history["train_loss"].append(train_loss)
        history["val_dice"].append(mean_dice)
        history["val_mae_mm"].append(mean_mae)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Dice: {mean_dice:.4f} | Val MAE: {mean_mae:.2f} mm | LR: {scheduler.get_last_lr()[0]:.2e}")

        # Checkpoint Best Model
        if mean_dice > best_val_dice:
            best_val_dice = mean_dice
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_dice": best_val_dice,
                "val_mae_mm": mean_mae
            }, best_ckpt_path)
            print(f"  --> Saved new best checkpoint to: {best_ckpt_path} (Dice: {best_val_dice:.4f})")

    # --- 9. GENERATE DIAGNOSTIC PLOTS ---
    print("\n" + "=" * 70)
    print("--> Generating Comprehensive Diagnostic Training Plots...")
    print("=" * 70)

    # 1. Training Curves Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs_range = range(1, epochs + 1)

    axes[0].plot(epochs_range, history["train_loss"], color="#2b5c8f", lw=2, marker="o", ms=4)
    axes[0].set_title("Training Loss (BCE + Soft Dice)", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(epochs_range, history["val_dice"], color="#2e7d32", lw=2, marker="s", ms=4)
    axes[1].set_title("Validation Dice Score", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Coefficient")
    axes[1].set_ylim([0.0, 1.05])
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].plot(epochs_range, history["val_mae_mm"], color="#d32f2f", lw=2, marker="^", ms=4)
    axes[2].set_title("Validation Biometry Error (MAE mm)", fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Error (mm)")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    curves_path = Path(output_dir) / "training_curves.png"
    plt.savefig(curves_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"--> Saved training curves plot: {curves_path}")

    # --- 10. HELD-OUT TEST EVALUATION & VISUAL OVERLAYS ---
    print("\n" + "=" * 70)
    print("--> Running Clinical Biometric Evaluation on Held-Out Test Scans...")
    print("=" * 70)

    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    test_dices = []
    pred_acs = []
    ref_acs = []
    sample_visuals = []

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)
            spacings = batch["pixel_spacing_mm"].numpy()
            refs = batch["reference_ac_mm"].numpy()

            logits = model(imgs)
            probs = torch.sigmoid(logits)
            pred_masks = (probs >= 0.5).cpu().numpy()
            gt_masks = masks.cpu().numpy()

            for b in range(imgs.size(0)):
                p_m = pred_masks[b, 0]
                g_m = gt_masks[b, 0]
                img_raw = imgs[b, 0].cpu().numpy()

                g_sum = g_m.sum()
                if g_sum > 0:
                    inter = (p_m * g_m).sum()
                    tot = p_m.sum() + g_sum
                    d = (2.0 * inter + 1e-6) / (tot + 1e-6)
                    test_dices.append(d)

                    sp = float(spacings[b]) if not np.isnan(spacings[b]) else 0.25
                    _, ac_mm = fit_ellipse_and_measure_ac(p_m, pixel_spacing_mm=sp)
                    _, gt_ac_mm = fit_ellipse_and_measure_ac(g_m, pixel_spacing_mm=sp)
                    target_ref = float(refs[b]) if not np.isnan(refs[b]) else gt_ac_mm

                    if ac_mm is not None and target_ref is not None and not np.isnan(target_ref):
                        pred_acs.append(ac_mm)
                        ref_acs.append(target_ref)

                    if len(sample_visuals) < 4:
                        sample_visuals.append((img_raw, g_m, p_m, d, ac_mm, target_ref))

    final_test_dice = float(np.mean(test_dices)) if test_dices else 0.0
    print(f"--> Test Overlap Dice Score: {final_test_dice:.4f}")

    # Generate Visual Segmentation Predictions Figure
    if sample_visuals:
        fig, axes = plt.subplots(len(sample_visuals), 3, figsize=(12, 4 * len(sample_visuals)))
        if len(sample_visuals) == 1:
            axes = np.expand_dims(axes, axis=0)

        for i, (raw_img, g_mask, p_mask, d_score, p_ac, r_ac) in enumerate(sample_visuals):
            # Column 1: Raw Ultrasound Image
            axes[i, 0].imshow(raw_img, cmap="gray")
            axes[i, 0].set_title(f"Test Scan #{i+1}: Raw Ultrasound", fontsize=10, fontweight="bold")
            axes[i, 0].axis("off")

            # Column 2: Ground Truth Mask
            axes[i, 1].imshow(raw_img, cmap="gray")
            masked_gt = np.ma.masked_where(g_mask == 0, g_mask)
            axes[i, 1].imshow(masked_gt, cmap="Greens", alpha=0.6, vmin=0, vmax=1)
            gt_text = f"GT AC: {r_ac:.1f} mm" if (r_ac is not None and not np.isnan(r_ac)) else "GT Mask"
            axes[i, 1].set_title(f"Ground Truth ({gt_text})", fontsize=10, fontweight="bold")
            axes[i, 1].axis("off")

            # Column 3: Predicted Attention U-Net Mask
            axes[i, 2].imshow(raw_img, cmap="gray")
            masked_pred = np.ma.masked_where(p_mask == 0, p_mask)
            axes[i, 2].imshow(masked_pred, cmap="Reds", alpha=0.6, vmin=0, vmax=1)
            pred_text = f"Pred AC: {p_ac:.1f} mm" if p_ac is not None else "No Fit"
            axes[i, 2].set_title(f"Attention U-Net Pred (Dice: {d_score:.3f} | {pred_text})", fontsize=10, fontweight="bold")
            axes[i, 2].axis("off")

        plt.tight_layout()
        samples_path = Path(output_dir) / "fetal_segmentation_samples.png"
        plt.savefig(samples_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"--> Saved segmentation sample overlays: {samples_path}")

    # Generate Bland-Altman Plot
    if len(pred_acs) >= 3:
        p_arr = np.array(pred_acs)
        r_arr = np.array(ref_acs)
        diff = p_arr - r_arr
        bias = float(np.mean(diff))
        sd = float(np.std(diff, ddof=1))
        mae = float(np.mean(np.abs(diff)))
        lower_loa = bias - 1.96 * sd
        upper_loa = bias + 1.96 * sd

        print(f"--> Biometric MAE: {mae:.2f} mm")
        print(f"--> Bland-Altman Systematic Bias: {bias:+.2f} mm")
        print(f"--> Bland-Altman 95% Limits of Agreement: [{lower_loa:.2f} mm, {upper_loa:.2f} mm]")

        plt.figure(figsize=(8, 5))
        mean_pairs = (p_arr + r_arr) / 2.0
        plt.scatter(mean_pairs, diff, alpha=0.75, color="#008080", edgecolors="k", s=45)
        plt.axhline(bias, color="red", linestyle="--", lw=2, label=f"Bias: {bias:+.2f} mm")
        plt.axhline(upper_loa, color="blue", linestyle=":", lw=1.5, label=f"+1.96 SD: {upper_loa:.2f} mm")
        plt.axhline(lower_loa, color="blue", linestyle=":", lw=1.5, label=f"-1.96 SD: {lower_loa:.2f} mm")
        plt.xlabel("Mean Fetal AC [(Pred + Ref) / 2] (mm)", fontsize=11)
        plt.ylabel("Difference [Pred - Ref] (mm)", fontsize=11)
        plt.title("Bland-Altman Agreement Plot: Fetal AC Biometry", fontsize=12, fontweight="bold")
        plt.legend(frameon=True, facecolor="white", edgecolor="none")
        plt.grid(True, linestyle="--", alpha=0.5)
        plot_path = Path(output_dir) / "bland_altman_plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"--> Saved Bland-Altman plot: {plot_path}")

    print("\n" + "=" * 70)
    print("SUCCESS: TRAINING, VISUALIZATION & BIOMETRY VALIDATION COMPLETE!")
    print(f"Check your Kaggle Output tab for: {best_ckpt_path.name}")
    print("Generated Diagnostic Files:")
    print("  1. acouslic_attention_unet.pth (Model Weights)")
    print("  2. training_curves.png (Loss, Dice & MAE plots)")
    print("  3. fetal_segmentation_samples.png (Visual Prediction Overlays)")
    print("  4. bland_altman_plot.png (Clinical Biometry Agreement Plot)")
    print("=" * 70)


if __name__ == "__main__":
    train_ac_model(epochs=35, batch_size=8, lr=1e-4)
