"""Fetal abdominal circumference (AC) from Acouslic-AI sweeps with best_attunet.pth.

Everything here is taken from the training notebook (aa-unet-training): model classes,
preprocessing, post-processing and the AC ellipse fit. The only addition is the frame
selector validated in the notebook: median AC of the 5 most confident frames per sweep.
"""
import math
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- constants (same values as the notebook) -------------------------------------------
H, W = 320, 416          # model input size
PIXEL_MM = 0.28          # official pixel spacing of Acouslic-AI frames
FRAMES_PER_SWEEP = 140   # 840 frames per case = 6 sweeps x 140
N_SWEEPS = 6
MIN_AREA = 2000          # smallest accepted mask area (pixels at 320x416)
TOP_K = 5                # frames averaged per sweep (median AC)


# ---- model (verbatim from the notebook; layer names match best_attunet.pth) -------------
class ConvBlock(nn.Module):
    def __init__(s, i, o):
        super().__init__()
        s.b = nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))

    def forward(s, x):
        return s.b(x)


class AttGate(nn.Module):
    def __init__(s, c, mid):
        super().__init__()
        s.Wg = nn.Sequential(nn.Conv2d(c, mid, 1, bias=False), nn.BatchNorm2d(mid))
        s.Wx = nn.Sequential(nn.Conv2d(c, mid, 1, bias=False), nn.BatchNorm2d(mid))
        s.psi = nn.Sequential(nn.Conv2d(mid, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(s, g, x):
        return x * s.psi(F.relu(s.Wg(g) + s.Wx(x)))


class AttUNet(nn.Module):
    def __init__(s, ch=(32, 64, 128, 256, 512)):
        super().__init__()
        s.enc, i = nn.ModuleList(), 1
        for c in ch:
            s.enc.append(ConvBlock(i, c))
            i = c
        s.pool = nn.MaxPool2d(2)
        s.up, s.att, s.dec = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for k in range(len(ch) - 1, 0, -1):
            s.up.append(nn.ConvTranspose2d(ch[k], ch[k - 1], 2, stride=2))
            s.att.append(AttGate(ch[k - 1], max(ch[k - 1] // 2, 8)))
            s.dec.append(ConvBlock(ch[k - 1] * 2, ch[k - 1]))
        s.out = nn.Conv2d(ch[0], 1, 1)

    def forward(s, x):
        skips = []
        for k, e in enumerate(s.enc):
            x = e(x)
            if k < len(s.enc) - 1:
                skips.append(x)
                x = s.pool(x)
        for u, a, d in zip(s.up, s.att, s.dec):
            x = u(x)
            sk = a(x, skips.pop())
            x = d(torch.cat([sk, x], 1))
        return s.out(x)


def load_model(weights_path, device=None):
    """Load best_attunet.pth (a plain state_dict). Returns (model, device)."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = AttUNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model.eval(), device


# ---- helpers (verbatim logic from the notebook) -----------------------------------------
def ac_full(m, px=PIXEL_MM):
    """AC in mm from a mask at ORIGINAL frame resolution (ellipse fit + Ramanujan)."""
    cnts, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return np.nan
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5:
        return np.nan
    (_, _), (w, h), _ = cv2.fitEllipse(c.reshape(-1, 2).astype(np.float32))
    a, b = w / 2 * px, h / 2 * px
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


def postprocess(p):
    """Keep the largest connected component; drop it if smaller than MIN_AREA."""
    p = p.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(p, connectivity=8)
    if n <= 1:
        return np.zeros_like(p)
    k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = (lab == k).astype(np.uint8)
    return out if out.sum() >= MIN_AREA else np.zeros_like(p)


def to_uint8(vol):
    """Same intensity scaling as training: whole-volume max -> 0..255."""
    if vol.dtype == np.uint8:
        return vol
    return (255 * vol.astype(np.float32) / max(float(vol.max()), 1)).astype(np.uint8)


@torch.no_grad()
def predict_probs(model, device, frames_u8, bs=32):
    """frames_u8: (N, H, W) uint8 already resized to (H, W). Returns (N, H, W) probabilities."""
    out = []
    for i in range(0, len(frames_u8), bs):
        x = torch.from_numpy(frames_u8[i:i + bs].astype(np.float32) / 255)[:, None].to(device)
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                lg = model(x)
        else:
            lg = model(x)
        out.append(torch.sigmoid(lg.float())[:, 0].cpu().numpy())
    return np.concatenate(out)


# ---- AC per sweep -----------------------------------------------------------------------
def sweep_ac(model, device, frames, top_k=TOP_K, px=PIXEL_MM):
    """frames: uint8 array (n_frames, orig_h, orig_w) for ONE sweep.
    Picks the top_k frames by confidence (mean probability inside the cleaned mask) and
    returns the median AC in mm, or NaN if the model finds no abdomen in the sweep."""
    orig_h, orig_w = frames.shape[1:]
    small = np.stack([cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) for f in frames])
    pr = predict_probs(model, device, small)
    pps = [postprocess(p > 0.5) for p in pr]
    areas = np.array([m.sum() for m in pps], float)
    conf = np.array([p[m > 0].mean() if m.any() else 0 for p, m in zip(pr, pps)])
    valid = np.where(areas > 0)[0]
    if len(valid) == 0:
        return float("nan")
    best = valid[np.argsort(-conf[valid])[:top_k]]
    acs = []
    for i in best:
        full = cv2.resize(pr[i], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) > 0.5
        acs.append(ac_full(full, px))
    return float(np.nanmedian(acs))


def case_acs(model, device, mha_path, frames_per_sweep=FRAMES_PER_SWEEP, px=PIXEL_MM):
    """Run one stacked case file (.mha). Returns a list of 6 ACs in mm, one per sweep."""
    vol = to_uint8(sitk.GetArrayFromImage(sitk.ReadImage(str(mha_path))))
    need = N_SWEEPS * frames_per_sweep
    if vol.ndim != 3 or vol.shape[0] != need:
        raise ValueError(f"{Path(mha_path).name}: expected {need} frames "
                         f"({N_SWEEPS} sweeps x {frames_per_sweep}), got shape {vol.shape}")
    return [sweep_ac(model, device, vol[k * frames_per_sweep:(k + 1) * frames_per_sweep], px=px)
            for k in range(N_SWEEPS)]


# ---- AC from ONE 2D frame ---------------------------------------------------------------
def image_ac(model, device, img, px=PIXEL_MM):
    """AC from a single 2D frame. No frame selection happens here, so this is less reliable
    than sweep_ac: the result is only as good as the plane the image shows.

    img: 2D array (H, W) or (H, W, C). px: mm per pixel of the ORIGINAL image.
    Returns dict(ac_mm, mask, conf). mask is uint8 at the input resolution. ac_mm is NaN
    and mask is empty when no abdomen is found. conf = mean probability inside the mask."""
    img = to_uint8(np.asarray(img))
    if img.ndim == 3:
        img = (cv2.cvtColor(np.ascontiguousarray(img[..., :3]), cv2.COLOR_RGB2GRAY)
               if img.shape[2] >= 3 else img[..., 0])
    orig_h, orig_w = img.shape
    small = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    pr = predict_probs(model, device, small[None])[0]
    m = postprocess(pr > 0.5)
    if not m.any():
        return dict(ac_mm=float("nan"), mask=np.zeros((orig_h, orig_w), np.uint8), conf=0.0)
    full = cv2.resize(pr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) > 0.5
    return dict(ac_mm=float(ac_full(full, px)), mask=full.astype(np.uint8), conf=float(pr[m > 0].mean()))
