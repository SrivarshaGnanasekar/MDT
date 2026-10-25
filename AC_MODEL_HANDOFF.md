# AC Model Handoff: Fetal Abdominal Circumference (AC) Segmentation

Part of the Maternal Digital Twin project.
Read this file first, before opening any code.

> Items marked **‹FILL›** are numbers or names only the project owner knows.
> Section 11 has a script that collects most of them automatically.

---

## 1. What this is

A deep-learning model that segments the fetal abdomen in an ultrasound image,
fits an ellipse to the mask, and converts the ellipse to an **abdominal
circumference (AC) in mm**. The AC is then compared with the expected AC for
the **gestational age (GA)**, and the result is labelled with a tier such as
*below expected*.

Pipeline in one line:

```
.mha ultrasound sweep -> pick frame(s) -> preprocess -> Attention U-Net mask
 -> ellipse fit -> pixel -> mm using real pixel spacing -> AC (mm)
 -> compare with expected AC for real GA -> tier / label
```

## 2. Folder layout (what is in the zip)

| Path | Purpose |
|---|---|
| `models/new_model/best_attunet.pth` | Trained weights (about 31.5 MB). **Do not lose or rename.** |
| `models/new_model/inference.py` | Loads the weights and runs segmentation on an image or frame. |
| `fetal/` | Wrapper code: ellipse fitting, mm conversion, GA comparison, tiers. |
| `config.py` | Settings. Database settings were removed or blanked for this copy. |
| `tests/` | Regression tests. |
| `tests/sample_mha/` | Sample `.mha` used by one test (the test skips itself if missing). |
| `AC_MODEL_HANDOFF.md` | This file. |

Model architecture: **Attention U-Net** (`best_attunet.pth`), PyTorch.
Total parameters: **7,857,603** (~7.86M parameters).
Input size: **320** x **416**, channels: **1** (grayscale), output: single-channel binary mask.

## 3. Dataset

| Item | Value |
|---|---|
| Dataset name / source | Acouslic-AI (Grand Challenge Benchmark) |
| License / usage terms | Research Use Only (Acouslic-AI Challenge License) |
| Imaging modality | 2D/3D freehand fetal ultrasound sweeps |
| Number of subjects/cases | Multi-sweep volumetric scans |
| Number of annotated images | Standard plane annotations per sweep |
| Train / val / test split | Standard challenge cross-validation partition |
| Annotation type | Abdomen segmentation mask & biometry ellipse fit |
| Total size on disk | Volumetric `.mha` (~335 MB per case volume) |

**The dataset is not included in the zip.** Ask the project owner for access
or the download link. Do not redistribute it.

### 3.1 The `.mha` files

`.mha` is a MetaImage file (header plus raw voxel data), read with SimpleITK.

| Property | Value |
|---|---|
| Frames per sweep | **140** frames per sweep (6 sweeps = **840** frames per case) |
| Frame size (pixels) | **562** (height) x **744** (width) |
| Pixel spacing (mm) | **0.28 mm**: always read it from header `img.GetSpacing()[:2]`, default 0.28 |
| Data type | **uint8** (8-bit unsigned integer) |
| Size per file | **~335 MB** (351,227,978 bytes) |

```python
import SimpleITK as sitk
img = sitk.ReadImage("sample.mha")
print(img.GetSize())      # (744, 562, 840) -> (width, height, frames)
print(img.GetSpacing())   # (0.28, 0.28, 0.28) in mm  <- use sx, sy for mm conversion
arr = sitk.GetArrayFromImage(img)   # shape (840, 562, 744)
```

### 3.2 Image files (png and others)

Standard ultrasound video frames exported from sweeps or clinical capture devices.
- Example sample included: `1dccf454_sw4_f478.png` (562 x 744 pixels, single-frame extraction).

## 4. Preprocessing

Exact steps as in training and `inference.py`, in order:
1. Grayscale conversion: If input has 3 or 4 channels, convert to 1-channel grayscale (`cv2.COLOR_BGR2GRAY`).
2. Type casting: Cast to `uint8` in range `[0, 255]`.
3. Spatial resizing: Resize frame from `(H_orig, W_orig)` to model input dimensions `320 x 416` (height=320, width=416) using bilinear interpolation (`cv2.INTER_LINEAR`).
4. Normalization: Scale pixels to float in `[0.0, 1.0]` by dividing by `255.0`.
5. Tensor conversion: Shape formatted as `(1, 1, 320, 416)` PyTorch tensor.
6. Post-processing: Apply Sigmoid activation -> threshold at `0.5` -> remove tiny contours `< 2000` pixels -> resize mask back to original image resolution `(W_orig, H_orig)` using nearest neighbor (`cv2.INTER_NEAREST`).
**Inference must use the same preprocessing as training.** If you change it, the results change.

## 5. Training process

| Item | Value |
|---|---|
| Framework | PyTorch |
| Architecture | Attention U-Net (`AttUNet`: 5-level encoder/decoder with Attention Gates) |
| Channel depths | `[32, 64, 128, 256, 512]` |
| Loss | Combined Dice Loss + Binary Cross Entropy (BCE) |
| Optimizer / learning rate | AdamW / lr 1e-4 with CosineAnnealingLR |
| Batch size / epochs | Batch size 8-16 / 50 epochs |
| Checkpoint selection | `models/new_model/best_attunet.pth` = Best Validation Dice score |

## 6. Results

| Metric | Value |
|---|---|
| Dice (validation) | > 0.94 |
| Pixel Spacing Precision | 0.28 mm / pixel calibrated |
| AC error vs reference (mean absolute, mm) | Not yet verified: run `fetal/evaluate_ac_accuracy.py` on the held-out case list (the earlier ~5.3 mm figure had no reproducible source) |

Known limitations: Frames where the abdomen is partially out of view, shadowing from fetal spine/limbs, or frames where confidence score < 0.98 (automatically flagged as `low` plane quality).

## 7. Three corrections that matter (read carefully)

These were found while verifying the pipeline. Keep them intact.

1. **Real pixel spacing.** The mm conversion must use the spacing stored in
   the file header, not a hard-coded or default value. A wrong spacing
   scales every AC value.
2. **Real gestational age.** The comparison must use the patient's actual GA,
   not a placeholder or default. Otherwise every case is compared against
   the wrong expected range.
3. **"Below expected" tier fix.** The tier logic that labels an AC as *below
   expected* was corrected. Do not revert it. The regression tests cover it.

## 8. How to use

### 8.1 Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install torch torchvision SimpleITK numpy opencv-python scikit-image pytest
```

Add any other packages listed in `requirements.txt` if present. Use a
**CPU** build of torch if there is no GPU.

### 8.2 Run inference on one image

```python
from fetal.ac_predict import FetalACPredictor

predictor = FetalACPredictor(use_new_model=True)
result = predictor.predict("1dccf454_sw4_f478.png", pixel_spacing_mm=0.28, gestational_age_weeks=28.0)

print("AC (mm):", result["actual_ac_mm"])
print("AC (cm):", result["actual_ac_cm"])
print("Confidence:", result["model_confidence"])
print("Plane Quality:", result["plane_quality"])
print("Screening Tier:", result["screening_tier"])
print("Hadlock EFW (g):", result["efw_hadlock_g"])
```

Or using the low-level inference module directly:
```python
import cv2
import models.new_model.inference as inf

model, device = inf.load_model("models/new_model/best_attunet.pth")
img = cv2.imread("1dccf454_sw4_f478.png", cv2.IMREAD_GRAYSCALE)
res = inf.image_ac(model, device, img, px=0.28)
print("AC (mm):", res["ac_mm"], "Confidence:", res["conf"])
```

### 8.3 Get AC from a `.mha` sweep

```python
from fetal.ac_predict import FetalACPredictor

predictor = FetalACPredictor(use_new_model=True)
# Computes median AC across top 5 confident frames for each of the 6 sweeps
sweeps_ac = predictor.predict_mha("tests/sample_mha/53056b82-e84c-404a-8d3d-29c1d24d94c8.mha", px=0.28)
print("Sweep AC measurements (mm):", sweeps_ac)
```

### 8.4 Run the tests

```powershell
python -m pytest tests/test_new_attunet_integration.py -v
python -m pytest tests/test_ac_macrosomia.py -v
python -m pytest tests/test_ac_biometry_pipeline.py -v
```

The test that needs the full `.mha` skips itself if the file is absent.

## 9. Security notes

- No database credentials are included. The original project connects to a
  MongoDB Atlas database that holds **real patient records**. You do not need
  it for the AC model, and you should not ask for access unless necessary.
- If you ever see a connection string in code, do not commit it. Use an
  environment variable instead.

## 10. Suggested next steps

1. Run `pytest tests -q` and confirm everything passes on your machine.
2. Reproduce one AC value end to end on the sample `.mha` and check it against
   the stored reference number.
3. Frame selection: automate choosing the best standard AC plane from a sweep
   (currently ‹FILL›: manual or heuristic).
4. Evaluate on more cases, including difficult ones (shadowing, poor views),
   and report per-GA-week error.
5. Add uncertainty or a quality check so poor masks are flagged rather than
   producing a confident wrong AC.
6. Validate the expected-AC reference table and tier thresholds with a
   clinician.
7. Document any change to preprocessing, and re-run the regression tests
   after every change.
8. Package inference as a small API or CLI once results are stable.

## 11. Script to fill the blanks (run on your original project)

```powershell
# Sizes
Get-ChildItem -Recurse -File -Include *.mha | Select-Object FullName, @{n='MB';e={[math]::Round($_.Length/1MB,1)}}
(Get-ChildItem -Recurse -File -Include *.png | Measure-Object).Count   # image count
```

```python
# .mha header facts
import SimpleITK as sitk, glob
for f in glob.glob("**/*.mha", recursive=True)[:5]:
    im = sitk.ReadImage(f)
    print(f, im.GetSize(), im.GetSpacing(), im.GetPixelIDTypeAsString())
```

```python
# model input shape and parameter count
import torch
sd = torch.load("models/new_model/best_attunet.pth", map_location="cpu")
print(sum(v.numel() for v in sd.values() if hasattr(v, "numel")))
```

---

*Owner: ‹FILL name and contact›. Last updated: ‹FILL date›.*
