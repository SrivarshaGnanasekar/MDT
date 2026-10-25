import math
from pathlib import Path
import pytest
import numpy as np
import SimpleITK as sitk

from fetal.ac_predict import FetalACPredictor

SAMPLE_MHA = Path(__file__).resolve().parent / "sample_mha" / "53056b82-e84c-404a-8d3d-29c1d24d94c8.mha"


@pytest.mark.skipif(not SAMPLE_MHA.exists(), reason="Sample MHA file not found")
def test_new_attunet_frame70():
    predictor = FetalACPredictor(use_new_model=True)
    img_sitk = sitk.ReadImage(str(SAMPLE_MHA))
    vol = sitk.GetArrayFromImage(img_sitk)
    frame70 = vol[70]

    res = predictor.predict(frame70, pixel_spacing_mm=0.28)
    assert res["actual_ac_mm"] is not None
    assert res["actual_ac_mm"] == pytest.approx(248.93, abs=0.5)
    assert res["model_confidence"] == pytest.approx(0.9851, abs=0.01)
    assert res["plane_quality"] == "ok"


@pytest.mark.skipif(not SAMPLE_MHA.exists(), reason="Sample MHA file not found")
def test_new_attunet_case_acs():
    predictor = FetalACPredictor(use_new_model=True)
    sweeps_ac = predictor.predict_mha(SAMPLE_MHA, px=0.28)
    assert len(sweeps_ac) == 6
    sweep2_model_ac = sweeps_ac[1]  # Sweep 2 model prediction ~274.16 mm
    assert sweep2_model_ac is not None
    assert sweep2_model_ac == pytest.approx(274.16, abs=0.5)
    
    # CSV reference for case 53056b82-e84c-404a-8d3d-29c1d24d94c8 sweep 2 is 268.86 mm
    csv_ref_sweep2 = 268.85562310673123
    assert abs(sweep2_model_ac - csv_ref_sweep2) <= 15.0


def test_new_attunet_black_image():
    predictor = FetalACPredictor(use_new_model=True)
    black_img = np.zeros((562, 744), dtype=np.uint8)
    res = predictor.predict(black_img, pixel_spacing_mm=0.28)
    assert res["actual_ac_mm"] is None
    assert res["actual_ac_cm"] is None
    assert res["efw_hadlock_g"] is None
    assert res["delta_ac_relative_pct"] is None
    assert res["screening_tier"] == "Unavailable (low plane quality)"
    assert res["model_confidence"] == 0.0
    assert res["plane_quality"] == "low"
    assert res["mask"].sum() == 0


def test_new_attunet_load_failure_graceful():
    predictor = FetalACPredictor(use_new_model=True)
    # Simulate load failure by unsetting new_model
    predictor.new_model = None
    # Intentionally corrupt NEW_MODEL_PATH
    import config
    original_path = config.NEW_MODEL_PATH
    config.NEW_MODEL_PATH = Path("non_existent_path.pth")
    try:
        black_img = np.zeros((562, 744), dtype=np.uint8)
        res = predictor.predict(black_img, pixel_spacing_mm=0.28)
        assert res["screening_tier"] == "Model Unavailable"
        assert res["error"] == "Model Unavailable"
        assert res["is_model_loaded"] is False
    finally:
        config.NEW_MODEL_PATH = original_path
