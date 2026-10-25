"""
tests/test_ac_accuracy_eval.py
==============================
Logic tests for fetal/evaluate_ac_accuracy.py using a fake predictor
(no model weights or ACOUSLIC-AI data needed).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fetal.evaluate_ac_accuracy as ev  # noqa: E402


def _csv(tmp_path):
    rows = [
        {"uuid": "c1", "subject_id": "001", "sweep_1_ac_mm": 250.0, "sweep_5_ac_mm": 260.0},
        {"uuid": "c2", "subject_id": "002", "sweep_2_ac_mm": 300.0},
        {"uuid": "c3", "subject_id": "002", "sweep_3_ac_mm": 280.0},
        {"uuid": "c4", "subject_id": "003", "sweep_1_ac_mm": 200.0, "sweep_6_ac_mm": 210.0},
    ]
    df = pd.DataFrame(rows).reindex(columns=["uuid", "subject_id", *ev.SWEEP_COLS])
    p = tmp_path / "circ.csv"
    df.to_csv(p, index=False)
    return p


def test_reference_long_format(tmp_path):
    ref = ev.load_reference(_csv(tmp_path))
    assert len(ref) == 6 and set(ref["sweep"]) == {1, 2, 3, 5, 6}


def test_constant_offset_gives_exact_bias():
    pairs = pd.DataFrame({"uuid": list("abcde"), "subject_id": list("abcde"),
                          "ref_ac_mm": [200, 250, 300, 280, 260.0]})
    pairs["pred_ac_mm"] = pairs["ref_ac_mm"] + 5
    m = ev.compute_metrics(pairs, n_boot=50)
    s = m["per_sweep"]
    assert s["Bias_mm"] == pytest.approx(5) and s["MAE_mm"] == pytest.approx(5)
    assert s["SD_diff_mm"] == pytest.approx(0, abs=1e-9)
    assert s["Within_10mm"] == 1.0


def test_detection_failures_counted_not_scored():
    pairs = pd.DataFrame({"uuid": list("abcd"), "subject_id": list("abcd"),
                          "ref_ac_mm": [200, 250, 300, 280.0], "pred_ac_mm": [205, np.nan, 290, 284.0]})
    m = ev.compute_metrics(pairs, n_boot=50)
    assert m["n_detection_failures"] == 1 and m["n_evaluated_sweeps"] == 3


def test_train_ids_exclude_cases_and_shared_subjects(tmp_path):
    ref = ev.load_reference(_csv(tmp_path))
    sel, mode, warn = ev.select_cases(ref, None, {"c2"})
    assert mode == "complement-of-train-ids"
    assert set(sel["uuid"]) == {"c1", "c4"}          # c3 dropped: same subject as training case c2
    assert any("subjects also in training" in w for w in warn)


def test_end_to_end_with_fake_predictor(tmp_path, monkeypatch):
    csv = _csv(tmp_path)
    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    (data / "masks").mkdir()
    for u in ["c1", "c2", "c3", "c4"]:
        (data / "images" / f"{u}.mha").write_bytes(b"x")
        (data / "masks" / f"{u}.mha").write_bytes(b"x")
    assert "images" in str(ev.find_mha(data, "c1"))

    truth = ev.load_reference(csv).set_index(["uuid", "sweep"])["ref_ac_mm"]

    def fake_make_predictor(weights, device):
        def predict(p):
            u = p.stem
            return {"pixel_spacing_mm": 0.28,
                    "sweep_ac_mm": [truth.get((u, k), np.nan) - 4 if (u, k) in truth else np.nan for k in range(1, 7)]}
        return predict

    monkeypatch.setattr(ev, "make_predictor", fake_make_predictor)
    ids = tmp_path / "test_ids.txt"
    ids.write_text("c1\nc2\nc3\nc4\n")
    out = tmp_path / "out"
    ev.main(["--data-dir", str(data), "--csv", str(csv), "--test-ids", str(ids), "--out", str(out),
             "--split-source", "unit test"])
    txt = (out / "ac_accuracy_report.txt").read_text()
    assert "HELD-OUT EVALUATION" in txt and "Bias (pred - ref): -4.0" in txt
    assert (out / "ac_bland_altman.png").exists() and (out / "ac_predictions_per_sweep.csv").exists()


def test_refuses_without_split(tmp_path):
    with pytest.raises(SystemExit):
        ev.main(["--data-dir", str(tmp_path), "--csv", str(_csv(tmp_path))])
