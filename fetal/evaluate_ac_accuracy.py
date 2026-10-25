"""
fetal/evaluate_ac_accuracy.py
=============================
Reproducible AC-accuracy evaluation of the deployed ACOUSLIC-AI Attention U-Net
(models/new_model/best_attunet.pth) on HELD-OUT ACOUSLIC-AI cases.

What it measures
----------------
For every held-out case (one stacked .mha = 6 sweeps x 140 frames) the deployed
pipeline from models/new_model/inference.py is run unchanged:
  segment every frame -> keep largest component (>= 2,000 px) -> pick the 5 most
  confident frames per sweep -> fit ellipse at original resolution -> Ramanujan
  perimeter x pixel spacing -> median AC (mm) per sweep.
The pixel spacing is read from each .mha header (not hard-coded).

Each predicted sweep AC is compared with the expert reference AC for that sweep
from the ACOUSLIC-AI circumference file (columns: uuid, subject_id,
sweep_1_ac_mm ... sweep_6_ac_mm; blank = no reference for that sweep).

Reported (per sweep with a reference; secondary: per case)
  * detection failures (reference exists but the model found no abdomen)
  * MAE, median absolute error, RMSE (mm)
  * Bland-Altman: bias (pred - ref), SD of differences, 95% limits of agreement,
    proportional bias (slope of difference on mean)
  * normalised absolute error |ref - pred| / max(ref, pred) (challenge-style NAE)
  * % of sweeps within +/-8.8% of the reference (interobserver AC limits of
    agreement, Sarris et al., UOG 2012;39:266-273) and within +/-10 mm
  95% CIs by cluster bootstrap over subjects (sweeps of one woman are correlated).

Held-out guarantee
------------------
The script will NOT label results as held-out unless you supply the split used to
train best_attunet.pth:
  --test-ids  file with the held-out case uuids (one per line), or
  --train-ids file with the uuids used for training/validation (all other cases in
              the circumference file are then evaluated).
Any subject that appears in both training and test lists is reported and dropped.
Without either file, use --unverified-split; the report is then titled
"NOT CONFIRMED HELD-OUT" so it cannot be mistaken for test-set evidence.

Usage
-----
python fetal/evaluate_ac_accuracy.py \
    --data-dir  /path/to/acouslic-ai-train-set \
    --csv       /path/to/circumferences/fetal_abdominal_circumferences_per_sweep.csv \
    --test-ids  splits/acouslic_test_uuids.txt \
    --split-source "aa-unet-training notebook, seed 42, 15% subjects"
Re-running resumes from the cached per-sweep predictions in the output folder.
GPU strongly recommended (each case is 840 frames, ~350 MB).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SWEEP_COLS = [f"sweep_{k}_ac_mm" for k in range(1, 7)]
INTEROBSERVER_PCT = 8.8   # Sarris et al. 2012, interobserver AC 95% LoA (+/- %)
ABS_TOL_MM = 10.0
N_BOOT = 2000


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_reference(csv_path: Path) -> pd.DataFrame:
    """Long format: one row per (uuid, sweep) that has an expert reference AC."""
    df = pd.read_csv(csv_path, dtype={"uuid": str, "subject_id": str})
    missing = {"uuid", "subject_id", *SWEEP_COLS} - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns {sorted(missing)}; found {list(df.columns)}")
    long = df.melt(id_vars=["uuid", "subject_id"], value_vars=SWEEP_COLS, var_name="sweep", value_name="ref_ac_mm")
    long["sweep"] = long["sweep"].str.extract(r"sweep_(\d)_", expand=False).astype(int)
    return long.dropna(subset=["ref_ac_mm"]).reset_index(drop=True)


def read_ids(path: Optional[Path]) -> Optional[set]:
    if path is None:
        return None
    return {ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip() and not ln.startswith("#")}


def select_cases(ref: pd.DataFrame, test_ids, train_ids):
    """Return (reference rows to evaluate, split description, warnings)."""
    warnings: List[str] = []
    all_cases = set(ref["uuid"])
    subj = ref.drop_duplicates("uuid").set_index("uuid")["subject_id"]
    if test_ids is not None:
        cases = all_cases & test_ids
        unknown = test_ids - all_cases
        if unknown:
            warnings.append(f"{len(unknown)} test uuids have no reference AC in the CSV and were skipped.")
        mode = "test-ids"
    elif train_ids is not None:
        cases = all_cases - train_ids
        mode = "complement-of-train-ids"
    else:
        cases, mode = all_cases, "UNVERIFIED (no split supplied)"
    if train_ids is not None:
        train_subjects = set(subj.reindex(list(train_ids & all_cases)).dropna())
        leaked = {c for c in cases if subj.get(c) in train_subjects}
        if leaked:
            warnings.append(f"{len(leaked)} evaluation cases belong to subjects also in training; dropped.")
            cases -= leaked
    return ref[ref["uuid"].isin(cases)].reset_index(drop=True), mode, warnings


def find_mha(data_dir: Path, uuid: str) -> Optional[Path]:
    """Image volume <uuid>.mha, ignoring files under mask/label folders (same file names)."""
    for p in data_dir.rglob(f"{uuid}.mha"):
        if not any(k in str(p).lower() for k in ("mask", "label", "annotation")):
            return p
    return None


# ---------------------------------------------------------------------------
# Prediction (deployed pipeline, unchanged)
# ---------------------------------------------------------------------------

def make_predictor(weights: Path, device: Optional[str]) -> Callable[[Path], Dict]:
    import SimpleITK as sitk
    from models.new_model import inference as inf

    model, dev = inf.load_model(str(weights), device=device)

    def predict_case(mha_path: Path) -> Dict:
        img = sitk.ReadImage(str(mha_path))
        px = float(img.GetSpacing()[0])
        vol = inf.to_uint8(sitk.GetArrayFromImage(img))
        n = inf.N_SWEEPS * inf.FRAMES_PER_SWEEP
        if vol.ndim != 3 or vol.shape[0] != n:
            raise ValueError(f"{mha_path.name}: expected {n} frames, got {vol.shape}")
        acs = [inf.sweep_ac(model, dev, vol[k * inf.FRAMES_PER_SWEEP:(k + 1) * inf.FRAMES_PER_SWEEP], px=px)
               for k in range(inf.N_SWEEPS)]
        return {"pixel_spacing_mm": px, "sweep_ac_mm": acs}

    return predict_case


def run_predictions(ref: pd.DataFrame, data_dir: Path, predict_case, cache: Path) -> pd.DataFrame:
    done = pd.read_csv(cache, dtype={"uuid": str, "subject_id": str}) if cache.exists() else pd.DataFrame()
    done_cases = set(done["uuid"]) if len(done) else set()
    rows = []
    cases = sorted(set(ref["uuid"]) - done_cases)
    for i, uuid in enumerate(cases, 1):
        p = find_mha(data_dir, uuid)
        sub = ref[ref["uuid"] == uuid]
        if p is None:
            print(f"[{i}/{len(cases)}] {uuid}: image .mha not found - skipped")
            continue
        t0 = time.time()
        res = predict_case(p)
        for _, r in sub.iterrows():
            rows.append({"uuid": uuid, "subject_id": r["subject_id"], "sweep": int(r["sweep"]),
                         "ref_ac_mm": float(r["ref_ac_mm"]),
                         "pred_ac_mm": float(res["sweep_ac_mm"][int(r["sweep"]) - 1]),
                         "pixel_spacing_mm": res["pixel_spacing_mm"]})
        print(f"[{i}/{len(cases)}] {uuid}: {len(sub)} reference sweep(s), {time.time() - t0:.0f}s")
        pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_csv(cache, index=False)
    out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    return out[out["uuid"].isin(set(ref["uuid"]))].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _stats(ref: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    d = pred - ref
    sd = float(np.std(d, ddof=1)) if len(d) > 1 else float("nan")
    mean = (pred + ref) / 2
    slope = float(np.polyfit(mean, d, 1)[0]) if len(d) > 2 else float("nan")
    return {
        "MAE_mm": float(np.mean(np.abs(d))), "MedianAE_mm": float(np.median(np.abs(d))),
        "RMSE_mm": float(np.sqrt(np.mean(d ** 2))), "Bias_mm": float(np.mean(d)), "SD_diff_mm": sd,
        "LoA_lower_mm": float(np.mean(d) - 1.96 * sd), "LoA_upper_mm": float(np.mean(d) + 1.96 * sd),
        "Proportional_bias_slope": slope,
        "NAE_mean": float(np.mean(np.abs(d) / np.maximum(np.maximum(ref, pred), 1e-6))),
        "Within_8.8pct": float(np.mean(np.abs(d) / ref * 100 <= INTEROBSERVER_PCT)),
        "Within_10mm": float(np.mean(np.abs(d) <= ABS_TOL_MM)),
    }


def compute_metrics(pairs: pd.DataFrame, n_boot: int = N_BOOT, seed: int = 42) -> Dict:
    detected = pairs.dropna(subset=["pred_ac_mm"])
    out = {
        "n_reference_sweeps": int(len(pairs)), "n_cases": int(pairs["uuid"].nunique()),
        "n_subjects": int(pairs["subject_id"].nunique()),
        "n_detection_failures": int(pairs["pred_ac_mm"].isna().sum()),
        "detection_failure_rate": float(pairs["pred_ac_mm"].isna().mean()) if len(pairs) else float("nan"),
        "n_evaluated_sweeps": int(len(detected)),
    }
    if len(detected) < 3:
        out["note"] = "Fewer than 3 evaluable sweeps; agreement statistics not computed."
        return out
    r, p = detected["ref_ac_mm"].to_numpy(), detected["pred_ac_mm"].to_numpy()
    out["per_sweep"] = _stats(r, p)

    # cluster bootstrap over subjects
    rng = np.random.default_rng(seed)
    groups = {s: g.index.to_numpy() for s, g in detected.reset_index(drop=True).groupby("subject_id")}
    keys = list(groups)
    det = detected.reset_index(drop=True)
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys), replace=True)])
        if len(idx) > 2:
            boots.append(_stats(det.loc[idx, "ref_ac_mm"].to_numpy(), det.loc[idx, "pred_ac_mm"].to_numpy()))
    bdf = pd.DataFrame(boots)
    out["per_sweep_95CI"] = {c: [float(bdf[c].quantile(.025)), float(bdf[c].quantile(.975))] for c in bdf.columns}

    case = det.groupby(["uuid", "subject_id"])[["ref_ac_mm", "pred_ac_mm"]].mean().reset_index()
    if len(case) >= 3:
        out["per_case"] = _stats(case["ref_ac_mm"].to_numpy(), case["pred_ac_mm"].to_numpy())
        out["per_case"]["n_cases"] = int(len(case))
    return out


def bland_altman_plot(pairs: pd.DataFrame, m: Dict, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pairs.dropna(subset=["pred_ac_mm"])
    r, p = d["ref_ac_mm"].to_numpy(), d["pred_ac_mm"].to_numpy()
    s = m["per_sweep"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    lim = (min(r.min(), p.min()) - 10, max(r.max(), p.max()) + 10)
    ax[0].scatter(r, p, s=14, alpha=.6, color="#2563EB")
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set(xlim=lim, ylim=lim, xlabel="Reference AC (mm)", ylabel="Predicted AC (mm)",
              title=f"Agreement (n={len(d)} sweeps, MAE {s['MAE_mm']:.1f} mm)")
    mean, diff = (r + p) / 2, p - r
    ax[1].scatter(mean, diff, s=14, alpha=.6, color="#2563EB")
    for y, lab, ls in [(s["Bias_mm"], f"Bias {s['Bias_mm']:+.1f}", "-"),
                       (s["LoA_upper_mm"], f"+1.96 SD {s['LoA_upper_mm']:+.1f}", "--"),
                       (s["LoA_lower_mm"], f"-1.96 SD {s['LoA_lower_mm']:+.1f}", "--")]:
        ax[1].axhline(y, color="#DC2626", ls=ls, lw=1.2)
        ax[1].text(mean.max(), y, f" {lab} mm", va="bottom", ha="right", fontsize=9, color="#DC2626")
    ax[1].axhline(0, color="k", lw=.8)
    ax[1].set(xlabel="Mean of predicted and reference AC (mm)", ylabel="Predicted - reference (mm)",
              title="Bland-Altman")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_report(m: Dict, meta: Dict, out_dir: Path) -> str:
    held = meta["split_mode"] in ("test-ids", "complement-of-train-ids")
    title = ("ACOUSLIC-AI AC ACCURACY - HELD-OUT EVALUATION" if held
             else "ACOUSLIC-AI AC ACCURACY - NOT CONFIRMED HELD-OUT (no training split supplied)")
    if held and meta.get("used_for_selection"):
        title = ("ACOUSLIC-AI AC ACCURACY - VALIDATION CASES (not used for training, but used for "
                 "checkpoint/pipeline selection: optimistic, not an independent test)")
    lines = [f"=== {title} ===",
             f"Generated {meta['generated_utc']} | weights {meta['weights']} | split: {meta['split_mode']}"
             + (f" ({meta['split_source']})" if meta.get("split_source") else ""),
             f"Reference sweeps {m['n_reference_sweeps']} in {m['n_cases']} cases / {m['n_subjects']} subjects; "
             f"detection failures {m['n_detection_failures']} ({m['detection_failure_rate']*100:.1f}%); "
             f"evaluated {m['n_evaluated_sweeps']}"]
    lines += [f"WARNING: {w}" for w in meta.get("warnings", [])]
    if "per_sweep" in m:
        s, ci = m["per_sweep"], m["per_sweep_95CI"]
        f = lambda k, fmt="{:.1f}": f"{fmt.format(s[k])} ({fmt.format(ci[k][0])} to {fmt.format(ci[k][1])})"
        lines += ["", "Per sweep (95% CI, cluster bootstrap over subjects):",
                  f"- MAE: {f('MAE_mm')} mm | median AE {s['MedianAE_mm']:.1f} mm | RMSE {f('RMSE_mm')} mm",
                  f"- Bias (pred - ref): {f('Bias_mm', '{:+.1f}')} mm | SD of differences {s['SD_diff_mm']:.1f} mm",
                  f"- 95% limits of agreement: {s['LoA_lower_mm']:+.1f} to {s['LoA_upper_mm']:+.1f} mm",
                  f"- Proportional bias (slope of difference on mean): {s['Proportional_bias_slope']:+.3f}",
                  f"- NAE (|ref - pred| / max): {f('NAE_mean', '{:.3f}')}",
                  f"- Within +/-8.8% of reference (interobserver LoA, Sarris 2012): {s['Within_8.8pct']*100:.1f}%",
                  f"- Within +/-10 mm: {s['Within_10mm']*100:.1f}%"]
        if "per_case" in m:
            c = m["per_case"]
            lines += ["", f"Per case (mean of sweeps, n={c['n_cases']}): MAE {c['MAE_mm']:.1f} mm, "
                          f"bias {c['Bias_mm']:+.1f} mm, LoA {c['LoA_lower_mm']:+.1f} to {c['LoA_upper_mm']:+.1f} mm"]
    else:
        lines += ["", m.get("note", "")]
    lines += ["", "Notes",
              "- Pipeline = models/new_model/inference.py (top-5 confident frames per sweep, median AC); "
              "pixel spacing read from each .mha header.",
              "- Failures are reported separately and excluded from the error statistics.",
              "- Reference = expert per-sweep AC from the ACOUSLIC-AI circumference file."]
    txt = "\n".join(lines) + "\n"
    (out_dir / "ac_accuracy_report.txt").write_text(txt)
    (out_dir / "ac_accuracy_report.json").write_text(json.dumps({"meta": meta, "metrics": m}, indent=2))
    return txt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--test-ids", type=Path)
    g.add_argument("--train-ids", type=Path)
    ap.add_argument("--unverified-split", action="store_true")
    ap.add_argument("--split-source", default="", help="Where the split came from (recorded in the report).")
    ap.add_argument("--used-for-selection", action="store_true",
                    help="The evaluated cases were used to pick the checkpoint, frame rule or thresholds.")
    ap.add_argument("--weights", type=Path, default=ROOT / "models" / "new_model" / "best_attunet.pth")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "validated" / "ac_accuracy")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Evaluate only the first N cases (smoke test).")
    a = ap.parse_args(argv)

    if a.test_ids is None and a.train_ids is None and not a.unverified_split:
        ap.error("Supply --test-ids or --train-ids (the split used to train best_attunet.pth), "
                 "or --unverified-split to run anyway with a 'NOT CONFIRMED HELD-OUT' report.")
    a.out.mkdir(parents=True, exist_ok=True)
    ref = load_reference(a.csv)
    ref, mode, warnings = select_cases(ref, read_ids(a.test_ids), read_ids(a.train_ids))
    if a.limit:
        keep = sorted(ref["uuid"].unique())[: a.limit]
        ref = ref[ref["uuid"].isin(keep)]
    print(f"Evaluating {ref['uuid'].nunique()} cases ({len(ref)} reference sweeps); split mode: {mode}")

    pairs = run_predictions(ref, a.data_dir, make_predictor(a.weights, a.device), a.out / "ac_predictions_per_sweep.csv")
    m = compute_metrics(pairs)
    meta = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "weights": str(a.weights.relative_to(ROOT)) if a.weights.is_relative_to(ROOT) else str(a.weights),
            "split_mode": mode, "split_source": a.split_source, "warnings": warnings,
            "used_for_selection": a.used_for_selection,
            "csv": a.csv.name, "limit": a.limit}
    if "per_sweep" in m:
        bland_altman_plot(pairs, m, a.out / "ac_bland_altman.png",
                          ("Validation cases (used for model selection)" if a.used_for_selection else "Held-out ACOUSLIC-AI cases")
                          if mode != "UNVERIFIED (no split supplied)" else "NOT confirmed held-out")
    print(write_report(m, meta, a.out))


if __name__ == "__main__":
    main()
