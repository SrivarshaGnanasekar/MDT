"""Run AC inference on a folder of stacked ultrasound cases (.mha) and write a CSV.

    python run.py --input path/to/images --weights best_attunet.pth --out ac_results.csv

--input can be a folder (searched recursively) or a single .mha file. Files whose path
contains "mask" are skipped, the same rule the training notebook used to pair files.
"""
import argparse
from pathlib import Path

import pandas as pd

from inference import case_acs, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--weights", default="best_attunet.pth")
    ap.add_argument("--out", default="ac_results.csv")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    a = ap.parse_args()

    p = Path(a.input)
    files = [p] if p.is_file() else sorted(p.rglob("*.mha"))
    files = [f for f in files if "mask" not in str(f).lower()]
    if not files:
        raise SystemExit(f"No image .mha files found under {p}")

    model, device = load_model(a.weights, a.device)
    print(f"device: {device} | cases: {len(files)}", flush=True)

    rows = []
    for n, f in enumerate(files, 1):
        try:
            acs = case_acs(model, device, f)
        except Exception as e:                       # keep going if one file is bad
            print(f"[{n}/{len(files)}] {f.name}: SKIPPED ({e})", flush=True)
            continue
        rows.append(dict(case=f.stem, **{f"sweep_{k + 1}_ac_mm": round(v, 2) for k, v in enumerate(acs)}))
        print(f"[{n}/{len(files)}] {f.name}: " + ", ".join("nan" if v != v else f"{v:.1f}" for v in acs), flush=True)

    pd.DataFrame(rows).to_csv(a.out, index=False)
    print("saved", a.out)


if __name__ == "__main__":
    main()
