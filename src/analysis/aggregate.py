"""Load every contract-compliant run into one DataFrame.

    python -m src.analysis.aggregate --runs runs_synth
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.classes import CLASSES
from src.common.contract import COLORSPACES, METHODS, validate_run

RUN_RE = re.compile(r"^(?:" + "|".join(sorted(METHODS)) + r")_"
                    r"(?:" + "|".join(sorted(COLORSPACES)) + r")_seed\d+$")

CLASSIFIERS = ["twostage", "uncropped", "effnet", "yolocls"]
PRETTY = {"twostage": "YOLOv8 + CNN (ours)", "uncropped": "CNN, no localization",
          "effnet": "YOLOv8 + EfficientNet-B0", "yolocls": "YOLOv8 classifier (end-to-end)",
          "stage1": "YOLOv8n detector"}


def load_runs(runs_root, strict=True):
    """One row per run. Raises on any contract violation when strict."""
    root = Path(runs_root)
    rows, problems = [], []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and RUN_RE.match(p.name)):
        probs = validate_run(d)
        if probs:
            problems += [f"{d.name}: {p}" for p in probs]
            if strict:
                continue
        m = json.loads((d / "metrics.json").read_text())
        row = {k: v for k, v in m.items() if k != "per_class_f1"}
        row["run"] = d.name
        row["run_path"] = str(d)
        row["per_class_f1"] = m.get("per_class_f1")
        rows.append(row)

    if problems:
        print(f"!! {len(problems)} contract problem(s):")
        for p in problems:
            print(f"   - {p}")
        if strict:
            raise SystemExit("Fix these with their owner before analysing. A run "
                             "that doesn't validate doesn't exist.")

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No valid runs under {root}")
    df["label"] = df["method"].map(PRETTY).fillna(df["method"])
    return df


def per_class_frame(df):
    """Long-format per-class F1: one row per (run, class)."""
    out = []
    for _, r in df.iterrows():
        pcf = r["per_class_f1"]
        if pcf is None or any(v is None for v in pcf):
            continue                      # stage1 has no per-class F1
        for i, v in enumerate(pcf):
            out.append({"method": r["method"], "colorspace": r["colorspace"],
                        "seed": r["seed"], "class_idx": i,
                        "class_name": CLASSES[i],
                        "kind": "digit" if i < 10 else "letter", "f1": v})
    return pd.DataFrame(out)


def coverage(df):
    """What's landed and what's still missing."""
    have = set(zip(df["method"], df["colorspace"], df["seed"]))
    missing = [f"{m}_{c}_seed{s}" for m in CLASSIFIERS for c in COLORSPACES
               for s in range(5) if (m, c, s) not in have]
    return sorted(have), missing


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--loose", action="store_true", help="don't stop on violations")
    a = ap.parse_args()
    df = load_runs(a.runs, strict=not a.loose)
    print(df.groupby(["method", "colorspace"])["macro_f1"]
            .agg(["count", "mean", "std"]).round(4).to_string())
    _, missing = coverage(df)
    print(f"\nmissing runs ({len(missing)}): {missing[:12]}"
          f"{' ...' if len(missing) > 12 else ''}")