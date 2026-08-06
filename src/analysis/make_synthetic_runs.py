"""Synthetic run directories, so the analysis pipeline can be built before any
real results exist. Structurally identical to the real thing; the numbers are
plausible fiction.

    python -m src.analysis.make_synthetic_runs --out runs_synth
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.contract import run_dir, write_run

# Rough expected ordering, so plots look like the real thing while you build.
PROFILE = {
    "twostage":  {"base": 0.94, "digit_penalty": 0.03},
    "effnet":    {"base": 0.93, "digit_penalty": 0.03},
    "uncropped": {"base": 0.81, "digit_penalty": 0.06},
    "yolocls":   {"base": 0.88, "digit_penalty": 0.10},   # no class weighting
}
COLOR_DROP = {"rgb": 0.0, "hsv": 0.18, "gray": 0.07}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs_synth")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-test", type=int, default=1363)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    for method, prof in PROFILE.items():
        for cs, drop in COLOR_DROP.items():
            for seed in range(args.seeds):
                f1 = np.clip(rng.normal(prof["base"] - drop, 0.012, 36), 0, 1)
                f1[:10] -= prof["digit_penalty"]           # digits are harder
                f1 = np.clip(f1, 0, 1)
                y_true = rng.integers(0, 36, args.n_test)
                y_pred = np.where(rng.random(args.n_test) < f1.mean(),
                                  y_true, rng.integers(0, 36, args.n_test))
                metrics = {
                    "method": method, "colorspace": cs, "seed": seed,
                    "split": "test",
                    "accuracy": float((y_true == y_pred).mean()),
                    "macro_precision": float(f1.mean() + rng.normal(0, 0.004)),
                    "macro_recall": float(f1.mean() + rng.normal(0, 0.004)),
                    "macro_f1": float(f1.mean()),
                    "per_class_f1": [float(v) for v in f1],
                    "train_seconds": None if cs != "rgb" else float(rng.normal(1500, 200)),
                    "n_samples": args.n_test,
                    "arch": "cnn" if method != "effnet" else "effnet",
                }
                if method in ("twostage", "effnet"):
                    metrics |= {"map50": 0.987 if cs == "rgb" else None,
                                "map50_95": 0.840 if cs == "rgb" else None,
                                "mean_iou": 0.909, "detection_miss_rate": 0.0017}
                preds = [[f"x/{i}.jpg", int(t), int(p), float(rng.uniform(0.4, 1.0))]
                         for i, (t, p) in enumerate(zip(y_true, y_pred))]
                out = run_dir(args.out, method, cs, seed)
                write_run(out, {"synthetic": True}, metrics, preds,
                          history=[[e, 1.0/e, 1.1/e, 0.5, 0.5] for e in range(1, 21)])
    print(f"wrote synthetic runs -> {args.out}")


if __name__ == "__main__":
    main()