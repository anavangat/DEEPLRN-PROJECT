"""Baseline 3: end-to-end YOLOv8 classifier on raw frames.

Ultralytics owns the training loop here, so this differs from the other
baselines in optimiser and schedule. That is unavoidable and must be declared
in the paper -- see the note in the guide.

    python -m src.baselines.train_yolocls --data data/frames/rgb/full \
        --colorspace rgb --seed 0 --epochs 60 --runs-root runs
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.classes import FOLDER_NAMES
from src.common.contract import run_dir, write_run
from src.common.seeding import set_all_seeds

LABELS = list(range(36))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data/frames/<cs>/full")
    ap.add_argument("--colorspace", required=True, choices=["rgb", "hsv", "gray"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model", default="yolov8n-cls.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lr0", type=float, default=1e-3)
    ap.add_argument("--runs-root", default="runs")
    args = ap.parse_args()

    set_all_seeds(args.seed)
    from ultralytics import YOLO

    data = Path(args.data).resolve()
    name = f"yolocls_{args.colorspace}_seed{args.seed}"

    t0 = time.time()
    model = YOLO(args.model)
    model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, optimizer="AdamW", lr0=args.lr0,
                patience=args.patience, device=0, workers=2, seed=args.seed,
                deterministic=True, project=f"{args.runs_root}/_ultralytics",
                name=name, exist_ok=True, verbose=True, plots=False, val=True)
    train_seconds = time.time() - t0

    assert [model.names[i] for i in range(36)] == FOLDER_NAMES, \
        "Ultralytics class order does not match the canonical ordering"

    # --- test pass: one prediction per test image ------------------------
    test_dir = data / "test"
    paths, y_true = [], []
    for ci, folder in enumerate(FOLDER_NAMES):
        for p in sorted((test_dir / folder).glob("*.jpg")):
            paths.append(str(p)); y_true.append(ci)

    y_pred, conf = [], []
    for i in range(0, len(paths), 128):
        chunk = paths[i:i + 128]
        for res in model.predict(chunk, imgsz=args.imgsz, device=0, verbose=False):
            probs = res.probs
            y_pred.append(int(probs.top1)); conf.append(float(probs.top1conf))

    y_true = np.array(y_true); y_pred = np.array(y_pred)
    metrics = {
        "method": "yolocls", "colorspace": args.colorspace, "seed": args.seed,
        "split": "test",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro",
                                                 labels=LABELS, zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro",
                                           labels=LABELS, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   labels=LABELS, zero_division=0)),
        "per_class_f1": [float(v) for v in f1_score(y_true, y_pred, average=None,
                                                    labels=LABELS, zero_division=0)],
        "train_seconds": train_seconds, "n_samples": len(paths),
        "arch": args.model, "lr0": args.lr0, "imgsz": args.imgsz,
        "class_weights": False,   # Ultralytics cls has no class-weight hook
    }

    out = run_dir(args.runs_root, "yolocls", args.colorspace, args.seed)
    write_run(out, vars(args), metrics,
              predictions=[[p, int(t), int(q), float(c)]
                           for p, t, q, c in zip(paths, y_true, y_pred, conf)],
              history=[])
    print(f"\nTEST  acc={metrics['accuracy']:.4f}  "
          f"macroF1={metrics['macro_f1']:.4f}  ({train_seconds/60:.1f} min)")
    print(f"run dir: {out}")


if __name__ == "__main__":
    main()
