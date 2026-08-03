"""Final Stage 1 training, one run per seed.

Uses the LR chosen by tune_stage1.py, trains to convergence, then evaluates on
VAL only. The test split stays sealed until eval_stage1.py.

    python -m src.stage1.train_stage1 --data data/prepared/rgb/data1.yaml \
        --colorspace rgb --seed 0 --best configs/stage1_best.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.contract import run_dir, write_run
from src.common.seeding import set_all_seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--colorspace", required=True, choices=["rgb", "hsv", "gray"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--best", default="configs/stage1_best.json")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--runs-root", default="runs")
    args = ap.parse_args()

    best = json.loads(Path(args.best).read_text())
    lr0 = best["best_params"]["lr0"]
    set_all_seeds(args.seed)

    from ultralytics import YOLO

    name = f"stage1_{args.colorspace}_seed{args.seed}"
    t0 = time.time()
    model = YOLO(args.model)
    res = model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        optimizer="SGD", lr0=lr0, momentum=0.937, cos_lr=True, patience=args.patience,
        device=0, workers=8, seed=args.seed, deterministic=True,
        project=f"{args.runs_root}/_ultralytics", name=name, exist_ok=True,
        verbose=True, plots=True, val=True,
    )
    train_seconds = time.time() - t0
    ul_dir = Path(res.save_dir)

    metrics = {
        "method": "stage1", "colorspace": args.colorspace, "seed": args.seed,
        "split": "val",
        "map50": float(res.box.map50), "map50_95": float(res.box.map),
        "mean_iou": None,             # filled by export_crops / eval_stage1
        "detection_miss_rate": None,  # filled by export_crops / eval_stage1
        "precision": float(res.box.mp), "recall": float(res.box.mr),
        # Stage 1 is single-class; the classification keys are placeholders that
        # keep the artifact contract uniform for Person 4's loader.
        "accuracy": None, "macro_precision": None, "macro_recall": None,
        "macro_f1": None, "per_class_f1": [None] * 36,
        "train_seconds": train_seconds, "n_samples": None,
        "lr0": lr0, "epochs_requested": args.epochs, "imgsz": args.imgsz,
    }

    out = run_dir(args.runs_root, "stage1", args.colorspace, args.seed)
    history = []
    csv_path = ul_dir / "results.csv"
    if csv_path.exists():
        import csv as _csv
        with csv_path.open() as fh:
            for i, row in enumerate(_csv.DictReader(fh), start=1):
                row = {k.strip(): v for k, v in row.items()}
                history.append([
                    i, row.get("train/box_loss"), row.get("val/box_loss"),
                    row.get("metrics/precision(B)"), row.get("metrics/mAP50-95(B)"),
                ])

    write_run(out, vars(args) | {"lr0": lr0}, metrics, predictions=[],
              history=history, weights_src=ul_dir / "weights" / "best.pt")
    print(f"\nval mAP50-95={res.box.map:.4f}  mAP50={res.box.map50:.4f}  "
          f"{train_seconds/60:.1f} min")
    print(f"run dir : {out}")
    print(f"weights : {out/'weights.pt'}")


if __name__ == "__main__":
    main()
