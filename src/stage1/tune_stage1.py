"""Optuna search for the Stage 1 learning rate.

SGD, momentum 0.937, cosine annealing, 15 trials over the initial LR.
Trials are short (default 10 epochs)

IMPORTANT: the sampler is seeded. Without seeding the TPESampler the search is
not reproducible even if torch/numpy are seeded.

NOTE: this optimises against the VALIDATION split. That is exactly why
the Kaggle test split is held out and touched only once, in eval_stage1.py.

    python -m src.stage1.tune_stage1 --data data/prepared/rgb/data1.yaml --trials 15
"""
import argparse
import json
import sys
from pathlib import Path

import optuna
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.seeding import make_sampler, set_all_seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="configs/stage1_best.json")
    args = ap.parse_args()

    set_all_seeds(args.seed)
    Path("runs/stage1_tune").mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    def objective(trial):
        lr0 = trial.suggest_float("lr0", 1e-4, 5e-2, log=True)
        model = YOLO(args.model)
        try:
            res = model.train(
                data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                optimizer="SGD", lr0=lr0, momentum=0.937, cos_lr=True,
                device=0, workers=8, seed=args.seed, deterministic=True,
                project="runs/stage1_tune", name=f"trial{trial.number:02d}",
                exist_ok=True, verbose=False, plots=False, val=True,
            )
            score = float(res.box.map)  # mAP@50-95 on val
        finally:
            del model
            torch.cuda.empty_cache()
        print(f"trial {trial.number:02d}  lr0={lr0:.5f}  val mAP50-95={score:.4f}", flush=True)
        return score

    study = optuna.create_study(
        direction="maximize", sampler=make_sampler(args.seed),
        study_name="stage1_lr", storage=f"sqlite:///runs/stage1_tune/study.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.trials)

    best = {
        "best_params": study.best_params,
        "best_value_val_map50_95": study.best_value,
        "n_trials": len(study.trials),
        "tune_epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "sampler_seed": args.seed,
        "fixed": {"optimizer": "SGD", "momentum": 0.937, "cos_lr": True},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(best, indent=2))
    print("\nBEST:", json.dumps(best["best_params"]), f"-> {args.out}")


if __name__ == "__main__":
    main()
