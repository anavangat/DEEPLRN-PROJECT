"""Optuna search for the Stage 2 classifier.

Objective = VALIDATION macro-F1. The test split is never touched here.

The sampler is seeded. Without seeding TPESampler the search is not
reproducible even with torch and numpy seeded, and the 
reproducibility statement would be false.

    python -m src.stage2.tune_stage2 --crop-root data/crops/rgb --trials 20
"""
import argparse
import json
import sys
from pathlib import Path

import optuna
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.seeding import make_sampler, set_all_seeds
from src.data.crop_dataset import class_weights, make_loaders
from src.stage2.engine import fit, get_device
from src.stage2.model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop-root", required=True)
    ap.add_argument("--arch", default="cnn")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr-min", type=float, default=1e-4)
    ap.add_argument("--lr-max", type=float, default=1e-2)
    ap.add_argument("--study-dir", default="runs/stage2_tune")
    ap.add_argument("--out", default="configs/stage2_best.json")
    args = ap.parse_args()

    set_all_seeds(args.seed)
    Path(args.study_dir).mkdir(parents=True, exist_ok=True)
    device = get_device()

    train_dl, val_dl, _, _ = make_loaders(
        args.crop_root, batch_size=args.batch, num_workers=args.workers,
        seed=args.seed)
    w = class_weights(train_dl.dataset).to(device)

    def objective(trial):
        lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
        dropout = trial.suggest_float("dropout", 0.2, 0.6)
        wd = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

        set_all_seeds(args.seed)          # same init for every trial
        model = build_model(args.arch, dropout=dropout).to(device)
        criterion = nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=3)
        try:
            out = fit(model, train_dl, val_dl, criterion, opt, sched,
                      epochs=args.epochs, patience=8, device=device,
                      trial=trial, verbose=False)
            score = out["best_val_macro_f1"]
        finally:
            del model
            torch.cuda.empty_cache()
        print(f"trial {trial.number:02d}  lr={lr:.5g} do={dropout:.3f} "
              f"wd={wd:.5g}  val macroF1={score:.4f}", flush=True)
        return score

    study = optuna.create_study(
        direction="maximize", sampler=make_sampler(args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        study_name=f"stage2_{args.arch}",
        storage=f"sqlite:///{args.study_dir}/study.db", load_if_exists=True)
    study.optimize(objective, n_trials=args.trials)

    best = {
        "best_params": study.best_params,
        "best_value_val_macro_f1": study.best_value,
        "n_trials": len(study.trials),
        "arch": args.arch, "tune_epochs": args.epochs, "batch": args.batch,
        "sampler_seed": args.seed,
        "search_space": {"lr": [args.lr_min, args.lr_max],
                         "dropout": [0.2, 0.6],
                         "weight_decay": [1e-5, 1e-2]},
        "fixed": {"optimizer": "AdamW", "scheduler": "ReduceLROnPlateau",
                  "selection": "val macro-F1", "class_weights": True},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(best, indent=2))

    lr = study.best_params["lr"]
    if lr < args.lr_min * 1.5 or lr > args.lr_max / 1.5:
        print(f"\n!! Best lr={lr:.5g} sits at the edge of "
              f"[{args.lr_min}, {args.lr_max}]. The optimum is probably outside "
              f"it -- widen and rerun before trusting this.")
    print("\nBEST:", json.dumps(study.best_params), f"-> {args.out}")


if __name__ == "__main__":
    main()