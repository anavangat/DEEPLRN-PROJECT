"""Stage 2 training -- one contract-compliant run per invocation.

    python -m src.stage2.train_stage2 --crop-root data/crops/rgb \
        --method twostage --colorspace rgb --seed 0 \
        --best configs/stage2_best.json --runs-root runs

Model selection is on VAL macro-F1. The TEST split is opened exactly once,
at the end, to produce the reported metrics.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.contract import run_dir, write_run
from src.common.seeding import set_all_seeds
from src.data.crop_dataset import class_weights, make_loaders
from src.stage2.detector_meta import detector_metrics
from src.stage2.engine import evaluate, fit, get_device
from src.stage2.model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop-root", required=True, help="data/crops/rgb or data/frames/rgb")
    ap.add_argument("--method", required=True,
                    choices=["twostage", "uncropped", "effnet"])
    ap.add_argument("--arch", default=None, help="cnn|effnet (defaults from --method)")
    ap.add_argument("--colorspace", required=True, choices=["rgb", "hsv", "gray"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--train-mode", default="gt")
    ap.add_argument("--eval-mode", default="pred")
    ap.add_argument("--best", default=None, help="configs/stage2_best.json")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--crops-root", default="data/crops")
    args = ap.parse_args()

    arch = args.arch or ("effnet" if args.method == "effnet" else "cnn")

    # Tuned hyperparameters override the defaults, and the file records which
    # search produced them.
    if args.best and Path(args.best).exists():
        best = json.loads(Path(args.best).read_text())["best_params"]
        args.lr = best.get("lr", args.lr)
        args.dropout = best.get("dropout", args.dropout)
        args.weight_decay = best.get("weight_decay", args.weight_decay)
        print(f"loaded {args.best}: lr={args.lr:.5g} dropout={args.dropout:.3g} "
              f"wd={args.weight_decay:.5g}")

    set_all_seeds(args.seed)
    device = get_device()

    train_dl, val_dl, test_dl, class_names = make_loaders(
        args.crop_root, train_mode=args.train_mode, eval_mode=args.eval_mode,
        batch_size=args.batch, num_workers=args.workers, seed=args.seed,
        augment=not args.no_augment)
    print(f"train={len(train_dl.dataset)} val={len(val_dl.dataset)} "
          f"test={len(test_dl.dataset)}  device={device}")

    model = build_model(arch, num_classes=36, dropout=args.dropout).to(device)

    w = None if args.no_class_weights else class_weights(train_dl.dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5)

    t0 = time.time()
    fit_out = fit(model, train_dl, val_dl, criterion, optimizer, scheduler,
                  epochs=args.epochs, patience=args.patience, device=device)
    train_seconds = time.time() - t0

    # ---- the one and only test pass -------------------------------------
    test = evaluate(model, test_dl, criterion, device)
    paths = [s[0] for s in test_dl.dataset.samples]     # loader is shuffle=False
    predictions = [[p, int(t), int(q), float(c)] for p, t, q, c in
                   zip(paths, test["y_true"], test["y_pred"], test["confidence"])]

    metrics = {
        "method": args.method, "colorspace": args.colorspace, "seed": args.seed,
        "split": "test",
        "accuracy": test["accuracy"],
        "macro_precision": test["macro_precision"],
        "macro_recall": test["macro_recall"],
        "macro_f1": test["macro_f1"],
        "per_class_f1": test["per_class_f1"],
        "train_seconds": train_seconds,
        "n_samples": len(test_dl.dataset),
        "val_macro_f1": fit_out["best_val_macro_f1"],
        "best_epoch": fit_out["best_epoch"],
        "epochs_run": fit_out["epochs_run"],
        "arch": arch, "lr": args.lr, "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "class_weights": not args.no_class_weights,
    }
    if args.method in ("twostage", "effnet"):
        metrics.update(detector_metrics(args.colorspace, "test",
                                        runs_root=args.runs_root,
                                        crops_root=args.crops_root))

    out = run_dir(args.runs_root, args.method, args.colorspace, args.seed)
    write_run(out, vars(args) | {"arch": arch}, metrics, predictions,
              history=fit_out["history"])
    torch.save(model.state_dict(), out / "weights.pt")

    print(f"\nTEST  acc={test['accuracy']:.4f}  macroF1={test['macro_f1']:.4f}  "
          f"({train_seconds/60:.1f} min, best epoch {fit_out['best_epoch']})")
    print(f"run dir: {out}")


if __name__ == "__main__":
    main()