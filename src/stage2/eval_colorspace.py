"""Colour-dependency ablation: RGB-trained weights, evaluated on HSV/grayscale.

Produces contract-compliant transfer runs, e.g. twostage_hsv_seed0, so that
the aggregator picks them up with no special-casing.

    python -m src.stage2.eval_colorspace --run runs/twostage_rgb_seed0 \
        --colorspaces hsv gray --crops-root data/crops
"""
#Script to test the model for color dependency
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.contract import run_dir, write_run
from src.common.seeding import set_all_seeds
from src.data.crop_dataset import make_loaders
from src.stage2.detector_meta import detector_metrics
from src.stage2.engine import evaluate, get_device
from src.stage2.model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a trained *_rgb_seed* run dir")
    ap.add_argument("--colorspaces", nargs="+", default=["hsv", "gray"])
    ap.add_argument("--crops-root", default="data/crops")
    ap.add_argument("--eval-mode", default="pred")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--runs-root", default=None)
    args = ap.parse_args()

    run = Path(args.run)
    src_metrics = json.loads((run / "metrics.json").read_text())
    method, seed = src_metrics["method"], int(src_metrics["seed"])
    arch = src_metrics.get("arch", "cnn")
    runs_root = args.runs_root or str(run.parent)

    set_all_seeds(seed)
    device = get_device()
    model = build_model(arch, dropout=src_metrics.get("dropout", 0.5)).to(device)
    model.load_state_dict(torch.load(run / "weights.pt", map_location=device))
    criterion = nn.CrossEntropyLoss()

    for cs in args.colorspaces:
        _, _, test_dl, _ = make_loaders(
            f"{args.crops_root}/{cs}", eval_mode=args.eval_mode,
            batch_size=args.batch, num_workers=args.workers, seed=seed)
        r = evaluate(model, test_dl, criterion, device)
        paths = [s[0] for s in test_dl.dataset.samples]
        preds = [[p, int(t), int(q), float(c)] for p, t, q, c in
                 zip(paths, r["y_true"], r["y_pred"], r["confidence"])]

        metrics = {
            "method": method, "colorspace": cs, "seed": seed, "split": "test",
            "accuracy": r["accuracy"], "macro_precision": r["macro_precision"],
            "macro_recall": r["macro_recall"], "macro_f1": r["macro_f1"],
            "per_class_f1": r["per_class_f1"],
            # No training happened: these weights were trained on RGB.
            "train_seconds": None, "n_samples": len(test_dl.dataset),
            "arch": arch, "transfer_from": run.name, "trained_on": "rgb",
        }
        if method in ("twostage", "effnet"):
            metrics.update(detector_metrics(cs, "test", runs_root=runs_root,
                                            crops_root=args.crops_root))

        out = run_dir(runs_root, method, cs, seed)
        write_run(out, {"transfer_from": str(run), "colorspace": cs,
                        "eval_mode": args.eval_mode, "arch": arch},
                  metrics, preds, history=[])
        print(f"{cs}: acc={r['accuracy']:.4f} macroF1={r['macro_f1']:.4f} -> {out}")


if __name__ == "__main__":
    main()
