"""The run-artifact contract.

EVERY experiment in this project -- Stage 1, Stage 2, and all baselines
-- must produce a directory shaped exactly like this:

    runs/{method}_{colorspace}_seed{n}/
        config.yaml       exact hyperparameters + git commit + env
        metrics.json      headline scalars (schema below)
        predictions.csv   img_path,y_true,y_pred,confidence
        history.csv       epoch,train_loss,val_loss,train_acc,val_acc
        weights.pt        final checkpoint

Person 4 builds every table and figure in the paper from metrics.json and
predictions.csv alone. If validate_run() passes, their code will work.
"""
import csv
import json
import platform
import subprocess
import sys
from pathlib import Path

METHODS = {
    "twostage",      # proposed: YOLOv8n crop -> custom CNN
    "uncropped",     # baseline 1: custom CNN on full 128x128 frames
    "effnet",        # baseline 2: YOLOv8n crop -> EfficientNet-B0
    "yolocls",       # baseline 3: end-to-end YOLOv8 classifier
    "stage1",        # detector-only run (mAP/IoU, no per-class F1)
}
COLORSPACES = {"rgb", "hsv", "gray"}

# Keys every classification run must provide.
REQUIRED_METRICS = [
    "method", "colorspace", "seed", "split",
    "accuracy", "macro_precision", "macro_recall", "macro_f1",
    "per_class_f1", "train_seconds", "n_samples",
]
# Extra keys required only for runs that involve the detector.
DETECTOR_METRICS = ["map50", "map50_95", "mean_iou", "detection_miss_rate"]

PRED_HEADER = ["img_path", "y_true", "y_pred", "confidence"]
HISTORY_HEADER = ["epoch", "train_loss", "val_loss", "train_acc", "val_acc"]


def run_dir(root, method: str, colorspace: str, seed: int) -> Path:
    assert method in METHODS, f"unknown method {method!r}, expected one of {sorted(METHODS)}"
    assert colorspace in COLORSPACES, f"unknown colorspace {colorspace!r}"
    return Path(root) / f"{method}_{colorspace}_seed{seed}"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def env_stamp() -> dict:
    stamp = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": git_commit(),
    }
    try:
        import torch

        stamp["torch"] = str(torch.__version__)
        stamp["cuda"] = str(torch.version.cuda) if torch.version.cuda else None
        stamp["gpu"] = str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu"
    except Exception:
        pass
    return stamp


def write_run(out_dir, config: dict, metrics: dict, predictions, history=None,
              weights_src=None) -> Path:
    """Write a complete, contract-compliant run directory."""
    import shutil

    import yaml

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(json.dumps(dict(config) | {"_env": env_stamp()}, default=str))
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))

    with (out / "predictions.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(PRED_HEADER)
        for row in predictions:
            w.writerow(row)

    if history:
        with (out / "history.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(HISTORY_HEADER)
            for row in history:
                w.writerow(row)

    if weights_src is not None and Path(weights_src).exists():
        shutil.copy2(weights_src, out / "weights.pt")

    return out


def validate_run(path, require_weights: bool = False) -> list:
    """Return a list of problems. Empty list == the run is compliant."""
    p = Path(path)
    problems = []

    if not p.is_dir():
        return [f"{p} is not a directory"]

    for f in ("config.yaml", "metrics.json", "predictions.csv"):
        if not (p / f).exists():
            problems.append(f"missing {f}")
    if require_weights and not (p / "weights.pt").exists():
        problems.append("missing weights.pt")

    if (p / "metrics.json").exists():
        try:
            m = json.loads((p / "metrics.json").read_text())
        except json.JSONDecodeError as e:
            return problems + [f"metrics.json is not valid JSON: {e}"]

        required = list(REQUIRED_METRICS)
        if m.get("method") in {"stage1", "twostage", "effnet"}:
            required += DETECTOR_METRICS
        for k in required:
            if k not in m:
                problems.append(f"metrics.json missing key {k!r}")

        pcf = m.get("per_class_f1")
        if pcf is not None and len(pcf) != 36:
            problems.append(f"per_class_f1 has {len(pcf)} entries, expected 36")
        if m.get("method") not in METHODS:
            problems.append(f"metrics.json method {m.get('method')!r} not in {sorted(METHODS)}")

    if (p / "predictions.csv").exists():
        with (p / "predictions.csv").open() as fh:
            r = csv.reader(fh)
            head = next(r, None)
            if head != PRED_HEADER:
                problems.append(f"predictions.csv header is {head}, expected {PRED_HEADER}")
            n = sum(1 for _ in r)
            # Stage 1 is detector-only: mAP/IoU live in metrics.json and there
            # are no per-image class predictions to write. Every other method
            # must produce one row per evaluated image.
            method = None
            if (p / "metrics.json").exists():
                try:
                    method = json.loads((p / "metrics.json").read_text()).get("method")
                except json.JSONDecodeError:
                    pass
            if n == 0 and method != "stage1":
                problems.append("predictions.csv has no rows")

    return problems


if __name__ == "__main__":
    for target in sys.argv[1:]:
        probs = validate_run(target)
        print(f"{'OK  ' if not probs else 'FAIL'} {target}")
        for prob in probs:
            print(f"       - {prob}")
