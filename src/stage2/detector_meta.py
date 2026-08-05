"""Detector metrics inherited from Stage 1, for contract compliance.

The detector is FIXED across all Stage 2 seeds: every crop in data/crops was
exported once from stage1_rgb_seed0's weights. Stage 2 seeds vary the
classifier only.
"""
import csv
import json
from pathlib import Path

DETECTOR_RUN = "stage1_rgb_seed0"   # the checkpoint every crop was exported from


def manifest_stats(crops_dir, split):
    """mean IoU over detected boxes + miss rate, from a `pred` crop manifest."""
    mf = Path(crops_dir) / f"manifest_{split}.csv"
    if not mf.exists():
        return None, None
    ious, detected, n = [], 0, 0
    with mf.open() as fh:
        for row in csv.DictReader(fh):
            n += 1
            if int(row["detected"]):
                detected += 1
                ious.append(float(row["iou_gt"]))
    return (sum(ious) / len(ious) if ious else 0.0), (n - detected) / max(1, n)


def detector_metrics(colorspace, split, runs_root="runs", crops_root="data/crops",
                     detector_run=DETECTOR_RUN):
    """map50/map50_95 come from the Stage 1 run; IoU and miss rate from the
    manifest of the colorspace actually being evaluated.

    For hsv/gray the detector was never *validated* on that colorspace -- only
    applied to it -- so mAP is left null rather than copied over from RGB and
    quietly misreported. mean_iou and miss rate ARE real for every colorspace,
    because export_crops measured them against that colorspace's own labels.
    """
    out = {"map50": None, "map50_95": None, "mean_iou": None,
           "detection_miss_rate": None, "detector_run": detector_run}

    mpath = Path(runs_root) / detector_run / "metrics.json"
    if mpath.exists() and colorspace == "rgb":
        m = json.loads(mpath.read_text())
        out["map50"] = m.get("map50")
        out["map50_95"] = m.get("map50_95")

    iou, miss = manifest_stats(Path(crops_root) / colorspace / "pred", split)
    out["mean_iou"], out["detection_miss_rate"] = iou, miss
    return out