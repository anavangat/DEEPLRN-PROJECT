"""The one time the test split is opened for Stage 1.

Computes test mAP with Ultralytics, then folds in mean IoU and detection miss
rate from the `pred` crop manifest produced by export_crops.py. Updates the
run's metrics.json in place.

    python -m src.stage1.eval_stage1 --run runs/stage1_rgb_seed0 \
        --data data/prepared/rgb/data1.yaml --crops data/crops/rgb/pred
"""
import argparse
import csv
import json
from pathlib import Path


def manifest_stats(crops_dir: Path, split: str):
    mf = crops_dir / f"manifest_{split}.csv"
    if not mf.exists():
        print(f"  (no {mf.name}; skipping IoU/miss-rate)")
        return None, None, None
    ious, detected, n = [], 0, 0
    with mf.open() as fh:
        for row in csv.DictReader(fh):
            n += 1
            if int(row["detected"]):
                detected += 1
                ious.append(float(row["iou_gt"]))
    return (sum(ious) / len(ious) if ious else 0.0), (n - detected) / max(1, n), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--crops", help="data/crops/<cs>/pred")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="defaults to the imgsz recorded in the run's metrics.json")
    args = ap.parse_args()

    run = Path(args.run)
    mpath = run / "metrics.json"
    metrics = json.loads(mpath.read_text())

    imgsz = args.imgsz or metrics.get("imgsz")
    if imgsz is None:
        raise SystemExit("No imgsz in metrics.json; pass --imgsz explicitly.")
    print(f"evaluating at imgsz={imgsz}")

    from ultralytics import YOLO

    model = YOLO(run / "weights.pt")
    res = model.val(data=args.data, split=args.split, imgsz=imgsz,
                device=0, verbose=False, plots=False,
                project="runs/_ultralytics", name=f"{run.name}_val", exist_ok=True)

    metrics["split"] = args.split
    metrics["map50"] = float(res.box.map50)
    metrics["map50_95"] = float(res.box.map)
    metrics["precision"] = float(res.box.mp)
    metrics["recall"] = float(res.box.mr)

    if args.crops:
        iou, miss, n = manifest_stats(Path(args.crops), args.split)
        if iou is not None:
            metrics["mean_iou"] = iou
            metrics["detection_miss_rate"] = miss
            metrics["n_samples"] = n

    mpath.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"TEST  mAP50={metrics['map50']:.4f}  mAP50-95={metrics['map50_95']:.4f}  "
          f"meanIoU={metrics['mean_iou']}  miss={metrics['detection_miss_rate']}")
    print(f"updated {mpath}")


if __name__ == "__main__":
    main()
