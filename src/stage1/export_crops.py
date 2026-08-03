"""Freeze Stage 1's output into a static crop dataset.

Two crop modes:

  --mode gt    boxes from the ground-truth labels. Used to TRAIN Stage 2, so the
               classifier learns from clean crops and is not handicapped by
               detector noise it had no part in causing.
  --mode pred  boxes from the trained detector (highest-confidence box only).
               Used to EVALUATE, so reported numbers reflect the real end-to-end
               pipeline including detector mistakes.

Train on gt/train, report on pred/test. This split is a deliberate design
choice and must be stated in the paper.

Output:
    crops/<colorspace>/<mode>/<split>/<NN_Name>/<stem>.jpg     (128x128)
    crops/<colorspace>/<mode>/manifest_<split>.csv

    python -m src.stage1.export_crops --mode pred --colorspace rgb \
        --data data/prepared/rgb --weights runs/stage1_rgb_seed0/weights/best.pt \
        --out data/crops --splits train val test
"""
import argparse
import csv
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.classes import CLASSES, FOLDER_NAMES, folder_name
from src.stage1.cropping import iou_xyxy, pad_and_clip, yolo_to_xyxy

MANIFEST_HEADER = [
    "split", "mode", "src_image", "class_idx", "class_name",
    "x1", "y1", "x2", "y2", "confidence", "iou_gt", "detected", "crop_path",
]


def read_gt(label_path: Path):
    """First (and normally only) GT box: (class_id, xc, yc, w, h) or None."""
    if not label_path.exists():
        return None
    for ln in label_path.read_text().splitlines():
        ln = ln.strip()
        if ln:
            p = ln.split()
            return int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
    return None


def save_crop(img: Image.Image, box, dst: Path, size: int):
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.crop(box).convert("RGB").resize((size, size), Image.BILINEAR).save(dst, quality=95)


def export_split(model, data_root: Path, split: str, mode: str, out_root: Path,
                 pad: float, size: int, conf: float, batch: int, imgsz: int):
    img_dir = data_root / "det36" / "images" / split
    lbl_dir = data_root / "det36" / "labels" / split
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise SystemExit(f"No images in {img_dir}")

    # Pre-create every class folder so ImageFolder sees all 36 even if one is empty.
    for name in FOLDER_NAMES:
        (out_root / split / name).mkdir(parents=True, exist_ok=True)

    rows, n_missed = [], 0

    def handle(img_path: Path, pred_box, pred_conf):
        nonlocal n_missed
        gt = read_gt(lbl_dir / f"{img_path.stem}.txt")
        if gt is None:
            return
        cid, xc, yc, bw, bh = gt
        with Image.open(img_path) as im:
            W, H = im.size
            gt_xyxy = yolo_to_xyxy(xc, yc, bw, bh, W, H)

            if mode == "gt":
                raw, cfd, iou, detected = gt_xyxy, 1.0, 1.0, True
            elif pred_box is not None:
                raw, cfd = pred_box, pred_conf
                iou = iou_xyxy(pred_box, gt_xyxy)
                detected = True
            else:
                # Detector found nothing. Fall back to the FULL FRAME rather than
                # dropping the sample, so the miss is counted as a pipeline error
                # instead of being quietly hidden from the metrics.
                raw, cfd, iou, detected = (0.0, 0.0, float(W), float(H)), 0.0, 0.0, False
                n_missed += 1

            box = pad_and_clip(*raw, W, H, pad) if detected else (0, 0, W, H)
            dst = out_root / split / folder_name(cid) / f"{img_path.stem}.jpg"
            save_crop(im, box, dst, size)

        rows.append([split, mode, img_path.name, cid, CLASSES[cid],
                     box[0], box[1], box[2], box[3], round(cfd, 5), round(iou, 5),
                     int(detected), str(dst.relative_to(out_root.parent.parent))])

    if mode == "gt":
        for ip in images:
            handle(ip, None, None)
    else:
        for i in range(0, len(images), batch):
            chunk = images[i:i + batch]
            # imgsz MUST match the value the detector was trained at. Ultralytics'
            # predict() defaults to 640 regardless of the checkpoint, so passing it
            # explicitly is not optional -- a mismatch silently degrades every box.
            results = model.predict([str(p) for p in chunk], conf=conf, verbose=False,
                                    device=0, max_det=10, imgsz=imgsz)
            for ip, res in zip(chunk, results):
                b = res.boxes
                if b is None or len(b) == 0:
                    handle(ip, None, None)
                else:
                    k = int(b.conf.argmax())
                    handle(ip, tuple(float(v) for v in b.xyxy[k].tolist()), float(b.conf[k]))
            if i % (batch * 20) == 0:
                print(f"    {split}: {min(i + batch, len(images))}/{len(images)}", flush=True)

    mf = out_root / f"manifest_{split}.csv"
    with mf.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(MANIFEST_HEADER)
        w.writerows(rows)

    miss_rate = n_missed / max(1, len(rows))
    ious = [r[10] for r in rows if r[11]]
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    print(f"  {split}: {len(rows)} crops | misses {n_missed} ({miss_rate:.3%}) | "
          f"mean IoU(detected) {mean_iou:.4f} -> {mf.name}")
    return {"split": split, "n": len(rows), "miss_rate": miss_rate, "mean_iou": mean_iou}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data/prepared/<colorspace>")
    ap.add_argument("--colorspace", required=True, choices=["rgb", "hsv", "gray"])
    ap.add_argument("--mode", required=True, choices=["gt", "pred"])
    ap.add_argument("--weights", help="required when --mode pred")
    ap.add_argument("--out", default="data/crops")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--pad", type=float, default=0.10)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--imgsz", type=int, default=None,
                    help="detector inference size; defaults to the checkpoint's "
                         "training imgsz, which is almost always what you want")
    args = ap.parse_args()

    model, imgsz = None, args.imgsz
    if args.mode == "pred":
        if not args.weights:
            raise SystemExit("--mode pred requires --weights")
        from ultralytics import YOLO
        model = YOLO(args.weights)

        trained_at = None
        try:
            trained_at = int(model.ckpt["train_args"]["imgsz"])
        except Exception:
            pass
        if imgsz is None:
            if trained_at is None:
                raise SystemExit(
                    "Could not read the training imgsz from the checkpoint. "
                    "Pass --imgsz explicitly, matching what you trained at.")
            imgsz = trained_at
            print(f"  imgsz {imgsz} (auto-detected from checkpoint)")
        elif trained_at is not None and trained_at != imgsz:
            print(f"  !! WARNING: detector was trained at imgsz={trained_at} but you "
                  f"passed --imgsz {imgsz}. Boxes will be worse. Ctrl-C now unless "
                  f"this mismatch is deliberate.")
        else:
            print(f"  imgsz {imgsz}")

    out_root = Path(args.out) / args.colorspace / args.mode
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {args.mode} crops for {args.colorspace} -> {out_root}")

    summary = [export_split(model, Path(args.data), s, args.mode, out_root,
                            args.pad, args.size, args.conf, args.batch, imgsz or 640)
               for s in args.splits]

    with (out_root / "export_summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "n", "miss_rate", "mean_iou"])
        for s in summary:
            w.writerow([s["split"], s["n"], f"{s['miss_rate']:.6f}", f"{s['mean_iou']:.6f}"])
    print("Done.")


if __name__ == "__main__":
    main()
