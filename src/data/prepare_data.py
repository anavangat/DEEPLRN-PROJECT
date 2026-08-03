"""Build the frozen dataset layout.

Takes the raw Kaggle download and produces, per colorspace:

    data/prepared/<colorspace>/
        det1/images/{train,val,test}    det1/labels/{train,val,test}   <- all boxes class 0 "hand"
        det36/images/{train,val,test}   det36/labels/{train,val,test}  <- original 36 classes
        data1.yaml     data36.yaml
        split_manifest.csv

Two self-contained trees so Ultralytics' images/ -> labels/ path rule works for
both without ever swapping a symlink mid-experiment.

Kaggle's own 80/20 train/test boundary is PRESERVED as the held-out test set.
The 80% training portion is split 90/10 into train/val, stratified by class,
fixed seed. Optuna tunes against val; test is touched exactly once at the end.

    python -m src.data.prepare_data --raw ~/asl_raw --colorspace rgb \
        --train-images RGB/train/images --train-labels RGB/train/labels \
        --test-images  RGB/test/images  --test-labels  RGB/test/labels
"""
import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.classes import CLASSES

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_pairs(img_dir: Path, lbl_dir: Path):
    """Return [(image_path, label_path, class_id)] for every matched pair."""
    if not img_dir.is_dir():
        raise SystemExit(f"Not a directory: {img_dir}")
    if not lbl_dir.is_dir():
        raise SystemExit(f"Not a directory: {lbl_dir}")

    pairs, missing = [], []
    for img in sorted(img_dir.rglob("*")):
        if img.suffix.lower() not in IMG_EXT:
            continue
        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            missing.append(img)
            continue
        lines = [ln.strip() for ln in lbl.read_text().splitlines() if ln.strip()]
        if not lines:
            missing.append(img)
            continue
        pairs.append((img, lbl, int(lines[0].split()[0])))

    if missing:
        print(f"  WARNING: {len(missing)} images had no usable label, dropped "
              f"(e.g. {missing[0].name})")
    return pairs


def to_single_class(text: str) -> str:
    out = []
    for ln in text.splitlines():
        ln = ln.strip()
        if ln:
            parts = ln.split()
            parts[0] = "0"
            out.append(" ".join(parts))
    return "\n".join(out) + "\n"


def link_or_copy(src: Path, dst: Path, copy: bool):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def emit(pairs, split, out_root: Path, manifest_rows, copy: bool):
    dirs = {}
    for tree in ("det1", "det36"):
        for kind in ("images", "labels"):
            d = out_root / tree / kind / split
            d.mkdir(parents=True, exist_ok=True)
            dirs[(tree, kind)] = d

    for img, lbl, cid in pairs:
        text = lbl.read_text()
        for tree in ("det1", "det36"):
            link_or_copy(img, dirs[(tree, "images")] / img.name, copy)
        (dirs[("det1", "labels")] / f"{img.stem}.txt").write_text(to_single_class(text))
        (dirs[("det36", "labels")] / f"{img.stem}.txt").write_text(text)
        manifest_rows.append([split, img.name, cid, CLASSES[cid], str(img.resolve())])


def write_yaml(out_root: Path, fname: str, tree: str, names):
    cfg = {
        "path": str((out_root / tree).resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": {i: n for i, n in enumerate(names)},
    }
    (out_root / fname).write_text(yaml.safe_dump(cfg, sort_keys=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default="data/prepared")
    ap.add_argument("--colorspace", required=True, choices=["rgb", "hsv", "gray"])
    ap.add_argument("--train-images", required=True)
    ap.add_argument("--train-labels", required=True)
    ap.add_argument("--test-images", required=True)
    ap.add_argument("--test-labels", required=True)
    ap.add_argument("--val-ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true",
                    help="copy instead of symlink (required on Google Drive mounts)")
    args = ap.parse_args()

    raw = Path(args.raw).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve() / args.colorspace
    out_root.mkdir(parents=True, exist_ok=True)

    def r(p):
        q = Path(p).expanduser()
        return q if q.is_absolute() else (raw / q)

    print(f"== {args.colorspace} ==")
    train_pairs = find_pairs(r(args.train_images), r(args.train_labels))
    test_pairs = find_pairs(r(args.test_images), r(args.test_labels))
    if not train_pairs or not test_pairs:
        raise SystemExit("No image/label pairs found -- check your --*-images/--*-labels paths")

    labels = [c for _, _, c in train_pairs]
    rare = [c for c, n in Counter(labels).items() if n < 2]
    if rare:
        raise SystemExit(f"Classes with <2 samples cannot be stratified: {rare}")

    tr, va = train_test_split(
        train_pairs, test_size=args.val_ratio, random_state=args.seed, stratify=labels
    )

    manifest_rows = []
    for split, pairs in (("train", tr), ("val", va), ("test", test_pairs)):
        emit(pairs, split, out_root, manifest_rows, copy=args.copy)
        n_cls = len({c for _, _, c in pairs})
        print(f"  {split:5s}: {len(pairs):6d} images, {n_cls:2d}/36 classes present")

    with (out_root / "split_manifest.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "filename", "class_id", "class_name", "source_path"])
        w.writerows(manifest_rows)

    write_yaml(out_root, "data1.yaml", "det1", ["hand"])
    write_yaml(out_root, "data36.yaml", "det36", CLASSES)

    print(f"\nWrote {out_root}")
    print(f"  Stage 1 localization : {out_root/'data1.yaml'}")
    print(f"  36-class detector    : {out_root/'data36.yaml'}")
    print(f"  Audit trail          : {out_root/'split_manifest.csv'}")


if __name__ == "__main__":
    main()
