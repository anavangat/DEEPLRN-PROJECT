"""Build the frozen dataset layout.

Produces, per colorspace:

    data/prepared/<colorspace>/
        det1/images/{train,val,test}    det1/labels/{...}   all boxes -> class 0 "hand"
        det36/images/{train,val,test}   det36/labels/{...}  original 36 classes
        data1.yaml   data36.yaml   split_manifest.csv

SPLITTING IS GROUPED BY SOURCE IMAGE. The dataset is a Roboflow export in which
many augmented files share one source photo. A naive per-file stratified split
scatters copies of the same photo across train and val, which inflates every
metric. StratifiedGroupKFold keeps all augmentations of a source together while
still balancing classes.

Two modes:
  default : vendor's val/ becomes our held-out TEST; vendor's train/ is carved
            90/10 into train/val (grouped). Use when check_leakage.py says the
            vendor split is group-clean.
  --pool  : discard the vendor split, pool everything, rebuild grouped 80/10/10.
            Use when check_leakage.py reports a LEAK.

    python -m src.data.prepare_data --raw ~/asl_raw --colorspace rgb \
        --train-images RGB/rgb/train/images --train-labels RGB/rgb/train/labels \
        --test-images  RGB/rgb/val/images   --test-labels  RGB/rgb/val/labels
"""
import argparse
import csv
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.classes import CLASSES
from src.data.check_leakage import source_key

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_pairs(img_dir: Path, lbl_dir: Path):
    """[(image_path, label_path, class_id, group_key)] for every usable pair."""
    for d in (img_dir, lbl_dir):
        if not d.is_dir():
            raise SystemExit(f"Not a directory: {d}")

    pairs, empty, multibox = [], [], []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXT:
            continue
        lbl = lbl_dir / f"{img.stem}.txt"
        if not lbl.exists():
            empty.append(img.name)
            continue
        lines = [ln.strip() for ln in lbl.read_text().splitlines() if ln.strip()]
        if not lines:
            empty.append(img.name)
            continue
        if len(lines) > 1:
            multibox.append(img.name)
        pairs.append((img, lbl, int(lines[0].split()[0]), source_key(img.stem)))

    if empty:
        print(f"    dropped {len(empty)} images with empty/missing labels "
              f"(e.g. {empty[0]})")
    if multibox:
        print(f"    {len(multibox)} images have >1 box; Stage 2 uses the first box only "
              f"({100*len(multibox)/max(1,len(pairs)):.2f}% of data)")
    return pairs


def grouped_split(pairs, val_fraction, seed):
    """Split off ~val_fraction, stratified by class, grouped by source image."""
    n_splits = max(2, int(round(1.0 / val_fraction)))
    y = np.array([c for _, _, c, _ in pairs])
    groups = np.array([g for _, _, _, g in pairs])
    X = np.zeros(len(pairs))

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    keep_idx, held_idx = next(sgkf.split(X, y, groups))
    keep = [pairs[i] for i in keep_idx]
    held = [pairs[i] for i in held_idx]

    # Grouping must be airtight -- assert rather than trust.
    assert not ({g for _, _, _, g in keep} & {g for _, _, _, g in held}), \
        "group leakage after StratifiedGroupKFold"
    return keep, held


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


def emit(pairs, split, out_root: Path, rows, copy: bool):
    dirs = {}
    for tree in ("det1", "det36"):
        for kind in ("images", "labels"):
            d = out_root / tree / kind / split
            d.mkdir(parents=True, exist_ok=True)
            dirs[(tree, kind)] = d

    for img, lbl, cid, grp in pairs:
        text = lbl.read_text()
        for tree in ("det1", "det36"):
            link_or_copy(img, dirs[(tree, "images")] / img.name, copy)
        (dirs[("det1", "labels")] / f"{img.stem}.txt").write_text(to_single_class(text))
        (dirs[("det36", "labels")] / f"{img.stem}.txt").write_text(text)
        rows.append([split, img.name, grp, cid, CLASSES[cid], str(img.resolve())])


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


def report(split, pairs, min_per_class=10):
    cls = Counter(c for _, _, c, _ in pairs)
    grp = {g for _, _, _, g in pairs}
    print(f"  {split:5s}: {len(pairs):6d} images | {len(grp):5d} source groups | "
          f"{len(cls):2d}/36 classes | per-class min={min(cls.values())} max={max(cls.values())}")
    missing = [CLASSES[c] for c in range(36) if cls.get(c, 0) == 0]
    thin = [CLASSES[c] for c in range(36) if 0 < cls.get(c, 0) < min_per_class]
    if missing:
        print(f"    !! ABSENT from {split}: {missing}")
        print(f"       Grouping is coarser than the class budget. Per-class F1 is")
        print(f"       undefined for these -- fix before running the real queue.")
    if thin:
        print(f"    ~  thin in {split} (<{min_per_class}): {thin}")


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
    ap.add_argument("--pool", action="store_true",
                    help="discard vendor split; rebuild grouped 80/10/10 from everything")
    ap.add_argument("--reuse-split", metavar="MANIFEST",
                    help="apply the source_group->split assignment from another "
                         "colorspace's split_manifest.csv. REQUIRED for hsv/gray so "
                         "all three colorspaces share one identical partition.")
    args = ap.parse_args()

    raw = Path(args.raw).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve() / args.colorspace
    out_root.mkdir(parents=True, exist_ok=True)

    def r(p):
        q = Path(p).expanduser()
        return q if q.is_absolute() else (raw / q)

    print(f"== {args.colorspace} ==")
    print("  scanning vendor train/")
    vendor_train = find_pairs(r(args.train_images), r(args.train_labels))
    print("  scanning vendor val/")
    vendor_val = find_pairs(r(args.test_images), r(args.test_labels))
    if not vendor_train or not vendor_val:
        raise SystemExit("No image/label pairs found -- check your path arguments")

    if args.reuse_split:
        print(f"\n  REUSE MODE: applying partition from {args.reuse_split}")
        assign = {}
        with open(args.reuse_split) as fh:
            for row in csv.DictReader(fh):
                assign[row["source_group"]] = row["split"]
        allp = vendor_train + vendor_val
        buckets = {"train": [], "val": [], "test": []}
        unknown = set()
        for rec in allp:
            s = assign.get(rec[3])
            if s is None:
                unknown.add(rec[3])
            else:
                buckets[s].append(rec)
        if unknown:
            raise SystemExit(
                f"{len(unknown)} source groups are not in the reference manifest "
                f"(e.g. {sorted(unknown)[:3]}). The colorspaces do not contain the "
                f"same source images; they cannot share a partition.")
        train, val, test = buckets["train"], buckets["val"], buckets["test"]

    elif args.pool:
        print("\n  POOLED MODE: vendor split discarded, rebuilding 80/10/10 grouped")
        allp = vendor_train + vendor_val
        rest, test = grouped_split(allp, 0.10, args.seed)
        train, val = grouped_split(rest, args.val_ratio / 0.90, args.seed)
    else:
        print("\n  DEFAULT MODE: vendor val/ -> our TEST; vendor train/ carved 90/10")
        test = vendor_val
        train, val = grouped_split(vendor_train, args.val_ratio, args.seed)

    rare = [c for c, n in Counter(cc for _, _, cc, _ in train).items() if n < 2]
    if rare:
        raise SystemExit(f"Classes with <2 training samples: {rare}")

    rows = []
    for split, pairs in (("train", train), ("val", val), ("test", test)):
        emit(pairs, split, out_root, rows, copy=args.copy)
        report(split, pairs)

    # Final leakage assertion across all three splits.
    gsets = defaultdict(set)
    for split, _fn, grp, *_ in rows:
        gsets[split].add(grp)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = gsets[a] & gsets[b]
        status = "CLEAN" if not shared else f"LEAK ({len(shared)} groups)"
        print(f"  group overlap {a}/{b}: {status}")
        if shared and not args.pool:
            print("    -> vendor split leaks. Re-run with --pool.")

    with (out_root / "split_manifest.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "filename", "source_group", "class_id", "class_name", "source_path"])
        w.writerows(rows)

    write_yaml(out_root, "data1.yaml", "det1", ["hand"])
    write_yaml(out_root, "data36.yaml", "det36", CLASSES)

    print(f"\nWrote {out_root}")
    print(f"  Stage 1 localization : {out_root/'data1.yaml'}")
    print(f"  36-class detector    : {out_root/'data36.yaml'}")
    print(f"  Audit trail          : {out_root/'split_manifest.csv'}")


if __name__ == "__main__":
    main()
