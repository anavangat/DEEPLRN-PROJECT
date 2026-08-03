"""Detect augmentation leakage

The dataset is a Roboflow export. Filenames look like

    train_100_jpg.rf.0265cb98582351a78a63ded2485c8a64.jpg
    ^^^^^^^^^         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    source image      augmentation hash

Many files can share one SOURCE image. If augmented copies of the same source
land in different splits, the model sees a near-duplicate of a validation image
during training and every reported metric is inflated. This is the single most
dangerous property of this dataset.

This script reports:
  1. the augmentation fan-out (files per source image)
  2. whether the vendor's own train/val split already leaks
  3. whether a source image ever carries more than one class label

    python -m src.data.check_leakage --root ~/asl_raw/RGB/rgb
"""
import argparse
import collections
import re
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ROBOFLOW = re.compile(r"^(?P<src>.+?)_jpg\.rf\.[0-9a-fA-F]+$")


def source_key(stem: str) -> str:
    """Collapse a Roboflow filename to its source image id."""
    m = ROBOFLOW.match(stem)
    return m.group("src") if m else stem


def scan(split_dir: Path):
    """Return {source_key: [(stem, class_id), ...]} for one split."""
    img_dir, lbl_dir = split_dir / "images", split_dir / "labels"
    if not img_dir.is_dir():
        raise SystemExit(f"missing {img_dir}")
    groups = collections.defaultdict(list)
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXT:
            continue
        lbl = lbl_dir / f"{img.stem}.txt"
        cid = None
        if lbl.exists():
            for ln in lbl.read_text().splitlines():
                if ln.strip():
                    cid = int(ln.split()[0])
                    break
        groups[source_key(img.stem)].append((img.stem, cid))
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="colorspace root containing train/ and val/, e.g. ~/asl_raw/RGB/rgb")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    per_split = {s: scan(root / s) for s in args.splits}

    print("=" * 64)
    print("1. AUGMENTATION FAN-OUT")
    for s, g in per_split.items():
        sizes = collections.Counter(len(v) for v in g.values())
        n_files = sum(len(v) for v in g.values())
        print(f"  {s:6s}: {n_files:6d} files from {len(g):6d} source images "
              f"(mean {n_files/max(1,len(g)):.2f}x)")
        print(f"          files-per-source histogram: {dict(sorted(sizes.items()))}")

    print("\n2. CROSS-SPLIT LEAKAGE (vendor's own split)")
    keys = {s: set(g) for s, g in per_split.items()}
    names = list(keys)
    leaked_any = False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            shared = keys[a] & keys[b]
            pct = 100 * len(shared) / max(1, len(keys[b]))
            flag = "LEAK" if shared else "clean"
            print(f"  {a} vs {b}: {len(shared)} shared source images "
                  f"({pct:.1f}% of {b}) -> {flag}")
            if shared:
                leaked_any = True
                print(f"    examples: {sorted(shared)[:5]}")

    print("\n3. LABEL CONSISTENCY WITHIN A SOURCE IMAGE")
    inconsistent = []
    for s, g in per_split.items():
        for k, items in g.items():
            cids = {c for _, c in items if c is not None}
            if len(cids) > 1:
                inconsistent.append((s, k, sorted(cids)))
    if inconsistent:
        print(f"  {len(inconsistent)} source images carry >1 class label:")
        for row in inconsistent[:5]:
            print(f"    {row}")
    else:
        print("  OK: every source image has a single consistent class")

    print("\n" + "=" * 64)
    print("VERDICT")
    if leaked_any:
        print("  The vendor split LEAKS across splits.")
        print("  -> Run prepare_data.py with --pool, which discards the vendor")
        print("     split entirely and rebuilds grouped 80/10/10 splits.")
    else:
        print("  Vendor split is group-clean; it can be used as a split boundary.")
        print("  -> Run prepare_data.py in default mode (vendor val becomes our test).")
    print("  Either way, the train/val carve-out MUST be grouped by source image.")


if __name__ == "__main__":
    main()
