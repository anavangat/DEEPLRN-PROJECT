"""Inspect the dataset before processing.
Prints the real on-disk layout of the Kaggle download, the class histogram, and
anything that would silently break later. Run this before ANY other data script
and paste the output into the team channel.

    python -m src.data.inspect_dataset --root /path/to/kaggle/download
"""
import argparse
import collections
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def tree(root: Path, max_depth: int = 3, depth: int = 0):
    if depth > max_depth:
        return
    entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name))
    dirs = [e for e in entries if e.is_dir()]
    files = [e for e in entries if e.is_file()]
    for d in dirs[:12]:
        n_img = sum(1 for f in d.rglob("*") if f.suffix.lower() in IMG_EXT)
        n_txt = sum(1 for f in d.rglob("*.txt"))
        print("  " * depth + f"[{d.name}]  images={n_img} labels={n_txt}")
        tree(d, max_depth, depth + 1)
    if len(dirs) > 12:
        print("  " * depth + f"... and {len(dirs) - 12} more directories")
    for f in files[:5]:
        print("  " * depth + f"- {f.name}")
    if len(files) > 5:
        print("  " * depth + f"... and {len(files) - 5} more files")


def label_stats(root: Path):
    label_files = list(root.rglob("*.txt"))
    label_files = [p for p in label_files if p.name != "classes.txt"]
    if not label_files:
        print("\n!! No .txt label files found. This dataset may NOT be YOLO format.")
        return

    per_class = collections.Counter()
    boxes_per_image = collections.Counter()
    malformed, empty, out_of_range = [], [], []

    for lf in label_files:
        lines = [ln.strip() for ln in lf.read_text().splitlines() if ln.strip()]
        boxes_per_image[len(lines)] += 1
        if not lines:
            empty.append(lf)
            continue
        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:
                malformed.append((lf, ln))
                continue
            try:
                cid = int(parts[0])
                vals = [float(v) for v in parts[1:]]
            except ValueError:
                malformed.append((lf, ln))
                continue
            per_class[cid] += 1
            if not all(0.0 <= v <= 1.0 for v in vals):
                out_of_range.append((lf, ln))

    print(f"\nLabel files: {len(label_files)}")
    print(f"Distinct class ids present: {len(per_class)} "
          f"(min={min(per_class) if per_class else '-'}, max={max(per_class) if per_class else '-'})")
    print(f"Boxes per image: {dict(sorted(boxes_per_image.items()))}")
    print(f"Empty label files: {len(empty)}   malformed lines: {len(malformed)}   "
          f"coords outside [0,1]: {len(out_of_range)}")

    print("\nclass_id : count")
    for cid in sorted(per_class):
        print(f"  {cid:>2} : {per_class[cid]}")

    counts = list(per_class.values())
    if counts:
        print(f"\nImbalance ratio (max/min): {max(counts) / min(counts):.2f}")
    for label, sample in (("malformed", malformed), ("out-of-range", out_of_range)):
        if sample:
            print(f"\nExample {label} line: {sample[0][0]}  ->  {sample[0][1]!r}")


def find_yamls(root: Path):
    yamls = list(root.rglob("*.yaml")) + list(root.rglob("*.yml"))
    for y in yamls:
        print(f"\n--- {y} ---")
        print(y.read_text()[:800])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--max-depth", type=int, default=3)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    print(f"ROOT: {root}\n")
    total_img = sum(1 for f in root.rglob("*") if f.suffix.lower() in IMG_EXT)
    print(f"Total images anywhere under root: {total_img}\n")
    print("LAYOUT")
    tree(root, args.max_depth)
    print("\nYAML FILES FOUND")
    find_yamls(root)
    print("\nLABEL STATISTICS")
    label_stats(root)
    print("\nCHECKLIST -- confirm before proceeding:")
    print("  [ ] I can see separate directories for RGB, HSV and grayscale")
    print("  [ ] Each split has matching images/ and labels/ subdirectories")
    print("  [ ] Class ids run 0..35 and 0-9 really are the digits")
    print("  [ ] Boxes per image is overwhelmingly 1")
    print("  [ ] No malformed or out-of-range label lines")


if __name__ == "__main__":
    main()
