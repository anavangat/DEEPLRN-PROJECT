"""Build the uncropped full-frame dataset for the localization ablation.

Same partition as the crops -- read straight from Person 1's split_manifest.csv
-- and the same 128x128 output size and folder naming, so it can be loaded by
src.data.crop_dataset.make_loaders() with train_mode=eval_mode="full".

    python -m src.data.export_frames --raw ~/asl_raw --colorspace rgb \
        --manifest data/prepared/rgb/split_manifest.csv --out data/frames
"""
import argparse
import csv
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.classes import CLASSES, FOLDER_NAMES, folder_name

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def index_by_filename(raw_root: Path, subdir: str):
    """filename -> path, for the one colorspace subtree we care about."""
    base = raw_root / subdir
    if not base.is_dir():
        raise SystemExit(f"Not a directory: {base}. Check --subdir against the "
                         f"real layout printed by inspect_dataset.py.")
    idx = {}
    for p in base.rglob("*"):
        if p.suffix.lower() in IMG_EXT and "_preview_gt" not in p.parts:
            idx.setdefault(p.name, p)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--colorspace", required=True, choices=["rgb", "hsv", "gray"])
    ap.add_argument("--subdir", default=None,
                    help="colorspace subtree under --raw; defaults per colorspace")
    ap.add_argument("--manifest", required=True, help="data/prepared/<cs>/split_manifest.csv")
    ap.add_argument("--out", default="data/frames")
    ap.add_argument("--size", type=int, default=128)
    args = ap.parse_args()

    default_sub = {"rgb": "RGB/rgb", "hsv": "HSV/hsv", "gray": "Grayscale/grey"}
    subdir = args.subdir or default_sub[args.colorspace]

    raw = Path(args.raw).expanduser().resolve()
    out_root = Path(args.out) / args.colorspace / "full"
    idx = index_by_filename(raw, subdir)
    print(f"indexed {len(idx)} images under {raw/subdir}")

    rows = sorted(csv.DictReader(open(args.manifest)), key=lambda r: r["filename"])
    for split in ("train", "val", "test"):
        for name in FOLDER_NAMES:               # pre-create all 36, always
            (out_root / split / name).mkdir(parents=True, exist_ok=True)

    written, missing = {"train": 0, "val": 0, "test": 0}, []
    manifests = {s: [] for s in ("train", "val", "test")}

    for r in rows:
        src = idx.get(r["filename"])
        if src is None:
            missing.append(r["filename"])
            continue
        split, cid = r["split"], int(r["class_id"])
        dst = out_root / split / folder_name(cid) / f"{Path(r['filename']).stem}.jpg"
        with Image.open(src) as im:
            im.convert("RGB").resize((args.size, args.size), Image.BILINEAR) \
              .save(dst, quality=95)
        written[split] += 1
        manifests[split].append([split, r["filename"], r["source_group"], cid,
                                 CLASSES[cid], str(dst)])

    for split, rws in manifests.items():
        with (out_root / f"manifest_{split}.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["split", "src_image", "source_group", "class_idx",
                        "class_name", "frame_path"])
            w.writerows(rws)

    print(f"wrote train={written['train']} val={written['val']} test={written['test']}")
    if missing:
        raise SystemExit(
            f"{len(missing)} manifest entries had no matching raw image "
            f"(e.g. {missing[:3]}). Your --raw/--subdir points at a different "
            f"copy of the dataset than Person 1 used. Stop and reconcile -- "
            f"a partially-built frame set makes the comparison invalid.")


if __name__ == "__main__":
    main()
