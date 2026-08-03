"""Handoff verification script. Checks that the crops and runs are present and correct.

Checks every promise in the handoff: crop trees complete and correctly ordered,
manifests present, Stage 1 runs contract-compliant, weights exported.

    python -m src.runner.verify_handoff --crops data/crops --runs runs
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.classes import FOLDER_NAMES
from src.common.contract import validate_run

SPLITS = ["train", "val", "test"]


def check_crops(root: Path, colorspaces, modes):
    problems = []
    for cs in colorspaces:
        for mode in modes:
            base = root / cs / mode
            if not base.is_dir():
                problems.append(f"MISSING crop tree {base}")
                continue
            for split in SPLITS:
                sd = base / split
                if not sd.is_dir():
                    problems.append(f"MISSING {sd}")
                    continue
                got = sorted(p.name for p in sd.iterdir() if p.is_dir())
                if got != FOLDER_NAMES:
                    problems.append(
                        f"{sd}: class dirs are not the canonical 36 (got {len(got)}). "
                        f"First mismatch at index "
                        f"{next((i for i, (a, b) in enumerate(zip(got, FOLDER_NAMES)) if a != b), -1)}: "
                        f"found {next((a for a, b in zip(got, FOLDER_NAMES) if a != b), 'n/a')!r}, "
                        f"expected {next((b for a, b in zip(got, FOLDER_NAMES) if a != b), 'n/a')!r}")
                n = sum(1 for _ in sd.rglob("*.jpg"))
                if n == 0:
                    problems.append(f"{sd}: no crops")
                else:
                    print(f"  {cs}/{mode}/{split}: {n} crops, 36 class dirs")
                if not (base / f"manifest_{split}.csv").exists():
                    problems.append(f"MISSING {base/f'manifest_{split}.csv'}")
    return problems


def check_runs(root: Path):
    problems = []
    runs = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_"))
    if not runs:
        problems.append(f"no run directories under {root}")
    for r in runs:
        probs = validate_run(r, require_weights=r.name.startswith("stage1"))
        if probs:
            problems += [f"{r.name}: {p}" for p in probs]
        else:
            print(f"  {r.name}: OK")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="data/crops")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--colorspaces", nargs="+", default=["rgb", "hsv", "gray"])
    ap.add_argument("--modes", nargs="+", default=["gt", "pred"])
    args = ap.parse_args()

    print("CROPS"); cp = check_crops(Path(args.crops), args.colorspaces, args.modes)
    print("RUNS");  rp = check_runs(Path(args.runs))

    problems = cp + rp
    print("\n" + "=" * 60)
    if problems:
        print(f"NOT READY TO HAND OFF -- {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("ALL CHECKS PASSED -- safe to hand off.")


if __name__ == "__main__":
    main()
