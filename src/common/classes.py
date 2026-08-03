"""Canonical class ordering for the whole project.

INDEX ORDER IS FROZEN: 0-9 are the digits, 10-35 are letters A-Z.
This must match the class indices already present in the Kaggle YOLO labels.

Crop folders are named "<zero-padded index>_<class name>" (e.g. "00_0", "10_A",
"35_Z") so that alphabetical sorting -- which is what torchvision's ImageFolder
does -- produces exactly this order. Never rename these folders.
"""

DIGITS = [str(d) for d in range(10)]
LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
CLASSES = DIGITS + LETTERS

NUM_CLASSES = len(CLASSES)
assert NUM_CLASSES == 36, NUM_CLASSES

NAME_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
IDX_TO_NAME = {i: name for i, name in enumerate(CLASSES)}


def folder_name(idx: int) -> str:
    """Crop-directory name for a class index, e.g. 10 -> '10_A'."""
    return f"{idx:02d}_{IDX_TO_NAME[idx]}"


FOLDER_NAMES = [folder_name(i) for i in range(NUM_CLASSES)]

# The invariant every Stage 2 loader depends on.
assert sorted(FOLDER_NAMES) == FOLDER_NAMES, "folder names must sort into index order"


def verify_imagefolder(dataset) -> None:
    """Raise if a torchvision ImageFolder did not pick up our canonical order."""
    got = list(dataset.classes)
    if got != FOLDER_NAMES:
        raise RuntimeError(
            "ImageFolder class order does not match the canonical ordering.\n"
            f"  expected[:12]: {FOLDER_NAMES[:12]}\n"
            f"  got[:12]:      {got[:12]}"
        )
