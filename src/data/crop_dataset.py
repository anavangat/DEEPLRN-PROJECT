"""Stage 2 must load crops through here.

Guarantees the canonical class order (0-9 digits, 10-35 A-Z) and fails loudly
rather than silently mislabelling everything -- the single most likely
project-ruining bug in this pipeline.

Usage on a free Colab T4:

    from src.data.crop_dataset import make_loaders
    train_dl, val_dl, test_dl, classes = make_loaders(
        crop_root="data/crops/rgb",
        train_mode="gt",     # train on clean ground-truth crops
        eval_mode="pred",    # evaluate on real detector output
        batch_size=64, seed=0,
    )
"""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.classes import CLASSES, FOLDER_NAMES, verify_imagefolder
from src.common.seeding import worker_init_fn

# ImageNet statistics: correct for the EfficientNet-B0 baseline, and a fine
# default for the custom CNN. Keep identical across methods or the comparison
# is confounded.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_transforms(size=128, train=False, augment=True):
    if train and augment:
        # NO horizontal flip: several ASL handshapes are chirality-sensitive and
        # flipping would create genuinely mislabelled training data.
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def make_dataset(crop_root, mode, split, size=128, train=False, augment=True):
    root = Path(crop_root) / mode / split
    if not root.is_dir():
        raise FileNotFoundError(
            f"{root} not found. Ask Person 1 for the crop archive, or check that "
            f"crop_root points at data/crops/<colorspace> (not the parent)."
        )
    ds = ImageFolder(str(root), transform=build_transforms(size, train, augment))
    verify_imagefolder(ds)  # hard fail on any ordering drift
    return ds


def make_loaders(crop_root, train_mode="gt", eval_mode="pred", size=128,
                 batch_size=64, num_workers=2, seed=0, augment=True):
    """Returns (train_dl, val_dl, test_dl, class_names).

    Train on `train_mode` crops, evaluate on `eval_mode` crops. The default
    (gt -> pred) trains on clean boxes but reports honest end-to-end numbers
    including detector errors. State this choice in the paper.
    """
    tr = make_dataset(crop_root, train_mode, "train", size, train=True, augment=augment)
    va = make_dataset(crop_root, eval_mode, "val", size)
    te = make_dataset(crop_root, eval_mode, "test", size)

    g = torch.Generator()
    g.manual_seed(seed)
    common = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                  worker_init_fn=worker_init_fn)

    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True, generator=g,
                   drop_last=False, **common),
        DataLoader(va, batch_size=batch_size, shuffle=False, **common),
        DataLoader(te, batch_size=batch_size, shuffle=False, **common),
        list(CLASSES),
    )


def class_weights(dataset):
    """Inverse-frequency weights for nn.CrossEntropyLoss(weight=...).
    Only use if inspect_dataset.py showed real imbalance."""
    import collections

    counts = collections.Counter(y for _, y in dataset.samples)
    total = sum(counts.values())
    return torch.tensor(
        [total / (len(FOLDER_NAMES) * max(1, counts.get(i, 0))) for i in range(len(FOLDER_NAMES))],
        dtype=torch.float,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Smoke-test a crop archive")
    ap.add_argument("--crop-root", required=True)
    ap.add_argument("--train-mode", default="gt")
    ap.add_argument("--eval-mode", default="pred")
    a = ap.parse_args()

    tr, va, te, names = make_loaders(a.crop_root, a.train_mode, a.eval_mode, num_workers=0)
    x, y = next(iter(tr))
    print(f"train={len(tr.dataset)} val={len(va.dataset)} test={len(te.dataset)}")
    print(f"batch {tuple(x.shape)} labels {y[:8].tolist()}")
    print(f"idx 0 -> {names[0]!r}   idx 10 -> {names[10]!r}   idx 35 -> {names[35]!r}")
    print("Class ordering verified.")
