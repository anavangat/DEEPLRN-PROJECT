"""Confusion matrix and per-class F1 figures.

    python -m src.analysis.figures --runs runs_synth --method twostage
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis.aggregate import PRETTY, load_runs, per_class_frame
from src.common.classes import CLASSES

FIGDIR = Path("figures")


def pooled_confusion(df, method, colorspace="rgb", normalize=True):
    """Confusion matrix summed over all seeds of one method."""
    cm = np.zeros((36, 36), dtype=float)
    runs = df[(df["method"] == method) & (df["colorspace"] == colorspace)]
    if runs.empty:
        raise SystemExit(f"no runs for {method}/{colorspace}")
    for _, r in runs.iterrows():
        with open(Path(r["run_path"]) / "predictions.csv") as fh:
            for row in csv.DictReader(fh):
                cm[int(row["y_true"]), int(row["y_pred"])] += 1
    if normalize:
        cm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    return cm


def plot_confusion(cm, method, colorspace="rgb"):
    FIGDIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, xticklabels=CLASSES, yticklabels=CLASSES, cmap="viridis",
                vmin=0, vmax=1, square=True, cbar_kws={"label": "row-normalised rate"},
                ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{PRETTY.get(method, method)} — {colorspace.upper()}, pooled over seeds")
    # Separate digits from letters -- the imbalance story is visible here.
    ax.axhline(10, color="w", lw=1.2); ax.axvline(10, color="w", lw=1.2)
    fig.tight_layout()
    out = FIGDIR / f"confusion_{method}_{colorspace}.png"
    fig.savefig(out, dpi=200); print(f"wrote {out}")
    return out


def top_confusions(cm, n=12):
    """The n most frequent off-diagonal errors, as (true, pred, rate)."""
    m = cm.copy(); np.fill_diagonal(m, 0)
    flat = np.dstack(np.unravel_index(np.argsort(m, axis=None)[::-1], m.shape))[0]
    return [(CLASSES[i], CLASSES[j], m[i, j]) for i, j in flat[:n]]


def plot_per_class_f1(df, colorspace="rgb"):
    FIGDIR.mkdir(exist_ok=True)
    pc = per_class_frame(df)
    pc = pc[pc["colorspace"] == colorspace]
    piv = pc.groupby(["method", "class_idx"])["f1"].mean().unstack()
    fig, ax = plt.subplots(figsize=(13, 4.5))
    for m in piv.index:
        ax.plot(range(36), piv.loc[m], marker="o", ms=3, lw=1.2,
                label=PRETTY.get(m, m))
    ax.axvline(9.5, color="k", ls=":", lw=1)
    ax.text(4.5, ax.get_ylim()[0], "digits", ha="center", va="bottom", fontsize=9)
    ax.text(22, ax.get_ylim()[0], "letters", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(36)); ax.set_xticklabels(CLASSES, fontsize=8)
    ax.set_ylabel("F1 (mean over seeds)"); ax.legend(fontsize=8)
    ax.set_title(f"Per-class F1 — {colorspace.upper()}")
    fig.tight_layout()
    out = FIGDIR / f"per_class_f1_{colorspace}.png"
    fig.savefig(out, dpi=200); print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--method", default="twostage")
    ap.add_argument("--colorspace", default="rgb")
    a = ap.parse_args()
    df = load_runs(a.runs)
    cm = pooled_confusion(df, a.method, a.colorspace)
    plot_confusion(cm, a.method, a.colorspace)
    plot_per_class_f1(df, a.colorspace)
    print("\nTop confusions:")
    for t, p, r in top_confusions(cm):
        print(f"  {t} -> {p}: {r:.3f}")