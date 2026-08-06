"""Friedman omnibus + Nemenyi post-hoc + critical-difference diagram.

Blocking factor is fixed by ANALYSIS_PREREGISTRATION.md and must not be
chosen after seeing results.

    python -m src.analysis.stats_tests --runs runs_synth --block class
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis.aggregate import CLASSIFIERS, PRETTY, load_runs, per_class_frame

# Studentised range statistic at alpha=0.05, divided by sqrt(2) -- the standard
# Nemenyi q_alpha table used for critical-difference diagrams (Demsar 2006).
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}


def build_matrix(df, block="class", colorspace="rgb"):
    """Returns (DataFrame blocks x methods, description)."""
    methods = [m for m in ["uncropped", "yolocls", "effnet", "twostage"]
               if m in set(df["method"])]
    if block == "class":
        pc = per_class_frame(df)
        pc = pc[pc["colorspace"] == colorspace]
        # average across seeds first -> one 36-vector per method
        mat = (pc.groupby(["method", "class_idx"])["f1"].mean()
                 .unstack(level=0)[methods])
        return mat, f"per-class F1, blocked on 36 classes ({colorspace})"
    if block == "seed":
        d = df[(df["colorspace"] == colorspace) & (df["method"].isin(methods))]
        mat = d.pivot_table(index="seed", columns="method", values="macro_f1")[methods]
        return mat, f"macro-F1, blocked on {len(mat)} seeds ({colorspace})"
    raise ValueError(block)


def friedman(mat):
    stat, p = friedmanchisquare(*[mat[c].values for c in mat.columns])
    return {"statistic": float(stat), "p_value": float(p),
            "k": mat.shape[1], "n_blocks": mat.shape[0]}


def average_ranks(mat):
    """Lower rank = better (we rank descending on F1)."""
    r = np.apply_along_axis(lambda row: rankdata(-row), 1, mat.values)
    return pd.Series(r.mean(axis=0), index=mat.columns).sort_values()


def critical_difference(k, n, alpha=0.05):
    q = Q05.get(k)
    if q is None:
        raise ValueError(f"no tabulated q for k={k}")
    return q * np.sqrt(k * (k + 1) / (6.0 * n))


def cd_diagram(ranks, nemenyi, cd, title, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 2.6))
    try:
        sp.critical_difference_diagram(
            ranks.rename(index=lambda m: PRETTY.get(m, m)),
            nemenyi.rename(index=lambda m: PRETTY.get(m, m),
                           columns=lambda m: PRETTY.get(m, m)))
        plt.title(f"{title}\nCD = {cd:.3f} (Nemenyi, alpha=0.05)", fontsize=10)
    except Exception as e:                       # older scikit-posthocs
        print(f"  (falling back to plain rank plot: {e})")
        plt.barh([PRETTY.get(m, m) for m in ranks.index], ranks.values)
        plt.xlabel("average rank (lower is better)")
        plt.title(f"{title} — CD = {cd:.3f}", fontsize=10)
    plt.tight_layout(); plt.savefig(out, dpi=200)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--block", default="class", choices=["class", "seed"])
    ap.add_argument("--colorspace", default="rgb")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()

    df = load_runs(a.runs)
    mat, desc = build_matrix(df, a.block, a.colorspace)
    print(f"\n{desc}\nmatrix: {mat.shape[0]} blocks x {mat.shape[1]} methods\n")
    print(mat.describe().round(4).to_string())

    fr = friedman(mat)
    print(f"\nFriedman: chi2={fr['statistic']:.4f}  p={fr['p_value']:.3e}  "
          f"(k={fr['k']}, N={fr['n_blocks']})")

    ranks = average_ranks(mat)
    print("\naverage ranks (lower is better):")
    for m, r in ranks.items():
        print(f"  {PRETTY.get(m, m):<32s} {r:.3f}")

    cd = critical_difference(fr["k"], fr["n_blocks"], a.alpha)
    print(f"\ncritical difference (Nemenyi, alpha={a.alpha}): {cd:.4f}")

    if fr["p_value"] >= a.alpha:
        print("\nFriedman is NOT significant. Report the omnibus result and the "
              "ranks; a post-hoc after a non-significant omnibus is not a valid "
              "basis for pairwise claims. Do NOT switch blocking factors now.")

    nem = sp.posthoc_nemenyi_friedman(mat.values)
    nem.index = nem.columns = mat.columns
    print("\nNemenyi pairwise p-values:")
    print(nem.round(4).to_string())

    out = Path("tables"); out.mkdir(exist_ok=True)
    nem.to_csv(out / f"nemenyi_{a.block}_{a.colorspace}.csv")
    pd.Series(fr).to_csv(out / f"friedman_{a.block}_{a.colorspace}.csv")
    ranks.to_csv(out / f"ranks_{a.block}_{a.colorspace}.csv")
    cd_diagram(ranks, nem, cd, desc, f"figures/cd_{a.block}_{a.colorspace}.png")


if __name__ == "__main__":
    main()