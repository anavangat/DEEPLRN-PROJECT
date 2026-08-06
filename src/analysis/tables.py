"""Paper tables. Writes Markdown (for drafting) and LaTeX (for the paper).

    python -m src.analysis.tables --runs runs_synth --colorspace rgb
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis.aggregate import CLASSIFIERS, PRETTY, load_runs, per_class_frame

ORDER = ["uncropped", "yolocls", "effnet", "twostage"]     # ours last


def ms(series):
    """mean ± std, 3 dp, the way it goes in the paper."""
    return f"{series.mean():.3f} ± {series.std(ddof=1):.3f}"


def main_table(df, colorspace="rgb"):
    d = df[(df["colorspace"] == colorspace) & (df["method"].isin(CLASSIFIERS))]
    rows = []
    for m in [x for x in ORDER if x in set(d["method"])]:
        g = d[d["method"] == m]
        rows.append({
            "Method": PRETTY[m],
            "Seeds": len(g),
            "Accuracy": ms(g["accuracy"]),
            "Macro-P": ms(g["macro_precision"]),
            "Macro-R": ms(g["macro_recall"]),
            "Macro-F1": ms(g["macro_f1"]),
            "Train (min)": ("—" if g["train_seconds"].isna().all()
                            else f"{g['train_seconds'].mean()/60:.1f}"),
        })
    return pd.DataFrame(rows)


def digit_letter_table(df, colorspace="rgb"):
    pc = per_class_frame(df)
    pc = pc[pc["colorspace"] == colorspace]
    rows = []
    for m in [x for x in ORDER if x in set(pc["method"])]:
        g = pc[pc["method"] == m]
        per_seed = g.groupby(["seed", "kind"])["f1"].mean().unstack()
        rows.append({"Method": PRETTY[m],
                     "Digit F1 (0–9)": ms(per_seed["digit"]),
                     "Letter F1 (A–Z)": ms(per_seed["letter"]),
                     "Gap": f"{per_seed['letter'].mean() - per_seed['digit'].mean():+.3f}"})
    return pd.DataFrame(rows)


def stage1_table(df):
    d = df[df["method"] == "stage1"]
    if d.empty:
        return None
    return pd.DataFrame([{
        "Model": "YOLOv8n (hand localization)", "Seeds": len(d),
        "mAP@50": ms(d["map50"]), "mAP@50-95": ms(d["map50_95"]),
        "Mean IoU": ms(d["mean_iou"]),
        "Miss rate": f"{d['detection_miss_rate'].mean():.4f}",
    }])

def colorspace_table(df, methods=("twostage",)):
    """Macro-F1 by colorspace, plus the drop relative to RGB."""
    rows = []
    for m in methods:
        d = df[df["method"] == m]
        if d.empty:
            continue
        base = d[d["colorspace"] == "rgb"]["macro_f1"].mean()
        row = {"Method": PRETTY.get(m, m)}
        for cs, nice in (("rgb", "RGB"), ("hsv", "HSV"), ("gray", "Grayscale")):
            g = d[d["colorspace"] == cs]
            row[nice] = ms(g["macro_f1"]) if len(g) else "—"
            if cs != "rgb" and len(g):
                row[f"Δ {nice}"] = f"{g['macro_f1'].mean() - base:+.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def emit(table, name, outdir="tables"):
    Path(outdir).mkdir(exist_ok=True)
    (Path(outdir) / f"{name}.md").write_text(table.to_markdown(index=False))
    (Path(outdir) / f"{name}.tex").write_text(
        table.to_latex(index=False, escape=False, column_format="l" + "c" * (len(table.columns) - 1)))
    print(f"\n=== {name} ===\n{table.to_markdown(index=False)}")

####
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--colorspace", default="rgb")
    a = ap.parse_args()
    df = load_runs(a.runs)
    emit(main_table(df, a.colorspace), f"main_results_{a.colorspace}")
    emit(digit_letter_table(df, a.colorspace), f"digit_letter_{a.colorspace}")
    s1 = stage1_table(df)
    if s1 is not None:
        emit(s1, "stage1_detection")
