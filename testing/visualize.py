#!/usr/bin/env python3
"""
visualize.py

Create four kinds of figures for every test-case (row) in outputs.csv:

1. Grade-distribution histogram (counts of Scam Probability 0-5).
2. 2×2 coloured confusion-matrix heat-map for the *default* threshold 3.
3. Per-category accuracy bar-chart.
4. Accuracy-vs-threshold plot for thresholds {2,3,4,5}
   (class = scam  ⇔  ScamProb ≥ threshold).

Each figure is written as <model_name>_<figure>.png in --out_dir
(default: ./figures).

USAGE
    python visualize.py \
        --csv outputs.csv \
        --logs_dir path/to/json_logs \
        --out_dir figs
"""

from __future__ import annotations
import argparse, json, os, re, textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
from sklearn.metrics import confusion_matrix

# ----------------------------------------------------------------------
# Helpers copied (lightly) from test.py so you don't need to import it
# ----------------------------------------------------------------------
_PROB_RE = re.compile(
    r'["\']?Scam\ Probability["\']?\s*:\s*(\d+)', re.I | re.VERBOSE
)

def extract_scam_probability(response_text: str) -> int | None:
    cleaned = response_text.strip().replace("\\", "")
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned).replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    m = _PROB_RE.search(cleaned)
    return int(m.group(1)) if m else None


def label_from_path(path: str) -> int | None:
    """Ground truth: 1 = scam, 0 = legit, None = unknown."""
    lp = path.lower()
    if "/scam" in lp:
        return 1
    if "/legit" in lp:
        return 0
    return None

# ----------------------------------------------------------------------
# Figure helpers
# ----------------------------------------------------------------------
def plot_histogram(probs: list[int], ax=None):
    ax = ax or plt.gca()
    ax.hist(probs, bins=[-0.5,0.5,1.5,2.5,3.5,4.5,5.5], rwidth=.8)
    plt.xticks(range(0,6))
    ax.set_xlabel("Scam Probability")
    ax.set_ylabel("Count")


def plot_confusion(y_true, y_pred, ax=None) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    ax = ax or plt.gca()
    im = ax.imshow(cm, cmap="Blues")
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}", ha="center", va="center",
                 fontweight="bold", color="white" if cm.max()/2 < v else "black")
    ax.set_xticks([0,1]); ax.set_xticklabels(["Scam","Legit"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["Scam","Legit"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")



def plot_category_bar(row: pd.Series, ax=None) -> None:
    cat_cols = [c for c in row.index if c.startswith("accuracy_")]
    cats      = [c.replace("accuracy_","") for c in cat_cols]
    accur     = [float(row[c]) for c in cat_cols]

    ax = ax or plt.gca()
    ax.barh(cats, accur)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel("Accuracy")


def plot_threshold(records, ax=None) -> None:
    y_true  = [gt for _, gt, _ in records]         # 1/0
    probs   = [p  for p, _, _ in records]          # 0-5 int

    threshes = [1,2,3,4,5]
    accs     = []
    for th in threshes:
        pred = [1 if p>=th else 0 for p in probs]
        correct = sum(int(p==t) for p,t in zip(pred, y_true))
        accs.append(correct / len(y_true))

    ax = ax or plt.gca()
    ax.bar([f"≥{t}" for t in threshes], accs)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Threshold (scam ⇔ ScamProb ≥ t)")
    ax.set_ylim(0,1)

# ----------------------------------------------------------------------
def load_records(json_path: Path):
    """Return list of (prob, gt_label, img_path)."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for rec in data:
        prob = extract_scam_probability(rec.get("response",""))
        gt   = label_from_path(rec.get("image",""))
        if prob is None or gt is None:
            continue
        records.append((prob, gt, rec.get("image","")))
    return records


def main(csv_path: Path, logs_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        model   = Path(row["json_filename"]).stem
        json_fp = logs_dir/row["json_filename"]
        if not json_fp.exists():
            print(f"[WARN] Skipping {model}: {json_fp} not found")
            continue

        records = load_records(json_fp)
        if not records:
            print(f"[WARN] Skipping {model}: no parsable records")
            continue

        probs   = [p for p,_,_ in records]
        y_true  = [gt for _,gt,_ in records]
        y_pred3 = [1 if p>=3 else 0 for p in probs]

        # -------- combined 2×2 figure --------
        fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)

        plot_histogram(probs,                    ax=axes[0,0])
        axes[0,0].set_title("Grade distribution")

        plot_confusion(y_true, y_pred3,          ax=axes[0,1])
        axes[0,1].set_title("Confusion (thr 3)")

        plot_category_bar(row,                   ax=axes[1,0])
        axes[1,0].set_title("Per-category accuracy")

        plot_threshold(records,                  ax=axes[1,1])
        axes[1,1].set_title("Accuracy vs threshold")

        fig.suptitle(f"{model}", fontsize=14)
        fig.savefig(out_dir / f"{model}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"✓ {model} → {out_dir / (model + '.png')}")

        print(f"✓ {model} → figures saved to {out_dir}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(__doc__))
    parser.add_argument("--csv",      type=Path, default="outputs.csv",
                        help="Path to outputs.csv generated by test.py")
    parser.add_argument("--logs_dir", type=Path, default=Path("."),
                        help="Directory containing the JSON log files")
    parser.add_argument("--out_dir",  type=Path, default=Path("figures"),
                        help="Where to write .png files")
    args = parser.parse_args()
    main(args.csv, args.logs_dir, args.out_dir)
