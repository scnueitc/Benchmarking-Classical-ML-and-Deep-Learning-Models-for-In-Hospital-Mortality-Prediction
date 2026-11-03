# -*- coding: utf-8 -*-
"""Generate all publication figures from exp.py outputs."""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

MODELS = ["LR", "RF", "XGBoost", "MLP", "1D-CNN", "LSTM", "FT-Transformer"]
COLORS = {"LR": "#4477AA", "RF": "#EE6677", "XGBoost": "#228833",
          "MLP": "#CCBB44", "1D-CNN": "#66CCEE", "LSTM": "#AA3377",
          "FT-Transformer": "#882255"}

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": "white"})

Z = np.load(os.path.join(HERE, "figdata", "predictions.npz"))

def get_oof(m):
    return Z[f"oof_{m}"]

def panel_sets():
    yield "A", Z["y_a"], [get_oof(m) for m in MODELS]
    yield "B", Z["y_b"], [Z[f"ext_{m}_B"] for m in MODELS]
    yield "C", Z["y_c"], [Z[f"ext_{m}_C"] for m in MODELS]

TITLES = {"A": "Set A (internal, out-of-fold)", "B": "Set B (external)",
          "C": "Set C (external)"}

# ---------------------------------------------------- Fig 1 ROC
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, (s, y, ps) in zip(axes, panel_sets()):
    for m, p in zip(MODELS, ps):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=COLORS[m], lw=1.4,
                label=f"{m} ({auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls="--")
    ax.set_title(TITLES[s]); ax.set_xlabel("1 - Specificity")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
axes[0].set_ylabel("Sensitivity")
axes[2].legend(loc="lower right", fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_roc.png"), bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------- Fig 2 PRC
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, (s, y, ps) in zip(axes, panel_sets()):
    prev = y.mean()
    for m, p in zip(MODELS, ps):
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, color=COLORS[m], lw=1.4,
                label=f"{m} ({auc(rc, pr):.3f})")
    ax.axhline(prev, color="#999999", lw=0.8, ls="--")
    ax.text(0.98, prev + 0.02, f"prevalence {prev:.2f}", ha="right",
            fontsize=7, color="#666666", transform=ax.get_yaxis_transform())
    ax.set_title(TITLES[s]); ax.set_xlabel("Recall")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1.02)
axes[0].set_ylabel("Precision")
axes[2].legend(loc="lower right", fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_prc.png"), bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------- Fig 3 calibration
def reliability(y, p, n_bins=10):
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    xs, ys_ = [], []
    for b in range(n_bins):
        m = bins == b
        if m.sum() == 0:
            continue
        xs.append(p[m].mean()); ys_.append(y[m].mean())
    return np.array(xs), np.array(ys_)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, (s, y, ps) in zip(axes, panel_sets()):
    for m, p in zip(MODELS, ps):
        xs, ys_ = reliability(y, p)
        ax.plot(xs, ys_, marker="o", ms=3, lw=1.2, color=COLORS[m], label=m)
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls="--")
    ax.set_title(TITLES[s]); ax.set_xlabel("Predicted probability")
    ax.set_xlim(0, 0.9); ax.set_ylim(0, 0.9)
axes[0].set_ylabel("Observed mortality rate")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_calibration.png"), bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------- Fig 4 DCA
def net_benefit(y, p, thr):
    n = len(y)
    pred = p >= thr
    tp = ((pred) & (y == 1)).sum()
    fp = ((pred) & (y == 0)).sum()
    return tp / n - fp / n * (thr / (1 - thr))

thr = np.linspace(0.01, 0.99, 99)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, (s, y, ps) in zip(axes, panel_sets()):
    n = len(y)
    ax.plot(thr, [y.mean()] * len(thr), color="#444444", lw=1, ls="--",
            label="Treat all")
    ax.plot(thr, np.zeros_like(thr), color="#999999", lw=1, ls=":",
            label="Treat none")
    for m, p in zip(MODELS, ps):
        nb = [net_benefit(y, p, t) for t in thr]
        ax.plot(thr, nb, color=COLORS[m], lw=1.3, label=m)
    ax.set_title(TITLES[s]); ax.set_xlabel("Threshold probability")
    ax.set_ylim(-0.05, 0.35)
axes[0].set_ylabel("Net benefit")
axes[2].legend(loc="upper right", fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig4_dca.png"), bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------- Fig 5 CV boxplot
fold = pd.read_csv(os.path.join(OUT, "cv_per_fold.csv"))
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, metric, ylab in zip(axes, ("AUROC", "AUPRC"), ("AUROC", "AUPRC")):
    data = [fold[fold.model == m][metric].values for m in MODELS]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                    medianprops=dict(color="black", lw=1.2))
    for patch, m in zip(bp["boxes"], MODELS):
        patch.set_facecolor(COLORS[m]); patch.set_alpha(0.55)
        patch.set_edgecolor("#555555")
    ax.set_xticklabels(MODELS, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylab)
    ax.set_title("Five-fold cross-validation on set A")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig5_cv_boxplot.png"), bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------- Fig 6 importance
imp = pd.read_csv(os.path.join(OUT, "feature_importances.csv"))
cols = [("LR_abscoef", "Logistic regression (|standardized $\\beta$|)"),
        ("RF_importance", "Random forest (Gini importance)"),
        ("XGB_importance", "XGBoost (gain importance)")]
fig, axes = plt.subplots(1, 3, figsize=(12, 5.4))
for ax, (col, title) in zip(axes, cols):
    d = imp.sort_values(col, ascending=True).tail(15)
    ax.barh(d["feature"], d[col] / d[col].max(), color="#4477AA", alpha=0.8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7.5)
    ax.set_xlabel("Relative importance")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig6_importance.png"), bbox_inches="tight")
plt.close(fig)

with open(os.path.join(OUT, "top15_intersection.json")) as f:
    inter = json.load(f)
print("Intersection of top-15:", inter)
print("Figures saved to", FIG)
