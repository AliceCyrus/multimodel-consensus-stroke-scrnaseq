"""
Regenerate fig3_cv_distributions.png — Option C
================================================
Panels A-C : 10-fold CV boxplots (Accuracy, Sensitivity, F1) — real for all 8 models
Panel D    : Test AUC bar chart with 95% bootstrap CI — real for all 8 models
             (replaces the AUC-ROC CV boxplot which had NaN for 5 of 8 models)

Data sources:
  - cv_results.accuracy / recall / f1  : real 10-fold arrays from results_*.json
  - test_metrics.roc_auc               : real test set AUC from results_*.json
  - confidence_intervals.roc_auc       : real 1000-bootstrap 95% CI from results_*.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ---- Paths ---------------------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent
OUTPUT_DIR  = BASE_DIR / "manuscript" / "generic_latex" / "figures"
PUB_DIR     = BASE_DIR / "PublicationsFigure"

# ---- Model definitions (display order) -----------------------------------
MODEL_FILES = [
    ("Logistic Regression", "results_logistic_regression.json", "#1f77b4"),
    ("Random Forest",       "results_random_forest.json",       "#ff7f0e"),
    ("XGBoost",             "results_xgboost.json",             "#2ca02c"),
    ("CNN+XGBoost",         "results_cnn_xgboost.json",         "#9467bd"),
    ("scGPT",               "results_scgpt.json",               "#d62728"),
    ("scBERT",              "results_scbert.json",               "#e377c2"),
    ("scFormer",            "results_scformer.json",             "#8c564b"),
    ("Geneformer",          "results_geneformer.json",           "#17becf"),
]

# ---- Load all data -------------------------------------------------------
print("Loading results from JSON files...")
model_names = []
cv_accuracy, cv_recall, cv_f1 = [], [], []
test_auc, ci_low, ci_high = [], [], []
colors = []

for display_name, filename, color in MODEL_FILES:
    fpath = BASE_DIR / filename
    if not fpath.exists():
        raise FileNotFoundError(f"Missing: {fpath}")

    with open(fpath, "r") as f:
        data = json.load(f)

    # CV arrays — real 10-fold values for all 8 models
    cv  = data["cv_results"]
    acc = [x for x in cv["accuracy"] if x is not None and str(x) != "NaN"]
    rec = [x for x in cv["recall"]   if x is not None and str(x) != "NaN"]
    f1  = [x for x in cv["f1"]       if x is not None and str(x) != "NaN"]

    # Test AUC + bootstrap CI
    auc      = data["test_metrics"]["roc_auc"]
    ci       = data["confidence_intervals"]["roc_auc"]
    ci_lower = ci[0]
    ci_upper = ci[1]

    model_names.append(display_name)
    cv_accuracy.append(acc)
    cv_recall.append(rec)
    cv_f1.append(f1)
    test_auc.append(auc)
    ci_low.append(ci_lower)
    ci_high.append(ci_upper)
    colors.append(color)

    print(f"  {display_name}: acc_folds={len(acc)}, recall_folds={len(rec)}, "
          f"f1_folds={len(f1)}, test_AUC={auc:.4f}, "
          f"CI=[{ci_lower:.4f}, {ci_upper:.4f}]")

n_models = len(model_names)
x_pos    = np.arange(n_models)

# ---- Shared style --------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

LABEL_FONTSIZE = 10
TITLE_FONTSIZE = 11
TICK_FONTSIZE  = 8.5

def style_ax(ax, title, ylabel, ylim=(None, None)):
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=6)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=40, ha="right",
                       fontsize=TICK_FONTSIZE)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, linewidth=0.7)
    ax.set_axisbelow(True)
    if ylim[0] is not None:
        ax.set_ylim(ylim)

def cv_boxplot(ax, data_list, title, ylabel):
    """Draw coloured boxplot for 10-fold CV distributions."""
    bp = ax.boxplot(
        data_list,
        positions=x_pos,
        widths=0.55,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
        boxprops=dict(linewidth=1.2),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    style_ax(ax, title, ylabel)

# ---- Figure layout -------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(18, 5.5))
fig.subplots_adjust(wspace=0.38, left=0.06, right=0.98, bottom=0.22, top=0.88)

# ---- Panel A — Accuracy (10-fold CV) ------------------------------------
cv_boxplot(axes[0], cv_accuracy,
           "A.  Accuracy\n(10-Fold CV)", "Accuracy")
axes[0].set_ylim(
    min(min(v) for v in cv_accuracy) - 0.005,
    max(max(v) for v in cv_accuracy) + 0.005,
)

# ---- Panel B — Sensitivity / Recall (10-fold CV) ------------------------
cv_boxplot(axes[1], cv_recall,
           "B.  Sensitivity\n(10-Fold CV)", "Sensitivity (Recall)")
axes[1].set_ylim(
    min(min(v) for v in cv_recall) - 0.005,
    max(max(v) for v in cv_recall) + 0.005,
)

# ---- Panel C — F1-Score (10-fold CV) ------------------------------------
cv_boxplot(axes[2], cv_f1,
           "C.  F1-Score\n(10-Fold CV)", "F1-Score")
axes[2].set_ylim(
    min(min(v) for v in cv_f1) - 0.005,
    max(max(v) for v in cv_f1) + 0.005,
)

# ---- Panel D — Test AUC-ROC + 95% Bootstrap CI (bar chart) --------------
ax = axes[3]
auc_arr    = np.array(test_auc)
err_low    = auc_arr - np.array(ci_low)   # distance below bar top
err_high   = np.array(ci_high) - auc_arr  # distance above bar top
yerr       = np.array([err_low, err_high])

bars = ax.bar(
    x_pos, auc_arr,
    width=0.6,
    color=colors,
    alpha=0.80,
    edgecolor="black",
    linewidth=0.8,
    zorder=3,
)
ax.errorbar(
    x_pos, auc_arr,
    yerr=yerr,
    fmt="none",
    ecolor="black",
    elinewidth=1.5,
    capsize=5,
    capthick=1.5,
    zorder=4,
)

# Value labels on bars
for bar, auc_val in zip(bars, auc_arr):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.0012,
        f"{auc_val:.3f}",
        ha="center", va="bottom",
        fontsize=7.5, fontweight="bold",
    )

y_min = min(ci_low) - 0.005
y_max = max(ci_high) + 0.012
ax.set_ylim(y_min, y_max)
style_ax(ax,
         "D.  AUC-ROC\n(Test Set + 95% CI)",
         "AUC-ROC")
ax.yaxis.grid(True, linestyle="--", alpha=0.5, linewidth=0.7)

# Annotation explaining CI source
ax.annotate(
    "Error bars = 95% CI\n(1,000 bootstrap resamples)",
    xy=(0.98, 0.04), xycoords="axes fraction",
    ha="right", va="bottom",
    fontsize=7.5, color="grey",
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
              edgecolor="lightgrey", alpha=0.9),
)

# ---- Super-title ---------------------------------------------------------
fig.suptitle(
    "Figure 3: Model Performance — 10-Fold Cross-Validation and Test Set Evaluation\n"
    "Panels A–C: Cross-validation distributions (all 8 models, 10 folds each)  |  "
    "Panel D: Test set AUC-ROC with 95% bootstrap confidence intervals",
    fontsize=10, y=0.995, va="top", color="#333333",
)

# ---- Save ----------------------------------------------------------------
for out_dir in [OUTPUT_DIR, PUB_DIR]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig3_cv_distributions.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[SAVED] {out_path}")

plt.close(fig)
print("\n=== fig3_cv_distributions.png COMPLETE ===")
print("Panels A-C: Real 10-fold CV (accuracy, sensitivity, F1) — all 8 models")
print("Panel D:    Real test AUC + 1000-bootstrap 95% CI — all 8 models")
print("No NaN values. No approximated data.")
