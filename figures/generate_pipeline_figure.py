"""
generate_pipeline_figure.py
============================
Generates Figure 1 – Study Pipeline Flowchart for the Ischemic Stroke
scRNA-seq Machine Learning analysis.

Outputs:
  figures/Figure1_Pipeline_Flowchart.png  (300 dpi)
  figures/Figure1_Pipeline_Flowchart.pdf

Run from the Phase4_RevisedModels directory:
    python figures/generate_pipeline_figure.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

# ── output paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG  = os.path.join(SCRIPT_DIR, "Figure1_Pipeline_Flowchart.png")
OUTPUT_PDF  = os.path.join(SCRIPT_DIR, "Figure1_Pipeline_Flowchart.pdf")

# ── colour palette ────────────────────────────────────────────────────────────
C_NAVY        = "#1B4F8A"
C_BLUE_MID    = "#2E86C1"
C_BLUE_LIGHT  = "#D6EAF8"
C_TEAL        = "#148F77"
C_TEAL_LIGHT  = "#D1F2EB"
C_PURPLE      = "#6C3483"
C_PURPLE_LIGHT= "#E8DAEF"
C_ORANGE      = "#BA4A00"
C_ORANGE_LIGHT= "#FAD7A0"
C_GREEN_DARK  = "#1E8449"
C_GREEN_LIGHT = "#D5F5E3"
C_RED_LIGHT   = "#FDEDEC"
C_RED_DARK    = "#922B21"
C_BG          = "#EBF5FB"

# ── helpers ───────────────────────────────────────────────────────────────────

def rbox(ax, cx, cy, w, h, fc, ec, lw=1.8, radius=0.025, zorder=3):
    """Draw a rounded rectangle; return the patch."""
    p = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad={radius}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=zorder
    )
    ax.add_patch(p)
    return p


def label(ax, x, y, txt, fs=9, fw="normal", color="black",
          ha="center", va="center", zorder=6):
    ax.text(x, y, txt, ha=ha, va=va, fontsize=fs,
            fontweight=fw, color=color, zorder=zorder,
            linespacing=1.4)


def arrow_v(ax, x, y0, y1, color=C_NAVY, lw=1.6):
    """Vertical arrow from (x,y0) → (x,y1)."""
    ax.annotate("",
                xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(
                    arrowstyle="-|>,head_length=0.22,head_width=0.13",
                    color=color, lw=lw),
                zorder=7)


def arrow_h(ax, x0, y, x1, color=C_NAVY, lw=1.6):
    """Horizontal arrow from (x0,y) → (x1,y)."""
    ax.annotate("",
                xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(
                    arrowstyle="-|>,head_length=0.22,head_width=0.13",
                    color=color, lw=lw),
                zorder=7)


# ── canvas ────────────────────────────────────────────────────────────────────
FW, FH = 10.5, 13.5
fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis("off")
fig.patch.set_facecolor("white")

# outer rounded border
rbox(ax, FW/2, FH/2, FW - 0.3, FH - 0.25,
     fc=C_BG, ec=C_BLUE_MID, lw=2.6, radius=0.08, zorder=0)

CX = FW / 2      # main centre x

# ══════════════════════════════════════════════════════════════════════════════
# 1. MOUSE MCAO MODEL
# ══════════════════════════════════════════════════════════════════════════════
Y1 = 12.55
BOX_W = 3.8; BOX_H = 0.72
rbox(ax, CX, Y1, BOX_W, BOX_H, fc=C_BLUE_LIGHT, ec=C_NAVY, lw=2)
label(ax, CX, Y1 + 0.12, "MOUSE MCAO MODEL", fs=11, fw="bold", color=C_NAVY)
label(ax, CX, Y1 - 0.19, "6 samples", fs=8.5, color=C_NAVY)

# stacked-rectangle icon
for dy in (0.16, 0.07, -0.02):
    rbox(ax, CX + 2.20, Y1 + dy, 0.28, 0.12,
         fc=C_BLUE_MID, ec=C_NAVY, lw=0.8, radius=0.01, zorder=5)

# ══════════════════════════════════════════════════════════════════════════════
# 2. 10X scRNA-seq
# ══════════════════════════════════════════════════════════════════════════════
Y2 = 11.35
arrow_v(ax, CX, Y1 - BOX_H/2, Y2 + 0.34)
rbox(ax, CX, Y2, 3.3, 0.62, fc=C_BLUE_LIGHT, ec=C_NAVY, lw=2)
label(ax, CX, Y2, "10X scRNA-seq", fs=10.5, fw="bold", color=C_NAVY)

# gear icon
gx, gy = CX + 1.9, Y2
circ = plt.Circle((gx, gy), 0.195, color=C_BLUE_MID, alpha=0.5, zorder=5)
ax.add_patch(circ)
for ang in np.linspace(0, 2*np.pi, 8, endpoint=False):
    ax.plot([gx + 0.16*np.cos(ang), gx + 0.27*np.cos(ang)],
            [gy + 0.16*np.sin(ang), gy + 0.27*np.sin(ang)],
            color=C_NAVY, lw=1.6, zorder=6)

# ══════════════════════════════════════════════════════════════════════════════
# 3. SEURAT PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
Y3 = 10.28
arrow_v(ax, CX, Y2 - 0.31, Y3 + 0.38)
rbox(ax, CX, Y3, 5.0, 0.70, fc=C_BLUE_LIGHT, ec=C_NAVY, lw=2)
label(ax, CX, Y3 + 0.13, "SEURAT PROCESSING", fs=10.5, fw="bold", color=C_NAVY)
label(ax, CX, Y3 - 0.17, "54,599 cells, 3,000 genes  →  50 PCs",
      fs=8.4, color=C_NAVY)

# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAIN/TEST  +  ML MODELS ROW
# ══════════════════════════════════════════════════════════════════════════════
Y_ROW = 8.35         # vertical centre of row

# arrow from Seurat → centre of row
arrow_v(ax, CX, Y3 - 0.35, Y_ROW + 1.15)

# ─── 4a. Subject-wise Train/Test Split (left) ─────────────────────────────
LX = 2.05           # left box centre x
LW, LH = 3.6, 2.35
rbox(ax, LX, Y_ROW, LW, LH, fc=C_BLUE_LIGHT, ec=C_NAVY, lw=2)
label(ax, LX, Y_ROW + LH/2 - 0.26,
      "SUBJECT-WISE\nTRAIN/TEST SPLIT",
      fs=9.5, fw="bold", color=C_NAVY)

# divider
ax.plot([LX - LW/2 + 0.15, LX + LW/2 - 0.15],
        [Y_ROW + 0.08, Y_ROW + 0.08],
        color=C_BLUE_MID, lw=0.9, linestyle="--", zorder=6)

label(ax, LX, Y_ROW + 0.60,
      "Train:",
      fs=7.8, fw="bold", color=C_NAVY)
label(ax, LX, Y_ROW + 0.32,
      "sham1, sham2, mcao1, mcao2\n(37,883 cells)",
      fs=7.6, color=C_NAVY)
label(ax, LX, Y_ROW - 0.60,
      "Test:",
      fs=7.8, fw="bold", color=C_NAVY)
label(ax, LX, Y_ROW - 0.87,
      "sham3, mcao3\n(16,716 cells)",
      fs=7.6, color=C_NAVY)

# ─── 4b. 8 ML Models (right / centre) ────────────────────────────────────
RX = 7.3            # right-side box centre x
RW, RH = 5.4, 2.35
rbox(ax, RX, Y_ROW, RW, RH, fc=C_BLUE_LIGHT, ec=C_NAVY, lw=2)
label(ax, RX, Y_ROW + RH/2 - 0.22,
      "8 MACHINE LEARNING MODELS",
      fs=10, fw="bold", color=C_NAVY)

# ── Classical ML sub-box ────────────────────────────────────────────────
scx, scy = RX - 1.42, Y_ROW + 0.07
sw, sh = 2.25, 1.32
rbox(ax, scx, scy, sw, sh, fc=C_TEAL_LIGHT, ec=C_TEAL, lw=1.5, zorder=4)
label(ax, scx, scy + sh/2 - 0.22, "Classical ML",
      fs=8.5, fw="bold", color=C_TEAL, zorder=7)
label(ax, scx, scy - 0.10,
      "• Logistic Regression\n• Random Forest\n• XGBoost",
      fs=7.8, color="#145a32", zorder=7)

# ── Transformer-inspired sub-box ────────────────────────────────────────
tcx, tcy = RX + 1.45, Y_ROW + 0.07
tw, th = 2.2, 1.32
rbox(ax, tcx, tcy, tw, th, fc=C_PURPLE_LIGHT, ec=C_PURPLE, lw=1.5, zorder=4)
label(ax, tcx, tcy + th/2 - 0.22, "Transformer-inspired",
      fs=8, fw="bold", color=C_PURPLE, zorder=7)
label(ax, tcx, tcy - 0.10,
      "• scGPT\n• scBERT\n• scFormer\n• Geneformer",
      fs=7.8, color="#4a235a", zorder=7)

# ── Hybrid sub-box ──────────────────────────────────────────────────────
hcx, hcy = RX + 0.02, Y_ROW - 0.77
hw, hh = 2.1, 0.52
rbox(ax, hcx, hcy, hw, hh, fc=C_ORANGE_LIGHT, ec=C_ORANGE, lw=1.5, zorder=4)
label(ax, hcx, hcy + 0.10, "Hybrid", fs=8.5, fw="bold", color=C_ORANGE, zorder=7)
label(ax, hcx, hcy - 0.11, "• CNN+XGBoost", fs=7.8, color="#6e2c00", zorder=7)

# arrow Train/Test → ML box
arrow_h(ax, LX + LW/2, Y_ROW, RX - RW/2, color=C_NAVY)

# ══════════════════════════════════════════════════════════════════════════════
# 5. GENE IMPORTANCE  +  CONSENSUS ROW
# ══════════════════════════════════════════════════════════════════════════════
Y_GI = 6.50

# arrow from ML box down to Gene Importance
arrow_v(ax, RX, Y_ROW - RH/2, Y_GI + 0.50)

# ─── 5a. Gene Importance Extraction ───────────────────────────────────────
GI_CX = 4.15
GI_W, GI_H = 4.5, 0.95
rbox(ax, GI_CX, Y_GI, GI_W, GI_H, fc=C_BLUE_LIGHT, ec=C_NAVY, lw=2)
label(ax, GI_CX, Y_GI + 0.18,
      "GENE IMPORTANCE EXTRACTION",
      fs=9.5, fw="bold", color=C_NAVY)
label(ax, GI_CX, Y_GI - 0.21,
      "Coefficients, L2 norm, SHAP  →  PCA loadings",
      fs=7.8, color=C_NAVY)

# ─── 5b. Multi-Model Consensus ────────────────────────────────────────────
CONS_CX = 8.85
CONS_W, CONS_H = 2.85, 0.95
rbox(ax, CONS_CX, Y_GI, CONS_W, CONS_H,
     fc=C_GREEN_LIGHT, ec=C_GREEN_DARK, lw=2)
label(ax, CONS_CX, Y_GI + 0.17,
      "MULTI-MODEL\nCONSENSUS",
      fs=8.8, fw="bold", color=C_GREEN_DARK)
label(ax, CONS_CX, Y_GI - 0.26, "(≥4/6 models)", fs=7.6, color=C_GREEN_DARK)

# 63 Consensus Genes badge (below consensus box)
BADGE_Y = Y_GI - 0.80
rbox(ax, CONS_CX, BADGE_Y, 2.6, 0.48,
     fc="#A9DFBF", ec=C_GREEN_DARK, lw=2.2, zorder=4)
label(ax, CONS_CX, BADGE_Y, "63 Consensus Genes",
      fs=9, fw="bold", color=C_GREEN_DARK)
# checkmark badge icon
circ_b = plt.Circle((CONS_CX + 1.38, BADGE_Y), 0.2,
                    color=C_GREEN_DARK, alpha=0.25, zorder=5)
ax.add_patch(circ_b)
label(ax, CONS_CX + 1.38, BADGE_Y, "✓", fs=9,
      fw="bold", color=C_GREEN_DARK, zorder=7)

# arrow Gene Importance → Consensus
arrow_h(ax, GI_CX + GI_W/2, Y_GI, CONS_CX - CONS_W/2, color=C_GREEN_DARK)

# ══════════════════════════════════════════════════════════════════════════════
# 6. BIOLOGICAL VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
Y_BV = 4.45

# arrows into bio-validation
arrow_v(ax, GI_CX, Y_GI - GI_H/2, Y_BV + 1.38)
arrow_v(ax, CONS_CX, BADGE_Y - 0.24, Y_BV + 1.38, color=C_GREEN_DARK)

BV_CX = FW / 2
BV_W, BV_H = 9.2, 2.90
rbox(ax, BV_CX, Y_BV, BV_W, BV_H, fc=C_RED_LIGHT, ec=C_RED_DARK, lw=2.4)

# section header inside
label(ax, BV_CX, Y_BV + BV_H/2 - 0.27,
      "BIOLOGICAL VALIDATION",
      fs=11.5, fw="bold", color=C_RED_DARK)

# GO/KEGG sub-header
label(ax, BV_CX - 0.6, Y_BV + 0.58,
      "GO/KEGG Enrichment",
      fs=9.5, fw="bold", color=C_RED_DARK)
# snowflake icons
for xi in (BV_CX + 1.1, BV_CX + 1.75):
    label(ax, xi, Y_BV + 0.58, "❄", fs=14, color=C_BLUE_MID)

# Key findings inner box
KF_CX, KF_CY = BV_CX - 1.35, Y_BV - 0.52
rbox(ax, KF_CX, KF_CY, 4.6, 1.12,
     fc="white", ec=C_RED_DARK, lw=1.4, zorder=4)
label(ax, KF_CX, KF_CY + 0.28,
      "Key findings:", fs=9, fw="bold", color=C_RED_DARK, zorder=7)
label(ax, KF_CX - 0.90, KF_CY - 0.18,
      "• IL-1 Pathway\n• Dual-Interferon Signature",
      fs=8.4, color="#641e16", ha="left", zorder=7)

# IL-1 badge
label(ax, KF_CX + 1.55, KF_CY - 0.12, "IL-1",
      fs=9, fw="bold", color=C_ORANGE, zorder=8)
rbox(ax, KF_CX + 1.55, KF_CY - 0.12, 0.65, 0.35,
     fc="#FDEBD0", ec=C_ORANGE, lw=1.3, radius=0.015, zorder=7)

# IFN annotation on the right
label(ax, BV_CX + 2.7, KF_CY + 0.10, "IFN-α type", fs=8.2, color=C_RED_DARK)
label(ax, BV_CX + 2.7, KF_CY - 0.22, "IFN-γ",       fs=8.2, color=C_RED_DARK)
for yi in (KF_CY + 0.10, KF_CY - 0.22):
    ax.annotate("",
                xy=(BV_CX + 4.25, yi), xytext=(BV_CX + 3.7, yi),
                arrowprops=dict(
                    arrowstyle="-|>,head_length=0.18,head_width=0.10",
                    color=C_RED_DARK, lw=1.3),
                zorder=8)

# ══════════════════════════════════════════════════════════════════════════════
# Title
# ══════════════════════════════════════════════════════════════════════════════
label(ax, CX, FH - 0.38, "Study Analysis Pipeline",
      fs=14, fw="bold", color=C_NAVY)
label(ax, 0.6, FH - 0.38, "Figure 1",
      fs=9.5, fw="bold", color=C_NAVY, ha="left")

# ── save ──────────────────────────────────────────────────────────────────────
for path, fmt in [(OUTPUT_PNG, "png"), (OUTPUT_PDF, "pdf")]:
    fig.savefig(path, format=fmt, dpi=300,
                bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")

plt.close(fig)
print("Done.")
