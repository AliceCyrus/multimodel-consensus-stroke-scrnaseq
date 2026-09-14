# ============================================================================
# SUPPLEMENTARY FIGURE S1 — QC Metrics (REAL DATA)
# ============================================================================
# Data source: Raw 10X Genomics data (barcodes.tsv.gz, features.tsv.gz,
#              matrix.mtx.gz) for all 6 samples in StrokeData/
#
# This script:
#   1. Loads all 6 raw samples via Read10X()
#   2. Creates Seurat objects with the SAME parameters as DP_Diff_Test&Train.R
#      (min.cells=3, min.features=200)
#   3. Computes percent.mt per cell
#   4. Exports real per-cell QC metrics to CSV (for transparency/reproducibility)
#   5. Plots real QC distributions coloured by sample and condition
#   6. Saves FigureS1_QC_Metrics.png at 300 DPI
#
# QC thresholds shown (same as DP_Diff_Test&Train.R):
#   nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 10
# ============================================================================

library(Seurat)
library(ggplot2)
library(patchwork)
library(dplyr)

set.seed(42)

# ---- Paths ---------------------------------------------------------------
base_dir   <- getwd()   # run from the repository root
data_dir   <- file.path(base_dir, "StrokeData")
output_dir <- file.path(base_dir, "PublicationsFigure")
manuscript_dir <- file.path(base_dir, "manuscript", "generic_latex", "figures")

if (!dir.exists(output_dir))     dir.create(output_dir,     recursive = TRUE)
if (!dir.exists(manuscript_dir)) dir.create(manuscript_dir, recursive = TRUE)

# ---- Fast path: reuse the committed per-cell QC table if present ----------
# qc_metrics_per_cell.csv is written by this script from the raw 10x data and
# is distributed with the repository. If it exists, the figure is regenerated
# from it directly, so the ~250 MB raw download is not needed for this step.
qc_csv <- file.path(base_dir, "qc_metrics_per_cell.csv")
if (file.exists(qc_csv)) {
  cat(sprintf("Using existing QC table: %s\n", qc_csv))
  qc_df <- read.csv(qc_csv, stringsAsFactors = FALSE)
  qc_df$passes_qc <- as.logical(qc_df$passes_qc)
  cat(sprintf("  %d cells loaded\n", nrow(qc_df)))
} else {
  # ---- Confirm all 6 sample folders exist ----------------------------------
  samples <- list(
    list(name = "sham1", condition = "Control"),
    list(name = "sham2", condition = "Control"),
    list(name = "sham3", condition = "Control"),
    list(name = "mcao1", condition = "Stroke"),
    list(name = "mcao2", condition = "Stroke"),
    list(name = "mcao3", condition = "Stroke")
  )

  for (s in samples) {
    p <- file.path(data_dir, s$name)
    if (!dir.exists(p)) {
      stop(paste("ERROR: Sample folder not found:", p))
    }
    required <- c("barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz")
    missing  <- required[!file.exists(file.path(p, required))]
    if (length(missing) > 0) {
      stop(paste("ERROR: Missing files in", p, ":", paste(missing, collapse = ", ")))
    }
    cat(sprintf("✓ Found: %s\n", p))
  }

  # ---- Load each sample with same params as DP_Diff_Test&Train.R ----------
  cat("\nLoading samples...\n")

  load_sample <- function(name, condition) {
    path <- file.path(data_dir, name)
    cat(sprintf("  Reading: %s\n", path))
    data <- Read10X(data.dir = path)
    obj  <- CreateSeuratObject(
      counts      = data,
      project     = name,
      min.cells   = 3,      # same as DP_Diff_Test&Train.R line 15
      min.features = 200    # same as DP_Diff_Test&Train.R line 15
    )
    obj$sample    <- name
    obj$condition <- condition
    return(obj)
  }

  sham1 <- load_sample("sham1", "Control")
  sham2 <- load_sample("sham2", "Control")
  sham3 <- load_sample("sham3", "Control")
  mcao1 <- load_sample("mcao1", "Stroke")
  mcao2 <- load_sample("mcao2", "Stroke")
  mcao3 <- load_sample("mcao3", "Stroke")

  cat(sprintf("\nCells loaded per sample (after min.cells=3, min.features=200):\n"))
  for (obj in list(sham1, sham2, sham3, mcao1, mcao2, mcao3)) {
    cat(sprintf("  %s: %d cells\n", unique(obj$sample), ncol(obj)))
  }

  # ---- Merge all 6 samples -------------------------------------------------
  cat("\nMerging all 6 samples...\n")
  all_samples <- merge(
    sham1,
    y = list(sham2, sham3, mcao1, mcao2, mcao3),
    add.cell.ids = c("sham1", "sham2", "sham3", "mcao1", "mcao2", "mcao3"),
    project = "StrokeProject"
  )
  cat(sprintf("Merged: %d cells, %d genes\n", ncol(all_samples), nrow(all_samples)))

  # ---- Compute percent mitochondrial (same as DP_Diff_Test&Train.R line 41) --
  cat("\nComputing percent.mt...\n")
  all_samples[["percent.mt"]] <- PercentageFeatureSet(all_samples, pattern = "^mt-")

  # ---- Extract real QC metrics per cell ------------------------------------
  cat("Extracting QC metrics per cell...\n")
  qc_df <- all_samples@meta.data %>%
    mutate(
      cell_id      = rownames(.),
      nFeature_RNA = nFeature_RNA,
      nCount_RNA   = nCount_RNA,
      percent_mt   = percent.mt,
      sample       = sample,
      condition    = condition,
      passes_qc    = (nFeature_RNA > 200 &
                      nFeature_RNA < 2500 &
                      percent.mt   < 10)
    ) %>%
    select(cell_id, sample, condition, nFeature_RNA, nCount_RNA, percent_mt, passes_qc)

  # Save real QC metrics CSV
  qc_csv <- file.path(base_dir, "qc_metrics_per_cell.csv")
  write.csv(qc_df, qc_csv, row.names = FALSE)
  cat(sprintf("✓ Saved real QC metrics: %s (%d cells)\n", qc_csv, nrow(qc_df)))
}


# Summary
cat(sprintf("\nQC Summary:\n"))
cat(sprintf("  Total cells before QC:  %d\n", nrow(qc_df)))
cat(sprintf("  Cells passing QC:       %d\n", sum(qc_df$passes_qc)))
cat(sprintf("  Cells removed by QC:    %d\n", sum(!qc_df$passes_qc)))
for (s in unique(qc_df$sample)) {
  n_pass <- sum(qc_df$passes_qc & qc_df$sample == s)
  n_all  <- sum(qc_df$sample == s)
  cat(sprintf("    %s: %d / %d pass QC\n", s, n_pass, n_all))
}

# Use only QC-passing cells for the figure (same cells used in analysis)
# Plot ALL cells (before filtering) so the reader can see the tails that
# the thresholds remove. The dashed lines then mark exactly where the cuts fall.
qc_pass <- qc_df
cat(sprintf("\nPlotting %d cells (all cells, before QC filtering)\n", nrow(qc_pass)))

# ---- Sample colour palette -----------------------------------------------
sample_colors <- c(
  "sham1" = "#3498db",
  "sham2" = "#2980b9",
  "sham3" = "#1a5276",
  "mcao1" = "#e74c3c",
  "mcao2" = "#c0392b",
  "mcao3" = "#7b241c"
)

# ---- Publication theme ---------------------------------------------------
theme_pub <- theme_classic() +
  theme(
    axis.text        = element_text(size = 10, color = "black"),
    axis.title       = element_text(size = 12, face = "bold"),
    plot.title       = element_text(size = 13, face = "bold", hjust = 0.5),
    legend.text      = element_text(size = 9),
    legend.title     = element_text(size = 10, face = "bold"),
    panel.grid.major.y = element_line(color = "grey92")
  )

# ---- Panel A: Genes per cell (nFeature_RNA) ------------------------------
p1 <- ggplot(qc_pass, aes(x = sample, y = nFeature_RNA, fill = sample)) +
  geom_violin(alpha = 0.85, scale = "width") +
  geom_boxplot(width = 0.12, fill = "white",
               outlier.size = 0.3, outlier.alpha = 0.3) +
  scale_fill_manual(values = sample_colors) +
  geom_hline(yintercept = 200,  linetype = "dashed",
             color = "grey40", linewidth = 0.7) +
  geom_hline(yintercept = 2500, linetype = "dashed",
             color = "grey40", linewidth = 0.7) +
  annotate("text", x = 0.55, y = 220,
           label = "200 min", hjust = 0, size = 2.8, color = "grey40") +
  annotate("text", x = 0.55, y = 2520,
           label = "2500 max", hjust = 0, size = 2.8, color = "grey40") +
  scale_x_discrete(limits = c("sham1","sham2","sham3","mcao1","mcao2","mcao3")) +
  labs(x = "", y = "Genes per Cell",
       title = "A. Genes per Cell (nFeature_RNA)",
       fill = "Sample") +
  theme_pub

# ---- Panel B: UMIs per cell (nCount_RNA, log10 scale) --------------------
p2 <- ggplot(qc_pass, aes(x = sample, y = nCount_RNA, fill = sample)) +
  geom_violin(alpha = 0.85, scale = "width") +
  geom_boxplot(width = 0.12, fill = "white",
               outlier.size = 0.3, outlier.alpha = 0.3) +
  scale_fill_manual(values = sample_colors) +
  scale_y_log10() +
  scale_x_discrete(limits = c("sham1","sham2","sham3","mcao1","mcao2","mcao3")) +
  labs(x = "", y = "Total UMIs per Cell (log10)",
       title = "B. Total UMI Counts (nCount_RNA)",
       fill = "Sample") +
  theme_pub

# ---- Panel C: Mitochondrial percentage -----------------------------------
p3 <- ggplot(qc_pass, aes(x = sample, y = percent_mt, fill = sample)) +
  geom_violin(alpha = 0.85, scale = "width") +
  geom_boxplot(width = 0.12, fill = "white",
               outlier.size = 0.3, outlier.alpha = 0.3) +
  scale_fill_manual(values = sample_colors) +
  geom_hline(yintercept = 10, linetype = "dashed",
             color = "grey40", linewidth = 0.7) +
  annotate("text", x = 0.55, y = 10.3,
           label = "10% threshold", hjust = 0, size = 2.8, color = "grey40") +
  scale_x_discrete(limits = c("sham1","sham2","sham3","mcao1","mcao2","mcao3")) +
  labs(x = "Sample", y = "Mitochondrial Reads (%)",
       title = "C. Mitochondrial Content (%MT)",
       fill = "Sample") +
  theme_pub

# ---- Combine panels ------------------------------------------------------
fig_s1 <- (p1 | p2 | p3) +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = "Supplementary Figure S1: Quality Control Metrics",
    subtitle = paste0(
      sprintf("All 6 samples | %d cells before QC, %d retained (nFeature 200\u20132500, %%MT < 10%%)",
              nrow(qc_pass), sum(qc_pass$passes_qc)),
      " | Dashed lines show QC thresholds"
    ),
    theme = theme(
      plot.title    = element_text(size = 16, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 10, hjust = 0.5, color = "grey40")
    )
  )

# ---- Save at 300 DPI -----------------------------------------------------
out1 <- file.path(output_dir,     "FigureS1_QC_Metrics.png")
out2 <- file.path(manuscript_dir, "FigureS1_QC_Metrics.png")

ggsave(out1, fig_s1, width = 16, height = 5.5, dpi = 300, bg = "white")
cat(sprintf("\n✓ Saved: %s\n", out1))

ggsave(out2, fig_s1, width = 16, height = 5.5, dpi = 300, bg = "white")
cat(sprintf("✓ Saved: %s\n", out2))

cat("\n=== FigureS1 COMPLETE ===\n")
cat("Data source: Real 10X raw data via Read10X() — same parameters as DP_Diff_Test&Train.R\n")
cat("No simulation used. Per-cell QC values saved to: qc_metrics_per_cell.csv\n")
cat(sprintf("Total cells plotted: %d (all cells, before QC)\n", nrow(qc_pass)))
