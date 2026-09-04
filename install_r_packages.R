#!/usr/bin/env Rscript
# ============================================================================
# R dependency installer
# ============================================================================
# Installs every CRAN and Bioconductor package used by the R scripts in this
# repository.
#
#   Rscript install_r_packages.R
#
# Tested with R 4.4.x and R 4.5.x.
#
# SEURAT VERSION: v5 is required.
# DP_Diff_Test_Train.R calls `LayerData(obj, assay = "RNA", layer = "scale.data")`,
# which was introduced in Seurat v5 / SeuratObject v5. Under Seurat v4 that
# function does not exist and the test-set PCA projection step will fail; the
# v4 equivalent is `GetAssayData(obj, slot = "scale.data")`.
#
# Note that the manuscript's Methods section states "Seurat v4.0". That text
# and this code disagree; see the "Known discrepancies" section of README.md.
# ============================================================================

cran_packages <- c(
  "Seurat",          # scRNA-seq preprocessing, QC, PCA  (v5.x required)
  "dplyr",
  "tidyverse",
  "ggplot2",
  "patchwork",
  "RColorBrewer",
  "circlize"
)

bioc_packages <- c(
  "clusterProfiler", # GO / KEGG over-representation analysis
  "org.Mm.eg.db",    # mouse gene annotation
  "enrichplot",      # dotplots, cnetplots
  "ComplexHeatmap"   # gene-pathway heatmap
)

# Not required. `pathview` draws annotated KEGG pathway diagrams. Earlier
# versions of this repository shipped five such diagrams, but no KEGG pathway
# is significantly enriched once the correct background is applied
# (see KEGG_Enrichment_Pathways/README.md), so those diagrams were removed and
# keggPathways.R does not call pathview. Install it only if you want to draw
# pathway maps yourself.
optional_bioc <- c("pathview")

message("Installing CRAN packages ...")
for (pkg in cran_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message("  installing ", pkg)
    install.packages(pkg, repos = "https://cloud.r-project.org")
  } else {
    message("  already present: ", pkg)
  }
}

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager", repos = "https://cloud.r-project.org")
}

message("\nInstalling Bioconductor packages ...")
for (pkg in bioc_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message("  installing ", pkg)
    BiocManager::install(pkg, update = FALSE, ask = FALSE)
  } else {
    message("  already present: ", pkg)
  }
}

message("\n--- Verifying ---")
all_pkgs <- c(cran_packages, bioc_packages)
missing <- all_pkgs[!vapply(all_pkgs, requireNamespace, logical(1), quietly = TRUE)]

if (length(missing) == 0) {
  message("All ", length(all_pkgs), " packages available.")
} else {
  message("STILL MISSING: ", paste(missing, collapse = ", "))
  quit(status = 1)
}

if (requireNamespace("Seurat", quietly = TRUE)) {
  v <- as.character(utils::packageVersion("Seurat"))
  message("Seurat version: ", v)
  if (substr(v, 1, 1) != "5") {
    message("WARNING: DP_Diff_Test_Train.R uses LayerData(), which requires ",
            "Seurat v5. Version ", v, " is installed and the test-set PCA ",
            "projection step will fail.")
  }
}
