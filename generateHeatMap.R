#!/usr/bin/env Rscript

# ============================================================================
# GENE-PATHWAY HEATMAP GENERATOR (FIXED)
# ============================================================================
# Purpose: Create heatmap showing which genes appear in which GO pathways
# Input: GO enrichment CSV files
# Output: Publication-quality heatmap (PDF and PNG)
# ============================================================================
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

packages <- c("ComplexHeatmap", "circlize")
for (pkg in packages) {
    if (!requireNamespace(pkg, quietly = TRUE))
        BiocManager::install(pkg, ask = FALSE)
}
# Load required libraries
library(tidyverse)
library(ComplexHeatmap)
library(circlize)
library(RColorBrewer)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

# Read the GO enrichment results.
# Run from the repository root (inputs in geneGoEnrichment/) or from inside
# geneGoEnrichment/ itself; both are supported.
find_go <- function(fname) {
    for (p in c(file.path("geneGoEnrichment", fname), fname)) {
        if (file.exists(p)) return(p)
    }
    stop("Could not find ", fname, ". Run go_enrichment.R first.")
}
go_bp <- read.csv(find_go("go_enrichment_BP.csv"), stringsAsFactors = FALSE)
go_mf <- read.csv(find_go("go_enrichment_MF.csv"), stringsAsFactors = FALSE)

# Write outputs into heatMap/ when run from the repository root.
out_dir <- if (dir.exists("heatMap")) "heatMap" else "."
op <- function(fname) file.path(out_dir, fname)

# ============================================================================
# 2. SELECT TOP PATHWAYS
# ============================================================================

# Select top 30 biological processes (can be adjusted)
top_pathways_bp <- go_bp %>%
  head(30) %>%
  select(ID, Description, geneID, p.adjust)

# Select top 15 molecular functions
top_pathways_mf <- go_mf %>%
  head(15) %>%
  select(ID, Description, geneID, p.adjust)

# Combine BP and MF
top_pathways <- bind_rows(
  top_pathways_bp %>% mutate(Category = "Biological Process"),
  top_pathways_mf %>% mutate(Category = "Molecular Function")
)

# ============================================================================
# 3. CREATE GENE-PATHWAY MATRIX
# ============================================================================

# Split gene IDs and create long format
pathway_genes <- top_pathways %>%
  mutate(genes = strsplit(geneID, "/")) %>%
  unnest(genes) %>%
  select(Description, genes, p.adjust, Category)

# Get all unique genes
all_genes <- unique(pathway_genes$genes)

# Create matrix: rows = genes, columns = pathways
gene_pathway_matrix <- pathway_genes %>%
  mutate(present = 1) %>%
  pivot_wider(
    id_cols = genes,
    names_from = Description,
    values_from = present,
    values_fill = 0
  )

# Convert to matrix
genes <- gene_pathway_matrix$genes
pathway_matrix <- as.matrix(gene_pathway_matrix[, -1])
rownames(pathway_matrix) <- genes

# ============================================================================
# 4. CLEAN COLUMN NAMES EARLY
# ============================================================================

# Clean column names to remove special characters that might cause issues
clean_colnames <- gsub("[^[:alnum:][:space:]-]", "", colnames(pathway_matrix))
clean_colnames <- gsub("\\s+", " ", clean_colnames)  # normalize whitespace
colnames(pathway_matrix) <- clean_colnames

# Update pathway descriptions in top_pathways to match cleaned names
top_pathways$Description <- clean_colnames

# ============================================================================
# 5. ANNOTATE WITH P-VALUES AND CATEGORIES
# ============================================================================

# Create column annotation (pathway p-values) - NOW with cleaned names
pathway_pvalues <- top_pathways %>%
  select(Description, p.adjust, Category) %>%
  distinct()

# Ensure order matches the matrix columns
pathway_pvalues <- pathway_pvalues %>%
  arrange(match(Description, colnames(pathway_matrix)))

# Create color mapping for p-values
col_fun_pval <- colorRamp2(
  c(1e-15, 1e-10, 1e-5, 0.05),
  c("#67001f", "#d6604d", "#fddbc7", "#f7f7f7")
)

# Category colors
category_colors <- c(
  "Biological Process" = "#4575b4",
  "Molecular Function" = "#d73027"
)

# Column annotation
col_annotation <- HeatmapAnnotation(
  `-log10(p.adjust)` = -log10(pathway_pvalues$p.adjust),
  Category = pathway_pvalues$Category,
  col = list(
    `-log10(p.adjust)` = col_fun_pval,
    Category = category_colors
  ),
  annotation_name_side = "left"
)

# ============================================================================
# 6. COUNT GENE FREQUENCY (ROW ANNOTATION)
# ============================================================================

# Count how many pathways each gene appears in
gene_frequency <- rowSums(pathway_matrix)

# Create row annotation
row_annotation <- rowAnnotation(
  `Pathway Count` = gene_frequency,
  col = list(`Pathway Count` = colorRamp2(c(1, 5, 10), c("#f7f7f7", "#fddbc7", "#d6604d"))),
  annotation_name_side = "top"
)

# ============================================================================
# 7. IDENTIFY STAR GENES
# ============================================================================

# Star genes (appear in many pathways)
star_genes <- names(gene_frequency[gene_frequency >= 5])

# Use asterisk (*) instead of star emoji for PDF compatibility
row_labels <- rownames(pathway_matrix)
row_labels[rownames(pathway_matrix) %in% star_genes] <-
  paste0(row_labels[rownames(pathway_matrix) %in% star_genes], " *")

# ============================================================================
# 8. CREATE HEATMAP
# ============================================================================

# Color for presence/absence
col_binary <- c("0" = "white", "1" = "#2166ac")

# Create main heatmap
ht <- Heatmap(
  pathway_matrix,
  name = "Gene Present",

  # Colors
  col = col_binary,

  # Clustering
  cluster_rows = TRUE,
  cluster_columns = TRUE,
  clustering_distance_rows = "binary",
  clustering_distance_columns = "binary",

  # Row settings
  row_labels = row_labels,
  row_names_side = "left",
  row_names_gp = gpar(fontsize = 8),
  show_row_dend = TRUE,
  row_dend_width = unit(2, "cm"),

  # Column settings - use cleaned names automatically
  column_names_side = "bottom",
  column_names_gp = gpar(fontsize = 7),
  column_names_rot = 45,
  show_column_dend = TRUE,
  column_dend_height = unit(2, "cm"),

  # Annotations
  top_annotation = col_annotation,
  right_annotation = row_annotation,

  # Title
  column_title = "Gene-Pathway Association Heatmap",
  column_title_gp = gpar(fontsize = 14, fontface = "bold"),

  # Legend
  heatmap_legend_param = list(
    title = "Gene Present",
    at = c(0, 1),
    labels = c("Absent", "Present")
  ),

  # Additional stability options
  use_raster = FALSE
)

# ============================================================================
# 9. SAVE HEATMAP
# ============================================================================

# PDF version (publication quality)
pdf(op("gene_pathway_heatmap.pdf"), width = 16, height = 12)
tryCatch({
  draw(ht, heatmap_legend_side = "right", annotation_legend_side = "right")
}, error = function(e) {
  cat("Error drawing heatmap:", conditionMessage(e), "\n")
})
dev.off()

# PNG version (for presentations)
png(op("gene_pathway_heatmap.png"), width = 1600, height = 1200, res = 150)
tryCatch({
  draw(ht, heatmap_legend_side = "right", annotation_legend_side = "right")
}, error = function(e) {
  cat("Error drawing heatmap:", conditionMessage(e), "\n")
})
dev.off()

cat("✅ Heatmaps created successfully!\n")
cat("   - gene_pathway_heatmap.pdf\n")
cat("   - gene_pathway_heatmap.png\n")

# ============================================================================
# 10. CREATE SIMPLIFIED VERSION (TOP GENES ONLY)
# ============================================================================

# Select genes that appear in ≥3 pathways
frequent_genes <- names(gene_frequency[gene_frequency >= 3])

# Subset matrix
pathway_matrix_subset <- pathway_matrix[frequent_genes, ]

# Recreate annotations for subset
gene_frequency_subset <- rowSums(pathway_matrix_subset)

row_annotation_subset <- rowAnnotation(
  `Pathway Count` = gene_frequency_subset,
  col = list(`Pathway Count` = colorRamp2(c(3, 6, 10), c("#f7f7f7", "#fddbc7", "#d6604d"))),
  annotation_name_side = "top"
)

# Star genes in subset
star_genes_subset <- names(gene_frequency_subset[gene_frequency_subset >= 5])
row_labels_subset <- rownames(pathway_matrix_subset)
row_labels_subset[rownames(pathway_matrix_subset) %in% star_genes_subset] <-
  paste0(row_labels_subset[rownames(pathway_matrix_subset) %in% star_genes_subset], " *")

# Create simplified heatmap
ht_simple <- Heatmap(
  pathway_matrix_subset,
  name = "Gene Present",
  col = col_binary,
  cluster_rows = TRUE,
  cluster_columns = TRUE,
  clustering_distance_rows = "binary",
  clustering_distance_columns = "binary",
  row_labels = row_labels_subset,
  row_names_side = "left",
  row_names_gp = gpar(fontsize = 10),
  show_row_dend = TRUE,
  row_dend_width = unit(2, "cm"),
  column_names_side = "bottom",
  column_names_gp = gpar(fontsize = 8),
  column_names_rot = 45,
  show_column_dend = TRUE,
  column_dend_height = unit(2, "cm"),
  top_annotation = col_annotation,
  right_annotation = row_annotation_subset,
  column_title = "Gene-Pathway Association (Frequent Genes ≥3 Pathways)",
  column_title_gp = gpar(fontsize = 14, fontface = "bold"),
  heatmap_legend_param = list(
    title = "Gene Present",
    at = c(0, 1),
    labels = c("Absent", "Present")
  ),
  use_raster = FALSE
)

# Save simplified version
pdf(op("gene_pathway_heatmap_simplified.pdf"), width = 16, height = 10)
tryCatch({
  draw(ht_simple, heatmap_legend_side = "right", annotation_legend_side = "right")
}, error = function(e) {
  cat("Error drawing simplified heatmap:", conditionMessage(e), "\n")
})
dev.off()

png(op("gene_pathway_heatmap_simplified.png"), width = 1600, height = 1000, res = 150)
tryCatch({
  draw(ht_simple, heatmap_legend_side = "right", annotation_legend_side = "right")
}, error = function(e) {
  cat("Error drawing simplified heatmap:", conditionMessage(e), "\n")
})
dev.off()

cat("✅ Simplified heatmaps created!\n")
cat("   - gene_pathway_heatmap_simplified.pdf\n")
cat("   - gene_pathway_heatmap_simplified.png\n")

# ============================================================================
# 11. GENERATE SUMMARY STATISTICS
# ============================================================================

cat("\n📊 HEATMAP STATISTICS:\n")
cat("======================\n")
cat(sprintf("Total genes: %d\n", nrow(pathway_matrix)))
cat(sprintf("Total pathways: %d\n", ncol(pathway_matrix)))
cat(sprintf("Genes in ≥5 pathways (*): %d\n", length(star_genes)))
cat(sprintf("Genes in ≥3 pathways: %d\n", length(frequent_genes)))

cat("\n* STAR GENES (appear in ≥5 pathways):\n")
star_gene_counts <- gene_frequency[star_genes]
star_gene_df <- data.frame(
  Gene = names(star_gene_counts),
  Pathway_Count = star_gene_counts
) %>% arrange(desc(Pathway_Count))

print(star_gene_df)

# Save star genes table
write.csv(star_gene_df, op("star_genes_pathway_counts.csv"), row.names = FALSE)
cat("\n✅ Star genes table saved: star_genes_pathway_counts.csv\n")

# ============================================================================
# 12. CREATE GENE-CENTRIC VIEW
# ============================================================================

# For each star gene, list all pathways it appears in
gene_pathway_list <- list()

for (gene in star_genes) {
  pathways <- colnames(pathway_matrix)[pathway_matrix[gene, ] == 1]
  gene_pathway_list[[gene]] <- pathways
}

# Save as readable text file
sink(op("star_genes_pathway_details.txt"))
cat("STAR GENES AND THEIR PATHWAYS\n")
cat("==============================\n\n")

for (gene in names(gene_pathway_list)) {
  cat(sprintf("%s * (appears in %d pathways)\n", gene, length(gene_pathway_list[[gene]])))
  cat("  Pathways:\n")
  for (pathway in gene_pathway_list[[gene]]) {
    cat(sprintf("    - %s\n", pathway))
  }
  cat("\n")
}
sink()

cat("✅ Star gene details saved: star_genes_pathway_details.txt\n")

cat("\n🎉 All heatmaps and summaries generated successfully!\n")