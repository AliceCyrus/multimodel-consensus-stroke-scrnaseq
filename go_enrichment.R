# GO Enrichment Analysis for Consensus Genes
# ==========================================

cat("Starting GO enrichment analysis...\n")

# Install packages if needed
cat("\nChecking/installing packages...\n")
if (!require("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager")
}

if (!require("clusterProfiler", quietly = TRUE)) {
    BiocManager::install("clusterProfiler", ask = FALSE)
}

if (!require("org.Mm.eg.db", quietly = TRUE)) {
    BiocManager::install("org.Mm.eg.db", ask = FALSE)
}

if (!require("enrichplot", quietly = TRUE)) {
    BiocManager::install("enrichplot", ask = FALSE)
}

# Load libraries
cat("\nLoading libraries...\n")
library(clusterProfiler)
library(org.Mm.eg.db)
library(enrichplot)

cat("✓ Libraries loaded\n")

# Check file exists and load
cat("\nLooking for consensus genes file...\n")

# Try different possible paths
possible_paths <- c(
    "/content/gene_level_results/consensus_genes.csv",
    "/content/consensus_genes.csv",
    "gene_level_results/consensus_genes.csv",
    "consensus_genes.csv"
)

file_found <- FALSE
for (path in possible_paths) {
    if (file.exists(path)) {
        cat(sprintf("✓ Found file at: %s\n", path))
        consensus <- read.csv(path, stringsAsFactors = FALSE)
        file_found <- TRUE
        break
    }
}

if (!file_found) {
    cat("❌ ERROR: consensus_genes.csv not found!\n")
    cat("\nSearched in:\n")
    for (path in possible_paths) {
        cat(sprintf("  • %s\n", path))
    }
    cat("\nCurrent working directory:\n")
    print(getwd())
    cat("\nFiles in current directory:\n")
    print(list.files())
    stop("File not found")
}

# Check data structure
cat("\nData structure:\n")
print(str(consensus))
cat("\nFirst few rows:\n")
print(head(consensus))

# Extract gene column
if (!"gene" %in% colnames(consensus)) {
    cat("❌ ERROR: 'gene' column not found!\n")
    cat("Available columns:\n")
    print(colnames(consensus))
    stop("Missing 'gene' column")
}

genes <- consensus$gene

# Remove any NA or empty values
genes <- genes[!is.na(genes) & genes != ""]

cat(sprintf("\n✓ Loaded %d genes\n", length(genes)))
cat("\nFirst 10 genes:\n")
print(head(genes, 10))

# Convert to Entrez IDs
cat("\nConverting gene symbols to Entrez IDs...\n")
gene_entrez <- bitr(
    genes,
    fromType = "SYMBOL",
    toType = "ENTREZID",
    OrgDb = org.Mm.eg.db
)

cat(sprintf("✓ Converted %d out of %d genes\n", nrow(gene_entrez), length(genes)))

if (nrow(gene_entrez) == 0) {
    stop("No genes could be converted to Entrez IDs")
}

# Check for duplicates
if (any(duplicated(gene_entrez$ENTREZID))) {
    cat("Removing duplicate Entrez IDs...\n")
    gene_entrez <- gene_entrez[!duplicated(gene_entrez$ENTREZID), ]
}

cat(sprintf("Using %d unique Entrez IDs for enrichment\n", nrow(gene_entrez)))

# ============================================================================
# BACKGROUND UNIVERSE
# ============================================================================
# Over-representation analysis compares the input list against a background.
# The consensus genes were selected by back-projection through the PCA
# loadings, which are defined only over the 3,000 highly variable genes.
# A gene outside that set had zero probability of ever being selected, so the
# HVG set - not the whole transcriptome - is the correct universe. Leaving
# `universe` unset makes clusterProfiler default to every annotated gene in
# org.Mm.eg.db (~25,700), which inflates enrichment significance.
# See Huang et al. 2009 (NAR) and Timmons et al. 2015 (Genome Biol).

hvg_paths <- c("pca_loadings.csv", "../pca_loadings.csv")
hvg_file <- NULL
for (p in hvg_paths) if (file.exists(p)) { hvg_file <- p; break }
if (is.null(hvg_file)) {
    stop("pca_loadings.csv not found - it defines the 3,000-HVG background universe.")
}

hvg_genes <- read.csv(hvg_file, stringsAsFactors = FALSE)$gene
hvg_genes <- unique(hvg_genes[!is.na(hvg_genes) & hvg_genes != ""])
cat(sprintf("\nBackground: %d highly variable genes from %s\n",
            length(hvg_genes), hvg_file))

hvg_entrez <- suppressWarnings(suppressMessages(bitr(
    hvg_genes, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Mm.eg.db)))
hvg_entrez <- hvg_entrez[!duplicated(hvg_entrez$ENTREZID), ]
background_ids <- hvg_entrez$ENTREZID
cat(sprintf("Background universe: %d unique Entrez IDs\n", length(background_ids)))

# ============================================================================
# CREATE OUTPUT DIRECTORY: geneGoEnrichment/
# ============================================================================
output_dir <- "geneGoEnrichment"   # relative to the repository root
if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
    cat(sprintf("\n✓ Created directory: %s\n", output_dir))
} else {
    cat(sprintf("\n✓ Using directory: %s\n", output_dir))
}

# GO enrichment - Biological Process
cat("\n" , rep("=", 70), "\n", sep="")
cat("RUNNING GO ENRICHMENT: BIOLOGICAL PROCESS\n")
cat(rep("=", 70), "\n", sep="")

ego_bp <- enrichGO(
    gene = gene_entrez$ENTREZID,
    universe = background_ids,
    OrgDb = org.Mm.eg.db,
    ont = "BP",
    pAdjustMethod = "BH",
    pvalueCutoff = 0.05,
    qvalueCutoff = 0.2,
    minGSSize = 10,
    maxGSSize = 500,
    readable = TRUE
)

cat(sprintf("\n✓ Found %d significant GO:BP terms\n", nrow(ego_bp)))

# GO enrichment - Molecular Function
cat("\n", rep("=", 70), "\n", sep="")
cat("RUNNING GO ENRICHMENT: MOLECULAR FUNCTION\n")
cat(rep("=", 70), "\n", sep="")

ego_mf <- enrichGO(
    gene = gene_entrez$ENTREZID,
    universe = background_ids,
    OrgDb = org.Mm.eg.db,
    ont = "MF",
    pAdjustMethod = "BH",
    pvalueCutoff = 0.05,
    qvalueCutoff = 0.2,
    minGSSize = 10,
    maxGSSize = 500,
    readable = TRUE
)

cat(sprintf("\n✓ Found %d significant GO:MF terms\n", nrow(ego_mf)))

# GO enrichment - Cellular Component
cat("\n", rep("=", 70), "\n", sep="")
cat("RUNNING GO ENRICHMENT: CELLULAR COMPONENT\n")
cat(rep("=", 70), "\n", sep="")

ego_cc <- enrichGO(
    gene = gene_entrez$ENTREZID,
    universe = background_ids,
    OrgDb = org.Mm.eg.db,
    ont = "CC",
    pAdjustMethod = "BH",
    pvalueCutoff = 0.05,
    qvalueCutoff = 0.2,
    minGSSize = 10,
    maxGSSize = 500,
    readable = TRUE
)

cat(sprintf("\n✓ Found %d significant GO:CC terms\n", nrow(ego_cc)))

# Display top results
if (nrow(ego_bp) > 0) {
    cat("\n", rep("=", 70), "\n", sep="")
    cat("TOP 10 BIOLOGICAL PROCESSES\n")
    cat(rep("=", 70), "\n", sep="")
    print(head(as.data.frame(ego_bp)[, c("Description", "pvalue", "Count")], 10))
}

if (nrow(ego_mf) > 0) {
    cat("\n", rep("=", 70), "\n", sep="")
    cat("TOP 10 MOLECULAR FUNCTIONS\n")
    cat(rep("=", 70), "\n", sep="")
    print(head(as.data.frame(ego_mf)[, c("Description", "pvalue", "Count")], 10))
}

# Save results
cat("\n", rep("=", 70), "\n", sep="")
cat("SAVING RESULTS\n")
cat(rep("=", 70), "\n", sep="")

write.csv(as.data.frame(ego_bp),
          file.path(output_dir, "go_enrichment_BP.csv"),
          row.names = FALSE)
cat("✓ Saved: go_enrichment_BP.csv\n")

write.csv(as.data.frame(ego_mf),
          file.path(output_dir, "go_enrichment_MF.csv"),
          row.names = FALSE)
cat("✓ Saved: go_enrichment_MF.csv\n")

write.csv(as.data.frame(ego_cc),
          file.path(output_dir, "go_enrichment_CC.csv"),
          row.names = FALSE)
cat("✓ Saved: go_enrichment_CC.csv\n")

# Create plots
cat("\n", rep("=", 70), "\n", sep="")
cat("CREATING PLOTS\n")
cat(rep("=", 70), "\n", sep="")

# Biological Process dotplot
if (nrow(ego_bp) > 0) {
    tryCatch({
        pdf(file.path(output_dir, "go_enrichment_BP_dotplot.pdf"),
            width = 12, height = 10)
        print(dotplot(ego_bp, showCategory = 20, title = "GO Biological Process"))
        dev.off()
        cat("✓ Saved: go_enrichment_BP_dotplot.pdf\n")
    }, error = function(e) {
        cat("✗ Error creating BP dotplot:", e$message, "\n")
    })

    tryCatch({
        pdf(file.path(output_dir, "go_enrichment_BP_barplot.pdf"),
            width = 12, height = 10)
        print(barplot(ego_bp, showCategory = 20))
        dev.off()
        cat("✓ Saved: go_enrichment_BP_barplot.pdf\n")
    }, error = function(e) {
        cat("✗ Error creating BP barplot:", e$message, "\n")
    })
}

# Molecular Function dotplot
if (nrow(ego_mf) > 0) {
    tryCatch({
        pdf(file.path(output_dir, "go_enrichment_MF_dotplot.pdf"),
            width = 12, height = 10)
        print(dotplot(ego_mf, showCategory = 20, title = "GO Molecular Function"))
        dev.off()
        cat("✓ Saved: go_enrichment_MF_dotplot.pdf\n")
    }, error = function(e) {
        cat("✗ Error creating MF dotplot:", e$message, "\n")
    })
}

# Cellular Component dotplot
if (nrow(ego_cc) > 0) {
    tryCatch({
        pdf(file.path(output_dir, "go_enrichment_CC_dotplot.pdf"),
            width = 12, height = 10)
        print(dotplot(ego_cc, showCategory = 20, title = "GO Cellular Component"))
        dev.off()
        cat("✓ Saved: go_enrichment_CC_dotplot.pdf\n")
    }, error = function(e) {
        cat("✗ Error creating CC dotplot:", e$message, "\n")
    })
}

# ============================================================================
# PART 2: PATHWAY ANALYSIS
# ============================================================================

cat("\n", rep("=", 70), "\n", sep="")
cat("ANALYZING SPECIFIC PATHWAYS\n")
cat(rep("=", 70), "\n", sep="")

# Load the BP enrichment results
bp_results <- read.csv(file.path(output_dir, "go_enrichment_BP.csv"))

# Get genes in top pathway
top_pathway <- bp_results[1, ]
cat("\nTop pathway:", top_pathway$Description, "\n")
cat("Genes involved:", top_pathway$geneID, "\n\n")

# Split the gene list
genes_in_pathway <- strsplit(as.character(top_pathway$geneID), "/")[[1]]
cat("Individual genes:\n")
for(gene in genes_in_pathway) {
  cat(sprintf("  • %s\n", gene))
}

# Get genes in leukocyte adhesion
adhesion_pathway <- bp_results[bp_results$Description == "leukocyte cell-cell adhesion", ]
if(nrow(adhesion_pathway) > 0) {
  cat("\n\nLeukocyte adhesion genes:\n")
  adhesion_genes <- strsplit(as.character(adhesion_pathway$geneID), "/")[[1]]
  for(gene in adhesion_genes) {
    cat(sprintf("  • %s\n", gene))
  }
}

# Get genes in cytokine receptor binding
cytokine_mf <- read.csv(file.path(output_dir, "go_enrichment_MF.csv"))
cytokine_pathway <- cytokine_mf[cytokine_mf$Description == "cytokine receptor binding", ]
if(nrow(cytokine_pathway) > 0) {
  cat("\n\nCytokine receptor binding genes:\n")
  cytokine_genes <- strsplit(as.character(cytokine_pathway$geneID), "/")[[1]]
  for(gene in cytokine_genes) {
    cat(sprintf("  • %s\n", gene))
  }
}

# ============================================================================
# PART 3: CREATE TABLE FOR MANUSCRIPT
# ============================================================================

cat("\n", rep("=", 70), "\n", sep="")
cat("CREATING MANUSCRIPT TABLE\n")
cat(rep("=", 70), "\n", sep="")

# Create Table for Manuscript
bp_top <- read.csv(file.path(output_dir, "go_enrichment_BP.csv"))
bp_top <- bp_top[1:10, c("Description", "pvalue", "Count", "GeneRatio")]

# Format p-values
bp_top$pvalue <- formatC(bp_top$pvalue, format = "e", digits = 2)

# Save for manuscript
write.csv(bp_top, file.path(output_dir, "Table_S3_GO_enrichment.csv"), row.names=FALSE)
cat("✓ Created: Table_S3_GO_enrichment.csv\n")

# ============================================================================
# CREATE README FILE
# ============================================================================

readme_content <- paste0(
"GO Enrichment Analysis Results
================================

Analysis Date: ", Sys.time(), "
Number of Input Genes: ", length(genes), "
Number of Converted Genes: ", nrow(gene_entrez), "

Files in this directory:
------------------------

1. CSV Files:
   - go_enrichment_BP.csv: Biological Process enrichment results
   - go_enrichment_MF.csv: Molecular Function enrichment results
   - go_enrichment_CC.csv: Cellular Component enrichment results
   - Table_S3_GO_enrichment.csv: Top 10 pathways for manuscript

2. PDF Plots:
   - go_enrichment_BP_dotplot.pdf: BP dotplot visualization
   - go_enrichment_BP_barplot.pdf: BP barplot visualization
   - go_enrichment_MF_dotplot.pdf: MF dotplot visualization
   - go_enrichment_CC_dotplot.pdf: CC dotplot visualization

Results Summary:
----------------
GO:BP terms found: ", nrow(ego_bp), "
GO:MF terms found: ", nrow(ego_mf), "
GO:CC terms found: ", nrow(ego_cc), "

Top 3 Biological Processes:
", paste(head(as.data.frame(ego_bp)$Description, 3), collapse="\n"), "

For questions or issues, contact your bioinformatics team.
"
)

writeLines(readme_content, file.path(output_dir, "README.txt"))
cat("✓ Created: README.txt\n")

# ============================================================================
# CREATE ZIP FILE FOR DOWNLOAD
# ============================================================================

cat("\n", rep("=", 70), "\n", sep="")
cat("CREATING DOWNLOADABLE ZIP FILE\n")
cat(rep("=", 70), "\n", sep="")

zip_file <- "geneGoEnrichment.zip"

# Remove old zip if exists
if (file.exists(zip_file)) {
    file.remove(zip_file)
    cat("Removed old zip file\n")
}

# Create zip file
# Portable: R's own zip() rather than a shell command with a Colab path.
tryCatch(
    utils::zip(zipfile = zip_file,
               files   = list.files(output_dir, full.names = TRUE),
               flags   = "-q"),
    error   = function(e) message("  (zip skipped: ", conditionMessage(e), ")"),
    warning = function(w) message("  (zip skipped: no zip utility on PATH)"))

if (file.exists(zip_file)) {
    file_size <- file.info(zip_file)$size / 1024  # Size in KB
    cat(sprintf("\n✓ Created zip file: geneGoEnrichment.zip (%.1f KB)\n", file_size))
    cat("\n📦Zip archive written to: geneGoEnrichment.zip\n")
} else {
    cat("\n✗ Failed to create zip file\n")
}

# ============================================================================
# FINAL SUMMARY
# ============================================================================

cat("\n", rep("=", 70), "\n", sep="")
cat("✅ ANALYSIS COMPLETE!\n")
cat(rep("=", 70), "\n", sep="")

cat("\nSummary:\n")
cat(sprintf("  • Genes analyzed: %d\n", length(genes)))
cat(sprintf("  • Genes converted: %d\n", nrow(gene_entrez)))
cat(sprintf("  • GO:BP terms: %d\n", nrow(ego_bp)))
cat(sprintf("  • GO:MF terms: %d\n", nrow(ego_mf)))
cat(sprintf("  • GO:CC terms: %d\n", nrow(ego_cc)))

cat("\nOutput directory: geneGoEnrichment/\n")
cat("\nFiles created:\n")
cat("  • go_enrichment_BP.csv\n")
cat("  • go_enrichment_MF.csv\n")
cat("  • go_enrichment_CC.csv\n")
cat("  • go_enrichment_BP_dotplot.pdf\n")
cat("  • go_enrichment_BP_barplot.pdf\n")
cat("  • go_enrichment_MF_dotplot.pdf\n")
cat("  • go_enrichment_CC_dotplot.pdf\n")
cat("  • Table_S3_GO_enrichment.csv\n")
cat("  • README.txt\n")

cat("\n📥 DOWNLOAD:\n")
cat("  •ZIP file: geneGoEnrichment.zip\n")

cat("\n🎉 Done! You can now download the entire folder as a zip file.\n")
cat("\nIn Google Colab, run:\n")
cat("  from google.colab import files\n")
cat("  (on Google Colab: files.download('geneGoEnrichment.zip'))\n")