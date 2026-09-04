#!/usr/bin/env Rscript

# ============================================================================
# KEGG PATHWAY ANALYSIS FOR CONSENSUS STROKE BIOMARKERS
# ============================================================================
#
# Purpose: KEGG over-representation analysis of the consensus gene signature,
#          run under the same conditions as the GO analysis so the two are
#          directly comparable.
#
# Input : gene_level_results/consensus_genes.csv   (63 consensus genes)
# Background : the 3,000 highly variable genes in pca_loadings.csv
# Output: KEGG_Enrichment_Pathways/
#
# ---------------------------------------------------------------------------
# EXPECTED RESULT: NO PATHWAY SURVIVES FDR CORRECTION.
#
# This is a real negative result, not a bug, and the script is committed so it
# can be verified. Two things drive it:
#
#   1. Correct background. Consensus genes are drawn from the 3,000 HVGs, a
#      pool already dominated by immune transcripts because immune activation
#      is what varies most after stroke. Against that honest background the
#      list is no longer unusually immune at the pathway level.
#
#   2. MHC class II saturation. Roughly half of the nominally enriched KEGG
#      pathways are driven entirely by the same three genes - H2-Aa, H2-Ab1,
#      H2-Eb1 - which belong to dozens of KEGG disease annotations. That
#      produces a long list of unrelated diseases (asthma, type I diabetes,
#      graft-versus-host disease) which reflects KEGG's annotation structure
#      rather than stroke biology.
#
# The manuscript therefore reports GO as the primary enrichment analysis and
# states the KEGG result as null. Full per-pathway statistics are written out
# so the negative result is inspectable.
# ============================================================================

suppressPackageStartupMessages({
    library(clusterProfiler)
    library(org.Mm.eg.db)
    library(dplyr)
})

set.seed(42)

cat(strrep("=", 70), "\n", sep = "")
cat("KEGG PATHWAY ENRICHMENT - CONSENSUS STROKE BIOMARKERS\n")
cat(strrep("=", 70), "\n\n", sep = "")

# ============================================================================
# 1. INPUT GENES
# ============================================================================

consensus_paths <- c("gene_level_results/consensus_genes.csv",
                     "consensus_genes.csv")
consensus_file <- NULL
for (p in consensus_paths) if (file.exists(p)) { consensus_file <- p; break }
if (is.null(consensus_file)) {
    stop("consensus_genes.csv not found. Run this script from the repository root.")
}

consensus <- read.csv(consensus_file, stringsAsFactors = FALSE)
if (!"gene" %in% colnames(consensus)) stop("Expected a 'gene' column in ", consensus_file)

consensus_genes <- consensus$gene
consensus_genes <- consensus_genes[!is.na(consensus_genes) & consensus_genes != ""]
cat(sprintf("Input   : %d consensus genes from %s\n", length(consensus_genes), consensus_file))

# ============================================================================
# 2. BACKGROUND UNIVERSE - the 3,000 HVGs (see header)
# ============================================================================

hvg_paths <- c("pca_loadings.csv", "../pca_loadings.csv")
hvg_file <- NULL
for (p in hvg_paths) if (file.exists(p)) { hvg_file <- p; break }
if (is.null(hvg_file)) stop("pca_loadings.csv not found - it defines the background universe.")

hvg_genes <- read.csv(hvg_file, stringsAsFactors = FALSE)$gene
hvg_genes <- unique(hvg_genes[!is.na(hvg_genes) & hvg_genes != ""])
cat(sprintf("Background: %d highly variable genes from %s\n\n", length(hvg_genes), hvg_file))

# ============================================================================
# 3. GENE ID CONVERSION
# ============================================================================

to_entrez <- function(symbols, label) {
    m <- suppressWarnings(suppressMessages(bitr(
        symbols, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Mm.eg.db)))
    m <- m[!duplicated(m$ENTREZID), ]
    cat(sprintf("  %-14s %5d symbols -> %5d Entrez IDs\n", label, length(symbols), nrow(m)))
    m
}

cat("Gene ID conversion:\n")
gene_entrez <- to_entrez(consensus_genes, "consensus")
hvg_entrez  <- to_entrez(hvg_genes,       "background")

not_converted <- setdiff(consensus_genes, gene_entrez$SYMBOL)
if (length(not_converted) > 0) {
    cat("  not converted: ", paste(not_converted, collapse = ", "), "\n")
}

entrez_ids     <- gene_entrez$ENTREZID
background_ids <- hvg_entrez$ENTREZID
cat("\n")

# ============================================================================
# 4. KEGG ENRICHMENT
# ============================================================================
# pvalueCutoff = 1 so that EVERY tested pathway is retained with its raw and
# adjusted p-value. Significance is applied afterwards, when writing outputs,
# which keeps the negative result fully inspectable.

cat("Running KEGG over-representation analysis (requires internet)...\n")

kegg <- tryCatch(
    enrichKEGG(
        gene            = entrez_ids,
        universe        = background_ids,
        organism        = "mmu",
        keyType         = "kegg",
        pAdjustMethod   = "BH",
        pvalueCutoff    = 1,
        qvalueCutoff    = 1,
        minGSSize       = 10,
        maxGSSize       = 500,
        use_internal_data = FALSE
    ),
    error = function(e) {
        cat("ERROR contacting the KEGG database:\n  ", conditionMessage(e), "\n")
        cat("KEGG requires an internet connection. Aborting.\n")
        NULL
    })

if (is.null(kegg)) quit(status = 1)

kegg_df <- as.data.frame(kegg)
if (nrow(kegg_df) == 0) stop("enrichKEGG returned no rows at all - check the input IDs.")

# Map Entrez back to symbols for readability
kegg_df$geneSymbol <- vapply(strsplit(kegg_df$geneID, "/"), function(ids) {
    paste(gene_entrez$SYMBOL[match(ids, gene_entrez$ENTREZID)], collapse = "/")
}, character(1))

kegg_df <- kegg_df %>%
    select(ID, Description, GeneRatio, BgRatio, pvalue, p.adjust, qvalue,
           Count, geneID, geneSymbol, everything()) %>%
    arrange(pvalue)

# ============================================================================
# 5. FLAG MHC-CLASS-II-ONLY PATHWAYS
# ============================================================================
# A pathway whose entire overlap is the MHC class II cluster carries no
# stroke-specific information: those genes sit in dozens of KEGG disease maps.

MHC_II <- c("H2-Aa", "H2-Ab1", "H2-Eb1", "Cd74")
kegg_df$MHCII_only <- vapply(strsplit(kegg_df$geneSymbol, "/"),
                             function(g) all(g %in% MHC_II), logical(1))

# ============================================================================
# 6. OUTPUT
# ============================================================================

output_dir <- "KEGG_Enrichment_Pathways"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

write.csv(kegg_df, file.path(output_dir, "KEGG_All_Tested_Pathways.csv"), row.names = FALSE)

nominal <- kegg_df[kegg_df$pvalue < 0.05, ]
write.csv(nominal, file.path(output_dir, "KEGG_Nominal_P05_NotFDRSignificant.csv"), row.names = FALSE)

significant <- kegg_df[kegg_df$p.adjust < 0.05, ]

cat("\n"); cat(strrep("=", 70), "\n", sep = "")
cat("RESULT\n")
cat(strrep("=", 70), "\n", sep = "")
cat(sprintf("  pathways tested            : %d\n", nrow(kegg_df)))
cat(sprintf("  background genes in KEGG   : %s\n", strsplit(kegg_df$BgRatio[1], "/")[[1]][2]))
cat(sprintf("  input genes mapped to KEGG : %s\n", strsplit(kegg_df$GeneRatio[1], "/")[[1]][2]))
cat(sprintf("  nominal p < 0.05           : %d\n", nrow(nominal)))
cat(sprintf("  FDR < 0.05                 : %d\n", nrow(significant)))
cat(sprintf("  best adjusted p-value      : %.3f\n", min(kegg_df$p.adjust)))

if (nrow(nominal) > 0) {
    cat(sprintf("  of the %d nominal hits, %d are driven ENTIRELY by MHC class II genes\n",
                nrow(nominal), sum(nominal$MHCII_only)))
    cat("\nTop 10 by raw p-value (NONE are FDR-significant):\n")
    top <- head(nominal, 10)
    for (i in seq_len(nrow(top))) {
        cat(sprintf("  %-46s p=%.4f padj=%.3f %s\n",
                    substr(top$Description[i], 1, 46), top$pvalue[i], top$p.adjust[i],
                    ifelse(top$MHCII_only[i], "[MHC-II only]", "")))
    }
}

if (nrow(significant) == 0) {
    cat("\nNo KEGG pathway survives FDR correction against the 3,000-HVG background.\n")
    cat("This is the result reported in the manuscript. GO is the primary analysis.\n")
} else {
    write.csv(significant, file.path(output_dir, "KEGG_Enrichment_Results.csv"), row.names = FALSE)
    cat(sprintf("\n%d pathway(s) FDR-significant - written to KEGG_Enrichment_Results.csv\n",
                nrow(significant)))
}

writeLines(capture.output(sessionInfo()), file.path(output_dir, "KEGG_Session_Info.txt"))
cat(sprintf("\nOutputs written to %s/\n", output_dir))
