# =============================================================================
# CORRECTED EXPORT: PCA Loadings for Python Analysis
# =============================================================================
# ALIGNMENT FIX: Now uses the EXACT same training split as the model training script.
# Training Set: sham1, sham2, mcao1, mcao2 (4 Samples)
# Excluded: sham3, mcao3 (Test Set - strictly hidden from PCA)

library(Seurat)
library(dplyr)

# Set seed for reproducibility
set.seed(42)

cat("\n========================================\n")
cat("EXPORTING PCA LOADINGS (STRICT TRAIN ONLY)\n")
cat("========================================\n\n")

# =============================================================================
# STEP 1: Load ONLY Training Samples
# =============================================================================

load_sample <- function(path, name, condition) {
  cat("Reading:", name, "\n")
  data <- Read10X(data.dir = path)
  obj <- CreateSeuratObject(counts = data, project = name, 
                           min.cells = 3, min.features = 200)
  obj$sample <- name
  obj$condition <- condition
  return(obj)
}

# Load ONLY the 4 training samples
sham1 <- load_sample("StrokeData/sham1", "sham1", "Control")
sham2 <- load_sample("StrokeData/sham2", "sham2", "Control")
mcao1 <- load_sample("StrokeData/mcao1", "mcao1", "Stroke")
mcao2 <- load_sample("StrokeData/mcao2", "mcao2", "Stroke")

# Merge into one Training Object
train_obj <- merge(
  sham1, 
  y = list(sham2, mcao1, mcao2),
  add.cell.ids = c("sham1", "sham2", "mcao1", "mcao2"),
  project = "StrokeProject_Train"
)

cat("\nTraining samples merged (Sham1, Sham2, MCAO1, MCAO2).\n")

# =============================================================================
# STEP 2: Preprocessing (Identical to Model Training)
# =============================================================================

cat("Running QC and Preprocessing...\n")

# QC filtering
train_obj[["percent.mt"]] <- PercentageFeatureSet(train_obj, pattern = "^mt-")
train_obj <- subset(
  train_obj, 
  subset = nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 10
)

# Normalize
train_obj <- NormalizeData(train_obj)

# Find variable features
train_obj <- FindVariableFeatures(train_obj, selection.method = "vst", 
                                 nfeatures = 3000)

# Scale data
train_obj <- ScaleData(train_obj, features = VariableFeatures(train_obj))

# Run PCA
cat("Computing PCA on Training Set...\n")
train_obj <- RunPCA(train_obj, features = VariableFeatures(train_obj))

cat("Preprocessing complete.\n")

# =============================================================================
# STEP 3: Export PCA Loadings
# =============================================================================

cat("\nExporting PCA loadings...\n")

# Get PCA loadings (genes x PCs)
pca_loadings <- Loadings(train_obj[["pca"]])

# Convert to dataframe
loadings_df <- as.data.frame(pca_loadings)
loadings_df$gene <- rownames(loadings_df)

# Reorder columns (gene first)
loadings_df <- loadings_df[, c("gene", colnames(pca_loadings))]

# Save
write.csv(loadings_df, "pca_loadings.csv", row.names = FALSE)

cat(paste("✓ Saved:", nrow(loadings_df), "genes x", 
          ncol(pca_loadings), "PCs\n"))
cat("✓ File: pca_loadings.csv\n")

# =============================================================================
# STEP 4: Export explained variance
# =============================================================================

cat("\nExporting explained variance...\n")

# Calculate explained variance
explained_var <- train_obj@reductions$pca@stdev^2 / 
                 sum(train_obj@reductions$pca@stdev^2)

# Create dataframe
var_df <- data.frame(
  PC = paste0("PC_", 1:length(explained_var)),
  explained_variance_ratio = explained_var,
  cumulative_variance = cumsum(explained_var)
)

# Save
write.csv(var_df, "pca_explained_variance.csv", row.names = FALSE)

cat("✓ File: pca_explained_variance.csv\n")

# Print summary
cat("\n========================================\n")
cat("SUMMARY\n")
cat("========================================\n\n")

cat("Total genes:", nrow(loadings_df), "\n")
cat("Number of PCs:", ncol(pca_loadings), "\n")
cat("Top 5 PCs explain:", round(sum(explained_var[1:5]) * 100, 2), "% variance\n")

cat("\nTop 10 genes for PC_1 (Training Set Definition):\n")
pc1_top <- loadings_df[order(abs(loadings_df$PC_1), decreasing = TRUE), ][1:10, c("gene", "PC_1")]
print(pc1_top, row.names = FALSE)

cat("\n========================================\n")
cat("READY FOR PYTHON!\n")
cat("========================================\n")
cat("NOTE: This file now perfectly aligns with 'DP_Diff_Test&Train.R'\n")