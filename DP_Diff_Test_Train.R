# %%
library(Seurat)
library(dplyr)

# Set seed for reproducibility
set.seed(42)

# ----------------------------- 
# 1. Load and preprocess samples
# -----------------------------
# %%
load_sample <- function(path, name, condition) {
  cat("\nReading:", path)
  data <- Read10X(data.dir = path)
  obj <- CreateSeuratObject(counts = data, project = name, min.cells = 3, min.features = 200)
  obj$sample <- name
  obj$condition <- condition
  return(obj)
}

# Load all samples
sham1 <- load_sample("StrokeData/sham1", "sham1", "Control")
sham2 <- load_sample("StrokeData/sham2", "sham2", "Control")
sham3 <- load_sample("StrokeData/sham3", "sham3", "Control")
mcao1 <- load_sample("StrokeData/mcao1", "mcao1", "Stroke")
mcao2 <- load_sample("StrokeData/mcao2", "mcao2", "Stroke")
mcao3 <- load_sample("StrokeData/mcao3", "mcao3", "Stroke")

# Merge into one Seurat object
all_samples <- merge(
  sham1, 
  y = list(sham2, sham3, mcao1, mcao2, mcao3),
  add.cell.ids = c("sham1", "sham2", "sham3", "mcao1", "mcao2", "mcao3"),
  project = "StrokeProject"
)

# -----------------------------
# 2. QC filtering
# -----------------------------
# %%
all_samples[["percent.mt"]] <- PercentageFeatureSet(all_samples, pattern = "^mt-")
all_samples <- subset(
  all_samples, 
  subset = nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 10
)

# -----------------------------
# 3. Train-test split by sample (no leakage)
# -----------------------------
# %%
train_samples <- c("sham1", "sham2", "mcao1", "mcao2")
test_samples  <- c("sham3", "mcao3")

train_obj <- subset(all_samples, subset = sample %in% train_samples)
test_obj  <- subset(all_samples, subset = sample %in% test_samples)

# -----------------------------
# 4. Train preprocessing
# -----------------------------
# %%
train_obj <- NormalizeData(train_obj)
train_obj <- FindVariableFeatures(train_obj, selection.method = "vst", nfeatures = 3000)
train_obj <- ScaleData(train_obj, features = VariableFeatures(train_obj))
train_obj <- RunPCA(train_obj, features = VariableFeatures(train_obj))

# -----------------------------
# 5. Align features between train & test
# -----------------------------
# %%
common_features <- intersect(rownames(train_obj), rownames(test_obj))
train_obj <- subset(train_obj, features = common_features)
test_obj  <- subset(test_obj,  features = common_features)

# -----------------------------
# 6. Project test data into train PCA space
# -----------------------------
# Normalize & scale test data using train's HVGs
# %%
test_obj <- NormalizeData(test_obj)
test_obj <- ScaleData(test_obj, features = VariableFeatures(train_obj))

# Get PCA loadings from train
pca_loadings <- Loadings(train_obj[["pca"]]) # genes x PCs
# %%
# Project scaled test matrix into train PCA space
test_scaled_mat <- as.matrix(
  LayerData(test_obj, assay = "RNA", layer = "scale.data")[rownames(pca_loadings), ]
)

# Transpose so cells are rows
test_pca_scores <- t(test_scaled_mat) %*% as.matrix(pca_loadings)


# Store projected PCA into test_obj
test_obj[["pca"]] <- CreateDimReducObject(
  embeddings = test_pca_scores,
  key = "PC_",
  assay = DefaultAssay(test_obj)
)

# -----------------------------
# 7. Save outputs
# -----------------------------
# Train PCA matrix
# %%
train_pca <- Embeddings(train_obj, reduction = "pca") %>% as.data.frame()
train_pca$cell_id <- rownames(train_pca)
write.csv(train_pca, "stroke_pca_train.csv", row.names = FALSE)

# Test PCA matrix
test_pca <- Embeddings(test_obj, reduction = "pca") %>% as.data.frame()
test_pca$cell_id <- rownames(test_pca)
write.csv(test_pca, "stroke_pca_test.csv", row.names = FALSE)

# Labels
train_labels <- train_obj@meta.data %>%
  mutate(cell_id = rownames(.)) %>%
  select(cell_id, sample, condition)
write.csv(train_labels, "stroke_labels_train.csv", row.names = FALSE)

test_labels <- test_obj@meta.data %>%
  mutate(cell_id = rownames(.)) %>%
  select(cell_id, sample, condition)
write.csv(test_labels, "stroke_labels_test.csv", row.names = FALSE)
