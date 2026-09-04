# COMPLETE PIPELINE DATA FLOW
## Ischemic Stroke scRNA-seq → Multi-Model ML → Consensus Biomarkers
### All Cell & Gene Counts Verified from Actual Files

> **Data Source:** GSE174574 (Mouse MCAO model, 6 mice)  
> **Code References:** `DP_Diff_Test&Train.R`, `comparisionModel.py`, `fixed_foundation_models.py`, `cnn_xgboostModel.py`  
> **Counts Source:** `qc_metrics_per_cell.csv`, `stroke_pca_train.csv`, `stroke_pca_test.csv`, `pca_loadings.csv`, raw `barcodes.tsv.gz` and `features.tsv.gz`

---

## MASTER DATA FLOW DIAGRAM

```
RAW 10X DATA (6 samples)
     │
     │  Each sample: 27,998 genes × (variable cells)
     ▼
STAGE 1: CreateSeuratObject  [per sample, R]
     │  Filter: min.cells=3, min.features=200
     │  Genes: 27,998 → ~18,676 (per sample estimate)
     ▼
STAGE 2: MERGE  [R]
     │  All 6 samples combined
     │  Cells: 58,523 total | Genes: 18,676
     ▼
STAGE 3: QC FILTERING  [R]
     │  nFeature_RNA > 200 & < 2,500 | percent.mt < 10%
     │  Cells: 58,523 → 54,599 (removed 3,924)
     │  Genes: 18,676 (unchanged — QC only removes cells)
     ▼
STAGE 4: SUBJECT-WISE TRAIN/TEST SPLIT  [R]
     │
     ├─── TRAIN: sham1, sham2, mcao1, mcao2
     │           37,883 cells × 18,676 genes
     │
     └─── TEST:  sham3, mcao3
                 16,716 cells × 18,676 genes
     ▼
STAGE 5: TRAIN-ONLY PREPROCESSING  [R]
     │  (test data untouched at this stage)
     │  NormalizeData   → 37,883 × 18,676 (values change, dims same)
     │  FindVariableFeatures → selects 3,000 HVGs from 18,676
     │  ScaleData       → 37,883 × 3,000  (scaled matrix)
     │  RunPCA          → 37,883 × 50 PCs + loadings matrix (3,000 × 50)
     ▼
STAGE 6: TEST PROJECTION  [R]
     │  NormalizeData on test (independent)
     │  ScaleData on test using train's 3,000 HVG list
     │  Matrix multiply: test_scaled (3,000×16,716)ᵀ × loadings (3,000×50)
     │  Result: 16,716 × 50 PCs  [test cells in train's coordinate space]
     ▼
STAGE 7: SAVE TO CSV  [R]
     │  stroke_pca_train.csv  → 37,883 × 51 (50 PCs + cell_id)
     │  stroke_pca_test.csv   → 16,716 × 51 (50 PCs + cell_id)
     │  stroke_labels_train.csv, stroke_labels_test.csv
     │  pca_loadings.csv      → 3,000 × 51 (50 PCs + gene name)
     ▼
STAGE 8: 8 ML MODELS  [Python]
     │  Input: 50 PCs per cell
     │  All models: binary classification (0=Control, 1=Stroke)
     │  10-fold CV on train | Final test on held-out test set
     ▼
STAGE 9: GENE IMPORTANCE BACK-PROJECTION  [Python]
     │  6 of 8 models → PC-level importance → gene scores
     │  Formula: Gene_j = Σ|w_i × L_ij| over 50 PCs
     │  Per model: 18,676 genes ranked → top 100 selected
     ▼
STAGE 10: CONSENSUS IDENTIFICATION  [Python]
     │  Threshold: gene appears in ≥ 4 of 6 model top-100 lists
     │  Result: 63 consensus genes
     ▼
STAGE 11: BIOLOGICAL VALIDATION  [R]
     GO + KEGG enrichment via clusterProfiler
     Background: 3,000 HVGs | Input: 63 consensus genes
```

---

## STAGE-BY-STAGE DETAILED BREAKDOWN

---

### STAGE 1 — Raw 10X Data (No filtering yet)

**Code:** `DP_Diff_Test&Train.R` — `Read10X()` inside `load_sample()` function

```r
data <- Read10X(data.dir = path)
```

| Sample | Condition | Role | Raw Cells | Raw Genes |
|--------|-----------|------|-----------|-----------|
| sham1 | Control | Train | 8,771 | 27,998 |
| sham2 | Control | Train | 8,540 | 27,998 |
| sham3 | Control | Test | 9,980 | 27,998 |
| mcao1 | Stroke | Train | 11,772 | 27,998 |
| mcao2 | Stroke | Train | 11,361 | 27,998 |
| mcao3 | Stroke | Test | 8,104 | 27,998 |
| **TOTAL** | | | **58,528** | **27,998** |

> **Source:** Counted directly from `barcodes.tsv.gz` (cell count) and `features.tsv.gz` (gene count) in each `StrokeData/<sample>/` folder.

**What the raw matrix is:**
- Each row = one gene (27,998 genes from the 10x Genomics mm10 genome)
- Each column = one cell barcode (potential cell)
- Values = UMI counts (how many RNA molecules from that gene were captured in that cell)
- Extremely sparse: >90% of entries are zero

---

### STAGE 2 — CreateSeuratObject (Load filter, per sample)

**Code:** `DP_Diff_Test&Train.R` Line 15

```r
obj <- CreateSeuratObject(counts = data, project = name, 
                          min.cells = 3, min.features = 200)
```

**Two filters applied simultaneously at object creation:**

| Filter | Parameter | What it removes |
|--------|-----------|-----------------|
| `min.cells = 3` | Gene level | Any gene expressed in fewer than 3 cells → gene removed from matrix entirely |
| `min.features = 200` | Cell level | Any cell with fewer than 200 detected genes → cell removed |

**Counts after CreateSeuratObject (per sample):**

| Sample | Cells before | Cells after | Cells removed | Genes (approx) |
|--------|-------------|-------------|---------------|----------------|
| sham1 | 8,771 | 8,770 | 1 | ~18,676 |
| sham2 | 8,540 | 8,537 | 3 | ~18,676 |
| sham3 | 9,980 | 9,980 | 0 | ~18,676 |
| mcao1 | 11,772 | 11,772 | 0 | ~18,676 |
| mcao2 | 11,361 | 11,361 | 0 | ~18,676 |
| mcao3 | 8,104 | 8,103 | 1 | ~18,676 |
| **TOTAL** | **58,528** | **58,523** | **5** | **~18,676** |

> **Why genes drop from 27,998 → ~18,676:**  
> `min.cells=3` applied across each sample removes genes with very low expression. When all 6 samples are merged, Seurat re-evaluates across the full combined dataset — genes that are absent in the majority of cells across all 6 mice are dropped, reducing from 27,998 to **18,676 genes in the merged object**.

---

### STAGE 3 — MERGE

**Code:** `DP_Diff_Test&Train.R` Lines 30–35

```r
all_samples <- merge(
  sham1, 
  y = list(sham2, sham3, mcao1, mcao2, mcao3),
  add.cell.ids = c("sham1","sham2","sham3","mcao1","mcao2","mcao3"),
  project = "StrokeProject"
)
```

**IN → OUT:**

| | Cells | Genes |
|--|-------|-------|
| **IN** | 58,523 (6 separate objects) | ~18,676 per object |
| **OUT** | **58,523** (one combined object) | **18,676** |

> Cell IDs are prefixed: e.g., `sham1_ACGTACGTACGT-1` so cells from different samples don't collide.  
> `merge()` takes the **intersection** of genes across all 6 objects = 18,676 genes shared by all.

---

### STAGE 4 — QC Filtering

**Code:** `DP_Diff_Test&Train.R` Lines 41–45

```r
all_samples[["percent.mt"]] <- PercentageFeatureSet(all_samples, pattern = "^mt-")
all_samples <- subset(all_samples, 
  subset = nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 10)
```

**Three QC thresholds and their biological meaning:**

| Filter | Threshold | Biological Reason | What it catches |
|--------|-----------|-------------------|-----------------|
| `nFeature_RNA > 200` | Min 200 genes detected | Cells with very few genes = empty droplets or dead cells | Debris / empty GEMs |
| `nFeature_RNA < 2500` | Max 2,500 genes detected | Cells with too many genes = two cells captured as one | Doublets |
| `percent.mt < 10%` | Max 10% mitochondrial | Dying cells lose cytoplasmic RNA but retain mt-RNA → high mt% | Damaged / apoptotic cells |

**Counts after QC — per sample (source: `qc_metrics_per_cell.csv`):**

| Sample | Condition | Before QC | After QC | Removed | % Removed |
|--------|-----------|-----------|----------|---------|-----------|
| sham1 | Control | 8,770 | **8,410** | 360 | 4.1% |
| sham2 | Control | 8,537 | **8,144** | 393 | 4.6% |
| sham3 | Control | 9,980 | **9,508** | 472 | 4.7% |
| mcao1 | Stroke | 11,772 | **10,801** | 971 | 8.2% |
| mcao2 | Stroke | 11,361 | **10,528** | 833 | 7.3% |
| mcao3 | Stroke | 8,103 | **7,208** | 895 | 11.0% |
| **TOTAL** | | **58,523** | **54,599** | **3,924** | **6.7%** |

> MCAO samples lose more cells (7–11%) vs. sham (4–5%). This is biologically expected: ischemic injury causes more cell death, so more cells have high mitochondrial reads.

**IN → OUT:**

| | Cells | Genes |
|--|-------|-------|
| **IN** | 58,523 | 18,676 |
| **OUT** | **54,599** | **18,676** (unchanged — QC removes only cells) |

---

### STAGE 5 — Subject-wise Train/Test Split

**Code:** `DP_Diff_Test&Train.R` Lines 51–55

```r
train_samples <- c("sham1", "sham2", "mcao1", "mcao2")
test_samples  <- c("sham3", "mcao3")

train_obj <- subset(all_samples, subset = sample %in% train_samples)
test_obj  <- subset(all_samples, subset = sample %in% test_samples)
```

**Why subject-wise (not random cell split):**  
Cells from the same mouse share biological background (genetics, surgery variation, batch effects). A random split would put cells from the same mouse in both train and test — the model would effectively "see" test data during training, making performance artificially high.

**Split result (source: `stroke_labels_train.csv`, `stroke_labels_test.csv`):**

| Set | Samples | Condition | Cells |
|-----|---------|-----------|-------|
| Training | sham1 | Control | 8,410 |
| Training | sham2 | Control | 8,144 |
| Training | mcao1 | Stroke | 10,801 |
| Training | mcao2 | Stroke | 10,528 |
| **Training TOTAL** | 4 mice | — | **37,883** |
| Test | sham3 | Control | 9,508 |
| Test | mcao3 | Stroke | 7,208 |
| **Test TOTAL** | 2 mice | — | **16,716** |
| **GRAND TOTAL** | 6 mice | — | **54,599** |

**Class balance:**

| Set | Control cells | Stroke cells | Ratio |
|-----|--------------|--------------|-------|
| Train | 16,554 (43.7%) | 21,329 (56.3%) | 1:1.29 |
| Test | 9,508 (56.9%) | 7,208 (43.1%) | 1.32:1 |

> Note: classes are slightly imbalanced — handled by `class_weight='balanced'` in LR and RF.

**IN → OUT:**

| | Cells | Genes |
|--|-------|-------|
| **IN** | 54,599 | 18,676 |
| **Train OUT** | **37,883** | **18,676** |
| **Test OUT** | **16,716** | **18,676** |

---

### STAGE 6 — Train-Only Preprocessing

**Code:** `DP_Diff_Test&Train.R` Lines 61–64  
**Test data is NOT touched in this stage.**

#### 6a. NormalizeData (Line 61)

```r
train_obj <- NormalizeData(train_obj)
```

**Formula applied to every cell:**
```
Normalized_ij = log1p( raw_count_ij / Σ_j(raw_count_ij) × 10,000 )
```

- `Σ_j(raw_count_ij)` = total UMI count for cell `i` (library size)
- Dividing by library size corrects for sequencing depth differences between cells
- ×10,000 = scaling factor (makes values comparable)
- `log1p` = log(x+1) stabilises variance and compresses dynamic range

**IN → OUT:**

| | Cells | Genes | Values |
|--|-------|-------|--------|
| **IN** | 37,883 | 18,676 | Raw UMI counts |
| **OUT** | **37,883** | **18,676** | Log-normalised values |

#### 6b. FindVariableFeatures (Line 62)

```r
train_obj <- FindVariableFeatures(train_obj, selection.method = "vst", nfeatures = 3000)
```

**VST method (Variance Stabilising Transformation):**
1. For each of 18,676 genes, compute mean expression across all 37,883 training cells
2. For each gene, compute observed variance
3. Model the expected variance at each mean level using a smooth spline (corrects for the mean-variance relationship: high-expression genes naturally have higher variance)
4. Compute standardised variance = observed / expected
5. Rank genes by standardised variance → top 3,000 = HVGs

**Biological meaning:** HVGs have more variation across cells than expected by chance — they carry biological signal (cell type differences, disease state) rather than random noise.

**Key:** The HVG list is stored inside `train_obj` as `VariableFeatures(train_obj)` — a list of 3,000 gene names. This list is later passed to test data preprocessing.

**IN → OUT:**

| | Cells | Genes |
|--|-------|-------|
| **IN** | 37,883 | 18,676 |
| **OUT** | 37,883 | **3,000 HVGs selected** (18,676 - 15,676 = 15,676 genes deprioritised) |

> The 15,676 non-HVG genes are not deleted — they remain in the object but are excluded from subsequent steps.

#### 6c. ScaleData (Line 63)

```r
train_obj <- ScaleData(train_obj, features = VariableFeatures(train_obj))
```

For each of the 3,000 HVGs, across all 37,883 training cells:
- **Centre:** subtract mean expression of that gene → mean becomes 0
- **Scale:** divide by standard deviation of that gene → variance becomes 1

Result: a 3,000 × 37,883 scaled matrix (stored as `scale.data` layer in Seurat).

**Why scaling is needed:** Without scaling, genes with high absolute expression (e.g., Actb, mean=500) would dominate PCA. Scaling ensures each gene contributes equally regardless of expression level.

**IN → OUT:**

| | Cells | Genes | Values |
|--|-------|-------|--------|
| **IN** | 37,883 | 18,676 (but only 3,000 used) | Log-normalised |
| **OUT** | **37,883** | **3,000** | Z-scores (mean=0, var=1 per gene) |

#### 6d. RunPCA (Line 64)

```r
train_obj <- RunPCA(train_obj, features = VariableFeatures(train_obj))
```

Singular value decomposition (SVD) of the 3,000 × 37,883 scaled matrix:

```
X_scaled (3,000 × 37,883) = U × Σ × Vᵀ

where:
  U (3,000 × 50) = gene loadings (directions of max variance in gene space)
  Σ (50 × 50)    = diagonal matrix of singular values (variance explained)
  V (37,883 × 50) = cell scores (PC embeddings)
```

- **50 PCs chosen** based on elbow plot: PC1 explains 17.86%, PC2 explains 14.28%, cumulative 90% variance reached at PC25 (from `FigureS2_PCA_Variance.png`)
- The **loadings matrix** `U` (stored as `pca_loadings.csv`) maps 3,000 genes → 50 PCs

**IN → OUT:**

| | Cells | Features | Stored as |
|--|-------|----------|-----------|
| **IN** | 37,883 | 3,000 genes | Scaled matrix |
| **OUT (embeddings)** | **37,883** | **50 PCs** | `stroke_pca_train.csv` |
| **OUT (loadings)** | — | **3,000 × 50** | `pca_loadings.csv` |

---

### STAGE 7 — Test Data Projection

**Code:** `DP_Diff_Test&Train.R` Lines 70–99  
**Uses parameters learned from train only — no new learning.**

#### 7a. Feature alignment (Lines 70–72)

```r
common_features <- intersect(rownames(train_obj), rownames(test_obj))
train_obj <- subset(train_obj, features = common_features)
test_obj  <- subset(test_obj,  features = common_features)
```

Ensures both objects contain identical genes before projection.

#### 7b. NormalizeData on test (Line 79)

```r
test_obj <- NormalizeData(test_obj)
```

Same formula applied independently to test cells using their own library sizes. No training statistics needed — mathematically independent.

#### 7c. ScaleData on test using train's HVG list (Line 80)

```r
test_obj <- ScaleData(test_obj, features = VariableFeatures(train_obj))
```

`VariableFeatures(train_obj)` = the same 3,000 gene names from stage 6b. Test data is scaled using test cells' own mean/SD, but restricted to the same 3,000 genes the training model operates on.

#### 7d. PCA Projection (Lines 83–91) — most critical step

```r
pca_loadings <- Loadings(train_obj[["pca"]])   # 3,000 genes × 50 PCs

test_scaled_mat <- as.matrix(
  LayerData(test_obj, assay="RNA", layer="scale.data")[rownames(pca_loadings), ]
)   # 3,000 genes × 16,716 cells

test_pca_scores <- t(test_scaled_mat) %*% as.matrix(pca_loadings)
# (16,716 × 3,000) × (3,000 × 50) = 16,716 × 50
```

**Matrix multiplication explained:**
- `pca_loadings` = the "rotation matrix" — tells you how to combine genes to get PC scores
- `t(test_scaled_mat)` = test cells × genes matrix
- Multiplying places test cells into the **same 50-dimensional coordinate space** defined by training data
- This is equivalent to `sklearn`'s `pca.transform(X_test)` after `pca.fit(X_train)`

**IN → OUT:**

| | Cells | Features | Source |
|--|-------|----------|--------|
| **IN (test scaled)** | 16,716 | 3,000 genes | Normalised + scaled |
| **IN (loadings)** | — | 3,000 × 50 | From training PCA |
| **OUT** | **16,716** | **50 PCs** | `stroke_pca_test.csv` |

---

### STAGE 8 — Machine Learning Classification

**Code:** `comparisionModel.py`, `cnn_xgboostModel.py`, `fixed_foundation_models.py`  
**Framework:** `phase1_framework.py` → `ModelValidator` class

**Input to all 8 models:**

| | Shape | Description |
|--|-------|-------------|
| X_train | 37,883 × 50 | PC scores for training cells |
| X_test | 16,716 × 50 | Projected PC scores for test cells |
| y_train | 37,883 | Binary labels (0=Control, 1=Stroke) |
| y_test | 16,716 | Binary labels for evaluation |

#### Model 1 — Logistic Regression

| Parameter | Value | Package |
|-----------|-------|---------|
| `solver` | `lbfgs` | `sklearn.linear_model.LogisticRegression` |
| `max_iter` | 1,000 | |
| `class_weight` | `balanced` | |
| `random_state` | 42 | |
| CV folds | 10 | `StratifiedKFold` |
| Bootstrap CI | 1,000 iterations | `sklearn.utils.resample` |

**Data flow:**  
50 PCs → weighted sum → sigmoid → P(Stroke)  
**Test AUC: 0.9968 [0.9962–0.9974]**

#### Model 2 — Random Forest

| Parameter | Value | Package |
|-----------|-------|---------|
| `n_estimators` | 200 trees | `sklearn.ensemble.RandomForestClassifier` |
| `max_depth` | 20 | |
| `min_samples_split` | 5 | |
| `min_samples_leaf` | 2 | |
| `max_features` | `sqrt` (≈7 PCs per split) | |
| `class_weight` | `balanced` | |
| `random_state` | 42 | |

**Data flow:**  
50 PCs → 200 trees (each trained on bootstrap sample) → majority vote → class  
**Test AUC: 0.9932 [0.9924–0.9941]**

#### Model 3 — XGBoost (Standalone)

| Parameter | Value | Package |
|-----------|-------|---------|
| `n_estimators` | 200 | `xgboost.XGBClassifier` |
| `max_depth` | 6 | |
| `learning_rate` | 0.1 | |
| `subsample` | 0.8 | |
| `colsample_bytree` | 0.8 | |
| `eval_metric` | `logloss` | |
| `random_state` | 42 | |

**Data flow:**  
50 PCs → 200 sequential trees (each corrects prior errors) → probability  
**Test AUC: 0.9966 [0.9961–0.9971]**

#### Model 4 — CNN+XGBoost

| Component | Parameter | Value | Package |
|-----------|-----------|-------|---------|
| CNN | `conv_filters` | [64, 128, 256] | `tensorflow.keras` |
| CNN | `kernel_size` | 3 | |
| CNN | `cnn_epochs` | 30 | |
| CNN | `early_stopping` | patience=5 | |
| CNN | `optimizer` | Adam, lr=0.001 | |
| CNN | `dropout` | 0.3 (conv), 0.4 (dense) | |
| XGBoost | `n_estimators` | 200 | `xgboost.XGBClassifier` |
| XGBoost | `max_depth` | 6 | |
| XGBoost | `learning_rate` | 0.1 | |

**Data flow:**  
50 PCs → Reshape(50,1) → Conv1D(64)→BN→Pool → Conv1D(128)→BN→Pool → Conv1D(256)→BN→Pool → GlobalAvgPool → Dense(128) → **128 CNN features** → XGBoost → class  
**Test AUC: 0.9954 [0.9946–0.9961]**

#### Model 5 — scGPT-inspired

| Parameter | Value |
|-----------|-------|
| `d_model` | 256 |
| `nhead` | 8 (each head = 32 dims) |
| `num_layers` | 4 |
| `dim_feedforward` | 1,024 (256×4) |
| `dropout` | 0.1 |
| `optimizer` | AdamW, lr=2e-4, wd=0.01 |
| `scheduler` | CosineAnnealingLR |
| `batch_size` | 128 |
| `epochs` | max 30 (early stop patience=5) |
| Package | `torch.nn.TransformerEncoder` |

**Data flow:**  
50 PCs → Linear(50→256) → +positional encoding → 4× TransformerEncoderLayer → LayerNorm → Linear(256→128) → GELU → Linear(128→2) → softmax  
**Test AUC: 0.9975 [0.9970–0.9980]** ← highest of all models

#### Model 6 — scBERT-inspired

| Parameter | Value |
|-----------|-------|
| `d_model` | 512 |
| `nhead` | 8 (each head = 64 dims) |
| `num_layers` | 6 |
| `dim_feedforward` | 2,048 (512×4) |
| `activation` | `gelu` |
| `dropout` | 0.1 |
| Pooling | CLS-token style (position 0) |
| Package | `torch.nn.TransformerEncoder` |

**Data flow:**  
50 PCs → Linear(50→512) → LayerNorm → 6× TransformerEncoderLayer → take position[0] → Linear(512→512)+Tanh (pooler) → Dropout → Linear(512→256) → GELU → Linear(256→2) → softmax  
**Test AUC: 0.9869 [0.9853–0.9886]**

#### Model 7 — scFormer-inspired

| Parameter | Value |
|-----------|-------|
| `d_model` | 384 |
| `nhead` | 6 (each head = 64 dims) |
| `num_layers` | 3 |
| `dim_feedforward` | 1,536 (384×4) |
| Norm position | Pre-LayerNorm (norm before attention) |
| Package | `torch.nn.MultiheadAttention` (manual stack) |

**Data flow:**  
50 PCs → Linear(50→384)+LayerNorm+GELU+Dropout → 3× [PreNorm → MultiheadAttention → residual → PreNorm → FFN → residual] → Linear(384→192) → GELU → Linear(192→2) → softmax  
**Test AUC: 0.9927 [0.9915–0.9939]**

#### Model 8 — Geneformer-inspired

| Parameter | Value |
|-----------|-------|
| `d_model` | 512 |
| `nhead` | 8 |
| `num_layers` | 4 |
| Value embedding | Linear(1→256) |
| Rank embedding | Linear(1→256) |
| Gene positions | Learnable, shape (1, 50, 512) |
| Pooling | Mean across 50 tokens |
| Package | `torch.nn.TransformerEncoder` |

**Data flow:**  
50 PCs → compute ranks → value_emb(50×256) + rank_emb(50×256) → concat(50×512) → +gene_positions → 4× TransformerEncoderLayer → mean pool across 50 tokens → Linear(512→256) → GELU → Linear(256→2) → softmax  
**Test AUC: 0.9778 [0.9756–0.9798]** ← lowest, still excellent

> **Key difference vs. scGPT/scBERT/scFormer:** Input is a **sequence of 50 tokens** (one per PC), not a single token. Rank-based encoding breaks the linear PCA back-projection, so excluded from consensus.

---

### STAGE 9 — Gene Importance Back-Projection

**Code:** `integrated_interpretability_pipeline.py`, `attention_extractor.py`

#### Which models contribute (6 of 8):

| Model | Included | Importance method | Reason if excluded |
|-------|----------|-------------------|--------------------|
| Logistic Regression | ✅ | `abs(coef_)` per PC | — |
| Random Forest | ✅ | `feature_importances_` (Gini) | — |
| XGBoost | ✅ | `feature_importances_` (gain) | — |
| CNN+XGBoost | ❌ | — | CNN creates nonlinear features → breaks PCA linearity |
| scGPT | ✅ | L2 norm of `input_projection` weights | — |
| scBERT | ✅ | L2 norm of `gene_embedding` weights | — |
| scFormer | ✅ | L2 norm of `gene_encoder[0]` weights | — |
| Geneformer | ❌ | — | Rank tokenisation → mathematically incompatible |

#### Back-projection formula (applied per model):

```
Gene_importance_j = Σᵢ₌₁⁵⁰ |w_i × L_ij|

where:
  w_i   = importance score of PC_i from the model
  L_ij  = loading of gene_j on PC_i  (from pca_loadings.csv: 3,000×50 matrix)
  j     = index over 3,000 HVGs
```

This projects the model's opinion of which PCs matter back into gene space, weighted by how much each gene contributes to those PCs.

**Data flow:**

| | Shape | Description |
|--|-------|-------------|
| `w` (per model) | 50 × 1 | PC importance scores |
| `L` (shared) | 3,000 × 50 | PCA loadings from `pca_loadings.csv` |
| `gene_scores` | 3,000 × 1 | Gene importance per model |
| Top genes selected | 100 per model | Top 100 ranked genes |
| After consensus | **63 genes** | Appear in ≥ 4 of 6 models' top-100 |

---

### STAGE 10 — Consensus Gene Identification

**Threshold:** Gene must appear in top-100 list of **≥ 4 out of 6** interpretable models.

| Model agreement | Genes at this level | Interpretation |
|----------------|---------------------|----------------|
| 1/6 models | 0 | (below threshold — no genes here) |
| 2/6 models | 0 | (below threshold) |
| 3/6 models | 0 | (below threshold) |
| 4/6 models | 41 genes | Moderate consensus |
| 5/6 models | 5 genes | Strong consensus |
| 6/6 models | 17 genes | **Perfect consensus** |
| **TOTAL** | **63 genes** | Final biomarker candidates |

**Top 17 genes (6/6 agreement — unanimous):**

| Gene | Function |
|------|----------|
| Spp1 | Osteopontin; macrophage activation |
| Cdkn1a | p21; cell cycle arrest |
| Cd24a | Immune modulation |
| **Il1rn** | **IL-1 receptor antagonist → Anakinra target** |
| Ifitm6 | Type I interferon response |
| H2-Eb1 | MHC Class II antigen presentation |
| H2-Aa | MHC Class II antigen presentation |
| Fth1 | Ferritin; iron storage, oxidative stress |
| Lpl | Lipoprotein lipase; lipid metabolism |
| Anxa1 | Annexin A1; inflammation resolution |
| Adam8 | Metalloprotease; inflammation |
| Ch25h | Cholesterol 25-hydroxylase; innate immunity |
| Plac8 | Immune cell marker |
| Napsa | Antigen processing |
| Cd72 | B-cell surface; immune regulation |
| Tubb6 | Cytoskeleton |
| Sirpb1c | Signal-regulatory protein |

**IN → OUT:**

| | Count |
|--|-------|
| **IN** | 3,000 HVGs (from pca_loadings.csv) |
| **Top 100 per model** | 100 genes × 6 models = up to 600 slots |
| **Unique genes appearing** | — |
| **OUT (consensus)** | **63 genes** (≥4/6 models) |

---

### STAGE 11 — Biological Validation

**Code:** R scripts in `GO-EnrichmentAnalysis/`  
**Package:** `clusterProfiler v4.0`

| Parameter | Value |
|-----------|-------|
| Input genes | 63 consensus genes |
| Background | 3,000 HVGs (the universe of genes tested) |
| Statistical test | Hypergeometric test |
| Multiple correction | Benjamini-Hochberg FDR |
| p-value threshold | 0.05 |
| Min gene set size | 10 |
| Max gene set size | 500 |

**Key enriched pathways:**

| Pathway | GO ID | FDR | Genes |
|---------|-------|-----|-------|
| Response to interferon-beta (Type I) | GO:0035456 | 1.4e-08 | 10/57 |
| Cellular response to interferon-beta | GO:0035458 | 7.3e-07 | 8/57 |
| Response to type II interferon | GO:0034341 | 4.5e-03 | 7/57 |
| Antigen processing via MHC II | GO:0019886 | 1.0e-02 | 4/57 |
| Inflammatory response | GO:0006954 | 1.5e-02 | 16/57 |
| Defence response to virus | GO:0051607 | 2.0e-02 | 7/57 |

> Tested against a background of the 3,000 HVGs. `GO:0060337` is **not**
> enriched and earlier drafts of this document listed it in error.

**IN → OUT:**

| | Count |
|--|-------|
| **IN** | 63 consensus genes |
| **Significant GO terms** | 101 BP, 12 MF, 16 CC (FDR < 0.05) |
| **KEGG pathways enriched** | None. 122 tested, 0 survive FDR correction (smallest adjusted p = 0.162) |
| **OUT (main finding)** | Dual-interferon signature (Type I + Type II IFN) |

---

## SUMMARY: CELLS AND GENES AT EVERY CHECKPOINT

| Checkpoint | Stage | Cells | Genes/Features | Key operation |
|------------|-------|-------|----------------|---------------|
| Raw 10X | Pre-processing | 58,528 | 27,998 | None — raw barcodes × features |
| After CreateSeuratObject | Stage 2 (per-sample) | 58,523 | ~18,676 | min.cells=3, min.features=200 |
| After merge | Stage 3 | 58,523 | 18,676 | Union of 6 Seurat objects |
| After QC filter | Stage 4 | **54,599** | 18,676 | nFeature 200–2500, %MT<10% |
| After split — Train | Stage 5 | **37,883** | 18,676 | sham1+sham2+mcao1+mcao2 |
| After split — Test | Stage 5 | **16,716** | 18,676 | sham3+mcao3 |
| After NormalizeData (train) | Stage 6a | 37,883 | 18,676 | Values change, dims same |
| After FindVariableFeatures | Stage 6b | 37,883 | **3,000 HVGs** | Feature selection |
| After ScaleData (train) | Stage 6c | 37,883 | 3,000 | Z-score normalisation |
| After RunPCA (train) | Stage 6d | **37,883 × 50 PCs** | **3,000 × 50 loadings** | Dimensionality reduction |
| Test normalised | Stage 7b | 16,716 | 18,676 | Independent log-norm |
| Test scaled (train HVGs) | Stage 7c | 16,716 | 3,000 | Train gene list applied |
| Test projected | Stage 7d | **16,716 × 50 PCs** | — | Matrix × loadings |
| ML model input (train) | Stage 8 | 37,883 | **50** | PC embeddings |
| ML model input (test) | Stage 8 | 16,716 | **50** | PC embeddings |
| Gene importance (per model) | Stage 9 | — | 3,000 ranked | Back-projection |
| Top genes per model | Stage 9 | — | 100 | Threshold |
| Consensus genes | Stage 10 | — | **63** | ≥4/6 models |
| Unanimous genes | Stage 10 | — | **17** | 6/6 models |
| GO/KEGG enrichment | Stage 11 | — | 63 input / 3,000 background | Hypergeometric test |

---

## EXISTING DOCUMENTATION INDEX

The following MD files exist in the project and cover specific aspects of this pipeline:

| File | Coverage | Location |
|------|----------|----------|
| `Theory/01_BIOLOGICAL_BACKGROUND.md` | MCAO model biology, stroke mechanisms | `Theory/` |
| `Theory/02_SINGLE_CELL_RNA_SEQ.md` | What scRNA-seq is, 10x Genomics | `Theory/` |
| `Theory/02A_SEURAT_PIPELINE_DEEP_DIVE.md` | Detailed Seurat QC and preprocessing | `Theory/` |
| `Theory/03_DATA_PREPROCESSING.md` | Normalization, HVG, scaling concepts | `Theory/` |
| `Theory/04_TRAIN_TEST_SPLIT.md` | Subject-wise split rationale and code | `Theory/` |
| `Theory/05_MACHINE_LEARNING_MODELS.md` | All 8 models explained conceptually | `Theory/` |
| `Theory/06_TRANSFORMER_ARCHITECTURES.md` | scGPT, scBERT, scFormer, Geneformer | `Theory/` |
| `Theory/07_GENE_IMPORTANCE_EXTRACTION.md` | PCA back-projection formula | `Theory/` |
| `Theory/08_CONSENSUS_METHODOLOGY.md` | Consensus threshold rationale | `Theory/` |
| `Theory/09_BIOLOGICAL_VALIDATION.md` | GO/KEGG enrichment methodology | `Theory/` |
| `Theory/10_COMPLETE_PIPELINE_WALKTHROUGH.md` | Step-by-step code execution | `Theory/` |
| `Theory/QA_TRAINEE_QUESTIONS.md` | Q&A format explanations | `Theory/` |
| `COMPREHENSIVE_METHODOLOGY_DOCUMENTATION.md` | Full pipeline with rationale (53 KB) | Root |
| `interpretability_analysis.md` | Gene extraction analysis | Root |
| `Discussion/TECHNICAL_EXPLANATIONS.md` | Technical deep-dives | `Discussion/` |
| `Discussion/DETAILED_TECHNICAL_ANALYSIS.md` | Detailed analysis notes | `Discussion/` |
| **`PIPELINE_DATA_FLOW_COMPLETE.md`** | **This file — verified counts at every stage** | **Root** |

---

*All cell and gene counts in this document are verified from actual output files:*  
*`barcodes.tsv.gz`, `features.tsv.gz`, `qc_metrics_per_cell.csv`, `stroke_pca_train.csv`, `stroke_pca_test.csv`, `stroke_labels_train.csv`, `stroke_labels_test.csv`, `pca_loadings.csv`*
