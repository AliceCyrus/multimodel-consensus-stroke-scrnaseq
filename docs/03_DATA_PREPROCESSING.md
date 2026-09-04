# Chapter 3: Data Preprocessing

## From Raw Counts to Clean Features

---

## 3.1 The Preprocessing Pipeline Overview

### Our Goal
Transform raw, noisy count data into clean, normalized features suitable for machine learning.

### The Complete Flow

```
RAW COUNT MATRIX
(genes × cells)
      ↓
┌─────────────────────────────────────┐
│  STEP 1: Quality Control (QC)       │
│  Remove dead/dying cells            │
│  Remove doublets                    │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│  STEP 2: Normalization              │
│  Make cells comparable              │
│  Log transform                      │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│  STEP 3: Feature Selection          │
│  Select 3,000 most variable genes   │
│  Remove uninformative genes         │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│  STEP 4: Scaling                    │
│  Standardize (mean=0, var=1)        │
│  Prepare for PCA                    │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│  STEP 5: PCA                        │
│  Reduce to 50 dimensions            │
│  Capture main variation             │
└─────────────────────────────────────┘
      ↓
CLEAN PCA MATRIX
(50 PCs × cells)
Ready for ML!
```

---

## 3.2 Step 1: Quality Control (QC)

### Why We Need QC

Not all "cells" in our data are real, healthy cells:
- **Empty droplets:** Barcoded beads with no cell (only ambient RNA)
- **Dead cells:** Dying cells leaking contents
- **Doublets:** Two cells in one droplet

### QC Metrics

#### Metric 1: nFeature_RNA (Genes Detected)
```
nFeature_RNA = number of genes with count > 0 in a cell
```

**Interpretation:**
- Too LOW (<200): Empty droplet or dead cell
- Too HIGH (>2500): Likely a doublet

**Our Filter:** 200 < nFeature_RNA < 2500

#### Metric 2: percent.mt (Mitochondrial Percentage)
```
percent.mt = (mitochondrial UMIs / total UMIs) × 100
```

**Why It Matters:**
- Dying cells lose cytoplasmic content first
- Mitochondria retained longer
- High %mt = dying cell

**Our Filter:** percent.mt < 10%

### QC Code (R/Seurat)

```r
# Calculate percent mitochondrial
# ^mt- matches mouse mitochondrial genes
seurat_obj[["percent.mt"]] <- PercentageFeatureSet(seurat_obj, pattern = "^mt-")

# Filter cells
seurat_obj <- subset(seurat_obj, 
                     subset = nFeature_RNA > 200 & 
                              nFeature_RNA < 2500 & 
                              percent.mt < 10)
```

### Visual Understanding

```
QC Filter Visualization:

         nFeature_RNA
              ↑
        2500--|--------- (Doublet zone)
              |    ✓ ✓
              |  ✓ ✓ ✓ ✓
              |✓ ✓ ✓ ✓ ✓   ← Keep these cells
              |  ✓ ✓ ✓
         200--|--------- (Dead cell zone)
              └──────────────→ percent.mt
                          10%
```

---

## 3.3 Step 2: Normalization

### Why Normalize?

**Problem:** Different cells have different sequencing depths (total reads).

```
Cell A: 10,000 total UMIs → Gene X has 100 counts
Cell B: 20,000 total UMIs → Gene X has 100 counts

Gene X in Cell A = 100/10,000 = 1.0% of transcripts
Gene X in Cell B = 100/20,000 = 0.5% of transcripts

Same raw count, different biological meaning!
```

### LogNormalize Method

**Formula:**
```
Normalized = log( (count / total_counts) × scale_factor + 1 )
           = log( (count / total_counts) × 10,000 + 1 )
```

**Why the log?**
- Gene expression is multiplicative, not additive
- Log makes it additive (easier for statistics)
- Reduces impact of extreme values

### Code

```r
seurat_obj <- NormalizeData(seurat_obj, 
                           normalization.method = "LogNormalize",
                           scale.factor = 10000)
```

### Before vs After

```
BEFORE (Raw):                    AFTER (Normalized):
Gene    Cell1  Cell2              Gene    Cell1   Cell2
GeneA    1000    500              GeneA    3.91    3.21
GeneB       5     10              GeneB    1.61    1.79
GeneC  100000  50000              GeneC    9.21    8.52
                                         ↑
                                   Values more comparable
```

---

## 3.4 Step 3: Feature Selection

### Why Select Features?

**Problem:** 20,000 genes is too many
- Most genes don't vary between cells (uninformative)
- More features = more noise = harder to learn

**Solution:** Keep only genes that VARY across cells (highly variable genes/HVGs)

### The VST Method

**VST = Variance-Stabilizing Transformation**

Steps:
1. For each gene, calculate mean expression
2. Model expected variance based on mean
3. Find genes with MORE variance than expected
4. These are "highly variable genes"

### Visual Understanding

```
                Variance
                    ↑
                    |     ★ ← Highly variable (keep!)
                    |   ★   ★
                    |  ★  ★
           Expected |--------★-------- Fitted line
                    |    ●
            trend   |  ●   ●
                    |●  ●
                    └────────────────→ Mean expression
                    
★ = Selected (above expected variance)
● = Not selected (expected variance)
```

### Our Settings

```r
seurat_obj <- FindVariableFeatures(seurat_obj, 
                                   selection.method = "vst", 
                                   nfeatures = 3000)
```

**We keep 3,000 most variable genes** - enough to capture biology, not so many to add noise.

---

## 3.5 Step 4: Scaling

### Why Scale?

**Problem:** Genes have different expression levels
- Gene A: ranges from 0-1000
- Gene B: ranges from 0-10

In PCA, Gene A would dominate just because of scale.

### Z-score Scaling

**Formula:**
```
scaled_value = (value - mean) / standard_deviation
```

**Result:**
- Every gene: mean = 0, variance = 1
- All genes now equally weighted

### Code

```r
seurat_obj <- ScaleData(seurat_obj, features = VariableFeatures(seurat_obj))
```

### Before vs After

```
BEFORE (Normalized):              AFTER (Scaled):
Gene    Cell1   Cell2             Gene    Cell1   Cell2
GeneA  103.5   102.1              GeneA    1.2    -0.3
GeneB    5.3     4.9              GeneB    0.8    -0.2
                                        ↑
                              Comparable scales
```

---

## 3.6 Step 5: Principal Component Analysis (PCA)

### Why PCA?

**Problem:** Even with 3,000 genes, that's still too many dimensions.

**PCA reduces dimensions while keeping important information.**

### What PCA Does (Simple Explanation)

Imagine your data as points in 3D space. PCA finds:
1. The direction of MOST variation (PC1)
2. The perpendicular direction with next-most variation (PC2)
3. And so on...

```
Original Data (3D)          After PCA (2D)
       z                           PC2
       ↑                            ↑
       |    ●                       |●
       | ● ● ●                      |●●●           
       |  ●                         |●
       └────────→ y        →        └────────→ PC1
      /
     x               Projected onto best-fit plane
```

### Mathematical Definition

Each PC is a **linear combination** of genes:
```
PC1 = 0.3×GeneA + 0.2×GeneB - 0.1×GeneC + ...
PC2 = 0.1×GeneA - 0.4×GeneB + 0.2×GeneC + ...
...
```

The coefficients (0.3, 0.2, -0.1, ...) are called **loadings**.

### Our Settings

```r
seurat_obj <- RunPCA(seurat_obj, features = VariableFeatures(seurat_obj))
# Default: 50 PCs
```

### Output

| What | Dimensions | Description |
|------|------------|-------------|
| **Cell embeddings** | cells × 50 | Each cell's position in PC space |
| **Gene loadings** | genes × 50 | How much each gene contributes to each PC |

### Why 50 PCs?

```
Variance Explained by PCs:
PC1:  ████████████████ 15%
PC2:  ████████████ 10%
PC3:  ██████████ 8%
PC4:  ████████ 6%
...
PC10: ████ 3%
...
PC50: █ 0.5%

Cumulative to PC50: ~90% of variance explained
```

---

## 3.7 The Critical Output Files

After preprocessing, we save these files:

| File | Content | Dimensions | Use |
|------|---------|------------|-----|
| `stroke_pca_train.csv` | Training cells in PC space | cells × 50 PCs | Model training |
| `stroke_pca_test.csv` | Test cells in PC space | cells × 50 PCs | Model testing |
| `stroke_labels_train.csv` | Training labels | cells × 1 | Supervision |
| `stroke_labels_test.csv` | Test labels | cells × 1 | Evaluation |
| `pca_loadings.csv` | Gene contributions | genes × 50 PCs | Interpretation |

---

## 3.8 Complete Preprocessing Code

```r
library(Seurat)
library(dplyr)
set.seed(42)

# Load sample (repeat for each)
load_sample <- function(path, name, condition) {
  data <- Read10X(data.dir = path)
  obj <- CreateSeuratObject(counts = data, project = name)
  obj$sample <- name
  obj$condition <- condition
  return(obj)
}

# Load all 6 samples
sham1 <- load_sample("data/sham1", "sham1", "Control")
sham2 <- load_sample("data/sham2", "sham2", "Control")
sham3 <- load_sample("data/sham3", "sham3", "Control")
mcao1 <- load_sample("data/mcao1", "mcao1", "Stroke")
mcao2 <- load_sample("data/mcao2", "mcao2", "Stroke")
mcao3 <- load_sample("data/mcao3", "mcao3", "Stroke")

# Merge all
combined <- merge(sham1, y = list(sham2, sham3, mcao1, mcao2, mcao3))

# QC
combined[["percent.mt"]] <- PercentageFeatureSet(combined, pattern = "^mt-")
combined <- subset(combined, 
                   nFeature_RNA > 200 & 
                   nFeature_RNA < 2500 & 
                   percent.mt < 10)

# Normalize
combined <- NormalizeData(combined)

# Feature selection
combined <- FindVariableFeatures(combined, nfeatures = 3000)

# Scale
combined <- ScaleData(combined, features = VariableFeatures(combined))

# PCA
combined <- RunPCA(combined, npcs = 50)
```

---

## 3.9 Key Terms Summary

| Term | Definition |
|------|------------|
| **QC** | Quality Control - removing bad cells |
| **nFeature_RNA** | Number of genes detected per cell |
| **percent.mt** | Percentage of mitochondrial genes |
| **Normalization** | Making cells comparable |
| **LogNormalize** | Log-transform normalized counts |
| **HVG** | Highly Variable Genes - most informative genes |
| **Scaling** | Z-score standardization (mean=0, var=1) |
| **PCA** | Principal Component Analysis - dimensionality reduction |
| **PC** | Principal Component - new feature |
| **Loadings** | Gene contributions to each PC |

---

## Navigation

← [Previous: scRNA-seq Technology](./02_SINGLE_CELL_RNA_SEQ.md) | [Return to Index](./00_README.md) | [Next: Train/Test Split →](./04_TRAIN_TEST_SPLIT.md)
