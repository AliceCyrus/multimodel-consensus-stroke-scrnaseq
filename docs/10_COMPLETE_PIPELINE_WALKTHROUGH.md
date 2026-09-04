# Chapter 10: Complete Pipeline Walkthrough

## Step-by-Step Code Execution Guide

---

## 10.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE ANALYSIS PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   PHASE 1: R Preprocessing                                              │
│   ├── DP_Diff_Test&Train.R                                             │
│   └── export_data_for_python.R                                         │
│                           ↓                                              │
│   PHASE 2: Python Model Training                                        │
│   ├── phase1_framework.py (Classical ML)                               │
│   ├── fixed_foundation_models.py (Transformers)                        │
│   └── cnn_xgboostModel.py (Hybrid)                                     │
│                           ↓                                              │
│   PHASE 3: Gene Importance Extraction                                   │
│   ├── integrated_interpretability_pipeline.py                          │
│   ├── attention_extractor.py                                           │
│   ├── fix_geneformer_extraction.py                                     │
│   └── shap_cnn_xgboost_extraction_fixed.py                             │
│                           ↓                                              │
│   PHASE 4: Biological Validation                                        │
│   ├── go_enrichment.R                                                  │
│   └── keggPathways.R                                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 10.2 Prerequisites

### Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| R | ≥ 4.0 | Preprocessing, enrichment |
| Python | ≥ 3.8 | ML models |
| CUDA (optional) | ≥ 11.0 | GPU acceleration |

### Python Packages

```bash
pip install numpy pandas scikit-learn scipy
pip install torch torchvision  # PyTorch for transformers
pip install tensorflow keras   # For CNN
pip install xgboost shap
pip install matplotlib seaborn
```

### R Packages

```r
install.packages(c("Seurat", "dplyr"))
# For Bioconductor packages:
BiocManager::install(c("clusterProfiler", "org.Mm.eg.db"))
```

---

## 10.3 Phase 1: R Preprocessing

### Step 1.1: Create Train/Test Split

**File:** `DP_Diff_Test&Train.R`

**Purpose:** Split data by subject, preprocess, compute PCA on training only.

```r
# Run in R
setwd("/path/to/multimodel-consensus-stroke-scrnaseq")
source("Phase4_RevisedModels/DP_Diff_Test&Train.R")
```

**Outputs:**
- `stroke_pca_train.csv` - Training cells in PCA space
- `stroke_pca_test.csv` - Test cells in PCA space
- `stroke_labels_train.csv` - Training labels
- `stroke_labels_test.csv` - Test labels

### Step 1.2: Export PCA Loadings

**File:** `export_data_for_python.R`

**Purpose:** Export gene-to-PC mapping for interpretability.

```r
source("Phase4_RevisedModels/export_data_for_python.R")
```

**Outputs:**
- `pca_loadings.csv` - (genes × 50 PCs) matrix

### Verify Phase 1

```r
# Check outputs exist
file.exists("Phase4_RevisedModels/stroke_pca_train.csv")  # TRUE
file.exists("Phase4_RevisedModels/pca_loadings.csv")      # TRUE
```

---

## 10.4 Phase 2: Model Training

### Step 2.1: Classical ML Models

**File:** `phase1_framework.py`

```python
# Navigate to project directory
cd Phase4_RevisedModels

# Run training
python phase1_framework.py
```

**What it does:**
1. Loads PCA data
2. Trains Logistic Regression, Random Forest, XGBoost
3. Performs 10-fold cross-validation
4. Computes bootstrap confidence intervals
5. Saves performance metrics and model checkpoints

**Outputs:**
- `checkpoints/logistic_regression_model.pkl`
- `checkpoints/random_forest_model.pkl`
- `checkpoints/xgboost_model.pkl`
- `results/performance_summary.json`

### Step 2.2: Foundation Model-Inspired Architectures

**File:** `fixed_foundation_models.py`

```python
python fixed_foundation_models.py
```

**What it does:**
1. Trains scGPT, scBERT, scFormer, Geneformer models
2. Uses early stopping and learning rate scheduling
3. Saves PyTorch checkpoints

**Outputs:**
- `checkpoints/scGPT_best.pt`
- `checkpoints/scBERT_best.pt`
- `checkpoints/scFormer_best.pt`
- `checkpoints/Geneformer_best.pt`

### Step 2.3: CNN+XGBoost Hybrid

**File:** `cnn_xgboostModel.py`

```python
python cnn_xgboostModel.py
```

**Outputs:**
- `checkpoints/cnn_xgboost_model.pkl`
- `checkpoints/cnn_feature_extractor.h5`

---

## 10.5 Phase 3: Gene Importance Extraction

### Step 3.1: Classical ML Importance

**File:** `integrated_interpretability_pipeline.py`

```python
python integrated_interpretability_pipeline.py
```

**What it does:**
1. Loads trained models
2. Extracts feature importance (coefficients, Gini, gain)
3. Maps PC importance to gene importance
4. Saves gene rankings

**Outputs:**
- `gene_level_results/gene_importance_LogisticRegression.csv`
- `gene_level_results/gene_importance_RandomForest.csv`
- `gene_level_results/gene_importance_XGBoost.csv`

### Step 3.2: Transformer Model Importance

**File:** `attention_extractor.py`

```python
python attention_extractor.py
```

**What it does:**
1. Loads transformer checkpoints
2. Extracts L2 norm of first layer weights
3. Maps to gene importance

**Outputs:**
- `transformer_importances/gene_importance_scGPT.csv`
- `transformer_importances/gene_importance_scBERT.csv`
- `transformer_importances/gene_importance_scFormer.csv`

### Step 3.3: Geneformer Special Extraction

**File:** `fix_geneformer_extraction.py`

```python
python fix_geneformer_extraction.py
```

**Why separate:** Geneformer uses `gene_positions` parameter, not `input_projection`.

**Output:**
- `transformer_importances/gene_importance_Geneformer.csv`

### Step 3.4: CNN+XGBoost Importance

**File:** `shap_cnn_xgboost_extraction_fixed.py`

```python
python shap_cnn_xgboost_extraction_fixed.py
```

**What it does:**
1. Computes SHAP values on XGBoost
2. Backpropagates through CNN
3. Maps to gene importance

**Output:**
- `gene_level_results/gene_importance_CNN_XGBoost.csv`

---

## 10.6 Phase 4: Consensus and Validation

### Step 4.1: Find Consensus Genes

**Part of:** `integrated_interpretability_pipeline.py`

The pipeline already computes consensus. To run standalone:

```python
# In Python
from integrated_interpretability_pipeline import find_consensus_genes

# Load all gene importance files
all_importance = load_all_gene_importance()

# Find consensus
consensus_genes, gene_counts = find_consensus_genes(
    all_importance, 
    min_models=4, 
    top_n=100
)

# Save
pd.DataFrame({'gene': consensus_genes}).to_csv('consensus_genes.csv', index=False)
```

**Output:**
- `gene_level_results/consensus_genes.csv`

### Step 4.2: GO Enrichment

**File:** `GO-EnrichmentAnalysis/go_enrichment.R`

```r
setwd("Phase4_RevisedModels")
source("GO-EnrichmentAnalysis/go_enrichment.R")
```

**Outputs:**
- `GO-EnrichmentAnalysis/go_results.csv`
- `GO-EnrichmentAnalysis/go_dotplot.pdf`

### Step 4.3: KEGG Pathways

**File:** `GO-EnrichmentAnalysis/keggPathways.R`

```r
source("GO-EnrichmentAnalysis/keggPathways.R")
```

**Outputs:**
- `GO-EnrichmentAnalysis/kegg_results.csv`
- `GO-EnrichmentAnalysis/kegg_dotplot.pdf`

---

## 10.7 Complete File Inventory

### Input Files (Created in Phase 1)

| File | Description |
|------|-------------|
| `stroke_pca_train.csv` | Training cells × 50 PCs |
| `stroke_pca_test.csv` | Test cells × 50 PCs |
| `stroke_labels_train.csv` | Training labels |
| `stroke_labels_test.csv` | Test labels |
| `pca_loadings.csv` | Genes × 50 PCs |

### Model Checkpoints (Created in Phase 2)

| File | Description |
|------|-------------|
| `checkpoints/*.pkl` | Classical ML models |
| `checkpoints/*.pt` | PyTorch transformer models |
| `checkpoints/*.h5` | CNN feature extractor |

### Gene Importance (Created in Phase 3)

| File | Description |
|------|-------------|
| `gene_level_results/gene_importance_*.csv` | Per-model gene rankings |
| `transformer_importances/gene_importance_*.csv` | Transformer gene rankings |

### Final Results (Created in Phase 4)

| File | Description |
|------|-------------|
| `consensus_genes.csv` | 63 consensus genes |
| `GO-EnrichmentAnalysis/*.csv` | Enrichment results |
| `GO-EnrichmentAnalysis/*.pdf` | Visualizations |

---

## 10.8 Troubleshooting

### Common Issues

**Issue 1: "CUDA out of memory"**
```python
# Reduce batch size
CONFIG['batch_size'] = 64  # instead of 128
```

**Issue 2: "File not found"**
```python
# Check working directory
import os
os.getcwd()  # Should be Phase4_RevisedModels
```

**Issue 3: "Module not found"**
```bash
pip install <missing_module>
```

**Issue 4: "R package not found"**
```r
install.packages("<package>")
# Or for Bioconductor:
BiocManager::install("<package>")
```

---

## 10.9 Quick Reference Commands

### Run Everything (Bash/PowerShell)

```bash
# Phase 1 (R)
Rscript DP_Diff_Test&Train.R
Rscript export_data_for_python.R

# Phase 2 (Python)
python phase1_framework.py
python fixed_foundation_models.py
python cnn_xgboostModel.py

# Phase 3 (Python)
python integrated_interpretability_pipeline.py
python attention_extractor.py
python fix_geneformer_extraction.py
python shap_cnn_xgboost_extraction_fixed.py

# Phase 4 (R)
Rscript GO-EnrichmentAnalysis/go_enrichment.R
Rscript GO-EnrichmentAnalysis/keggPathways.R
```

### Verify Results

```python
import pandas as pd

# Check consensus genes
consensus = pd.read_csv('gene_level_results/consensus_genes.csv')
print(f"Found {len(consensus)} consensus genes")

# Check top genes
print(consensus.head(10))
```

---

## 10.10 Summary

### The Complete Journey

```
RAW DATA (10X)
     ↓
[Phase 1: R Preprocessing]
     ↓
PCA EMBEDDINGS (50-dim)
     ↓
[Phase 2: Model Training]
     ↓
8 TRAINED MODELS
     ↓
[Phase 3: Gene Extraction]
     ↓
8 GENE IMPORTANCE LISTS
     ↓
[Consensus Voting]
     ↓
63 CONSENSUS GENES
     ↓
[Phase 4: Validation]
     ↓
BIOLOGICAL PATHWAYS (IL-1, IFN, Phagocytosis)
     ↓
PUBLICATION-READY RESULTS
```

---

## Congratulations! 🎉

You've completed the Theory documentation. You now understand:

1. ✅ The biological context (stroke, MCAO model)
2. ✅ The data technology (scRNA-seq)
3. ✅ The preprocessing (QC, normalization, PCA)
4. ✅ The critical split (subject-wise, no leakage)
5. ✅ All 8 ML models
6. ✅ Transformer architectures in detail
7. ✅ Gene importance extraction methods
8. ✅ The consensus voting system
9. ✅ Biological validation
10. ✅ How to run everything

**Next step:** Write your manuscript!

---

## Navigation

← [Previous: Biological Validation](./09_BIOLOGICAL_VALIDATION.md) | [Return to Index](./00_README.md)
