# Chapter 7: Gene Importance Extraction

## How We Identify Important Genes

---

## 7.1 The Central Challenge

### The Problem

We trained models on **50 PCA components**.
We want to identify important **GENES** (~3,000).

**How do we go from PC importance → Gene importance?**

### The Solution

```
MODEL                PCA LOADINGS
  ↓                      ↓
PC Importance    ×    Gene-to-PC
(50 values)          (genes × 50)
                          ↓
                  Gene Importance
                  (~3,000 values)
```

---

## 7.2 Step 1: Extract PC Importance (Different for Each Model)

### For Classical ML Models

#### Logistic Regression

```python
# Coefficients directly give PC importance
pc_importance = np.abs(model.coef_[0])  # Shape: (50,)
```

**Interpretation:**
- `coef[i]` = how much does PC_i influence the prediction
- Larger absolute value = more important

#### Random Forest / XGBoost

```python
# Built-in feature importance
pc_importance = model.feature_importances_  # Shape: (50,)
```

**What it measures:**
- **Gini importance** (RF): How much does splitting on this PC reduce impurity?
- **Gain importance** (XGB): How much does this PC improve predictions?

---

### For Transformer Models: L2 Norm Method

#### The Intuition

The first layer of a transformer projects 50 PCs → 256 dimensions.

```
INPUT           TRANSFORMATION           HIDDEN
[PC1]  ───→     [w1,1  w2,1  ...  w256,1]
[PC2]  ───→     [w1,2  w2,2  ...  w256,2]    =    [h1, h2, ... h256]
...
[PC50] ───→     [w1,50 w2,50 ... w256,50]
```

If a PC is important:
- The model will learn LARGE weights connecting it to hidden dimensions
- The MAGNITUDE (L2 norm) of its weight column will be high

#### The Math

```python
# Get first layer weights
weights = model.input_projection.weight.data  # Shape: (256, 50)

# Compute L2 norm for each PC (column)
pc_importance = np.linalg.norm(weights, axis=0)  # Shape: (50,)

# L2 norm formula:
# pc_importance[i] = sqrt(w[0,i]² + w[1,i]² + ... + w[255,i]²)
```

#### Visual Explanation

```
Weight Matrix (256 × 50):

           PC_1    PC_2    PC_3   ...  PC_50
dim_0    [ 0.3     0.1     0.8    ...   0.2 ]
dim_1    [ 0.5     0.2     0.9    ...   0.1 ]
dim_2    [ 0.4     0.1     0.7    ...   0.3 ]
  ...       ↓       ↓       ↓            ↓
dim_255  [ 0.2     0.3     0.6    ...   0.1 ]
         ─────────────────────────────────────
L2 norm:   2.1     1.0     5.3    ...   0.8

→ PC_3 is MOST important (highest norm: 5.3)
→ PC_50 is LEAST important (lowest norm: 0.8)
```

---

### For Geneformer: Special Case

Geneformer doesn't have a simple `input_projection`. Instead, it uses `gene_positions`.

```python
# Geneformer stores learned position embeddings
gene_positions = model.gene_positions.data  # Shape: (1, 50, 512)
gene_positions = gene_positions.squeeze(0)   # Shape: (50, 512)

# L2 norm per position (each position = one PC)
pc_importance = np.linalg.norm(gene_positions, axis=1)  # Shape: (50,)
```

**Interpretation:** PCs with larger position embeddings are more important.

---

### For CNN+XGBoost: SHAP + Gradients

This is the most complex extraction:

```
STEP 1: SHAP on XGBoost
────────────────────────
XGBoost takes 128 CNN features as input.
Use SHAP TreeExplainer to get importance of each CNN feature.

cnn_feature_importance: (128,)


STEP 2: Gradient Backprop through CNN
──────────────────────────────────────
For each CNN feature, compute:
d(CNN_feature_i) / d(PCA_input_j)

This tells us: "How sensitive is CNN feature i to PC j?"

gradients: (128, 50)


STEP 3: Combine
───────────────
pc_importance[j] = Σ_i cnn_importance[i] × |gradient[i, j]|

Weighted sum: Important CNN features × their sensitivity to each PC
```

#### Code

```python
# SHAP values for XGBoost
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(cnn_features)
cnn_feature_importance = np.abs(shap_values).mean(axis=0)

# Gradient backprop through CNN
pca_importance = np.zeros(50)
for feature_idx in range(128):
    with tf.GradientTape() as tape:
        tape.watch(X_input)
        cnn_output = cnn_model(X_input)[:, feature_idx]
    gradients = tape.gradient(cnn_output, X_input)
    avg_gradient = np.mean(np.abs(gradients), axis=0)
    pca_importance += cnn_feature_importance[feature_idx] * avg_gradient
```

---

## 7.3 Step 2: Map PC Importance to Gene Importance

### The PCA Loadings Matrix

When we ran PCA, we got:
```
X_cells_pca = X_cells_genes × PCA_loadings
```

Where `PCA_loadings` is a (genes × PCs) matrix.

```
             PC1    PC2    PC3   ...   PC50
GeneA      [ 0.05  -0.02   0.15  ...   0.01]  ← How much does GeneA
GeneB      [ 0.10   0.08  -0.03  ...   0.07]       contribute to each PC?
GeneC      [-0.15   0.20   0.05  ...  -0.12]
...
Gene3000   [ 0.03  -0.05   0.11  ...   0.04]
```

### The Mapping Formula

```
For each gene g:

gene_importance[g] = Σ_j |loading[g, j]| × pc_importance[j]
```

**In plain English:**
- A gene is important if it contributes heavily to important PCs
- We sum across all PCs, weighted by PC importance

### Code

```python
def map_pc_to_genes(pc_importance, pca_loadings):
    """
    Map PC importance to gene importance.
    
    Args:
        pc_importance: (50,) array - importance of each PC
        pca_loadings: DataFrame with genes as rows, PCs as columns
    
    Returns:
        gene_importance: (n_genes,) array
    """
    # Get loading matrix (genes × PCs)
    loading_matrix = pca_loadings.iloc[:, 1:].values  # Skip 'gene' column
    
    # Absolute loadings × PC importance
    gene_importance = np.abs(loading_matrix) @ pc_importance
    
    # Normalize to sum to 1
    gene_importance = gene_importance / gene_importance.sum()
    
    return gene_importance
```

### Visual Example

```
Suppose:
- PC1 is very important (importance = 0.5)
- PC2 is less important (importance = 0.1)

Gene A: loads heavily on PC1 (0.3), lightly on PC2 (0.05)
Gene B: loads lightly on PC1 (0.02), heavily on PC2 (0.4)

Gene A importance: |0.3| × 0.5 + |0.05| × 0.1 = 0.15 + 0.005 = 0.155
Gene B importance: |0.02| × 0.5 + |0.4| × 0.1 = 0.01 + 0.04 = 0.05

→ Gene A is more important because it loads on the important PC!
```

---

## 7.4 The Complete Extraction Pipeline

```
MODEL TRAINING COMPLETE
          ↓
┌─────────────────────────────────────────────────┐
│  FOR EACH MODEL:                                │
│                                                 │
│  1. Extract PC importance                       │
│     - Classical: coefficients/feature_importances│
│     - Transformers: L2 norm of first layer     │
│     - CNN+XGB: SHAP + gradient backprop        │
│                                                 │
│  2. Load PCA loadings (genes × PCs)            │
│                                                 │
│  3. Compute gene importance:                   │
│     gene_imp = |loadings| × pc_importance      │
│                                                 │
│  4. Save ranked gene list                      │
│                                                 │
└─────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────┐
│  OUTPUT FILES:                                  │
│  - gene_importance_LogisticRegression.csv      │
│  - gene_importance_RandomForest.csv            │
│  - gene_importance_XGBoost.csv                 │
│  - gene_importance_CNN_XGBoost.csv             │
│  - gene_importance_scGPT.csv                   │
│  - gene_importance_scBERT.csv                  │
│  - gene_importance_scFormer.csv                │
│  - gene_importance_Geneformer.csv              │
└─────────────────────────────────────────────────┘
```

---

## 7.5 Quality Check: Do Models Agree?

### Jaccard Similarity

```
For two models' top-100 gene lists:

Jaccard = |Intersection| / |Union|
        = |genes in BOTH lists| / |genes in EITHER list|

Range: 0 (no overlap) to 1 (identical lists)
```

### Our Results

| Model Pair | Jaccard |
|------------|---------|
| scGPT ↔ scBERT | 0.98 |
| scGPT ↔ scFormer | 0.99 |
| RF ↔ XGBoost | 0.83 |
| LR ↔ RF | 0.45 |
| Geneformer ↔ scGPT | 0.77 |

### Interpretation

- **Transformer models agree strongly** (0.77-0.99)
- **Tree-based models agree** (0.83)
- **Linear vs. tree models differ more** (0.45)

This is expected and good—different model families capture different patterns!

---

## 7.6 Summary

| Model Type | PC Extraction Method | Validity |
|------------|---------------------|----------|
| Logistic Regression | Absolute coefficients | ★★★★★ Direct |
| Random Forest | Gini importance | ★★★★★ Direct |
| XGBoost | Gain importance | ★★★★★ Direct |
| CNN+XGBoost | SHAP + Gradients | ★★★★☆ Indirect but valid |
| Transformers | L2 norm of first layer | ★★★★☆ Heuristic but reasonable |
| Geneformer | L2 norm of gene_positions | ★★★★☆ Heuristic but reasonable |

---

## Navigation

← [Previous: Transformer Architectures](./06_TRANSFORMER_ARCHITECTURES.md) | [Return to Index](./00_README.md) | [Next: Consensus Methodology →](./08_CONSENSUS_METHODOLOGY.md)
