# Chapter 5: Machine Learning Models

## All 8 Models Explained Simply

---

## 5.1 Our Model Zoo

We use **8 different models** from 3 families:

| Family | Models | Paradigm |
|--------|--------|----------|
| **Classical ML** | Logistic Regression, Random Forest, XGBoost | Established, interpretable |
| **Hybrid** | CNN + XGBoost | Deep features + tree classifier |
| **Transformer-inspired** | scGPT, scBERT, scFormer, Geneformer | Neural networks inspired by foundation models |

### Why So Many Models?

**The consensus principle:** A gene is more likely to be a genuine biomarker if MULTIPLE DIFFERENT algorithms identify it as important.

---

## 5.2 Model 1: Logistic Regression

### Complexity: ★☆☆☆☆ (Simplest)

### What It Does
Draws a line (or plane) to separate stroke from control cells.

### Mathematical Intuition

```
Input: 50 PCs (PC1, PC2, ..., PC50)

Formula:
score = w1×PC1 + w2×PC2 + ... + w50×PC50 + bias

If score > 0 → Predict STROKE
If score < 0 → Predict CONTROL
```

### Visual Explanation

```
        PC2
         ↑
         |    ○ ○ ○ (Stroke)
         |   ○ ○
         |  ○ ○
         |_______~~~~~~~~~~~  ← Decision boundary (line)
         | ● ● ●
         |  ● ● ● (Control)
         |   ● ●
         └────────────────→ PC1
```

### Interpretation
The weights (w1, w2, ...) directly tell us which PCs are important:
- Large positive w → Higher PC value → More likely stroke
- Large negative w → Higher PC value → More likely control

### Code

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Get PC importance directly!
pc_importance = np.abs(model.coef_[0])
```

---

## 5.3 Model 2: Random Forest

### Complexity: ★★☆☆☆

### What It Does
Creates many decision trees, each voting on stroke/control.

### Intuition: 20 Questions

Each tree plays "20 questions":

```
Tree 1:
Is PC5 > 0.3?
├── YES: Is PC12 < -0.5?
│   ├── YES: STROKE (80%)
│   └── NO: Is PC3 > 0.1?
│       ├── YES: CONTROL (70%)
│       └── NO: STROKE (60%)
└── NO: Is PC28 > 0.8?
    ├── YES: STROKE (75%)
    └── NO: CONTROL (85%)
```

### The "Random" Part

1. Each tree trained on random subset of cells (bootstrap)
2. Each split considers random subset of PCs
3. Makes trees diverse → better ensemble

### The "Forest" Part

```
Tree 1: STROKE
Tree 2: CONTROL  
Tree 3: STROKE     →  Majority vote: STROKE (6/10)
Tree 4: STROKE
...
Tree 200: STROKE
```

### Interpretation
**Gini Importance:** How often a PC is used to make splits

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Get PC importance
pc_importance = model.feature_importances_
```

---

## 5.4 Model 3: XGBoost

### Complexity: ★★★☆☆

### What It Does
Like Random Forest, but trees are built sequentially to fix previous mistakes.

### Key Difference from Random Forest

```
Random Forest:           XGBoost:
Build trees in parallel  Build trees sequentially
                         
Tree 1 ─┐                Tree 1
Tree 2 ─┼→ Average       ↓ (focus on errors)
Tree 3 ─┤                Tree 2
...     │                ↓ (focus on remaining errors)
Tree N ─┘                Tree 3
                         ...
```

### The "Boosting" Principle

```
Round 1: Build tree, find mistakes
Round 2: Build tree focusing on Round 1 mistakes
Round 3: Build tree focusing on Round 2 mistakes
...
Round N: Ensemble of all trees
```

### Why XGBoost Often Wins

- Regularization (prevents overfitting)
- Handles missing values
- Fast implementation
- State-of-the-art for tabular data

### Code

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# Get PC importance (gain-based)
pc_importance = model.feature_importances_
```

---

## 5.5 Model 4: CNN + XGBoost (Hybrid)

### Complexity: ★★★★☆

### What It Does
1. CNN learns new features from PCA
2. XGBoost classifies using CNN features

### Architecture

```
INPUT: 50 PCs
       ↓
┌──────────────────────────────────┐
│         CNN FEATURE EXTRACTOR     │
├──────────────────────────────────┤
│ Reshape: [50] → [50, 1]          │
│ Conv1D(64) + BatchNorm + Pool    │
│ Conv1D(128) + BatchNorm + Pool   │
│ Conv1D(256) + BatchNorm + Pool   │
│ GlobalAveragePool                 │
│ Dense(128)                        │
├──────────────────────────────────┤
│ Output: 128 CNN features          │
└──────────────────────────────────┘
       ↓
┌──────────────────────────────────┐
│         XGBOOST CLASSIFIER        │
│ Input: 128 CNN features          │
│ Output: Stroke vs Control        │
└──────────────────────────────────┘
```

### Why Hybrid?
- CNN: Learns non-linear feature combinations
- XGBoost: Good at classification with tree structure
- Together: Best of both worlds

### Interpretation Challenge

How to get gene importance from this?

```
SHAP on XGBoost → CNN feature importance (128 features)
       ↓
Gradient backprop → PC importance (50 PCs)
       ↓
PCA loadings → Gene importance (~3000 genes)
```

---

## 5.6 Models 5-8: Transformer-Inspired Architectures

### (See Chapter 6 for detailed explanations)

Quick summary:

| Model | Inspired By | Key Feature | d_model | Layers |
|-------|-------------|-------------|---------|--------|
| scGPT | GPT | Single-token transformer | 256 | 4 |
| scBERT | BERT | Pooler + classification | 512 | 6 |
| scFormer | Novel | Multi-head attention layers | 384 | 3 |
| Geneformer | Geneformer | Rank-based embeddings | 512 | 4 |

---

## 5.7 Model Comparison Summary

### Architecture Comparison

| Model | Type | Parameters | Training Time |
|-------|------|------------|---------------|
| Logistic Regression | Linear | ~100 | <1 min |
| Random Forest | Ensemble trees | ~500K | ~5 min |
| XGBoost | Boosted trees | ~300K | ~3 min |
| CNN+XGBoost | Hybrid NN | ~400K | ~15 min |
| scGPT | Transformer | ~1.3M | ~25 min |
| scBERT | Transformer | ~5.2M | ~50 min |
| scFormer | Transformer | ~2.1M | ~20 min |
| Geneformer | Transformer | ~4.8M | ~30 min |

### Performance Comparison

| Model | Test AUC | Interpretation Ease |
|-------|----------|---------------------|
| scGPT | 0.9975 | Medium |
| Logistic Regression | 0.9968 | ★★★★★ Easy |
| XGBoost | 0.9966 | ★★★★☆ Good |
| Random Forest | 0.9932 | ★★★★☆ Good |
| CNN+XGBoost | 0.9954 | ★★☆☆☆ Hard |
| scFormer | 0.9927 | ★★☆☆☆ Hard |
| scBERT | 0.9869 | ★★☆☆☆ Hard |
| Geneformer | 0.9778 | ★★☆☆☆ Hard |

### Key Insight

**Simple models match complex ones!**

This tells us:
1. The stroke signal is strong and linearly separable
2. Complex models don't hurt—they provide diversity
3. Consensus across diverse models is robust

---

## 5.8 Validation: 10-Fold Cross-Validation

### What It Is

```
TRAINING DATA (4 samples, ~X,XXX cells)
       ↓
Split into 10 equal parts (folds)

Round 1: Train on folds 2-10, test on fold 1
Round 2: Train on folds 1,3-10, test on fold 2
...
Round 10: Train on folds 1-9, test on fold 10

Report: Average performance across all 10 rounds
```

### Why We Do This

- Uses all training data for both training and validation
- Gets robust performance estimate
- Identifies if model is stable

### Our Implementation

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for train_idx, val_idx in cv.split(X_train, y_train):
    X_fold_train = X_train[train_idx]
    y_fold_train = y_train[train_idx]
    X_fold_val = X_train[val_idx]
    y_fold_val = y_train[val_idx]
    
    model.fit(X_fold_train, y_fold_train)
    predictions = model.predict(X_fold_val)
    # Calculate metrics...
```

---

## Navigation

← [Previous: Train/Test Split](./04_TRAIN_TEST_SPLIT.md) | [Return to Index](./00_README.md) | [Next: Transformer Architectures →](./06_TRANSFORMER_ARCHITECTURES.md)
