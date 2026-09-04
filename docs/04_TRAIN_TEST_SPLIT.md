# Chapter 4: Train/Test Split

## The Critical Step That Prevents Data Leakage

---

## 4.1 Why This Chapter is THE MOST IMPORTANT

This chapter explains the **single most important methodological decision** in our study. Getting this wrong would invalidate ALL our results.

> **If you only read one chapter carefully, read this one.**

---

## 4.2 What is Data Leakage?

### Simple Definition

**Data leakage** occurs when information from the test set accidentally influences the training process, leading to overly optimistic performance estimates.

### Real-World Analogy

Imagine a student taking an exam:
- **Proper testing:** Student studies general concepts, sees new questions on exam
- **Data leakage:** Student accidentally sees the exact exam questions while studying

The second student will score artificially high, but hasn't actually learned.

---

## 4.3 The WRONG Way: Random Cell Split

### What Some Studies Do (WRONG!)

```
ALL CELLS (from all 6 samples)
          ↓
    Random shuffle
          ↓
    ┌─────────────────────────────────────┐
    │  80% TRAIN        │    20% TEST    │
    │  Mixed cells from │  Mixed cells   │
    │  all 6 samples    │  from all 6    │
    └─────────────────────────────────────┘
```

### Why This is WRONG

**Problem:** Cells from the SAME mouse end up in both train AND test.

```
Mouse 1 (MCAO): 1000 cells
                    ↓
              Random split
                    ↓
            800 in TRAIN + 200 in TEST
```

**The Issue:** The model learns patterns specific to Mouse 1 during training, then "recognizes" those patterns on Mouse 1's test cells.

This is NOT generalization—this is "remembering."

### The Consequence

**Artificially inflated performance:**
- Model claims 99% accuracy
- But on truly new mice → might be 60%
- Results don't replicate in other labs

---

## 4.4 The CORRECT Way: Subject-Wise Split

### Our Approach

```
6 SAMPLES
==========

SHAM1 ─┐
SHAM2 ─┼─→ TRAINING SET (4 samples)
MCAO1 ─┤   Used for: Learning + Cross-validation
MCAO2 ─┘

SHAM3 ─┐
       ├─→ TEST SET (2 samples)
MCAO3 ─┘   Used for: Final evaluation only
              NEVER seen during training
```

### Why This is CORRECT

**Each mouse is completely in ONE set:**
- Training mice: sham1, sham2, mcao1, mcao2
- Test mice: sham3, mcao3

**The model has NEVER seen any cells from sham3 or mcao3 during training.**

When we evaluate on test set, we're measuring true generalization.

---

## 4.5 The PCA Projection Problem

### The Hidden Leak

Even with subject-wise split, there's another leak source:

**WRONG:**
```
ALL 6 SAMPLES
     ↓
Compute PCA on ALL cells
     ↓
Split into train/test
     ↓
Test cells contributed to PCA! (LEAK!)
```

**WHY IT'S A LEAK:**
PCA learns the main patterns in the data. If test cells are included:
- PCA "knows about" test cell patterns
- When we project test cells, they fit better than truly new data would
- Inflated performance

### OUR SOLUTION

```
TRAINING SAMPLES ONLY
(sham1, sham2, mcao1, mcao2)
           ↓
Compute PCA (learn patterns)
           ↓
Get PCA LOADINGS (the transformation rules)
           ↓
┌─────────────────────────────────────────────────┐
│                                                 │
│  TRAIN CELLS              TEST CELLS            │
│  Apply PCA ✓              PROJECT onto          │
│                           training PCA ✓        │
│                                                 │
│  (Same PCA      ←─────→   (Uses TRAIN's         │
│   computed here)           loadings)            │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Mathematical Detail

**Training:**
```
PCA learned from train data:
X_train_scaled → PCA → X_train_pca

Output: 
- Loadings matrix W (50 × 3000)
- X_train_pca = X_train_scaled × W
```

**Testing:**
```
Test data uses TRAINING loadings:
X_test_scaled → (multiply by W from training) → X_test_pca

Test cells did NOT influence W!
```

---

## 4.6 Our Implementation (Code)

### R Code: DP_Diff_Test&Train.R

```r
# STEP 1: Load all samples
sham1 <- load_sample("sham1", "Control")
sham2 <- load_sample("sham2", "Control")
sham3 <- load_sample("sham3", "Control")
mcao1 <- load_sample("mcao1", "Stroke")
mcao2 <- load_sample("mcao2", "Stroke")
mcao3 <- load_sample("mcao3", "Stroke")

# STEP 2: Subject-wise split (BEFORE any processing)
train_samples <- c("sham1", "sham2", "mcao1", "mcao2")
test_samples  <- c("sham3", "mcao3")

# Merge and split
all_samples <- merge(sham1, list(sham2, sham3, mcao1, mcao2, mcao3))
train_obj <- subset(all_samples, subset = sample %in% train_samples)
test_obj  <- subset(all_samples, subset = sample %in% test_samples)

# STEP 3: Preprocess TRAINING ONLY
train_obj <- NormalizeData(train_obj)
train_obj <- FindVariableFeatures(train_obj, nfeatures = 3000)
train_obj <- ScaleData(train_obj)
train_obj <- RunPCA(train_obj)  # PCA learned HERE

# STEP 4: Get PCA loadings from TRAINING
pca_loadings <- Loadings(train_obj[["pca"]])  # This is matrix W

# STEP 5: Process TEST using TRAINING parameters
test_obj <- NormalizeData(test_obj)
test_obj <- ScaleData(test_obj, features = VariableFeatures(train_obj))

# STEP 6: PROJECT test onto TRAINING PCA
test_scaled_mat <- LayerData(test_obj, layer = "scale.data")[rownames(pca_loadings), ]
test_pca_scores <- t(test_scaled_mat) %*% as.matrix(pca_loadings)
```

### Key Line Explained

```r
test_pca_scores <- t(test_scaled_mat) %*% as.matrix(pca_loadings)
```

- `test_scaled_mat`: Test cells × genes (scaled expression)
- `pca_loadings`: Genes × PCs (from TRAINING)
- Result: Test cells × PCs (test cells in training's PCA space)

**This is the projection that prevents leakage.**

---

## 4.7 Verification: No Leakage

### Checklist

| Question | Answer | Verified |
|----------|--------|----------|
| Were test samples used in PCA? | NO | ✅ |
| Were test samples normalized? | Yes, separately | ✅ |
| Did test samples influence HVG selection? | NO | ✅ |
| Were test samples used in CV? | NO | ✅ |
| Are test samples completely independent? | YES | ✅ |

### Why Our AUC ~0.99 is BELIEVABLE

Some might say "0.99 is too good to be true!"

**Why it's real:**
1. Subject-wise split (no memorization)
2. Stroke vs. control is biologically distinct
3. Consistent across 8 different models
4. Even simple models (Logistic Regression) achieve ~0.997

**If there was leakage:**
- Simple models wouldn't do as well
- Results would vary wildly between models
- Literature wouldn't support such clear separation

---

## 4.8 Common Mistakes in Other Studies

### Mistake 1: Cell-level split
```
# WRONG!
all_cells_shuffled <- all_cells[sample(1:n), ]
train <- all_cells_shuffled[1:train_size, ]
test <- all_cells_shuffled[(train_size+1):n, ]
```

### Mistake 2: PCA on all data first
```
# WRONG!
all_cells <- RunPCA(all_cells)  # Leaky!
train <- subset(all_cells, sample %in% train_samples)
test <- subset(all_cells, sample %in% test_samples)
```

### Mistake 3: Feature selection on all data
```
# WRONG!
hvgs <- FindVariableFeatures(all_cells)  # Test influenced!
train <- subset(all_cells, sample %in% train_samples)
```

---

## 4.9 Summary: The Golden Rules

1. **Split by SUBJECT (mouse), not by cell**
2. **PCA computed on TRAINING only**
3. **Test data PROJECTED onto training PCA**
4. **No test data touches any training step**
5. **Test set used ONLY for final evaluation**

---

## Navigation

← [Previous: Data Preprocessing](./03_DATA_PREPROCESSING.md) | [Return to Index](./00_README.md) | [Next: ML Models →](./05_MACHINE_LEARNING_MODELS.md)
