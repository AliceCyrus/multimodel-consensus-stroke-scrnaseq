# Trainee Q&A: Detailed Explanations

## Questions Asked During Learning Sessions

*Last Updated: January 17, 2026*

---

## Table of Contents

1. [What does min.cells=3 mean?](#q1-what-does-mincells3-mean)
2. [How do cells and genes generate PCs?](#q2-how-do-cells-and-genes-generate-pcs)
3. [Is each cell mapped to its own PC?](#q3-is-each-cell-mapped-to-its-own-pc)
4. [What are UMIs and why normalize?](#q4-what-are-umis-and-why-normalize)
5. [Why scale after normalization?](#q5-why-scale-after-normalization)
6. [Does Z-score scale values to [-1, 1]?](#q6-does-z-score-scale-values-to--1-1)
7. [PCA deep dive: embeddings, loadings, and calculations](#q7-pca-deep-dive-embeddings-loadings-and-calculations)
8. [Why exclude Geneformer and CNN+XGBoost from consensus?](#q8-why-exclude-geneformer-and-cnnxgboost-from-consensus)

---

## Q1: What does min.cells=3 mean?

### Question
> "What does min.cells=3 means? How it removing 12k genes?"

### Answer

When you create a Seurat object, `min.cells = 3` filters out **genes that are expressed in fewer than 3 cells**.

#### The Logic

```r
CreateSeuratObject(counts = data, min.cells = 3, min.features = 200)
```

**For each gene, Seurat asks:**
> "In how many cells does this gene have at least 1 count (non-zero)?"

- If **≥ 3 cells** → **Keep the gene**
- If **< 3 cells** → **Remove the gene**

#### Why Remove These Genes?

A gene expressed in only 1 or 2 cells is:
1. **Statistically unreliable** - Could be noise or sequencing error
2. **Not informative** - Can't distinguish cell types or conditions
3. **Computationally wasteful** - Adds zeros without adding signal

#### Visual Example

```
BEFORE min.cells filtering:

           Cell1  Cell2  Cell3  Cell4  Cell5  ...  Cell65000
GeneA        5      0      3      0      7    ...      2      ← Expressed in many cells ✓
GeneB        0      1      0      0      0    ...      0      ← Expressed in 1 cell only ✗
GeneC        0      0      0      2      0    ...      1      ← Expressed in 2 cells ✗
GeneD        3      0      1      0      5    ...      0      ← Expressed in 3+ cells ✓

AFTER min.cells = 3:
- GeneA and GeneD → KEPT
- GeneB and GeneC → REMOVED (expressed in < 3 cells)
```

#### Why ~12,000 Genes Get Removed

- Mouse genome: ~32,000 annotated genes
- Many genes are not expressed in brain, very lowly expressed, or pseudogenes
- ~12,000 expressed in < 3 cells → removed
- **~20,000 genes remain**

#### The Two Parameters Together

| Parameter | Filters | Meaning |
|-----------|---------|---------|
| `min.cells = 3` | **Genes** | Gene must be detected in ≥3 cells |
| `min.features = 200` | **Cells** | Cell must have ≥200 genes detected |

---

## Q2: How do cells and genes generate PCs?

### Question
> "Explain how selected cells and genes is generating PCs. Please explain the mechanism with example."

### Answer

#### Starting Point: The Scaled Data Matrix

After preprocessing, we have:
- **Rows** = 54,599 cells
- **Columns** = 3,000 genes (selected HVGs)
- **Values** = Scaled expression (Z-scores)

#### The Problem PCA Solves

3,000 genes is too many dimensions! But many genes are correlated:
- If Gene1 goes up, Gene2 also goes up (co-expressed)

**PCA finds these patterns and creates new "super-features" (PCs).**

#### Tiny Example: 2 Genes → 1 PC

```
Scaled Data:
           Gene1   Gene2
Cell_A      1.0     0.8
Cell_B      0.5     0.4
Cell_C     -0.5    -0.6
Cell_D     -1.0    -0.9
Cell_E      0.0     0.1
```

Gene1 and Gene2 are correlated! PCA finds this pattern.

#### The PCA Process

**Step 1:** Find direction of maximum variance

PCA asks: "What linear combination of genes captures the most variation?"

**Step 2:** Define PC1

```
PC1 = 0.71 × Gene1 + 0.71 × Gene2
```

These weights (0.71, 0.71) are called **loadings**.

**Step 3:** Project each cell onto PC1

```
Cell_A: PC1 = 0.71×(1.0) + 0.71×(0.8) = 1.28
Cell_B: PC1 = 0.71×(0.5) + 0.71×(0.4) = 0.64
Cell_C: PC1 = 0.71×(-0.5) + 0.71×(-0.6) = -0.79
Cell_D: PC1 = 0.71×(-1.0) + 0.71×(-0.9) = -1.35
Cell_E: PC1 = 0.71×(0.0) + 0.71×(0.1) = 0.07
```

**Result:** 2 genes → 1 PC while keeping most information!

#### Scaling Up: 3,000 Genes → 50 PCs

Same principle, just bigger:
- Each PC is a weighted sum of all 3,000 genes
- 50 directions are found that capture ~90% of variance

---

## Q3: Is each cell mapped to its own PC?

### Question
> "If there is 54,599 cells and 3,000 genes then how 50 PCs are calculated... for cell 54,599 it will be PC54599?"

### Answer

#### The Misconception

**❌ WRONG thinking:**
```
Cell_1 → PC1
Cell_2 → PC2
...
Cell_54599 → PC54599
```

**This is NOT how PCA works!**

#### The Correct Understanding

**✅ CORRECT:** PCs are **shared axes** that ALL cells are measured on.

```
                   PC1    PC2    PC3   ...   PC50
Cell_1          [  1.2   -0.5    0.8  ...    0.3 ]
Cell_2          [ -0.3    1.4    0.2  ...   -0.7 ]
Cell_3          [  0.9   -0.8    1.1  ...    0.5 ]
...
Cell_54599      [  0.1    0.6   -0.2  ...    1.2 ]
```

**Every cell gets a score for ALL 50 PCs!**

#### Analogy: Height and Weight

```
Person    Height(cm)   Weight(kg)
─────────────────────────────────
Alice        165          55
Bob          180          80
```

The axes (Height, Weight) are **SHARED**. Each person has values on the same axes.

Similarly: PC1-PC50 are shared axes, each cell has its own values.

#### Where Does "50" Come From?

Maximum possible PCs = **min(cells, genes)** - 1 = 2,999

But we CHOOSE only 50 because:
- PC1-PC50 capture ~90% of variance
- PC51-PC2999 are mostly noise

#### Summary Table

| Concept | Value |
|---------|-------|
| Number of cells | 54,599 |
| Number of genes | 3,000 |
| Number of PCs | 50 (we choose) |
| PC1 formula | 1 (applied to ALL cells) |
| PC1 scores | 54,599 values (one per cell) |

---

## Q4: What are UMIs and why normalize?

### Question
> "If UMI is Unique Molecular Identifier, what does it mean 'Cell A has 10,000 UMIs'? And why does same raw count have different biological meaning?"

### Answer

#### What is a UMI?

A UMI is a short random DNA barcode that gets attached to **each original mRNA molecule** BEFORE amplification.

#### The Problem UMIs Solve

**Without UMIs:** PCR amplification is random - some molecules amplify more than others.

```
Original: 3 mRNA molecules
After PCR: 16 reads (random amplification)
Without UMI: We'd think there are 16 molecules!
```

**With UMIs:** Each original molecule has a unique tag.

```
Original: 3 mRNA molecules with UMI_001, UMI_002, UMI_003
After PCR: 16 reads, but only 3 unique UMIs
With UMI: We correctly count 3 original molecules!
```

#### What "10,000 UMIs in Cell A" Means

**Total UMIs = Total unique mRNA molecules detected**

```
Cell A has 10,000 total UMIs:
  GeneX:  100 UMIs
  GeneY:  500 UMIs
  GeneZ:   50 UMIs
  ...
  Total: 10,000 UMIs
```

#### The Normalization Problem

```
Cell A: 10,000 total UMIs → Gene X has 100 counts
Cell B: 20,000 total UMIs → Gene X has 100 counts
```

Both show "100 counts" but:
- Cell A: GeneX is 100/10,000 = **1.0%** of transcripts
- Cell B: GeneX is 100/20,000 = **0.5%** of transcripts

**Same raw count, different biological meaning!**

#### Visual: Pie Charts

```
Cell A (10,000 UMIs):     Cell B (20,000 UMIs):
  GeneX = 1% ████         GeneX = 0.5% ██
  Other = 99%             Other = 99.5%
```

#### After Normalization

```
Cell A: log(100/10000 × 10000 + 1) = log(101) ≈ 4.62
Cell B: log(100/20000 × 10000 + 1) = log(51) ≈ 3.93
```

Now they're correctly different!

---

## Q5: Why scale after normalization?

### Question
> "After normalization genes still have different values (GeneA~3.5, GeneC~9). Why do we need scaling?"

### Answer

#### After Normalization: Still Different Scales!

```
AFTER NORMALIZATION:
Gene    Cell1   Cell2   Absolute Scale
──────────────────────────────────────
GeneA    3.91    3.21   ~3.5
GeneB    1.61    1.79   ~1.7
GeneC    9.21    8.52   ~9.0
```

GeneC values (~9) are still much higher than GeneB (~1.7)!

#### The PCA Problem

PCA finds directions of **maximum variance**. If we don't scale:
- GeneC would dominate PC1 (bigger absolute values)
- GeneB would be ignored (smaller values)
- PCA would find "highly expressed genes" not "variable genes"

#### The Solution: Z-Score Scaling

```
scaled = (normalized - gene_mean) / gene_std
```

**Result:**

```
SCALED (Z-score):
Gene    Cell1   Cell2
─────────────────────
GeneA    1.0    -1.0
GeneB   -1.0     1.0
GeneC    1.0    -1.0
```

Now ALL genes have mean=0, variance=1. Equal contribution to PCA!

#### The Complete Picture

| Step | Purpose | What it fixes |
|------|---------|---------------|
| **Normalization** | Account for sequencing depth | Different total UMIs per cell |
| **Scaling** | Equal weight to all genes | Different expression levels per gene |

```
RAW → [Normalization] → NORMALIZED → [Scaling] → SCALED → [PCA]
           ↓                              ↓
    Fixes cell-to-cell             Fixes gene-to-gene
    depth differences              scale differences
```

---

## Q6: Does Z-score scale values to [-1, 1]?

### Question
> "The Z-score scales value between -1 to 1, is this correct?"

### Answer

#### The Misconception

**❌ WRONG:** Z-score values are always between -1 and 1.

**✅ CORRECT:** Z-score values can be **any number** (unbounded).

#### What Z-Score Actually Means

```
Z = (value - mean) / standard_deviation
```

The result tells you: **"How many standard deviations away from the mean?"**

| Z-score | Interpretation |
|---------|----------------|
| Z = 0 | Exactly at the mean |
| Z = 1 | 1 standard deviation above mean |
| Z = -1 | 1 standard deviation below mean |
| Z = 2.5 | 2.5 standard deviations above |
| Z = -3 | 3 standard deviations below |

**No fixed bounds!**

#### Example with Real Data

Suppose GeneA across 10,000 cells has:
- Mean = 5.0
- Standard deviation = 2.0

```
Cell with value 5.0:  Z = (5.0 - 5.0) / 2.0 = 0
Cell with value 7.0:  Z = (7.0 - 5.0) / 2.0 = 1.0
Cell with value 3.0:  Z = (3.0 - 5.0) / 2.0 = -1.0
Cell with value 11.0: Z = (11.0 - 5.0) / 2.0 = 3.0  ← Can exceed 1!
Cell with value 0.5:  Z = (0.5 - 5.0) / 2.0 = -2.25 ← Can be less than -1!
```

#### What Z-Score Guarantees

| Property | Value |
|----------|-------|
| Mean across cells | **0** (exactly) |
| Variance across cells | **1** (exactly) |
| Range | **Unbounded** (-∞ to +∞) |

#### Comparison with Other Scaling Methods

| Method | Range | Used by Seurat? |
|--------|-------|-----------------|
| **Z-score** | Unbounded | ✅ Yes |
| Min-Max | [0, 1] | No |
| Tanh | [-1, 1] | No |

---

## Q7: PCA deep dive: embeddings, loadings, and calculations

### Question
> "I'm confused with PCA calculations. First you show PC1 calculated for each cell, then you show a matrix where every cell has all 50 PCs. Also, how do 54,599 PC1 values accumulate? And how are loadings calculated?"

### Answer

#### Part 1: Two Matrices of PCA

PCA produces TWO separate matrices:

**MATRIX 1: CELL EMBEDDINGS (Scores)**
```
Dimensions: 54,599 cells × 50 PCs

              PC1      PC2      PC3    ...    PC50
Cell_1      [ 1.28   -0.50    0.80   ...    0.30 ]
Cell_2      [ 0.64    1.40    0.22   ...   -0.70 ]
Cell_3      [-0.79   -0.80    1.10   ...    0.50 ]
...
Cell_54599  [-1.35    0.60   -0.20   ...    1.20 ]
```

**MATRIX 2: GENE LOADINGS (Weights)**
```
Dimensions: 3,000 genes × 50 PCs

              PC1      PC2      PC3    ...    PC50
Gene1       [ 0.71    0.10   -0.05   ...    0.02 ]
Gene2       [ 0.71   -0.15    0.08   ...    0.01 ]
Gene3       [ 0.05    0.80    0.10   ...   -0.03 ]
...
Gene3000    [-0.02    0.05    0.90   ...    0.05 ]
```

#### Part 2: Reading by Row vs Column

It's the SAME embeddings matrix, just different perspectives:

**By ROW (one cell):**
```
Cell_1: [1.28, -0.50, 0.80, ..., 0.30]
       = Cell_1's position in 50-dimensional PC space
```

**By COLUMN (one PC):**
```
PC1: [1.28, 0.64, -0.79, ..., -1.35]
    = All 54,599 cells' scores on PC1
```

#### Part 3: Do 54,599 PC1 Values Accumulate?

**NO! They are INDEPENDENT scores.**

```
PC1 column:
├── Cell_1:     1.28  ← Cell_1's PC1 score (independent)
├── Cell_2:     0.64  ← Cell_2's PC1 score (independent)
├── Cell_3:    -0.79  ← Cell_3's PC1 score (independent)
...
└── Cell_54599: -1.35 ← Cell_54599's PC1 score (independent)

These don't add up. They're just 54,599 different measurements.
```

**What IS PC1?**

PC1 is a FORMULA (recipe):
```
PC1 = 0.71×Gene1 + 0.71×Gene2 + 0.05×Gene3 + ... + (-0.02)×Gene3000
```

This ONE formula is applied to each cell with its own gene values:
```
Cell_1:    PC1 = 0.71×(1.2)  + 0.71×(-0.5) + ... = 1.28
Cell_2:    PC1 = 0.71×(-0.3) + 0.71×(1.4)  + ... = 0.64
                 ↑ Same weights   ↑ Different values
```

#### Part 4: How Are Loadings Calculated?

**Step 1: Start with scaled data matrix**
```
X = 54,599 cells × 3,000 genes
```

**Step 2: Compute covariance matrix**
```
Cov = (1/n) × X^T × X

Result: 3,000 × 3,000 matrix showing how gene pairs co-vary
```

**Step 3: Eigenvalue decomposition**
```
Cov × v = λ × v

v = eigenvector → becomes LOADINGS
λ = eigenvalue → becomes VARIANCE EXPLAINED
```

**Step 4: Sort eigenvectors by eigenvalue**
- Largest eigenvalue → PC1 loadings
- Second largest → PC2 loadings
- ...and so on

**Step 5: Cell embeddings = Data × Loadings**
```
Embeddings = X × Loadings
(54,599 × 50) = (54,599 × 3,000) × (3,000 × 50)
```

#### Summary Table

| Concept | Dimensions | What It Contains |
|---------|------------|------------------|
| **Loadings** | 3,000 × 50 | Gene weights for each PC |
| **Embeddings** | 54,599 × 50 | Cell scores on each PC |
| **One PC column** | 54,599 values | All cells' scores (independent) |
| **One cell row** | 50 values | One cell's position in PC space |

---

## Q8: Why exclude Geneformer and CNN+XGBoost from consensus?

### Question
> "Why are Geneformer and CNN+XGBoost not used for consensus? For CNN+XGBoost we already use SHAP for indirect extraction, so why isn't it included?"

### Answer

#### Part 1: Why Geneformer is Excluded

**Different Gene Importance Extraction Method**

| Model | Extraction Layer | Method |
|-------|------------------|--------|
| scGPT | `input_projection` | L2 norm of weights |
| scBERT | `input_projection` | L2 norm of weights |
| scFormer | `input_projection` | L2 norm of weights |
| **Geneformer** | `gene_positions` | L2 norm of position embeddings |

**The Problem:**
```
scGPT/scBERT/scFormer:
  Input → input_projection(50→256) → Transformer
          └─ Extract weights here

Geneformer:
  Input → value_embedding + rank_embedding + gene_positions → Transformer
                                              └─ DIFFERENT extraction point
```

The `gene_positions` captures **which PC positions matter**, not **how genes map to features**. This is philosophically different.

**If we included Geneformer:**
- Transformers would have 4/8 votes (over-represented)
- Geneformer's importance is computed differently
- We'd be comparing "apples and oranges"

#### Part 2: Why CNN+XGBoost is Excluded

**Indirect Multi-Step Extraction Chain**

```
Step 1: XGBoost trained on CNN features (128 features)
        ↓
Step 2: SHAP on XGBoost → CNN feature importance (128 values)
        ↓
Step 3: Gradient backprop through CNN → PC importance (50 values)
        ↓
Step 4: PCA loadings → Gene importance (3000 values)
```

**Compare to Classical ML (Direct):**
```
Logistic Regression:
  Coefficients → Gene importance (1 step via PCA)
```

**Problems with 4-step chain:**
1. **Error accumulation** - Errors compound through steps
2. **Interpretation artifacts** - Gradient approximations aren't exact
3. **Signal distortion** - SHAP → gradient backprop is approximate

#### Part 3: Why Not CNN+XGBoost Even With SHAP?

SHAP on XGBoost IS valid, but:

```
SHAP gives importance of CNN FEATURES (128 dimensions)
                    ↓
But we need GENE importance (3000 dimensions)
                    ↓
Must backpropagate through CNN (APPROXIMATE)
                    ↓
Then through PCA (multiplication)
```

The issue is the **CNN backpropagation step**:
```python
# This is approximate, not exact
gradients = tape.gradient(cnn_output, X_input)
```
Gradients give "local sensitivity" not necessarily "global importance."

#### Part 4: The Balance Argument

**With 6 models (our choice):**
```
Classical ML:     3/6 = 50%
Transformers:     3/6 = 50%
→ Equal representation!
```

**If we included all 8:**
```
Classical ML:     3/8 = 37.5%
Hybrid:           1/8 = 12.5%
Transformers:     4/8 = 50%
→ Transformers over-represented
```

#### Summary

| Model | Exclusion Reason |
|-------|------------------|
| **Geneformer** | Different extraction method (`gene_positions` vs `input_projection`) |
| **CNN+XGBoost** | 4-step indirect chain introduces potential artifacts |

**Key Principle:** Consensus works best when all models use **comparable extraction methods**.

---

## Navigation

← [Return to Index](./00_README.md)
