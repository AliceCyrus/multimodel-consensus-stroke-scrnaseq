# Chapter 2: Single-Cell RNA Sequencing

## The Technology Behind Our Data

---

## 2.1 What is Single-Cell RNA-Seq?

### The Problem with "Bulk" RNA-Seq

Imagine you want to understand what people in a city are saying:

**Bulk RNA-seq:** Record ALL voices at once → Get garbled noise
**Single-cell RNA-seq:** Interview each person individually → Understand each perspective

### Simple Definition

**Single-cell RNA sequencing (scRNA-seq)** measures gene expression in individual cells, rather than averaging across millions of cells.

### Why This Matters for Stroke

In stroke tissue:
- Microglia might be highly inflammatory
- Neurons might be activating death pathways
- Astrocytes might be trying to protect

Bulk RNA-seq would **average** these signals. scRNA-seq lets us see **each cell type's response**.

---

## 2.2 The 10X Genomics Chromium Technology

### Our Platform
We used **10X Genomics Chromium** - the most widely used scRNA-seq platform.

### How It Works (Step by Step)

```
STEP 1: CELL SUSPENSION
========================
Dissociate tissue into
individual cells
        ↓
      (~~~)
     (~ ~ ~)  Single cells
      (~~~)   in liquid


STEP 2: GEL BEAD EMULSION (GEM)
===============================
Each cell captured with
one barcoded gel bead
in a tiny oil droplet

    ┌─────────────────┐
    │  ○ ← Cell       │
    │  ◈ ← Gel bead   │← Oil droplet
    │     with barcode│
    └─────────────────┘
    
Each droplet = 1 cell + 1 unique barcode


STEP 3: CELL LYSIS & BARCODING
==============================
Inside each droplet:
- Cell breaks open
- mRNA captured on bead
- Barcode attached to each mRNA

    Cell A mRNA → BARCODE-A + mRNA sequence
    Cell B mRNA → BARCODE-B + mRNA sequence


STEP 4: SEQUENCING
==================
All barcoded mRNA pooled
and sequenced together

Read: [BARCODE][UMI][GENE]
      ↓        ↓     ↓
      Which   Which  Which
      cell?   molecule? gene?


STEP 5: COMPUTATIONAL DEMULTIPLEXING
====================================
Group reads by barcode →
Reconstruct each cell's transcriptome
```

### Key Terms

| Term | Meaning |
|------|---------|
| **Barcode** | Unique DNA sequence identifying each cell |
| **UMI** | Unique Molecular Identifier - counts individual mRNA molecules |
| **GEM** | Gel bead-in-Emulsion - the oil droplet |
| **Read** | One sequenced fragment |

---

## 2.3 From Raw Reads to Count Matrix

### The Processing Pipeline

```
RAW FASTQ FILES
===============
Billions of short DNA sequences
(~100 base pairs each)
        ↓
CELL RANGER (10X Software)
===========================
1. Demultiplex by barcode
2. Align to reference genome
3. Count UMIs per gene per cell
        ↓
COUNT MATRIX
============
Rows: ~20,000 genes
Columns: ~10,000 cells
Values: UMI counts
```

### What the Count Matrix Looks Like

```
              Cell_1  Cell_2  Cell_3  Cell_4  ...
Gene_A           15       0       7      23   ...
Gene_B            0       0       0       0   ...
Gene_C          103       2      45      67   ...
Gene_D            5       8       3       2   ...
...             ...     ...     ...     ...   ...
```

**This is our starting point!**

---

## 2.4 Why scRNA-seq Data is Challenging

### The Problems

1. **Sparsity (Zeros Everywhere)**
   - Most matrix entries are 0
   - Why? Didn't sequence that molecule by chance
   - Called "dropout"

2. **Technical Noise**
   - Batch effects (different days/runs)
   - Doublets (two cells in one droplet)
   - Ambient RNA (floating mRNA from dead cells)

3. **High Dimensionality**
   - ~20,000 genes × ~10,000 cells
   - Too many features for many algorithms
   - Need dimensionality reduction (→ PCA)

4. **Cell-to-Cell Variability**
   - Even identical cells have noise
   - Need statistical approaches

### Our Solutions

| Problem | Solution | Implementation |
|---------|----------|----------------|
| Zeros | Log normalization | `NormalizeData()` |
| Noise | Quality filtering | QC metrics |
| Dimensionality | PCA reduction | 50 principal components |
| Variability | Multi-model consensus | 8 algorithms voting |

> 📚 **For detailed explanations of WHY these solutions and alternatives considered:**
> See [Chapter 2A: Seurat Pipeline Deep Dive](./02A_SEURAT_PIPELINE_DEEP_DIVE.md)

---

## 2.5 Our Dataset Summary

### Sample Information

| Sample | Condition | Surgery | Cells After QC |
|--------|-----------|---------|----------------|
| sham1 | Control | Sham surgery | ~X,XXX |
| sham2 | Control | Sham surgery | ~X,XXX |
| sham3 | Control | Sham surgery | ~X,XXX |
| mcao1 | Stroke | MCAO surgery | ~X,XXX |
| mcao2 | Stroke | MCAO surgery | ~X,XXX |
| mcao3 | Stroke | MCAO surgery | ~X,XXX |

### Key Numbers

- **Total cells analyzed:** ~XX,XXX
- **Genes measured:** ~20,000
- **Genes after filtering:** ~3,000 (highly variable)
- **Final features:** 50 (PCA components)

---

## 2.6 Comparison to Other Technologies

### scRNA-seq vs. Bulk RNA-seq

| Aspect | Bulk RNA-seq | scRNA-seq |
|--------|--------------|-----------|
| Resolution | Population average | Individual cells |
| Cell types | Mixed signal | Can identify each |
| Sensitivity | Higher per gene | Lower per gene |
| Cost | Lower | Higher |
| Data size | Smaller | Massive |

### Why scRNA-seq for Stroke?

Stroke tissue contains many cell types responding differently:
- Some cells dying
- Some cells fighting
- Some cells healing

Only scRNA-seq can distinguish these responses.

---

## 2.7 The Analysis Software: Seurat

### What is Seurat?

**Seurat** is an R package for scRNA-seq analysis. It's the most widely used tool in the field.

### Named After

Georges Seurat, pointillist painter who made images from individual dots - just like we make insights from individual cells!

### What Seurat Does for Us

1. **CreateSeuratObject** - Load the data
2. **QC filtering** - Remove bad cells
3. **NormalizeData** - Make cells comparable
4. **FindVariableFeatures** - Select informative genes
5. **ScaleData** - Standardize for math
6. **RunPCA** - Reduce dimensions

We'll cover each step in the next chapter.

> 📚 **For comprehensive technical details of each step, parameters, and alternatives:**
> See [Chapter 2A: Seurat Pipeline Deep Dive](./02A_SEURAT_PIPELINE_DEEP_DIVE.md)

---

## 2.8 Key Concepts Summary

| Concept | Simple Explanation | Why It Matters |
|---------|-------------------|----------------|
| **scRNA-seq** | Gene expression per cell | See individual cell responses |
| **Barcode** | Cell ID tag | Know which reads came from which cell |
| **UMI** | Molecule counter | Accurate quantification |
| **Count matrix** | Genes × Cells table | Our raw data |
| **Sparsity** | Many zeros | Must handle carefully |
| **Seurat** | Analysis software | Does the heavy lifting |

---

## Navigation

← [Previous: Biological Background](./01_BIOLOGICAL_BACKGROUND.md) | [Return to Index](./00_README.md) | [Next: Data Preprocessing →](./03_DATA_PREPROCESSING.md)
