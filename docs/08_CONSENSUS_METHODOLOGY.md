# Chapter 8: Consensus Methodology

## The Multi-Model Voting System

---

## 8.1 The Core Insight

### The Problem with Single-Model Results

If you run ONE algorithm and it says "Gene X is important":
- Is this real biology?
- Or a quirk of that specific algorithm?

**You can't tell!**

### The Consensus Solution

If MULTIPLE DIFFERENT algorithms agree on Gene X:
- Less likely to be an algorithm artifact
- More likely to be genuine biology

> **Our principle:** Trust genes that multiple independent methods identify.

---

## 8.2 Our Consensus Strategy

### The Voting System

```
STEP 1: Get Top 100 Genes from Each Model
─────────────────────────────────────────

Model 1 (LR):      [GeneA, GeneB, GeneC, ...]  Top 100
Model 2 (RF):      [GeneA, GeneD, GeneC, ...]  Top 100
Model 3 (XGB):     [GeneA, GeneC, GeneE, ...]  Top 100
Model 4 (scGPT):   [GeneA, GeneC, GeneF, ...]  Top 100
Model 5 (scBERT):  [GeneA, GeneC, GeneF, ...]  Top 100
Model 6 (scFormer):[GeneA, GeneC, GeneG, ...]  Top 100


STEP 2: Count Votes for Each Gene
─────────────────────────────────

GeneA: appears in 6/6 models → 6 votes
GeneC: appears in 6/6 models → 6 votes
GeneF: appears in 2/6 models → 2 votes
GeneB: appears in 1/6 models → 1 vote
...


STEP 3: Apply Consensus Threshold
──────────────────────────────────

Threshold: Gene must appear in ≥ 4/6 models

GeneA: 6 votes ✓ CONSENSUS
GeneC: 6 votes ✓ CONSENSUS
GeneF: 2 votes ✗ Not consensus
GeneB: 1 vote  ✗ Not consensus
```

---

## 8.3 Why 6 Models (Not 8)?

### The Decision

We have 8 models, but use only 6 for consensus:

| Model | Used for Consensus? | Reason |
|-------|---------------------|--------|
| Logistic Regression | ✅ Yes | Linear baseline |
| Random Forest | ✅ Yes | Tree-based ensemble |
| XGBoost | ✅ Yes | Boosted trees |
| scGPT | ✅ Yes | Transformer |
| scBERT | ✅ Yes | Transformer |
| scFormer | ✅ Yes | Transformer |
| Geneformer | ❌ No | Different extraction method |
| CNN+XGBoost | ❌ No | Hybrid, indirect extraction |

### Justification

**Geneformer exclusion:**
- Uses different extraction method (gene_positions vs. input_projection)
- After correction, still showed different pattern
- Including would give transformers 4/8 votes (over-representation)

**CNN+XGBoost exclusion:**
- Indirect extraction (SHAP → gradients → PCA)
- More complex chain → more potential for artifacts

### Balance of Model Families

```
With 6 models:
- Classical ML: 3/6 (LR, RF, XGB)
- Transformers: 3/6 (scGPT, scBERT, scFormer)

Equal representation of fundamentally different approaches!
```

---

## 8.4 Why Top 100? Why Threshold 4?

### Why Top 100 Genes per Model?

**Trade-off:**
- Too few (e.g., 10): Might miss real biomarkers
- Too many (e.g., 500): Includes too much noise

**Choosing 100:**
- Captures ~10% of variable genes
- Standard in literature
- Balances sensitivity and specificity

### Why Threshold of 4/6?

```
Threshold sensitivity analysis:

Threshold  Consensus Genes  Strictness
────────────────────────────────────────
≥ 2 models      500+        Too lenient (noise)
≥ 3 models      ~200        Lenient
≥ 4 models      63          Moderate ← Our choice
≥ 5 models      ~30         Strict
= 6 models      ~15         Very strict (might miss real ones)
```

**Choosing ≥4/6 (majority):**
- Requires majority agreement
- Still allows some model disagreement
- 63 genes is a reasonable number for follow-up

---

## 8.5 Implementation Code

```python
def find_consensus_genes(all_gene_importance, min_models=4, top_n=100):
    """
    Find genes that appear in top-N of at least min_models models.
    
    Args:
        all_gene_importance: dict of {model_name: DataFrame with 'gene', 'importance'}
        min_models: minimum number of models a gene must appear in
        top_n: number of top genes to consider from each model
    
    Returns:
        consensus_genes: list of gene names
        gene_counts: dict of {gene: count}
    """
    gene_counts = {}
    
    for model_name, gene_df in all_gene_importance.items():
        # Get top N genes for this model
        top_genes = gene_df.nlargest(top_n, 'importance')['gene'].tolist()
        
        # Count votes
        for gene in top_genes:
            gene_counts[gene] = gene_counts.get(gene, 0) + 1
    
    # Filter by threshold
    consensus_genes = [gene for gene, count in gene_counts.items() 
                       if count >= min_models]
    
    return consensus_genes, gene_counts

# Usage
consensus, counts = find_consensus_genes(
    all_gene_importance={
        'LogisticRegression': lr_genes,
        'RandomForest': rf_genes,
        'XGBoost': xgb_genes,
        'scGPT': scgpt_genes,
        'scBERT': scbert_genes,
        'scFormer': scformer_genes,
    },
    min_models=4,
    top_n=100
)

print(f"Found {len(consensus)} consensus genes")
```

---

## 8.6 Our Results

### The 63 Consensus Genes

```
Genes with 6/6 votes: ~15 genes (strongest signal)
Genes with 5/6 votes: ~20 genes
Genes with 4/6 votes: ~28 genes
────────────────────────────────────
Total consensus:      63 genes
```

### Top Consensus Genes (by vote count)

| Gene | Vote Count | Known Function |
|------|------------|----------------|
| Spp1 | 6/6 | Inflammation, macrophage activation |
| Lgals3 | 6/6 | Phagocytosis, microglia |
| Ctsb | 6/6 | Lysosomal protease |
| Ctsl | 6/6 | Lysosomal protease |
| Fth1 | 6/6 | Iron storage |
| Gbp2 | 6/6 | Interferon response |
| Igtp | 6/6 | Interferon response |
| Il1rn | 5/6 | IL-1 receptor antagonist |
| ... | ... | ... |

---

## 8.7 Validation: Do These Genes Make Sense?

### Biological Coherence Check

If our consensus genes are real, they should:
1. Be enriched in stroke-related pathways
2. Include known stroke genes
3. Form coherent functional groups

**Result:** ✅ All three confirmed! (See Chapter 9)

### Model Agreement Pattern

```
Gene Agreement Heatmap:

           LR    RF   XGB  scGPT scBERT scFormer
Spp1      [✓]   [✓]   [✓]   [✓]   [✓]    [✓]     = 6/6
Il1rn     [✓]   [✓]   [✓]   [✓]   [✓]    [✗]     = 5/6
MyGene    [✗]   [✓]   [✓]   [✓]   [✓]    [✓]     = 5/6
...

Pattern: Strong genes seen by ALL methods
         Weaker genes seen by most methods
```

---

## 8.8 Why This Approach is Strong

### Advantage 1: Robustness to Model-Specific Artifacts

```
Single model:       "Gene X is #1!"
                    (But maybe just an artifact?)

Multi-model:        "Gene X is top-100 in 6/6 models!"
                    (Unlikely to be artifact in ALL models)
```

### Advantage 2: Methodological Diversity

```
Linear methods:     Capture linear relationships
Tree methods:       Capture non-linear, interaction effects
Neural networks:    Capture complex patterns

If all agree → Signal is robust to methodology
```

### Advantage 3: Defensible to Reviewers

```
Reviewer: "How do you know Gene X isn't just noise?"

Answer: "Gene X was independently identified by 6 different 
        algorithms with fundamentally different mathematical
        foundations. The probability of this occurring by chance
        for a noise gene is very low."
```

---

## 8.9 Limitations of Consensus

### Limitation 1: May Miss Real Genes

Some real biomarkers might be:
- Captured by only 1-2 models
- Below top-100 in most models

**Trade-off:** We prioritize precision over recall.

### Limitation 2: Correlated Models

If models are too similar:
- scGPT ↔ scBERT: 98% agreement
- They're almost "voting twice"

**Mitigation:** We balance model families (3 classical, 3 transformer).

### Limitation 3: Threshold is Arbitrary

Why 4/6? Why not 3/6 or 5/6?

**Answer:** It's a reasonable middle ground. We report sensitivity analysis.

---

## Navigation

← [Previous: Gene Importance](./07_GENE_IMPORTANCE_EXTRACTION.md) | [Return to Index](./00_README.md) | [Next: Biological Validation →](./09_BIOLOGICAL_VALIDATION.md)
