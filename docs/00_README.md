# THE COMPLETE THEORY GUIDE

## Ischemic Stroke Biomarker Discovery Using Multi-Model Machine Learning

**For Research Trainees**  
*A Comprehensive Guide from Biology to Publication*

---

## TABLE OF CONTENTS

1. [00_README.md](./00_README.md) - Start here
2. [01_BIOLOGICAL_BACKGROUND.md](./01_BIOLOGICAL_BACKGROUND.md) - Understanding stroke and the MCAO model
3. [02_SINGLE_CELL_RNA_SEQ.md](./02_SINGLE_CELL_RNA_SEQ.md) - The technology behind our data
4. [03_DATA_PREPROCESSING.md](./03_DATA_PREPROCESSING.md) - From raw reads to PCA
5. [04_TRAIN_TEST_SPLIT.md](./04_TRAIN_TEST_SPLIT.md) - The critical step that prevents data leakage
6. [05_MACHINE_LEARNING_MODELS.md](./05_MACHINE_LEARNING_MODELS.md) - All 8 models explained
7. [06_TRANSFORMER_ARCHITECTURES.md](./06_TRANSFORMER_ARCHITECTURES.md) - Deep dive into foundation model-inspired approaches
8. [07_GENE_IMPORTANCE_EXTRACTION.md](./07_GENE_IMPORTANCE_EXTRACTION.md) - How we identify important genes
9. [08_CONSENSUS_METHODOLOGY.md](./08_CONSENSUS_METHODOLOGY.md) - The multi-model voting system
10. [09_BIOLOGICAL_VALIDATION.md](./09_BIOLOGICAL_VALIDATION.md) - GO/KEGG enrichment analysis
11. [10_COMPLETE_PIPELINE_WALKTHROUGH.md](./10_COMPLETE_PIPELINE_WALKTHROUGH.md) - Step-by-step code execution guide

---

## QUICK START

**If you're brand new:** Start with `01_BIOLOGICAL_BACKGROUND.md`

**If you understand biology but not ML:** Start with `05_MACHINE_LEARNING_MODELS.md`

**If you want to run the code:** Go to `10_COMPLETE_PIPELINE_WALKTHROUGH.md`

---

## EXPERIMENT SUMMARY

### What We Did
Built 8 machine learning models to classify stroke vs. control cells from single-cell RNA-seq data, then identified genes that ALL models agree are important.

### What We Found
63 "consensus genes" that multiple independent algorithms identified as stroke biomarkers, including a novel dual-interferon signature.

### Why It Matters
These genes are robust candidates for stroke diagnosis and treatment, validated by computational agreement across diverse methodologies.

---

*Created: January 13, 2026*
