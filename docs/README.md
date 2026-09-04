# Documentation

Method documentation for the multi-model consensus pipeline. Written as
explanatory walkthroughs rather than API reference — they explain *why* each
step is done the way it is, which is usually the harder question.

## Start here

**[`PIPELINE_DATA_FLOW.md`](PIPELINE_DATA_FLOW.md)** — the single most useful
document. Traces every stage from the six raw 10x matrices through to the
63-gene consensus, with the cell and gene count at each step and the file each
number was read from. Use it to check any figure in the paper against the data.

## Background

| File | Covers |
|---|---|
| [`00_README.md`](00_README.md) | Orientation to the documentation set |
| [`01_BIOLOGICAL_BACKGROUND.md`](01_BIOLOGICAL_BACKGROUND.md) | Ischemic stroke pathophysiology, why biomarkers are needed |
| [`02_SINGLE_CELL_RNA_SEQ.md`](02_SINGLE_CELL_RNA_SEQ.md) | What scRNA-seq measures and how droplet capture works |
| [`02A_SEURAT_PIPELINE_DEEP_DIVE.md`](02A_SEURAT_PIPELINE_DEEP_DIVE.md) | Seurat internals: normalisation, HVG selection, scaling |

## Method

| File | Covers |
|---|---|
| [`03_DATA_PREPROCESSING.md`](03_DATA_PREPROCESSING.md) | QC thresholds and the reasoning behind each cutoff |
| [`04_TRAIN_TEST_SPLIT.md`](04_TRAIN_TEST_SPLIT.md) | Subject-wise splitting, data leakage, pseudoreplication |
| [`05_MACHINE_LEARNING_MODELS.md`](05_MACHINE_LEARNING_MODELS.md) | All eight architectures and their hyperparameters |
| [`06_TRANSFORMER_ARCHITECTURES.md`](06_TRANSFORMER_ARCHITECTURES.md) | How the transformer-inspired models are built |
| [`07_GENE_IMPORTANCE_EXTRACTION.md`](07_GENE_IMPORTANCE_EXTRACTION.md) | PCA back-projection, and why two models are excluded |
| [`08_CONSENSUS_METHODOLOGY.md`](08_CONSENSUS_METHODOLOGY.md) | Voting rule, threshold selection, Jaccard convergence |
| [`09_BIOLOGICAL_VALIDATION.md`](09_BIOLOGICAL_VALIDATION.md) | GO and KEGG enrichment, background set choice |
| [`10_COMPLETE_PIPELINE_WALKTHROUGH.md`](10_COMPLETE_PIPELINE_WALKTHROUGH.md) | End-to-end run, command by command |
| [`methodology_details.txt`](methodology_details.txt) | Condensed parameter reference |

## Questions

[`QA_TRAINEE_QUESTIONS.md`](QA_TRAINEE_QUESTIONS.md) — questions that come up
repeatedly when someone new reads this pipeline, with answers.

---

**A note on the transformer models.** They are described throughout as
"transformer-inspired", and that wording is deliberate. They use transformer
architectures trained from scratch on PCA-reduced features. They are *not* the
published pretrained scGPT, scBERT, scFormer or Geneformer models, and no
pretrained weights were fine-tuned. This choice makes the comparison across
architectures fair — every model sees the same 50-dimensional input — but it
means the results say nothing about what those foundation models achieve with
their original pretraining. See
[`06_TRANSFORMER_ARCHITECTURES.md`](06_TRANSFORMER_ARCHITECTURES.md).
