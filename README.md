# Multi-Model Machine Learning Consensus Identifies a Dual-Interferon Gene Signature in Ischemic Stroke

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](requirements.txt)
[![R >= 4.4](https://img.shields.io/badge/R-%3E%3D4.4-276DC3.svg)](install_r_packages.R)
[![Data: GSE174574](https://img.shields.io/badge/data-GSE174574-brightgreen.svg)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174574)

Code and derived results for the paper:

> **Multi-Model Machine Learning Consensus Identifies a Dual-Interferon Gene
> Signature in Ischemic Stroke: An Integrative Single-Cell Transcriptomic
> Analysis**
> Alisha Shamim, Angana Chakroborty

Eight machine learning models spanning three architectural families are trained
to separate stroke from control cells in mouse MCAO single-cell RNA-seq data.
Gene importance is back-projected from PCA space through the loadings matrix,
and genes selected by four or more of the six interpretable models are retained
as a consensus signature. The rationale is that a gene picked out by models with
genuinely different inductive biases is more likely to reflect biology than the
quirks of any one algorithm.

---

## Headline results

| | |
|---|---|
| Cells after QC | 54,599 (37,883 train / 16,716 test) |
| Animals | 6 (4 train, 2 held-out test) — **subject-wise split** |
| Models trained | 8 (3 classical, 1 hybrid, 4 transformer-inspired) |
| Test-set AUC range | 0.9778 – 0.9975 |
| Interpretable models used for consensus | 6 |
| **Consensus genes (≥4/6 models)** | **63** — 17 at 6/6, 5 at 5/6, 41 at 4/6 |

The 17 genes selected unanimously by all six models:

```
Spp1  Cdkn1a  Cd24a  Cd72  Fth1  Lpl  H2-Eb1  Ifitm6  H2-Aa
Tubb6  Il1rn  Napsa  Sirpb1c  Adam8  Plac8  Ch25h  Anxa1
```

GO enrichment against the 3,000-HVG background is dominated by interferon
response: `response to interferon-beta` is the single most enriched term
(FDR = 1.4e-08), and `response to type II interferon` is also significant
(FDR = 4.5e-03) — the dual-interferon signature. MHC class II antigen
presentation is significant too. `Il1rn`, the target of the FDA-approved drug
Anakinra, is selected unanimously by all six models.

---

## Repository layout

```
.
├── DP_Diff_Test_Train.R              Seurat QC, subject-wise split, PCA        [STAGE 1]
├── export_data_for_python.R          Export PCA loadings + variance            [STAGE 1]
│
├── phase1_framework.py               Validation framework: 10-fold CV, bootstrap CI
├── repo_paths.py                     Path resolution helper (see "Colab" note below)
├── comparisionModel.py               Logistic Regression, Random Forest, XGBoost [STAGE 2]
├── cnn_xgboostModel.py               CNN + XGBoost hybrid                        [STAGE 2]
├── fixed_foundation_models.py        scGPT/scBERT/scFormer/Geneformer-inspired   [STAGE 2]
│
├── gene_importance_extractor.py      PC importance -> gene scores via loadings   [STAGE 3]
├── integrated_interpretability_pipeline.py   Consensus driver                    [STAGE 3]
├── attention_extractor.py            Transformer input-projection importances    [STAGE 3]
├── fix_geneformer_extraction.py      Geneformer rank-encoding special case
│
├── go_enrichment.R                   GO over-representation (clusterProfiler)    [STAGE 4]
├── keggPathways.R                    KEGG over-representation (negative result)  [STAGE 4]
├── generateHeatMap.R                 Gene x pathway heatmap                      [STAGE 4]
├── generate_FigureS1_real.R          Supplementary Figure S1 (QC metrics)
├── generate_FigureS2_real.R          Supplementary Figure S2 (PCA variance)
├── generate_supplementary_figures.R  Both supplementary figures in one pass
├── regenerate_*.py                   Re-render published figures from saved results
│
├── map_mouse_to_human_orthologs.py   Mouse -> human ortholog mapping (exploratory)
├── validate_human_orthology_strict.py
│
├── data/                             Download instructions + GEO fetch script
├── StrokeData/                       Empty sample folders; raw data goes here
├── docs/                             Method walkthroughs and pipeline data flow
│
├── gene_level_results/               Per-model gene rankings + the 63-gene consensus
├── transformer_importances/          Per-transformer gene importance tables
├── geneGoEnrichment/                 GO enrichment results (CSV + plots)
├── KEGG_Enrichment_Pathways/         KEGG statistics (negative result — see its README)
├── heatMap/                          Gene-pathway heatmap outputs
├── figures_all_models/               ROC, PR, CV, radar plots across all 8 models
├── figures/                          Pipeline flowchart + its generator
├── PublicationsFigure/               Publication-ready figures and tables
└── results_*.json                    Full metrics for each of the 8 models
```

---

## Getting started

### Option A — inspect the published results (no setup)

Everything downstream of model training is committed. You can verify the
paper's findings without downloading data or running anything:

| Question | File |
|---|---|
| What are the 63 consensus genes? | `gene_level_results/consensus_genes.csv` |
| How much did models agree? | `gene_level_results/gene_signature_convergence.csv` |
| How did each model perform? | `performance_table_all_models.csv`, `results_*.json` |
| Which GO terms were enriched? | `geneGoEnrichment/go_enrichment_BP.csv` |
| Which KEGG pathways? | None significant — see [`KEGG_Enrichment_Pathways/README.md`](KEGG_Enrichment_Pathways/README.md) |
| Per-model gene rankings | `gene_level_results/genes_*.csv`, `transformer_importances/` |

### Option B — retrain the models

Needs the two PCA matrices, which are regenerated by the R preprocessing step
(Option C) or restored from the Zenodo archive.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python comparisionModel.py            # LR, RF, XGBoost
python cnn_xgboostModel.py            # CNN + XGBoost
python fixed_foundation_models.py     # 4 transformer-inspired models
python integrated_interpretability_pipeline.py   # gene back-projection + consensus
```

### Option C — run the whole pipeline from raw data

```bash
# 1. Dependencies
Rscript install_r_packages.R
pip install -r requirements.txt

# 2. Raw data (~250 MB) - read data/README.md first, the sample mapping is manual
bash data/download_GSE174574.sh

# 3. Preprocessing: QC, subject-wise split, PCA
Rscript DP_Diff_Test_Train.R
Rscript export_data_for_python.R

# 4. Model training  (see Option B)
# 5. Gene extraction and consensus  (see Option B)

# 6. Biological validation
Rscript go_enrichment.R
Rscript keggPathways.R      # needs internet; returns a negative result
Rscript generateHeatMap.R
```

Run every script **from the repository root** — paths are resolved relative to
it.

---

## Pipeline

| Stage | Language | Script | Output |
|---|---|---|---|
| 1 | R | `DP_Diff_Test_Train.R` | 58,528 raw cells → 54,599 after QC; 3,000 HVGs; 50 PCs |
| 2 | R | `export_data_for_python.R` | `pca_loadings.csv` (3,000 × 50) |
| 3 | Python | `comparisionModel.py`, `cnn_xgboostModel.py`, `fixed_foundation_models.py` | `results_*.json` for 8 models |
| 4 | Python | `gene_importance_extractor.py` | PC importance → gene scores |
| 5 | Python | `integrated_interpretability_pipeline.py` | 63 consensus genes |
| 6 | R | `go_enrichment.R` | 101 BP / 12 MF / 16 CC terms (3,000-HVG background) |
| 7 | R | `keggPathways.R` | 122 pathways tested, 0 FDR-significant |

Full stage-by-stage numbers, with every cell and gene count traced to the file
it came from, are in [`docs/PIPELINE_DATA_FLOW.md`](docs/PIPELINE_DATA_FLOW.md).

### The consensus rule

Gene-level importance is recovered from PCA space as

```
Gene_j = Σ over 50 PCs of | w_i × L_ij |
```

where `w_i` is a model's importance for principal component *i* and `L_ij` is
the loading of gene *j* on that component. Each model's top 100 genes are
taken, and genes appearing in at least four of the six interpretable lists are
kept.

Two of the eight models are excluded from the consensus, for reasons of
mathematics rather than performance:

- **Geneformer-inspired** uses rank-value tokenisation, which breaks the linear
  correspondence between input features and PCA components that back-projection
  requires.
- **CNN+XGBoost** applies nonlinear convolutions before classification, so the
  PCA loadings no longer map cleanly onto its learned features.

Both scored well (AUC 0.9778 and 0.9954 respectively) and are reported in the
performance tables. Their exclusion affects interpretation only.

---

## Requirements

**Python 3.7.9** — the exact environment used for the published numbers.
Pins are in [`requirements.txt`](requirements.txt). Key versions: numpy 1.21.6,
pandas 1.3.5, scikit-learn 1.0.2, xgboost 1.6.2, torch 1.13.1,
tensorflow 1.15.0, shap 0.42.1.

Python 3.7 is end-of-life. [`requirements-modern.txt`](requirements-modern.txt)
provides a Python 3.10+ set that reproduces the analysis structurally, with the
caveats listed in that file.

**R ≥ 4.4 with Seurat v5.** `Rscript install_r_packages.R` installs everything.
Bioconductor packages used: clusterProfiler, org.Mm.eg.db, enrichplot,
ComplexHeatmap.

A GPU is optional. The four transformer models train on CPU in a reasonable
time at this scale (50 input dimensions).

---

## Reproducibility

`set.seed(42)` in R and seed 42 in Python are set throughout. The train/test
split is fixed by animal, not sampled, so it is identical on every run:

- **Train** — sham1, sham2, mcao1, mcao2 (37,883 cells)
- **Test** — sham3, mcao3 (16,716 cells)

PCA is fitted on training cells only; test cells are projected into the training
PCA space. No test cell influences any fitted parameter.

What is *not* fully deterministic: XGBoost with `subsample=0.8` and PyTorch GPU
kernels introduce small run-to-run variation. AUC values should reproduce to
about three decimal places. The consensus gene list is stable in composition;
the ordering of genes tied at the same vote count may vary.

---

## Known discrepancies

Recorded here rather than quietly fixed, so anyone comparing the repository
against the paper can see them.

1. **Seurat v5 is required.** `DP_Diff_Test_Train.R` calls
   `LayerData(obj, assay = "RNA", layer = "scale.data")`, which exists only in
   Seurat v5; the v4 equivalent is `GetAssayData(obj, slot = "scale.data")`.
   The manuscript Methods section states v5 accordingly.

2. **GO input gene count.** `go_enrichment.R` reads the 63-gene consensus
   file. Of those 63, 57 map to Entrez identifiers and are used for testing.
   The paper describes the analysis as being "of 63 consensus genes", which is
   the input count, and states the 57 explicitly in Results.

3. **KEGG is a negative result.** No KEGG pathway survives FDR correction
   against the 3,000-HVG background (122 tested, 0 significant, smallest
   adjusted p = 0.162). Ten of the 21 nominally significant pathways are driven
   entirely by three MHC class II genes annotated to many unrelated KEGG
   disease maps. The manuscript reports GO as the primary enrichment analysis
   and states the KEGG result as null. See
   [`KEGG_Enrichment_Pathways/README.md`](KEGG_Enrichment_Pathways/README.md).

4. **Figure 1 has no generator.** `figures/generate_pipeline_figure.py`
   produces `Figure1_Pipeline_Flowchart.png`, which is *not* the workflow figure
   used in the paper (`Figure1_Study_Workflow.png`). The published Figure 1 was
   assembled outside this codebase.

5. **Colab origins.** Several scripts were originally written for Google Colab
   with hardcoded `/content/drive/MyDrive/` paths. These have been replaced with
   repository-relative resolution via `repo_paths.py`; the Colab locations are
   retained as fallbacks so the scripts still run there unchanged.

---

## What is deliberately not included

- **Raw sequencing data** (~251 MB) — public at GEO
  [GSE174574](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174574).
  See [`data/README.md`](data/README.md).
- **`stroke_pca_train.csv` / `stroke_pca_test.csv`** (~48 MB) — deterministic
  outputs of `DP_Diff_Test_Train.R`.
- **Trained weights** (`.pt`, `.pkl`, ~190 MB) — retrainable, and pickle files
  execute arbitrary code on load, so they are poor things to distribute.
- **A supplementary 7-model / 78-gene consensus.** An exploratory variant added
  CNN+XGBoost to the voting pool. It is not the analysis reported in the paper
  and has been left out to avoid two contradictory consensus lists sitting side
  by side. The paper's result is the 6-model, 63-gene consensus.

---

## Citation

If you use this code, please cite both the paper and the original dataset.

```bibtex
@article{shamim2026multimodel,
  title   = {Multi-Model Machine Learning Consensus Identifies a Dual-Interferon
             Gene Signature in Ischemic Stroke: An Integrative Single-Cell
             Transcriptomic Analysis},
  author  = {Shamim, Alisha and Chakroborty, Angana},
  year    = {2026},
  note    = {Manuscript}
}

@article{zheng2022single,
  title   = {Single-cell {RNA-seq} reveals the transcriptional landscape in
             ischemic stroke},
  author  = {Zheng, Kexin and Lin, Lan and Jiang, Wei and Chen, Lin and
             Zhang, Xiaojun and Zhang, Qian and Ren, Yi and Hao, Junwei},
  journal = {Journal of Cerebral Blood Flow \& Metabolism},
  volume  = {42}, number = {1}, pages = {56--73}, year = {2022},
  note    = {GEO: GSE174574}
}
```

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).

### Archiving with Zenodo

To mint a DOI for this repository:

1. Sign in at [zenodo.org](https://zenodo.org) with your GitHub account.
2. Under **GitHub**, toggle this repository **on**.
3. In GitHub, create a release (for example `v1.0.0`).
4. Zenodo archives the release and issues a DOI.
5. Add the DOI badge here and to the paper's Data Availability Statement.

`.zenodo.json` pre-fills the deposition metadata.

---

## License

MIT — see [LICENSE](LICENSE). The licence covers the code. Raw data remain
subject to the terms of the original GEO deposition.

---

## Contact

Questions about the code: open an
[issue](https://github.com/AliceCyrus/multimodel-consensus-stroke-scrnaseq/issues).
Correspondence about the study should go to the corresponding author listed in
the paper.

The authors thank Md Kashif Akram for technical guidance during the
computational analysis.
