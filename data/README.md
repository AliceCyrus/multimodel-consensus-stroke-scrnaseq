# Data

No raw sequencing data is stored in this repository. This page explains where
each input comes from and which files you need for which task.

---

## What is and is not in the repository

| Data | Size | In repo? | How to get it |
|---|---|---|---|
| Raw 10x matrices (6 samples) | ~251 MB | No | Download from GEO — see below |
| `stroke_pca_train.csv` / `stroke_pca_test.csv` | ~48 MB | No | Regenerate with `Rscript DP_Diff_Test_Train.R` |
| `stroke_labels_train.csv` / `stroke_labels_test.csv` | 2.4 MB | **Yes** | — |
| `pca_loadings.csv` (3,000 genes × 50 PCs) | 3.0 MB | **Yes** | — |
| `pca_explained_variance.csv` | 2 KB | **Yes** | — |
| `qc_metrics_per_cell.csv` | 4.5 MB | **Yes** | — |
| Trained model weights (`.pt`, `.pkl`) | ~190 MB | No | Regenerate by training, or download from the Zenodo archive |
| Per-model results (`results_*.json`) | ~10 MB | **Yes** | — |
| Gene importance and consensus tables | ~1 MB | **Yes** | — |

The two PCA matrices are the only missing piece needed to rerun the Python
models. Everything downstream of them — gene importances, the consensus list,
enrichment results — is committed, so you can inspect and verify the published
findings without downloading or recomputing anything.

---

## Source dataset

**GEO accession:** [GSE174574](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174574)

Mouse brain single-cell RNA-seq, transient middle cerebral artery occlusion
(tMCAO) model: 60 minutes occlusion followed by 24 hours reperfusion, versus
sham-operated controls. Six adult C57BL/6 mice, three per condition.

Please cite the original data generators alongside this repository:

> Zheng K, Lin L, Jiang W, Chen L, Zhang X, Zhang Q, Ren Y, Hao J.
> Single-cell RNA-seq reveals the transcriptional landscape in ischemic stroke.
> *Journal of Cerebral Blood Flow & Metabolism*. 2022;42(1):56–73.

---

## Downloading

From the repository root:

```bash
bash data/download_GSE174574.sh
```

The script downloads the GEO supplementary archive, unpacks it, and prints the
files it found. It does **not** guess how those files map onto sample folders,
because GEO filenames are assigned by the submitter and can change. You do that
in the next step.

---

## Expected layout

`DP_Diff_Test_Train.R` calls `Read10X()` on six directories. Each must contain
exactly three gzipped files with these exact names:

```
StrokeData/
├── sham1/    barcodes.tsv.gz   features.tsv.gz   matrix.mtx.gz
├── sham2/    barcodes.tsv.gz   features.tsv.gz   matrix.mtx.gz
├── sham3/    barcodes.tsv.gz   features.tsv.gz   matrix.mtx.gz
├── mcao1/    barcodes.tsv.gz   features.tsv.gz   matrix.mtx.gz
├── mcao2/    barcodes.tsv.gz   features.tsv.gz   matrix.mtx.gz
└── mcao3/    barcodes.tsv.gz   features.tsv.gz   matrix.mtx.gz
```

`Read10X()` requires those three filenames literally. GEO files usually arrive
prefixed with a GSM accession (for example
`GSM5319989_sham1_barcodes.tsv.gz`), so you must strip the prefix when moving
each file into place.

**The sham1/sham2/sham3 and mcao1/mcao2/mcao3 assignment matters.** The
train/test split is subject-wise and hardcoded: `sham1`, `sham2`, `mcao1`,
`mcao2` are training; `sham3` and `mcao3` are the held-out test set. Check each
GSM's title on the GEO page and map it to the correct folder — if you shuffle
the replicates, you will still get a working pipeline but different numbers,
and the test set will no longer be the one used in the paper.

---

## Verifying your download

After arranging the files, these are the raw counts the paper's pipeline saw:

| Sample | Condition | Split | Raw cells | Raw genes |
|---|---|---|---|---|
| sham1 | Control | Train | 8,771 | 27,998 |
| sham2 | Control | Train | 8,540 | 27,998 |
| sham3 | Control | **Test** | 9,980 | 27,998 |
| mcao1 | Stroke | Train | 11,772 | 27,998 |
| mcao2 | Stroke | Train | 11,361 | 27,998 |
| mcao3 | Stroke | **Test** | 8,104 | 27,998 |

Cell counts come from the line count of each `barcodes.tsv.gz`; the gene count
comes from `features.tsv.gz`. Check one with:

```bash
zcat StrokeData/sham1/barcodes.tsv.gz | wc -l    # expect 8771
zcat StrokeData/sham1/features.tsv.gz | wc -l    # expect 27998
```

After QC filtering (200 < nFeature < 2,500; percent.mt < 10%) the pipeline
retains **54,599** cells: 37,883 training and 16,716 test. If your numbers
differ, the sample-to-folder mapping is probably wrong.
