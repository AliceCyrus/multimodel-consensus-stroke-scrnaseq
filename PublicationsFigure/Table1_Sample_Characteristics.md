# Table 1: Sample Characteristics

**Study Dataset:** GSE174574

## Sample Overview

| Sample ID | Condition | Group | Set | Cell Count |
|-----------|-----------|-------|-----|------------|
| sham1 | Control | Sham Surgery | Training | TBD |
| sham2 | Control | Sham Surgery | Training | TBD |
| sham3 | Control | Sham Surgery | Test | 9,508 |
| mcao1 | Stroke | MCAO | Training | TBD |
| mcao2 | Stroke | MCAO | Training | TBD |
| mcao3 | Stroke | MCAO | Test | 7,208 |

## After Quality Control

| Set | Condition | Total Cells |
|-----|-----------|-------------|
| Training | Control | 16,554 |
| Training | Stroke | 21,329 |
| **Training Total** | | **37,883** |
| Test | Control | 9,508 |
| Test | Stroke | 7,208 |
| **Test Total** | | **16,716** |
| **Grand Total** | | **54,599** |

## Quality Control Parameters

| Parameter | Threshold | Rationale |
|-----------|-----------|-----------|
| Minimum genes per cell | 200 | Remove empty droplets |
| Maximum genes per cell | 2,500 | Remove potential doublets |
| Mitochondrial % | < 10% | Remove dying cells |
| Minimum cells per gene | 3 | Remove rarely expressed genes |

## Feature Engineering

| Step | Output |
|------|--------|
| Variable features selected | 3,000 genes |
| PCA components retained | 50 PCs |
| Variance explained (PC1-50) | ~90% |
