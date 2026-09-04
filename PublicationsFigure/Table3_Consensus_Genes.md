# Table 3: Top 20 Consensus Genes

**Threshold:** Genes appearing in ≥4/6 models' top-100 gene lists

## Consensus Genes Overview

| Total Consensus Genes | 63 |
|----------------------|-----|
| 6/6 Models Agreement | 17 genes |
| 5/6 Models Agreement | 5 genes |
| 4/6 Models Agreement | 41 genes |

## Top 20 Genes with Highest Model Agreement

| Rank | Gene | Model Count | Models | Function |
|------|------|-------------|--------|----------|
| 1 | **Spp1** | 6/6 | LR, RF, XGB, scGPT, scBERT, scFormer | Osteopontin, inflammation |
| 2 | **Cdkn1a** | 6/6 | LR, RF, XGB, scGPT, scBERT, scFormer | Cell cycle inhibitor (p21) |
| 3 | **Cd24a** | 6/6 | LR, RF, XGB, scGPT, scBERT, scFormer | Cell adhesion, immune modulation |
| 4 | **Cd72** | 6/6 | LR, RF, XGB, scGPT, scBERT, scFormer | B-cell activation, immune |
| 5 | **Fth1** | 6/6 | LR, RF, XGB, scGPT, scBERT, scFormer | Ferritin heavy chain, iron storage |
| 6 | **Lpl** | 6/6 | LR, RF, XGB, scGPT, scBERT, scFormer | Lipoprotein lipase, lipid metabolism |
| 7 | **H2-Eb1** | 6/6 | LR, RF, XGB, scGPT, scBERT, scFormer | MHC class II, antigen presentation |
| 8 | **Ifitm6** | 6/6 | LR, RF, XGB, scGPT, scBERT, scFormer | Interferon-induced transmembrane |
| 9 | **H2-Aa** | 6/6 | LR, RF, XGB, scGPT, scBERT, scFormer | MHC class II, antigen presentation |
| 10 | **Tubb6** | 6/6 | LR, RF, XGB, scGPT, scBERT, scFormer | Tubulin, cytoskeleton |
| 11 | **Il1rn** | 6/6 | LR, RF, XGB, scGPT, scBERT, scFormer | **IL-1 receptor antagonist** |
| 12 | **Napsa** | 6/6 | LR, RF, XGB, scGPT, scBERT, scFormer | Aspartic peptidase |
| 13 | **Sirpb1c** | 6/6 | LR, RF, XGB, scGPT, scBERT, scFormer | Signal regulatory protein |
| 14 | **Adam8** | 6/6 | LR, RF, XGB, scGPT, scBERT, scFormer | Metalloprotease, inflammation |
| 15 | **Plac8** | 6/6 | LR, RF, XGB, scGPT, scBERT, scFormer | Immune cell marker |
| 16 | **Ch25h** | 6/6 | LR, RF, XGB, scGPT, scBERT, scFormer | Cholesterol 25-hydroxylase |
| 17 | **Anxa1** | 6/6 | LR, RF, XGB, scGPT, scBERT, scFormer | Annexin A1, anti-inflammatory |
| 18 | Sepp1 | 5/6 | LR, XGB, scGPT, scBERT, scFormer | Selenoprotein P |
| 19 | Dbp | 5/6 | LR, XGB, scGPT, scBERT, scFormer | D-box binding protein |
| 20 | Emp1 | 5/6 | LR, XGB, scGPT, scBERT, scFormer | Epithelial membrane protein 1 |

## Key Biological Themes

### 1. Interferon Response Signature (Novel Finding)
- **Ifitm6** (6/6) - Interferon-induced transmembrane protein
- **Gbp2** (4/6) - Guanylate binding protein 2
- **Igtp** (4/6) - Interferon gamma induced GTPase
- **Ifit3** (4/6) - Interferon-induced protein

### 2. IL-1 Pathway (Therapeutic Target)
- **Il1rn** (6/6) - IL-1 receptor antagonist
  - *Clinical relevance:* Target of Anakinra, tested in stroke trials

### 3. Antigen Presentation (MHC Class II)
- **H2-Eb1** (6/6)
- **H2-Aa** (6/6)
- **H2-Ab1** (4/6)
- **Cd74** (4/6)

### 4. Cell Signaling & Adhesion
- **Spp1** (6/6) - Osteopontin
- **Cd24a** (6/6)
- **Anxa1** (6/6) - Anti-inflammatory

## Model Legend

| Abbreviation | Full Name |
|--------------|-----------|
| LR | Logistic Regression |
| RF | Random Forest |
| XGB | XGBoost |
| scGPT | scGPT-inspired transformer |
| scBERT | scBERT-inspired transformer |
| scFormer | scFormer-inspired transformer |
