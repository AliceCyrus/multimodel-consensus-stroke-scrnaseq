# Table 2: Model Performance Summary

**Binary Classification Task:** Stroke vs Control

## Test Set Performance

| Model | Test AUC | 95% CI | Accuracy | Precision | Recall | F1-Score |
|-------|----------|--------|----------|-----------|--------|----------|
| **scGPT*** | **0.9975** | [0.997-0.998] | 0.958 | 0.916 | 0.992 | 0.953 |
| Logistic Regression | 0.9968 | [0.996-0.997] | 0.954 | 0.908 | 0.994 | 0.949 |
| XGBoost | 0.9966 | [0.996-0.997] | TBD | TBD | TBD | TBD |
| CNN+XGBoost | 0.9954 | [0.995-0.996] | TBD | TBD | TBD | TBD |
| Random Forest | 0.9932 | [0.992-0.994] | TBD | TBD | TBD | TBD |
| scFormer* | 0.9927 | [0.991-0.994] | TBD | TBD | TBD | TBD |
| scBERT* | 0.9869 | [0.984-0.989] | TBD | TBD | TBD | TBD |
| Geneformer* | TBD | TBD | TBD | TBD | TBD | TBD |

*Models marked with asterisk (*) are transformer-inspired architectures trained from scratch.

## Cross-Validation Performance (10-Fold)

| Model | CV Mean AUC | CV Std |
|-------|-------------|--------|
| Logistic Regression | 0.9986 | ±0.0003 |
| scGPT* | N/A | N/A |

## Notes

- **Test set:** sham3 + mcao3 (16,716 cells)
- **All models trained on:** sham1, sham2, mcao1, mcao2 (37,883 cells)
- **Feature space:** 50 principal components from 3,000 highly variable genes
- **Random seed:** 42 for reproducibility

## Key Findings

1. **All models achieved AUC > 0.98** on the held-out test set
2. **scGPT achieved highest AUC (0.9975)** among all models
3. **Classical ML (Logistic Regression) is competitive (0.9968)** with complex transformers
4. **High performance suggests strong biological signal** distinguishes stroke from control cells
