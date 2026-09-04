"""
Regenerate Gene Convergence Heatmap with Corrected Geneformer Data
===================================================================
This script regenerates the heatmap showing Jaccard similarity between 
model gene importance rankings, using the corrected Geneformer extraction 
that shows 77% overlap with other foundation models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set paths
BASE_DIR = Path(__file__).resolve().parent
GENE_RESULTS_DIR = BASE_DIR / "gene_level_results"
TRANSFORMER_DIR = BASE_DIR / "transformer_importances"
OUTPUT_DIR = BASE_DIR / "PublicationsFigure"

# Load gene importance files for all models
def load_top_genes(filepath, n=100):
    """Load top N genes from a gene importance file."""
    df = pd.read_csv(filepath)
    if 'gene' in df.columns:
        return set(df.head(n)['gene'].tolist())
    elif df.columns[0] == 'Unnamed: 0':
        return set(df.head(n).iloc[:, 0].tolist())
    else:
        return set(df.head(n)[df.columns[0]].tolist())

def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0

# Load all model gene lists (ALL 8 MODELS)
print("Loading gene importance files...")

models = {}

# Classical ML (3 models)
models['Logistic Regression'] = load_top_genes(GENE_RESULTS_DIR / "genes_logistic_regression.csv")
models['Random Forest'] = load_top_genes(GENE_RESULTS_DIR / "genes_random_forest.csv")
models['XGBoost'] = load_top_genes(GENE_RESULTS_DIR / "genes_xgboost.csv")

# Hybrid Deep Learning (1 model) - ADDING THIS
models['CNN+XGBoost'] = load_top_genes(GENE_RESULTS_DIR / "genes_cnn_xgboost.csv")

# Foundation models (4 models)
models['scGPT'] = load_top_genes(TRANSFORMER_DIR / "gene_importance_scGPT.csv")
models['scBERT'] = load_top_genes(TRANSFORMER_DIR / "gene_importance_scBERT.csv")
models['scFormer'] = load_top_genes(TRANSFORMER_DIR / "gene_importance_scFormer.csv")
models['Geneformer'] = load_top_genes(TRANSFORMER_DIR / "gene_importance_Geneformer.csv")

print(f"Loaded {len(models)} models")
for name, genes in models.items():
    print(f"  {name}: {len(genes)} genes")

# Calculate pairwise Jaccard similarity
model_names = list(models.keys())
n = len(model_names)
similarity_matrix = np.zeros((n, n))

for i, name1 in enumerate(model_names):
    for j, name2 in enumerate(model_names):
        similarity_matrix[i, j] = jaccard_similarity(models[name1], models[name2])

# Create DataFrame
similarity_df = pd.DataFrame(similarity_matrix, index=model_names, columns=model_names)

print("\nGene Signature Convergence Matrix (Jaccard Similarity):")
print(similarity_df.round(2))

# Save the updated convergence matrix
similarity_df.to_csv(GENE_RESULTS_DIR / "gene_signature_convergence_CORRECTED.csv")
print(f"\nSaved convergence matrix to: {GENE_RESULTS_DIR / 'gene_signature_convergence_CORRECTED.csv'}")

# Create the heatmap
plt.figure(figsize=(10, 8))

# Use a diverging colormap
cmap = sns.color_palette("RdYlBu_r", as_cmap=True)

# Create heatmap with annotations
ax = sns.heatmap(
    similarity_df, 
    annot=True, 
    fmt='.2f',
    cmap='YlOrRd',
    vmin=0, 
    vmax=1,
    square=True,
    cbar_kws={'label': 'Jaccard Similarity', 'shrink': 0.8},
    annot_kws={'size': 10}
)

plt.title('Gene Signature Convergence (Top 100 genes)\n(Corrected Geneformer Extraction)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()

# Save the figure
output_path = OUTPUT_DIR / "gene_convergence_heatmap_CORRECTED.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nSaved heatmap to: {output_path}")

# Also save to manuscript figures folders
for folder in ['frontiers_latex', 'generic_latex']:
    manuscript_path = BASE_DIR / "manuscript" / folder / "figures" / "gene_convergence_heatmap.png"
    plt.savefig(manuscript_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Updated: {manuscript_path}")

plt.close()

# Print summary
print("\n" + "="*60)
print("SUMMARY: Gene Signature Convergence")
print("="*60)
print("\nCLASSICAL ML CLUSTER:")
print(f"  LR ↔ RF:  {similarity_df.loc['Logistic Regression', 'Random Forest']:.1%}")
print(f"  LR ↔ XGB: {similarity_df.loc['Logistic Regression', 'XGBoost']:.1%}")
print(f"  RF ↔ XGB: {similarity_df.loc['Random Forest', 'XGBoost']:.1%}")

print("\nFOUNDATION MODEL CLUSTER:")
print(f"  scGPT ↔ scBERT:    {similarity_df.loc['scGPT', 'scBERT']:.1%}")
print(f"  scGPT ↔ scFormer:  {similarity_df.loc['scGPT', 'scFormer']:.1%}")
print(f"  scBERT ↔ scFormer: {similarity_df.loc['scBERT', 'scFormer']:.1%}")

print("\nGENEFORMER (CORRECTED):")
print(f"  Geneformer ↔ scGPT:    {similarity_df.loc['Geneformer', 'scGPT']:.1%}")
print(f"  Geneformer ↔ scBERT:   {similarity_df.loc['Geneformer', 'scBERT']:.1%}")
print(f"  Geneformer ↔ scFormer: {similarity_df.loc['Geneformer', 'scFormer']:.1%}")

print("\nCROSS-FAMILY (Classical ↔ Foundation):")
print(f"  LR ↔ scGPT:  {similarity_df.loc['Logistic Regression', 'scGPT']:.1%}")
print(f"  RF ↔ scGPT:  {similarity_df.loc['Random Forest', 'scGPT']:.1%}")
print(f"  XGB ↔ scGPT: {similarity_df.loc['XGBoost', 'scGPT']:.1%}")

print("\n" + "="*60)
print("Done! Heatmap regenerated with corrected Geneformer data.")
print("="*60)
