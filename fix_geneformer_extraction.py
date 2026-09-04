"""
FIX GENEFORMER EXTRACTION
=========================
This script properly extracts gene importance from the trained Geneformer model
using its gene_positions parameters.

Run this ONLY for Geneformer to replace the incorrect extraction.
"""

import os
import numpy as np
import pandas as pd
import torch

# Import model
from fixed_foundation_models import GeneformerModel

print("="*70)
print("GENEFORMER GENE IMPORTANCE - CORRECTED EXTRACTION")
print("="*70)

# Load PCA loadings
print("\n1. Loading PCA loadings...")
loadings = pd.read_csv('pca_loadings.csv', index_col=0)
print(f"   Loaded: {loadings.shape[0]} genes x {loadings.shape[1]} PCs")

# Load Geneformer checkpoint
print("\n2. Loading Geneformer checkpoint...")
checkpoint_path = 'model_checkpoints/geneformer_best.pt'
device = torch.device('cpu')
checkpoint = torch.load(checkpoint_path, map_location=device)
config = checkpoint['model_config']

print(f"   Config: d_model={config['d_model']}, layers={config['num_layers']}")

# Initialize model
model = GeneformerModel(
    input_dim=config['input_dim'],
    d_model=config['d_model'],
    nhead=config['nhead'],
    num_layers=config['num_layers'],
    dropout=config['dropout']
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"   Model loaded successfully!")

# Extract PC importance from gene_positions
print("\n3. Extracting PC importance from gene_positions...")
gene_positions = model.gene_positions.data.cpu().numpy()
print(f"   gene_positions shape: {gene_positions.shape}")

# Remove batch dimension: (1, 50, 512) -> (50, 512)
gene_positions = gene_positions.squeeze(0)
print(f"   After squeeze: {gene_positions.shape}")

# Compute L2 norm for each PC's position embedding
pc_importance = np.linalg.norm(gene_positions, axis=1)
print(f"   PC importance shape: {pc_importance.shape}")

# Normalize
pc_importance = pc_importance / pc_importance.sum()

print(f"\n   Stats:")
print(f"   - Mean: {pc_importance.mean():.6f}")
print(f"   - Max: {pc_importance.max():.6f}")
print(f"   - Top PC: PC_{np.argmax(pc_importance)+1}")
print(f"   - Top 5 PCs: {(np.argsort(pc_importance)[-5:][::-1] + 1).tolist()}")

# Map to genes
print("\n4. Mapping PC importance to genes...")
n_pcs = min(len(pc_importance), loadings.shape[1])
gene_importance = np.zeros(loadings.shape[0])

for i, pc_name in enumerate(loadings.columns[:n_pcs]):
    pc_loadings = np.abs(loadings[pc_name].values)
    gene_importance += pc_importance[i] * pc_loadings

# Create DataFrame
gene_df = pd.DataFrame({
    'gene': loadings.index,
    'importance': gene_importance,
    'rank': 0,
    'method': 'Geneformer'
})

# Sort by importance
gene_df = gene_df.sort_values('importance', ascending=False).reset_index(drop=True)
gene_df['rank'] = range(1, len(gene_df) + 1)

print(f"   Mapped to {len(gene_df)} genes")

# Save
output_file = 'transformer_importances/gene_importance_Geneformer.csv'
gene_df.to_csv(output_file, index=False)
print(f"\n5. SAVED: {output_file}")

# Display top 10
print("\n6. Top 10 genes from CORRECTED Geneformer extraction:")
print(gene_df.head(10)[['rank', 'gene', 'importance']].to_string(index=False))

# Compare with PC_1
print("\n7. Comparison check - Top 10 genes by PC_1 loadings:")
pc1_genes = loadings.iloc[:, 0].abs().sort_values(ascending=False).head(10)
print(pc1_genes.to_string())

print("\n" + "="*70)
print("DONE! Geneformer extraction corrected.")
print("="*70)
print("\nNext step: python integrated_interpretability_pipeline.py")
