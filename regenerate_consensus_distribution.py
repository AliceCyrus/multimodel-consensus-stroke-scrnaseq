"""
Regenerate Consensus Distribution with All 1-6 Bars
===================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set paths
BASE_DIR = Path(__file__).resolve().parent
GENE_RESULTS_DIR = BASE_DIR / "gene_level_results"
OUTPUT_DIR = BASE_DIR / "manuscript" / "generic_latex" / "figures"

# Load consensus genes data
consensus_file = GENE_RESULTS_DIR / "consensus_genes.csv"
print(f"Loading consensus genes from: {consensus_file}")

consensus_df = pd.read_csv(consensus_file)
print(f"Loaded {len(consensus_df)} genes")
print(consensus_df.head())

# Count genes by number of models
if 'count' in consensus_df.columns:
    counts = consensus_df['count'].value_counts().sort_index()
else:
    # If no count column, assume 'models' column has comma-separated models
    consensus_df['count'] = consensus_df['models'].str.count(',') + 1
    counts = consensus_df['count'].value_counts().sort_index()

print("\nGenes by model count:")
print(counts)

# Create complete distribution from 1-6
distribution = {}
for i in range(1, 7):
    distribution[i] = counts.get(i, 0)

print("\nComplete distribution (1-6):")
for k, v in distribution.items():
    print(f"  {k} models: {v} genes")

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

x = list(distribution.keys())
y = list(distribution.values())

# Color bars based on threshold
colors = ['lightcoral' if i < 4 else 'steelblue' for i in x]

bars = ax.bar(x, y, color=colors, edgecolor='black', linewidth=1.2)

# Add value labels on bars
for bar, val in zip(bars, y):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add threshold line
ax.axvline(x=3.5, color='red', linestyle='--', linewidth=2, label='Consensus threshold (>=4)')

# Formatting
ax.set_xlabel('Number of Models', fontsize=12)
ax.set_ylabel('Number of Genes', fontsize=12)
ax.set_title('Consensus Gene Distribution (Top 100 per model)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{i}/6' for i in x], fontsize=11)
ax.legend(loc='upper right')

# Add annotations
total_consensus = sum(v for k, v in distribution.items() if k >= 4)
ax.annotate(f'Consensus genes (>=4/6): {total_consensus}',
            xy=(5, max(y)*0.8),
            fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()

# Save figure
output_path = OUTPUT_DIR / "consensus_distribution.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nSaved to: {output_path}")

# Also save to frontiers folder
frontiers_path = BASE_DIR / "manuscript" / "frontiers_latex" / "figures" / "consensus_distribution.png"
plt.savefig(frontiers_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved to frontiers: {frontiers_path}")

# Save to PublicationsFigure
pub_path = BASE_DIR / "PublicationsFigure" / "consensus_distribution_FULL.png"
plt.savefig(pub_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved to publications: {pub_path}")

plt.close()
print("\nDone!")
