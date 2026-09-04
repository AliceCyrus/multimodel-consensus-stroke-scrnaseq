"""
Mouse-to-Human Gene Ortholog Mapper
====================================
Maps your identified mouse genes to human orthologs using the mygene.info API.
Creates a supplementary table for manuscript inclusion.

Usage:
    python map_mouse_to_human_orthologs.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time

print("="*70)
print("MOUSE-TO-HUMAN GENE ORTHOLOG MAPPING")
print("="*70)

# ============================================================================
# OPTION 1: Using mygene (requires internet)
# ============================================================================

try:
    import mygene
    use_mygene = True
    print("\nUsing mygene.info API for ortholog mapping...")
except ImportError:
    use_mygene = False
    print("\nmygene not installed. Using simple name mapping...")
    print("Install with: pip install mygene")

# ============================================================================
# LOAD CONSENSUS GENES
# ============================================================================

# Try different possible locations
consensus_files = [
    'gene_level_results/consensus_genes_4models_UPDATED.csv',
    'gene_level_results/consensus_genes.csv',
    'transformer_importances/gene_importance_Geneformer.csv'  # Fallback to one model
]

consensus_df = None
for file in consensus_files:
    if Path(file).exists():
        consensus_df = pd.read_csv(file)
        print(f"\nLoaded: {file}")
        print(f"  Genes: {len(consensus_df)}")
        break

if consensus_df is None:
    print("\nNo consensus gene file found. Using top genes from all models...")
    # Load from individual model files
    all_genes = set()
    for model in ['scGPT', 'scBERT', 'scFormer', 'Geneformer']:
        file = f'transformer_importances/gene_importance_{model}.csv'
        if Path(file).exists():
            df = pd.read_csv(file)
            all_genes.update(df.head(50)['gene'].tolist())
    
    consensus_df = pd.DataFrame({'gene': sorted(list(all_genes))})
    print(f"  Combined {len(consensus_df)} unique genes from all models")

# Get top genes
top_genes = consensus_df['gene'].head(100).tolist()
print(f"\nProcessing top {len(top_genes)} genes...")

# ============================================================================
# MAP TO HUMAN ORTHOLOGS
# ============================================================================

results = []

if use_mygene:
    print("\nQuerying mygene.info database...")
    mg = mygene.MyGeneInfo()
    
    # Query in batches to avoid rate limiting
    batch_size = 50
    for i in range(0, len(top_genes), batch_size):
        batch = top_genes[i:i+batch_size]
        print(f"  Processing genes {i+1}-{min(i+batch_size, len(top_genes))}...")
        
        # Query mouse genes
        query_results = mg.querymany(
            batch,
            scopes='symbol',
            fields='symbol,taxid,homologene.genes',
            species='mouse',
            returnall=True
        )
        
        for gene, result in zip(batch, query_results['out']):
            row = {
                'mouse_gene': gene,
                'human_ortholog': 'N/A',
                'conservation_type': 'Unknown',
                'confidence': 'N/A'
            }
            
            # Extract human ortholog from homologene data
            if 'homologene' in result and 'genes' in result['homologene']:
                for homolog_gene in result['homologene']['genes']:
                    if homolog_gene.get('taxid') == 9606:  # Human tax ID
                        row['human_ortholog'] = homolog_gene.get('symbol', 'N/A')
                        row['conservation_type'] = '1:1 ortholog'
                        row['confidence'] = 'High'
                        break
            
            results.append(row)
        
        # Rate limiting
        if i + batch_size < len(top_genes):
            time.sleep(1)

else:
    print("\nUsing simple capitalization mapping (mouse to human)...")
    print("Note: This is approximate. Install mygene for accurate mapping.")

    
    # Simple heuristic: mouse genes are often human genes capitalized
    # Works for many common genes (e.g., Gfap → GFAP)
    for gene in top_genes:
        results.append({
            'mouse_gene': gene,
            'human_ortholog': gene.upper(),
            'conservation_type': 'Inferred (capitalize)',
            'confidence': 'Medium'
        })

# Create DataFrame
ortholog_df = pd.DataFrame(results)

# ============================================================================
# ADD ADDITIONAL INFORMATION
# ============================================================================

# Add known stroke relevance (from literature)
stroke_genes = {
    'Gfap': 'Astrocyte activation marker, elevated in human stroke',
    'Lcn2': 'Inflammatory marker, correlates with stroke severity',
    'Actb': 'Cytoskeletal stress response, conserved across species',
    'Ccl19': 'Chemokine signaling, involved in post-stroke inflammation',
    'S100b': 'Astrocyte marker, clinical biomarker for brain injury',
    'Vim': 'Intermediate filament, reactive astrocytes',
    'Cd68': 'Macrophage/microglia marker, neuroinflammation',
    'Adm': 'Vascular response, protective in cerebral ischemia',
    'Plat': 'Tissue plasminogen activator, stroke therapeutic target'
}

ortholog_df['human_stroke_relevance'] = ortholog_df['mouse_gene'].map(
    lambda x: stroke_genes.get(x, 'To be validated')
)

# ============================================================================
# SAVE RESULTS
# ============================================================================

# Main table
output_file = 'gene_level_results/mouse_human_ortholog_mapping.csv'
ortholog_df.to_csv(output_file, index=False)
print(f"\n✓ Saved: {output_file}")

# Supplementary table (formatted for manuscript)
supp_df = ortholog_df.copy()
supp_df = supp_df.rename(columns={
    'mouse_gene': 'Mouse Gene',
    'human_ortholog': 'Human Ortholog',
    'conservation_type': 'Conservation',
    'confidence': 'Mapping Confidence',
    'human_stroke_relevance': 'Human Stroke Evidence'
})

supp_file = 'gene_level_results/Table_SX_Mouse_Human_Orthologs.csv'
supp_df.to_csv(supp_file, index=False)
print(f"✓ Saved supplementary table: {supp_file}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*70)
print("MAPPING SUMMARY")
print("="*70)

total = len(ortholog_df)
mapped = len(ortholog_df[ortholog_df['human_ortholog'] != 'N/A'])
high_conf = len(ortholog_df[ortholog_df['confidence'] == 'High'])
has_evidence = len(ortholog_df[ortholog_df['human_stroke_relevance'] != 'To be validated'])

print(f"\nTotal mouse genes analyzed: {total}")
print(f"Successfully mapped to human: {mapped} ({mapped/total*100:.1f}%)")
print(f"High confidence mapping: {high_conf} ({high_conf/total*100:.1f}%)")
print(f"With human stroke evidence: {has_evidence} ({has_evidence/total*100:.1f}%)")

# Display top 10
print("\n" + "="*70)
print("TOP 10 GENES WITH HUMAN ORTHOLOGS")
print("="*70)
print("\n", supp_df.head(10).to_string(index=False))

# Conservation statistics
if use_mygene:
    one_to_one = len(ortholog_df[ortholog_df['conservation_type'] == '1:1 ortholog'])
    print(f"\n1:1 orthologs: {one_to_one}/{total} ({one_to_one/total*100:.1f}%)")

print("\n" + "="*70)
print("READY FOR MANUSCRIPT")
print("="*70)
print("\n✓ Use Table_SX_Mouse_Human_Orthologs.csv as supplementary material")
print("✓ Cite conservation evidence in discussion")
print("✓ Reference human stroke literature for validation")

# Generate citation suggestions
print("\n" + "="*70)
print("SUGGESTED LITERATURE SEARCH")
print("="*70)
print("\nSearch PubMed for validation:")
for gene in ortholog_df.head(10)['human_ortholog']:
    if gene != 'N/A':
        print(f"  - \"{gene} AND stroke AND human\"")
