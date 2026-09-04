"""
INTEGRATED INTERPRETABILITY PIPELINE
=====================================
This script integrates:
1. Your existing phase2_interpretability.py (SHAP & simulated attention)
2. The new gene_importance_extractor.py (PC → Gene mapping)
3. Real attention extraction from real_attention_extraction.py

This is the COMPLETE solution for your paper!

Usage:
    python integrated_interpretability_pipeline.py
"""

import os
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import the gene extractor
from gene_importance_extractor import GeneImportanceExtractor

print("="*80)
print("INTEGRATED INTERPRETABILITY PIPELINE")
print("Foundation Models vs Interpretable Methods: Gene-Level Comparison")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'data_dir': '.',
    'pca_loadings': 'pca_loadings.csv',
    'output_dir': 'gene_level_results',
    'checkpoint_dir': 'model_checkpoints',
    'max_samples_shap': 500,
    'max_samples_attention': 500,
    'top_n_genes': 100,
    'consensus_threshold': 4  # Genes must appear in ≥4 models
}

# Create output directory
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# ============================================================================
# STEP 1: EXTRACT GENE IMPORTANCE FROM BASELINE MODELS
# ============================================================================

def extract_baseline_models_genes():
    """
    Extract gene-level importance from baseline models using trained models.
    """
    print("\n" + "="*80)
    print("STEP 1: EXTRACTING GENES FROM BASELINE MODELS")
    print("="*80)
    
    # Check if PCA loadings exist
    if not os.path.exists(CONFIG['pca_loadings']):
        print("\n❌ ERROR: pca_loadings.csv not found!")
        print("\nYou need to run the R script first:")
        print("  1. Open the export_pca_loadings.R script")
        print("  2. Run it in R/RStudio")
        print("  3. It will create pca_loadings.csv")
        print("\nThen run this script again.")
        return {}
    
    # Initialize extractor
    extractor = GeneImportanceExtractor(CONFIG['pca_loadings'])
    
    # Load data
    print("\nLoading data...")
    train_pca = pd.read_csv(f"{CONFIG['data_dir']}/stroke_pca_train.csv")
    test_pca = pd.read_csv(f"{CONFIG['data_dir']}/stroke_pca_test.csv")
    train_labels = pd.read_csv(f"{CONFIG['data_dir']}/stroke_labels_train.csv")
    test_labels = pd.read_csv(f"{CONFIG['data_dir']}/stroke_labels_test.csv")
    
    X_train = train_pca.drop(columns=['cell_id']).values
    X_test = test_pca.drop(columns=['cell_id']).values
    y_train = (train_labels['condition'] == 'Stroke').astype(int).values
    
    model_file = 'model_random_forest.pkl'
    if os.path.exists(model_file):
        print(f"  Loading saved model: {model_file}")
        with open(model_file, 'rb') as f:
            rf_model = pickle.load(f)
    else:
        print(f"  Training new model...")
        from sklearn.ensemble import RandomForestClassifier
        rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        with open(model_file, 'wb') as f:
            pickle.dump(rf_model, f)
    
    # Extract genes from feature importances
    gene_df_rf = extractor.from_feature_importances(
        rf_model.feature_importances_,
        "Random Forest"
    )
    all_gene_importance['Random Forest'] = gene_df_rf
    
    # Save
    gene_df_rf.to_csv(f"{CONFIG['output_dir']}/genes_random_forest.csv", index=False)
    print(f"  ✓ Saved: genes_random_forest.csv")
    
    # ========================================================================
    # 1C: XGBOOST
    # ========================================================================
    print("\n[3/3] XGBoost...")
    
    model_file = 'model_xgboost.pkl'
    if os.path.exists(model_file):
        print(f"  Loading saved model: {model_file}")
        with open(model_file, 'rb') as f:
            xgb_model = pickle.load(f)
    else:
        print(f"  Training new model...")
        from xgboost import XGBClassifier
        xgb_model = XGBClassifier(n_estimators=200, random_state=42, verbosity=0)
        xgb_model.fit(X_train, y_train)
        with open(model_file, 'wb') as f:
            pickle.dump(xgb_model, f)
    
    # Extract genes from feature importances
    gene_df_xgb = extractor.from_feature_importances(
        xgb_model.feature_importances_,
        "XGBoost"
    )
    all_gene_importance['XGBoost'] = gene_df_xgb
    
    # Save
    gene_df_xgb.to_csv(f"{CONFIG['output_dir']}/genes_xgboost.csv", index=False)
    print(f"  ✓ Saved: genes_xgboost.csv")
    
    print(f"\n✓ Extracted genes from {len(all_gene_importance)} baseline models")
    
    return all_gene_importance, extractor, (X_train, X_test, y_train, y_test)


# ============================================================================
# STEP 2: EXTRACT GENE IMPORTANCE FROM SHAP ANALYSIS
# ============================================================================

def extract_shap_based_genes(extractor, models_dict, X_train, X_test):
    """
    Run SHAP analysis and extract gene-level importance.
    """
    print("\n" + "="*80)
    print("STEP 2: EXTRACTING GENES FROM SHAP ANALYSIS")
    print("="*80)
    
    import shap
    
    all_shap_genes = {}
    
    for model_name, model in models_dict.items():
        if not hasattr(model, 'predict_proba'):
            continue
        
        print(f"\n[Processing] {model_name}...")
        
        try:
            # Sample data for SHAP
            if len(X_test) > CONFIG['max_samples_shap']:
                indices = np.random.choice(len(X_test), CONFIG['max_samples_shap'], replace=False)
                X_explain = X_test[indices]
            else:
                X_explain = X_test
            
            # Compute SHAP values
            print(f"  Computing SHAP values...")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_explain)
            
            # Handle multi-output
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class
            
            print(f"  SHAP values shape: {shap_values.shape}")
            
            # Extract genes from SHAP
            gene_df = extractor.from_shap_values(
                shap_values,
                f"{model_name} SHAP"
            )
            all_shap_genes[f"{model_name} SHAP"] = gene_df
            
            # Save
            filename = f"{CONFIG['output_dir']}/genes_{model_name.lower().replace(' ', '_')}_shap.csv"
            gene_df.to_csv(filename, index=False)
            print(f"  ✓ Saved: {filename}")
            
            # Also save SHAP values themselves
            shap_file = f"{CONFIG['output_dir']}/shap_values_{model_name.lower().replace(' ', '_')}.npy"
            np.save(shap_file, shap_values)
            print(f"  ✓ Saved SHAP values: {shap_file}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\n✓ Extracted genes from {len(all_shap_genes)} SHAP analyses")
    
    return all_shap_genes


# ============================================================================
# STEP 3: LOAD ATTENTION-BASED GENES FROM TRANSFORMERS
# ============================================================================

def load_transformer_genes(extractor):
    """
    Load pre-computed transformer gene importance (from real_attention_extraction.py).
    If not available, provide instructions.
    """
    print("\n" + "="*80)
    print("STEP 3: LOADING TRANSFORMER GENE IMPORTANCE")
    print("="*80)
    
    transformer_dir = Path(CONFIG['data_dir']) / 'transformer_importances'
    
    if not transformer_dir.exists():
        print("\n⚠ Transformer importance not yet extracted!")
        print("\nTo extract attention from transformers:")
        print("  1. Run: python real_attention_extraction.py")
        print("  2. This will extract attention from saved checkpoints")
        print("  3. Then run this script again")
        print("\nFor now, continuing with baseline models only...")
        return {}
    
    all_transformer_genes = {}
    
    models = ['scGPT', 'scBERT', 'scFormer', 'Geneformer']
    
    for model_name in models:
        gene_file = transformer_dir / f"gene_importance_{model_name}.csv"
        
        if gene_file.exists():
            print(f"[Loading] {model_name}...")
            gene_df = pd.read_csv(gene_file)
            
            # Ensure correct format
            if 'gene' not in gene_df.columns:
                gene_df = gene_df.rename(columns={'Feature': 'gene', 'Importance': 'importance'})
            
            # Add method column if missing
            if 'method' not in gene_df.columns:
                gene_df['method'] = model_name
            
            # Add rank if missing
            if 'rank' not in gene_df.columns:
                gene_df = gene_df.sort_values('importance', ascending=False).reset_index(drop=True)
                gene_df['rank'] = range(1, len(gene_df) + 1)
            
            all_transformer_genes[model_name] = gene_df
            
            print(f"  ✓ Loaded: {len(gene_df)} genes")
            print(f"  Top gene: {gene_df.iloc[0]['gene']}")
        else:
            print(f"[Skipping] {model_name} - file not found")
    
    if len(all_transformer_genes) > 0:
        print(f"\n✓ Loaded genes from {len(all_transformer_genes)} transformer models")
    else:
        print("\n⚠ No transformer genes loaded")
    
    return all_transformer_genes


# ============================================================================
# STEP 4: CROSS-METHOD COMPARISON
# ============================================================================

def compare_gene_signatures(all_gene_importance, top_n=100):
    """
    Compare gene signatures across all methods.
    """
    print("\n" + "="*80)
    print("STEP 4: CROSS-METHOD GENE SIGNATURE COMPARISON")
    print("="*80)
    
    if len(all_gene_importance) < 2:
        print("⚠ Need at least 2 models for comparison")
        return None
    
    # Get top genes from each method
    top_genes_dict = {}
    for method, gene_df in all_gene_importance.items():
        top_genes = gene_df.head(top_n)['gene'].tolist()
        top_genes_dict[method] = set(top_genes)
    
    # Compute pairwise Jaccard similarity
    methods = list(top_genes_dict.keys())
    n_methods = len(methods)
    
    jaccard_matrix = np.zeros((n_methods, n_methods))
    
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            set1 = top_genes_dict[method1]
            set2 = top_genes_dict[method2]
            
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            jaccard_matrix[i, j] = intersection / union if union > 0 else 0
    
    # Create DataFrame
    jaccard_df = pd.DataFrame(
        jaccard_matrix,
        index=methods,
        columns=methods
    )
    
    print("\nJaccard Similarity Matrix (Top {} genes):".format(top_n))
    print(jaccard_df.round(3))
    
    # Save
    jaccard_df.to_csv(f"{CONFIG['output_dir']}/gene_signature_convergence.csv")
    print(f"\n✓ Saved: gene_signature_convergence.csv")
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(jaccard_df, annot=True, fmt='.2f', cmap='YlOrRd', 
               square=True, cbar_kws={'label': 'Jaccard Similarity'})
    plt.title(f'Gene Signature Convergence (Top {top_n} genes)')
    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/gene_convergence_heatmap.png", dpi=300)
    plt.close()
    print(f"✓ Saved: gene_convergence_heatmap.png")
    
    return jaccard_df, top_genes_dict


# ============================================================================
# STEP 5: CONSENSUS GENES
# ============================================================================

def find_consensus_genes(all_gene_importance, min_models=4, top_n=100):
    """
    Find genes that appear in top-N of multiple models.
    """
    print("\n" + "="*80)
    print(f"STEP 5: FINDING CONSENSUS GENES (≥{min_models} models)")
    print("="*80)
    
    # Count occurrences of each gene in top-N
    gene_counts = {}
    gene_models = {}
    
    for method, gene_df in all_gene_importance.items():
        top_genes = gene_df.head(top_n)['gene'].tolist()
        
        for gene in top_genes:
            if gene not in gene_counts:
                gene_counts[gene] = 0
                gene_models[gene] = []
            gene_counts[gene] += 1
            gene_models[gene].append(method)
    
    # Filter for consensus genes
    consensus_list = []
    for gene, count in gene_counts.items():
        if count >= min_models:
            consensus_list.append({
                'gene': gene,
                'count': count,
                'models': ', '.join(gene_models[gene]),
                'frequency': count / len(all_gene_importance)
            })
    
    # Create DataFrame
    consensus_df = pd.DataFrame(consensus_list)
    consensus_df = consensus_df.sort_values('count', ascending=False)
    
    print(f"\nFound {len(consensus_df)} consensus genes")
    print(f"(appearing in ≥{min_models} out of {len(all_gene_importance)} models)")
    
    if len(consensus_df) > 0:
        print("\nTop 20 consensus genes:")
        print(consensus_df.head(20)[['gene', 'count', 'frequency']].to_string(index=False))
        
        # Save
        consensus_df.to_csv(f"{CONFIG['output_dir']}/consensus_genes.csv", index=False)
        print(f"\n✓ Saved: consensus_genes.csv")
        
        # Plot distribution
        plt.figure(figsize=(10, 6))
        count_dist = consensus_df['count'].value_counts().sort_index()
        plt.bar(count_dist.index, count_dist.values, color='steelblue', alpha=0.7)
        plt.xlabel('Number of Models')
        plt.ylabel('Number of Genes')
        plt.title(f'Consensus Gene Distribution (Top {top_n} per model)')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{CONFIG['output_dir']}/consensus_distribution.png", dpi=300)
        plt.close()
        print(f"✓ Saved: consensus_distribution.png")
    
    return consensus_df


# ============================================================================
# STEP 6: IL-1 PATHWAY VALIDATION
# ============================================================================

def validate_il1_pathway(consensus_df):
    """
    Check for IL-1 pathway genes in consensus list.
    """
    print("\n" + "="*80)
    print("STEP 6: IL-1 PATHWAY VALIDATION")
    print("="*80)
    
    # Known IL-1 pathway genes (mouse)
    il1_genes = [
        'Il1a', 'Il1b', 'Il1r1', 'Il1rap', 'Il1rn',
        'Myd88', 'Irak1', 'Irak2', 'Irak4',
        'Nfkb1', 'Nfkb2', 'Rela', 'Relb',
        'Tnf', 'Il6', 'Ccl2', 'Cxcl1', 'Cxcl2'
    ]
    
    # Check overlap
    consensus_genes = set(consensus_df['gene'].tolist())
    found_il1_genes = [g for g in il1_genes if g in consensus_genes]
    
    print(f"\nIL-1 pathway genes checked: {len(il1_genes)}")
    print(f"Found in consensus: {len(found_il1_genes)}")
    
    if len(found_il1_genes) > 0:
        print(f"\n✅ IL-1 PATHWAY GENES FOUND:")
        for gene in found_il1_genes:
            gene_info = consensus_df[consensus_df['gene'] == gene].iloc[0]
            print(f"  • {gene:10s} - in {gene_info['count']} models ({gene_info['frequency']:.1%})")
        
        # Enrichment test (hypergeometric)
        from scipy.stats import hypergeom
        
        M = len(consensus_df)  # Total genes in consensus
        n = len(il1_genes)     # IL-1 genes
        N = len(found_il1_genes)  # Overlap
        
        # This is simplified - proper test needs background gene set size
        print(f"\n📊 IL-1 Pathway Enrichment:")
        print(f"  Overlap: {N}/{n} IL-1 genes found")
        print(f"  Enrichment: {(N/n * 100):.1f}% of IL-1 genes in consensus")
    else:
        print("\n⚠ No IL-1 pathway genes found in consensus")
        print("  Note: This could mean:")
        print("  1. IL-1 pathway not strongly implicated")
        print("  2. Gene names might differ (check synonyms)")
        print("  3. Pathway genes rank lower (check full list)")
    
    # Save results
    il1_results = {
        'il1_genes_checked': il1_genes,
        'il1_genes_found': found_il1_genes,
        'enrichment': len(found_il1_genes) / len(il1_genes) if len(il1_genes) > 0 else 0
    }
    
    with open(f"{CONFIG['output_dir']}/il1_validation.json", 'w') as f:
        json.dump(il1_results, f, indent=2)
    
    print(f"\n✓ Saved: il1_validation.json")
    
    return il1_results


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Complete integrated interpretability pipeline.
    """
    print("\n" + "="*80)
    print("STARTING INTEGRATED PIPELINE")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  PCA loadings: {CONFIG['pca_loadings']}")
    print(f"  Output directory: {CONFIG['output_dir']}")
    print(f"  Top N genes: {CONFIG['top_n_genes']}")
    print(f"  Consensus threshold: >={CONFIG['consensus_threshold']} models")
    
    # Step 1: Baseline models
    baseline_genes, extractor, data = extract_baseline_models_genes()
    
    if len(baseline_genes) == 0:
        print("\n❌ Failed to extract baseline genes. Check PCA loadings file.")
        return
    
    X_train, X_test, y_train, y_test = data
    
    # Step 2: SHAP analysis
    shap_genes = extract_shap_based_genes(extractor, baseline_genes, X_train, X_test)
    
    # Step 3: Transformer attention
    transformer_genes = load_transformer_genes(extractor)
    
    # Combine all
    all_gene_importance = {**baseline_genes, **shap_genes, **transformer_genes}
    
    print(f"\n{'='*80}")
    print(f"TOTAL METHODS WITH GENE IMPORTANCE: {len(all_gene_importance)}")
    print(f"{'='*80}")
    for method in all_gene_importance.keys():
        print(f"  ✓ {method}")
    
    # Step 4: Cross-method comparison
    jaccard_df, top_genes_dict = compare_gene_signatures(
        all_gene_importance,
        top_n=CONFIG['top_n_genes']
    )
    
    # Step 5: Consensus genes
    consensus_df = find_consensus_genes(
        all_gene_importance,
        min_models=CONFIG['consensus_threshold'],
        top_n=CONFIG['top_n_genes']
    )
    
    # Step 6: IL-1 validation
    if len(consensus_df) > 0:
        il1_results = validate_il1_pathway(consensus_df)
    
    # Generate summary report
    generate_summary_report(all_gene_importance, consensus_df, jaccard_df)
    
    print("\n" + "="*80)
    print("✅ INTEGRATED PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nAll results saved in: {CONFIG['output_dir']}/")
    print("\nKey files created:")
    print(f"  • genes_*.csv - Gene importance for each model")
    print(f"  • consensus_genes.csv - Top consensus biomarkers")
    print(f"  • gene_signature_convergence.csv - Method similarity matrix")
    print(f"  • il1_validation.json - IL-1 pathway validation")
    print(f"  • summary_report.txt - Complete analysis summary")


def generate_summary_report(all_gene_importance, consensus_df, jaccard_df):
    """Generate text summary report"""
    
    report = f"""
{'='*80}
GENE-LEVEL INTERPRETABILITY ANALYSIS - SUMMARY REPORT
{'='*80}

METHODS ANALYZED: {len(all_gene_importance)}
{chr(10).join('  • ' + m for m in all_gene_importance.keys())}

CONVERGENCE ANALYSIS:
--------------------
Average Jaccard Similarity: {jaccard_df.values[np.triu_indices_from(jaccard_df.values, k=1)].mean():.3f}
Range: [{jaccard_df.values[np.triu_indices_from(jaccard_df.values, k=1)].min():.3f} - {jaccard_df.values[np.triu_indices_from(jaccard_df.values, k=1)].max():.3f}]

CONSENSUS GENES:
---------------
Total consensus genes (≥{CONFIG['consensus_threshold']} models): {len(consensus_df) if len(consensus_df) > 0 else 0}

Top 10 Consensus Genes:
{consensus_df.head(10)[['gene', 'count', 'frequency']].to_string(index=False) if len(consensus_df) > 0 else 'None found'}

BIOLOGICAL INTERPRETATION:
-------------------------
These consensus genes represent high-confidence biomarkers identified by 
multiple independent interpretability methods. They warrant:

1. Literature validation (known stroke markers)
2. GO enrichment analysis  
3. Pathway analysis (IL-1, inflammation, apoptosis)
4. Experimental validation (RT-qPCR, Western blot)

FILES GENERATED:
---------------
{chr(10).join('  • ' + f for f in os.listdir(CONFIG['output_dir']) if f.endswith(('.csv', '.png', '.json')))}

NEXT STEPS FOR PUBLICATION:
---------------------------
1. ✅ Gene-level importance extracted
2. ✅ Cross-method convergence analyzed
3. ⏳ GO enrichment on consensus genes
4. ⏳ Literature comparison
5. ⏳ Experimental validation plan

{'='*80}
"""
    
    with open(f"{CONFIG['output_dir']}/summary_report.txt", 'w') as f:
        f.write(report)
    
    print(report)


if __name__ == "__main__":
    main()