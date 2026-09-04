"""
EXTRACT GENE IMPORTANCE FROM TRANSFORMER CHECKPOINTS
====================================================
This WORKING version loads your actual model architectures.

Usage:
    python extract_transformer_genes.py
"""

import os
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ✅ IMPORT YOUR ACTUAL MODEL CLASSES
from fixed_foundation_models import (
    scGPTModel, 
    scBERTModel, 
    scFormerModel, 
    GeneformerModel
)

print("="*70)
print("TRANSFORMER GENE IMPORTANCE EXTRACTION")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'checkpoint_dir': 'model_checkpoints',
    'output_dir': 'transformer_importances',
    'pca_loadings': 'pca_loadings.csv',
    'data_dir': __import__('repo_paths').data_dir(),
    'max_samples': 1000,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Create output directory
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# ============================================================================
# LOAD PCA LOADINGS
# ============================================================================

def load_pca_loadings():
    """Load PCA loadings matrix"""
    pca_path = CONFIG['pca_loadings']
    
    if not os.path.exists(pca_path):
        print(f"\n❌ ERROR: {pca_path} not found!")
        print("\nYou need to run the R script first:")
        print("  1. Open export_pca_loadings.R in R/RStudio")
        print("  2. Run it to create pca_loadings.csv")
        return None
    
    print(f"\n✓ Loading PCA loadings: {pca_path}")
    loadings = pd.read_csv(pca_path, index_col=0)
    print(f"  Shape: {loadings.shape}")
    print(f"  Genes: {loadings.shape[0]}, PCs: {loadings.shape[1]}")
    
    return loadings

# ============================================================================
# LOAD TEST DATA
# ============================================================================

def load_data():
    """Load test data for extraction"""
    print("\n✓ Loading test data...")
    
    test_pca = pd.read_csv(f"{CONFIG['data_dir']}/stroke_pca_test.csv")
    test_labels = pd.read_csv(f"{CONFIG['data_dir']}/stroke_labels_test.csv")
    
    X_test = test_pca.drop(columns=['cell_id']).values
    y_test = (test_labels['condition'] == 'Stroke').astype(int).values
    
    # Sample if too large
    if len(X_test) > CONFIG['max_samples']:
        indices = np.random.choice(len(X_test), CONFIG['max_samples'], replace=False)
        X_test = X_test[indices]
        y_test = y_test[indices]
    
    print(f"  Test samples: {X_test.shape}")
    
    return X_test, y_test

# ============================================================================
# EXTRACT EMBEDDING IMPORTANCE
# ============================================================================

def extract_embedding_importance(model, model_name):
    """
    Extract PC-level importance from model's embedding layer.
    
    This works for models with joint PC embedding (scGPT, scBERT, scFormer).
    For Geneformer, uses special extraction method.
    
    Method: Use L2 norm of input embedding weights
    """
    
    # ⚡ SPECIAL HANDLING FOR GENEFORMER
    if model_name == 'Geneformer' or hasattr(model, 'gene_positions'):
        return extract_geneformer_importance(model, model_name)
    print(f"\n[{model_name}] Extracting from embedding layer...")
    
    # Get first layer weights (input projection/embedding)
    first_layer = None
    
    if hasattr(model, 'input_projection'):
        first_layer = model.input_projection
        layer_name = "input_projection"
    elif hasattr(model, 'gene_embedding'):
        first_layer = model.gene_embedding
        layer_name = "gene_embedding"
    elif hasattr(model, 'gene_encoder') and isinstance(model.gene_encoder, nn.Sequential):
        first_layer = model.gene_encoder[0]
        layer_name = "gene_encoder[0]"
    elif hasattr(model, 'value_embedding'):
        first_layer = model.value_embedding
        layer_name = "value_embedding"
    else:
        print(f"  ⚠ No recognizable embedding layer found")
        print(f"  Available attributes: {[n for n, _ in model.named_children()]}")
        return None
    
    print(f"  ✓ Using layer: {layer_name}")
    
    # Get weights
    weights = first_layer.weight.data.cpu().numpy()  # Shape: (d_model, input_dim)
    print(f"  ✓ Weight shape: {weights.shape}")
    
    # Compute L2 norm for each input feature (PC)
    # This tells us how much each PC contributes to the model
    pc_importance = np.linalg.norm(weights, axis=0)  # Shape: (input_dim,)
    
    # Normalize to sum to 1
    pc_importance = pc_importance / pc_importance.sum()
    
    print(f"  ✓ PC importance extracted: {pc_importance.shape}")
    print(f"  ✓ Mean: {pc_importance.mean():.6f}, Max: {pc_importance.max():.6f}")
    print(f"  ✓ Top PC: PC_{np.argmax(pc_importance)+1} (importance: {pc_importance.max():.6f})")
    
    return pc_importance

# ============================================================================
# MAP PC IMPORTANCE TO GENES
# ============================================================================

def map_pc_to_genes(pc_importance, loadings, model_name):
    """
    Map PC-level importance to gene-level importance.
    
    Formula:
    gene_importance[i] = Σ_j |loading[i,j]| × pc_importance[j]
    """
    print(f"\n  Mapping to genes...")
    
    # Ensure same length
    n_pcs = min(len(pc_importance), loadings.shape[1])
    pc_importance = pc_importance[:n_pcs]
    
    # Compute gene importance
    gene_importance = np.zeros(loadings.shape[0])
    
    for i, pc_name in enumerate(loadings.columns[:n_pcs]):
        pc_loadings = np.abs(loadings[pc_name].values)
        gene_importance += pc_importance[i] * pc_loadings
    
    # Create DataFrame
    gene_df = pd.DataFrame({
        'gene': loadings.index,
        'importance': gene_importance,
        'rank': 0,
        'method': model_name
    })
    
    # Sort by importance
    gene_df = gene_df.sort_values('importance', ascending=False).reset_index(drop=True)
    gene_df['rank'] = range(1, len(gene_df) + 1)
    
    print(f"  ✓ Mapped to {len(gene_df)} genes")
    print(f"  ✓ Top gene: {gene_df.iloc[0]['gene']} (importance: {gene_df.iloc[0]['importance']:.6f})")
    
    return gene_df

# ============================================================================
# LOAD MODEL FROM CHECKPOINT
# ============================================================================

def load_model_from_checkpoint(model_class, checkpoint_path, device):
    """Load trained model from checkpoint"""
    
    if not os.path.exists(checkpoint_path):
        print(f"  ❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    print(f"  Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['model_config']
    
    print(f"  ✓ Config: d_model={config['d_model']}, layers={config['num_layers']}")
    
    # ✅ Initialize REAL model with saved config
    model = model_class(
        input_dim=config['input_dim'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"  ✓ Model loaded successfully!")
    print(f"  ✓ Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
    
    return model

# ============================================================================
# PROCESS SINGLE MODEL
# ============================================================================

def process_model(model_class, model_name, checkpoint_name, loadings):
    """
    Process single transformer model:
    1. Load checkpoint
    2. Extract PC importance
    3. Map to genes
    4. Save results
    """
    print(f"\n{'='*70}")
    print(f"PROCESSING: {model_name}")
    print(f"{'='*70}")
    
    checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], checkpoint_name)
    device = torch.device(CONFIG['device'])
    
    # Load model
    model = load_model_from_checkpoint(model_class, checkpoint_path, device)
    
    if model is None:
        print(f"  ⚠ Skipping {model_name} (checkpoint not found)")
        return None
    
    # Extract PC importance
    pc_importance = extract_embedding_importance(model, model_name)
    
    if pc_importance is None:
        print(f"  ❌ Failed to extract importance for {model_name}")
        return None
    
    # Map to genes
    gene_df = map_pc_to_genes(pc_importance, loadings, model_name)
    
    # Save
    output_file = os.path.join(
        CONFIG['output_dir'],
        f"gene_importance_{model_name}.csv"
    )
    gene_df.to_csv(output_file, index=False)
    print(f"\n  ✅ SAVED: {output_file}")
    
    # Display top 10 genes
    print(f"\n  Top 10 genes for {model_name}:")
    print(gene_df.head(10)[['rank', 'gene', 'importance']].to_string(index=False))
    
    return gene_df

# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Extract gene importance from all transformer checkpoints.
    """
    print("\nConfiguration:")
    print(f"  Device: {CONFIG['device']}")
    print(f"  Checkpoint dir: {CONFIG['checkpoint_dir']}")
    print(f"  Output dir: {CONFIG['output_dir']}")
    print(f"  Max samples: {CONFIG['max_samples']}")
    
    # Load PCA loadings
    loadings = load_pca_loadings()
    if loadings is None:
        return
    
    # Load data (optional - not used for embedding method)
    X_test, y_test = load_data()
    
    # ✅ Define models with CORRECT CLASS REFERENCES
    models_config = [
        {
            'class': scGPTModel,
            'name': 'scGPT',
            'checkpoint': 'scgpt_best.pt'
        },
        {
            'class': scBERTModel,
            'name': 'scBERT',
            'checkpoint': 'scbert_best.pt'
        },
        {
            'class': scFormerModel,
            'name': 'scFormer',
            'checkpoint': 'scformer_best.pt'
        },
        {
            'class': GeneformerModel,
            'name': 'Geneformer',
            'checkpoint': 'geneformer_best.pt'
        }
    ]
    
    # Process each model
    results = {}
    for config in models_config:
        gene_df = process_model(
            config['class'],
            config['name'],
            config['checkpoint'],
            loadings
        )
        
        if gene_df is not None:
            results[config['name']] = gene_df
    
    # Summary
    print(f"\n{'='*70}")
    print(f"✅ EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"\nSuccessfully processed: {len(results)}/{len(models_config)} models")
    print(f"\nFiles created in: {CONFIG['output_dir']}/")
    
    for name in results.keys():
        print(f"  ✓ gene_importance_{name}.csv")
    
    print(f"\n🔬 Ready for interpretability analysis!")
    print(f"   Next step: python integrated_interpretability_pipeline.py")

if __name__ == "__main__":
    main()