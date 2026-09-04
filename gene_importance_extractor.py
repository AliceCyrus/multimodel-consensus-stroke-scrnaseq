"""
GENE IMPORTANCE EXTRACTOR
==========================
The critical bridge: Maps PC importance → Gene importance using PCA loadings

This is THE KEY FILE that your paper needs!

Usage:
    from gene_importance_extractor import GeneImportanceExtractor
    
    extractor = GeneImportanceExtractor('pca_loadings.csv')
    
    # For baseline models
    gene_importance = extractor.from_model_coefficients(lr_model.coef_[0])
    
    # For SHAP values
    gene_importance = extractor.from_shap_values(shap_values)
    
    # For attention weights
    gene_importance = extractor.from_attention(attention_weights)
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

class GeneImportanceExtractor:
    """
    Extract gene-level importance from any model that produces PC-level importance.
    
    This is the CRITICAL class that connects:
    - Model predictions (on 50 PCs) → Gene biomarkers (3000 genes)
    
    Methods available:
    - from_coefficients: For linear models (LR)
    - from_feature_importances: For tree models (RF, XGBoost)
    - from_shap_values: For SHAP analysis
    - from_attention: For transformer models
    - from_gradients: For neural networks
    """
    
    def __init__(self, pca_loadings_path='pca_loadings.csv'):
        """
        Initialize with PCA loadings file.
        
        Parameters:
        -----------
        pca_loadings_path : str
            Path to PCA loadings CSV file
            Format: rows=genes, columns=PC_1 to PC_50
        """
        print(f"\n{'='*70}")
        print("INITIALIZING GENE IMPORTANCE EXTRACTOR")
        print(f"{'='*70}")
        
        # Load PCA loadings
        if not os.path.exists(pca_loadings_path):
            raise FileNotFoundError(
                f"PCA loadings file not found: {pca_loadings_path}\n"
                f"Please run the R script to export PCA loadings first."
            )
        
        self.pca_loadings = pd.read_csv(pca_loadings_path, index_col=0)
        
        print(f"✓ Loaded PCA loadings: {self.pca_loadings.shape}")
        print(f"  - Genes: {self.pca_loadings.shape[0]}")
        print(f"  - PCs: {self.pca_loadings.shape[1]}")
        print(f"  - PC columns: {list(self.pca_loadings.columns[:5])} ... {list(self.pca_loadings.columns[-2:])}")
        print(f"  - Gene examples: {list(self.pca_loadings.index[:5])}")
        
        # Store gene names and PC names
        self.gene_names = self.pca_loadings.index.tolist()
        self.pc_names = self.pca_loadings.columns.tolist()
        self.n_genes = len(self.gene_names)
        self.n_pcs = len(self.pc_names)
        
        # Convert to numpy for faster computation
        self.loadings_matrix = self.pca_loadings.values  # Shape: (n_genes, n_pcs)
        
        print(f"\n✓ Ready to extract gene importance from PC importance!")
    
    def _map_to_genes(self, pc_importance, method_name="Unknown"):
        """
        Core mapping function: PC importance → Gene importance
        
        Formula:
        gene_importance[i] = Σ_j |loading[i,j]| × pc_importance[j]
        
        Parameters:
        -----------
        pc_importance : array-like, shape (n_pcs,)
            Importance score for each PC
        method_name : str
            Name of method (for logging)
        
        Returns:
        --------
        gene_importance_df : pd.DataFrame
            DataFrame with columns: ['gene', 'importance', 'rank']
            Sorted by importance (descending)
        """
        # Convert to numpy array
        pc_importance = np.array(pc_importance)
        
        # Validate shape
        if pc_importance.shape[0] != self.n_pcs:
            raise ValueError(
                f"PC importance has {pc_importance.shape[0]} values, "
                f"but PCA loadings has {self.n_pcs} PCs"
            )
        
        # Compute gene importance using weighted sum of absolute loadings
        # gene_importance = |loadings| @ pc_importance
        gene_scores = np.abs(self.loadings_matrix) @ pc_importance
        
        # Create DataFrame
        gene_df = pd.DataFrame({
            'gene': self.gene_names,
            'importance': gene_scores,
            'rank': 0
        })
        
        # Sort by importance (descending) and add ranks
        gene_df = gene_df.sort_values('importance', ascending=False).reset_index(drop=True)
        gene_df['rank'] = range(1, len(gene_df) + 1)
        
        # Add method name
        gene_df['method'] = method_name
        
        # Summary statistics
        print(f"\n✓ Mapped PC importance → Gene importance ({method_name})")
        print(f"  - Total genes: {len(gene_df)}")
        print(f"  - Top gene: {gene_df.iloc[0]['gene']} (importance: {gene_df.iloc[0]['importance']:.6f})")
        print(f"  - Mean importance: {gene_df['importance'].mean():.6f}")
        print(f"  - Std importance: {gene_df['importance'].std():.6f}")
        
        return gene_df
    
    # ========================================================================
    # METHOD 1: From Linear Model Coefficients (Logistic Regression)
    # ========================================================================
    
    def from_coefficients(self, coefficients, model_name="Logistic Regression"):
        """
        Extract gene importance from linear model coefficients.
        
        For Logistic Regression: model.coef_[0]
        
        Parameters:
        -----------
        coefficients : array, shape (n_pcs,)
            Model coefficients (one per PC)
        model_name : str
            Name of model for labeling
        
        Returns:
        --------
        gene_df : pd.DataFrame
            Gene importance DataFrame
        """
        print(f"\n{'='*70}")
        print(f"EXTRACTING GENES FROM COEFFICIENTS ({model_name})")
        print(f"{'='*70}")
        
        # Use absolute values of coefficients as PC importance
        pc_importance = np.abs(coefficients)
        
        print(f"Coefficient statistics:")
        print(f"  - Mean |coef|: {pc_importance.mean():.6f}")
        print(f"  - Max |coef|: {pc_importance.max():.6f}")
        print(f"  - Top PC: PC_{np.argmax(pc_importance)+1}")
        
        return self._map_to_genes(pc_importance, method_name=model_name)
    
    # ========================================================================
    # METHOD 2: From Tree Feature Importances (Random Forest, XGBoost)
    # ========================================================================
    
    def from_feature_importances(self, importances, model_name="Random Forest"):
        """
        Extract gene importance from tree model feature importances.
        
        For Random Forest: model.feature_importances_
        For XGBoost: model.feature_importances_
        
        Parameters:
        -----------
        importances : array, shape (n_pcs,)
            Feature importances (one per PC)
        model_name : str
            Name of model for labeling
        
        Returns:
        --------
        gene_df : pd.DataFrame
            Gene importance DataFrame
        """
        print(f"\n{'='*70}")
        print(f"EXTRACTING GENES FROM FEATURE IMPORTANCES ({model_name})")
        print(f"{'='*70}")
        
        pc_importance = np.array(importances)
        
        print(f"Feature importance statistics:")
        print(f"  - Mean importance: {pc_importance.mean():.6f}")
        print(f"  - Max importance: {pc_importance.max():.6f}")
        print(f"  - Top PC: PC_{np.argmax(pc_importance)+1}")
        
        return self._map_to_genes(pc_importance, method_name=model_name)
    
    # ========================================================================
    # METHOD 3: From SHAP Values
    # ========================================================================
    
    def from_shap_values(self, shap_values, model_name="SHAP Analysis"):
        """
        Extract gene importance from SHAP values.
        
        SHAP values shape: (n_samples, n_pcs)
        We take the mean absolute SHAP value for each PC.
        
        Parameters:
        -----------
        shap_values : array, shape (n_samples, n_pcs)
            SHAP values from explainer
        model_name : str
            Name of model for labeling
        
        Returns:
        --------
        gene_df : pd.DataFrame
            Gene importance DataFrame
        """
        print(f"\n{'='*70}")
        print(f"EXTRACTING GENES FROM SHAP VALUES ({model_name})")
        print(f"{'='*70}")
        
        # Average absolute SHAP values across samples
        pc_importance = np.abs(shap_values).mean(axis=0)
        
        print(f"SHAP value statistics:")
        print(f"  - Samples: {shap_values.shape[0]}")
        print(f"  - Mean |SHAP|: {pc_importance.mean():.6f}")
        print(f"  - Max |SHAP|: {pc_importance.max():.6f}")
        print(f"  - Top PC: PC_{np.argmax(pc_importance)+1}")
        
        return self._map_to_genes(pc_importance, method_name=model_name)
    
    # ========================================================================
    # METHOD 4: From Attention Weights (Transformers)
    # ========================================================================
    
    def from_attention(self, attention_weights, model_name="Transformer"):
        """
        Extract gene importance from attention weights.
        
        Attention weights can be:
        - 1D array: (n_pcs,) - already averaged
        - 2D array: (n_samples, n_pcs) - need to average
        - 3D array: (n_samples, n_heads, n_pcs) - average across samples & heads
        
        Parameters:
        -----------
        attention_weights : array
            Attention weights from transformer
        model_name : str
            Name of model for labeling
        
        Returns:
        --------
        gene_df : pd.DataFrame
            Gene importance DataFrame
        """
        print(f"\n{'='*70}")
        print(f"EXTRACTING GENES FROM ATTENTION ({model_name})")
        print(f"{'='*70}")
        
        attention = np.array(attention_weights)
        
        print(f"Attention shape: {attention.shape}")
        
        # Reduce to 1D if needed
        if len(attention.shape) == 3:
            # (samples, heads, pcs) → (pcs,)
            pc_importance = attention.mean(axis=(0, 1))
            print(f"  Averaged across samples and heads")
        elif len(attention.shape) == 2:
            # (samples, pcs) → (pcs,)
            pc_importance = attention.mean(axis=0)
            print(f"  Averaged across samples")
        else:
            # Already 1D
            pc_importance = attention
        
        # Ensure correct length
        if len(pc_importance) != self.n_pcs:
            if len(pc_importance) > self.n_pcs:
                # Trim (remove special tokens)
                print(f"  Trimming attention from {len(pc_importance)} to {self.n_pcs}")
                pc_importance = pc_importance[:self.n_pcs]
            else:
                raise ValueError(
                    f"Attention has {len(pc_importance)} values, "
                    f"but need {self.n_pcs} PCs"
                )
        
        print(f"Attention statistics:")
        print(f"  - Mean attention: {pc_importance.mean():.6f}")
        print(f"  - Max attention: {pc_importance.max():.6f}")
        print(f"  - Top PC: PC_{np.argmax(pc_importance)+1}")
        
        return self._map_to_genes(pc_importance, method_name=model_name)
    
    # ========================================================================
    # METHOD 5: From Gradients (Neural Networks)
    # ========================================================================
    
    def from_gradients(self, gradients, model_name="Neural Network"):
        """
        Extract gene importance from input gradients.
        
        Gradients shape: (n_samples, n_pcs) or (n_pcs,)
        We use mean absolute gradient as PC importance.
        
        Parameters:
        -----------
        gradients : array
            Gradients w.r.t. input features
        model_name : str
            Name of model for labeling
        
        Returns:
        --------
        gene_df : pd.DataFrame
            Gene importance DataFrame
        """
        print(f"\n{'='*70}")
        print(f"EXTRACTING GENES FROM GRADIENTS ({model_name})")
        print(f"{'='*70}")
        
        grads = np.array(gradients)
        
        # Average if 2D
        if len(grads.shape) == 2:
            pc_importance = np.abs(grads).mean(axis=0)
        else:
            pc_importance = np.abs(grads)
        
        print(f"Gradient statistics:")
        print(f"  - Mean |gradient|: {pc_importance.mean():.6f}")
        print(f"  - Max |gradient|: {pc_importance.max():.6f}")
        print(f"  - Top PC: PC_{np.argmax(pc_importance)+1}")
        
        return self._map_to_genes(pc_importance, method_name=model_name)
    
    # ========================================================================
    # BATCH PROCESSING: Extract from Multiple Models
    # ========================================================================
    
    def extract_from_all_models(self, models_dict, X_train=None, X_test=None, 
                                output_dir='gene_importance_results'):
        """
        Batch extraction from multiple models.
        
        Parameters:
        -----------
        models_dict : dict
            {model_name: trained_model}
        X_train : array, optional
            Training data for SHAP background
        X_test : array, optional
            Test data for SHAP/attention extraction
        output_dir : str
            Directory to save results
        
        Returns:
        --------
        all_gene_importance : dict
            {model_name: gene_importance_df}
        """
        print(f"\n{'='*70}")
        print("BATCH EXTRACTION FROM ALL MODELS")
        print(f"{'='*70}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        all_gene_importance = {}
        
        for model_name, model in models_dict.items():
            print(f"\n[Processing] {model_name}...")
            
            try:
                # Determine model type and extract accordingly
                if hasattr(model, 'coef_'):
                    # Linear model
                    gene_df = self.from_coefficients(model.coef_[0], model_name)
                    
                elif hasattr(model, 'feature_importances_'):
                    # Tree-based model
                    gene_df = self.from_feature_importances(
                        model.feature_importances_, model_name
                    )
                    
                elif hasattr(model, 'model_') and hasattr(model.model_, 'state_dict'):
                    # PyTorch model - use gradients or attention
                    if X_test is not None:
                        print(f"  Extracting from neural network...")
                        # This requires the real_attention_extraction functions
                        # For now, skip or use placeholder
                        print(f"  ⚠ Neural network extraction requires specialized code")
                        continue
                    else:
                        print(f"  ⚠ Need X_test for neural network extraction")
                        continue
                
                else:
                    print(f"  ⚠ Unknown model type, skipping")
                    continue
                
                # Save results
                output_path = os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_genes.csv')
                gene_df.to_csv(output_path, index=False)
                print(f"  ✓ Saved: {output_path}")
                
                all_gene_importance[model_name] = gene_df
                
            except Exception as e:
                print(f"  ✗ Error processing {model_name}: {e}")
        
        print(f"\n✓ Extracted gene importance from {len(all_gene_importance)} models")
        print(f"✓ Results saved in: {output_dir}/")
        
        return all_gene_importance
    
    # ========================================================================
    # UTILITY: Get Top Genes
    # ========================================================================
    
    def get_top_genes(self, gene_df, top_n=50):
        """
        Get top N most important genes.
        
        Parameters:
        -----------
        gene_df : pd.DataFrame
            Gene importance DataFrame
        top_n : int
            Number of top genes to return
        
        Returns:
        --------
        top_genes : pd.DataFrame
            Top N genes
        """
        return gene_df.head(top_n)
    
    def get_genes_by_names(self, gene_df, gene_list):
        """
        Get importance scores for specific genes.
        
        Parameters:
        -----------
        gene_df : pd.DataFrame
            Gene importance DataFrame
        gene_list : list
            List of gene names to look up
        
        Returns:
        --------
        subset : pd.DataFrame
            Subset of gene_df for requested genes
        """
        return gene_df[gene_df['gene'].isin(gene_list)]


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def extract_genes_from_saved_models(results_dir='.',
                                   pca_loadings_path='pca_loadings.csv',
                                   output_dir='gene_importance_results'):
    """
    Extract gene importance from all saved model result files.
    
    This function:
    1. Looks for saved model .pkl files or results .json files
    2. Loads each model
    3. Extracts gene importance
    4. Saves results
    
    Parameters:
    -----------
    results_dir : str
        Directory containing saved models
    pca_loadings_path : str
        Path to PCA loadings CSV
    output_dir : str
        Directory to save gene importance results
    
    Returns:
    --------
    all_gene_importance : dict
        {model_name: gene_importance_df}
    """
    print(f"\n{'='*70}")
    print("AUTO-EXTRACTING FROM SAVED MODELS")
    print(f"{'='*70}")
    
    extractor = GeneImportanceExtractor(pca_loadings_path)
    
    # Look for saved model files
    model_files = {
        'Logistic Regression': 'model_logistic_regression.pkl',
        'Random Forest': 'model_random_forest.pkl',
        'XGBoost': 'model_xgboost.pkl'
    }
    
    models_to_process = {}
    
    for model_name, filename in model_files.items():
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            print(f"✓ Found: {filename}")
            with open(filepath, 'rb') as f:
                models_to_process[model_name] = pickle.load(f)
        else:
            print(f"✗ Not found: {filename}")
    
    if len(models_to_process) == 0:
        print("\n⚠ No saved models found!")
        print("Please ensure model .pkl files are in the results directory")
        return {}
    
    # Extract from all found models
    return extractor.extract_from_all_models(
        models_to_process,
        output_dir=output_dir
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of GeneImportanceExtractor
    """
    
    print("\n" + "="*70)
    print("GENE IMPORTANCE EXTRACTOR - EXAMPLE USAGE")
    print("="*70)
    
    # Initialize extractor
    try:
        extractor = GeneImportanceExtractor('pca_loadings.csv')
    except FileNotFoundError:
        print("\n⚠ ERROR: pca_loadings.csv not found!")
        print("\nPlease run the R script first:")
        print("  1. Open R/RStudio")
        print("  2. Run the export_pca_loadings.R script")
        print("  3. This will create pca_loadings.csv")
        print("\nThen run this Python script again.")
        exit(1)
    
    # Example 1: From coefficients (simulated)
    print("\n" + "="*70)
    print("EXAMPLE 1: Extract from Logistic Regression coefficients")
    print("="*70)
    
    # Simulate LR coefficients (in practice, use: model.coef_[0])
    fake_coef = np.random.randn(50)
    gene_df_lr = extractor.from_coefficients(fake_coef, "Logistic Regression")
    
    print("\nTop 10 genes:")
    print(gene_df_lr.head(10)[['rank', 'gene', 'importance']].to_string(index=False))
    
    # Example 2: From feature importances (simulated)
    print("\n" + "="*70)
    print("EXAMPLE 2: Extract from Random Forest importances")
    print("="*70)
    
    # Simulate RF importances (in practice, use: model.feature_importances_)
    fake_importance = np.random.rand(50)
    fake_importance = fake_importance / fake_importance.sum()  # Normalize
    gene_df_rf = extractor.from_feature_importances(fake_importance, "Random Forest")
    
    print("\nTop 10 genes:")
    print(gene_df_rf.head(10)[['rank', 'gene', 'importance']].to_string(index=False))
    
    # Example 3: From SHAP values (simulated)
    print("\n" + "="*70)
    print("EXAMPLE 3: Extract from SHAP values")
    print("="*70)
    
    # Simulate SHAP values (in practice, from shap explainer)
    fake_shap = np.random.randn(500, 50)  # 500 samples, 50 PCs
    gene_df_shap = extractor.from_shap_values(fake_shap, "XGBoost SHAP")
    
    print("\nTop 10 genes:")
    print(gene_df_shap.head(10)[['rank', 'gene', 'importance']].to_string(index=False))
    
    # Save examples
    os.makedirs('example_outputs', exist_ok=True)
    gene_df_lr.to_csv('example_outputs/example_lr_genes.csv', index=False)
    gene_df_rf.to_csv('example_outputs/example_rf_genes.csv', index=False)
    gene_df_shap.to_csv('example_outputs/example_shap_genes.csv', index=False)
    
    print("\n" + "="*70)
    print("✓ EXAMPLES COMPLETE!")
    print("="*70)
    print("\nExample outputs saved in: example_outputs/")
    print("\nNow you can use this class with your REAL models:")
    print("  1. Load your trained models")
    print("  2. Extract PC importance (coef, feature_importances, SHAP, attention)")
    print("  3. Use extractor methods to map to genes")
    print("  4. Compare gene signatures across methods!")