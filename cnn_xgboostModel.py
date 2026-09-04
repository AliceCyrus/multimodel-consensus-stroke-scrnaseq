"""
Model 1: CNN+XGBoost Ensemble for Single-Cell Stroke Classification
=====================================================================
SKLearn-compatible version for Phase 1 framework integration

Architecture:
    1. 1D CNN extracts hierarchical features from PCA embeddings
    2. XGBoost performs final classification on CNN features
    3. SHAP explains XGBoost decisions (Phase 2)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin  # ✅ CRITICAL: Added this import
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: Load and Prepare Data
# ============================================================================

def load_stroke_data():
    """Load preprocessed stroke data"""
    print("Loading stroke data...")
    
    # Load PCA embeddings
    # NOTE: Using relative paths for local execution
    # For Colab, mount drive and update paths accordingly:
    # '/content/drive/MyDrive/stroke_project/Phase4_RevisedModels/stroke_pca_train.csv'
    train_pca = pd.read_csv('stroke_pca_train.csv')
    test_pca = pd.read_csv('stroke_pca_test.csv')
    
    # Load labels
    train_labels = pd.read_csv('stroke_labels_train.csv')
    test_labels = pd.read_csv('stroke_labels_test.csv')
    
    # Prepare features (drop cell_id)
    X_train = train_pca.drop(columns=['cell_id']).values
    X_test = test_pca.drop(columns=['cell_id']).values
    
    # Prepare labels (1 = Stroke, 0 = Control)
    y_train = (train_labels['condition'] == 'Stroke').astype(int).values
    y_test = (test_labels['condition'] == 'Stroke').astype(int).values
    
    print(f"✓ Train: {X_train.shape[0]} cells, {X_train.shape[1]} PCs")
    print(f"✓ Test: {X_test.shape[0]} cells, {X_test.shape[1]} PCs")
    print(f"✓ Train distribution: {np.sum(y_train)} Stroke, {len(y_train)-np.sum(y_train)} Control")
    print(f"✓ Test distribution: {np.sum(y_test)} Stroke, {len(y_test)-np.sum(y_test)} Control")
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# STEP 2: Build CNN Feature Extractor
# ============================================================================

def build_cnn_feature_extractor(input_dim, num_filters=[64, 128, 256]):
    """
    Build 1D CNN for feature extraction from PCA embeddings.
    
    Architecture:
        Input (PCA) → Conv1D → BN → Pool → Conv1D → BN → Pool → 
        Conv1D → GlobalAvgPool → Dense → Features
    
    Parameters:
    -----------
    input_dim : int, number of PCs (e.g., 50)
    num_filters : list, filters for each conv layer
    
    Returns:
    --------
    model : Keras model
    """
    from tensorflow import keras
    from tensorflow.keras import layers
    
    inputs = layers.Input(shape=(input_dim,), name='pca_input')
    
    # Reshape for 1D convolution: (batch, features, channels)
    x = layers.Reshape((input_dim, 1))(inputs)
    
    # Convolutional blocks
    for i, filters in enumerate(num_filters):
        x = layers.Conv1D(filters, kernel_size=3, padding='same', 
                         activation='relu', name=f'conv{i+1}')(x)
        x = layers.BatchNormalization(name=f'bn{i+1}')(x)
        x = layers.MaxPooling1D(pool_size=2, name=f'pool{i+1}')(x)
        x = layers.Dropout(0.3, name=f'dropout{i+1}')(x)
    
    # Global pooling to get fixed-size features
    x = layers.GlobalAveragePooling1D(name='global_pool')(x)
    
    # Dense feature layer
    features = layers.Dense(128, activation='relu', name='feature_dense')(x)
    features = layers.Dropout(0.4, name='feature_dropout')(features)
    
    # Output layer for CNN training
    outputs = layers.Dense(2, activation='softmax', name='output')(features)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='CNN_FeatureExtractor')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ============================================================================
# STEP 3: SKLearn-Compatible CNN+XGBoost Ensemble Class
# ============================================================================

class CNNXGBoostClassifier(BaseEstimator, ClassifierMixin):  # ✅ CRITICAL: Inherit from sklearn base classes
    """
    SKLearn-compatible two-stage ensemble:
        Stage 1: CNN learns hierarchical features
        Stage 2: XGBoost classifies using CNN features
    """
    
    def __init__(self, input_dim=50, cnn_filters=None, xgb_params=None, 
                 cnn_epochs=30, cnn_batch_size=64, verbose=1):  # ✅ CRITICAL: All params with defaults
        """
        Parameters:
        -----------
        input_dim : int, number of input features (PCs)
        cnn_filters : list, filters for CNN layers
        xgb_params : dict, XGBoost hyperparameters
        cnn_epochs : int, epochs for CNN training
        cnn_batch_size : int, batch size for CNN
        verbose : int, verbosity level
        """
        # ✅ CRITICAL: Assign all parameters to self with same name
        self.input_dim = input_dim
        self.cnn_filters = cnn_filters if cnn_filters is not None else [64, 128, 256]
        self.xgb_params = xgb_params if xgb_params is not None else {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        self.cnn_epochs = cnn_epochs
        self.cnn_batch_size = cnn_batch_size
        self.verbose = verbose
        
        # These will be initialized during fit (use trailing underscore for fitted attributes)
        self.cnn_model_ = None
        self.xgb_model_ = None
        self.feature_extractor_ = None
        self.is_fitted_ = False
    
    def get_params(self, deep=True):  # ✅ CRITICAL: Required for sklearn compatibility
        """
        Get parameters for this estimator.
        Required for sklearn compatibility and cross-validation.
        """
        return {
            'input_dim': self.input_dim,
            'cnn_filters': self.cnn_filters,
            'xgb_params': self.xgb_params,
            'cnn_epochs': self.cnn_epochs,
            'cnn_batch_size': self.cnn_batch_size,
            'verbose': self.verbose
        }
    
    def set_params(self, **params):  # ✅ CRITICAL: Required for sklearn compatibility
        """
        Set parameters for this estimator.
        Required for sklearn compatibility.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def _build_models(self):
        """Build CNN and XGBoost models"""
        print("Building CNN feature extractor...")
        self.cnn_model_ = build_cnn_feature_extractor(self.input_dim, self.cnn_filters)
        
        from xgboost import XGBClassifier
        self.xgb_model_ = XGBClassifier(**self.xgb_params)
    
    def fit(self, X, y, X_val=None, y_val=None):  # ✅ CRITICAL: Must return self
        """
        Train the ensemble in two stages.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target labels
        X_val : array, validation features (optional, for display only)
        y_val : array, validation labels (optional, for display only)
        
        Returns:
        --------
        self : object
            Returns self for sklearn compatibility
        """
        from tensorflow import keras
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Update input_dim if different
        if X.shape[1] != self.input_dim:
            self.input_dim = X.shape[1]
        
        # Build models if not already built
        if self.cnn_model_ is None or self.xgb_model_ is None:
            self._build_models()
        
        if self.verbose > 0:
            print("\n" + "="*60)
            print("STAGE 1: Training CNN Feature Extractor")
            print("="*60)
        
        # Prepare validation data for display purposes only
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Train CNN with early stopping
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=5,
                restore_best_weights=True,
                verbose=0
            )
        ]
        
        verbose_level = 1 if self.verbose > 0 else 0
        
        self.cnn_model_.fit(
            X, y,
            validation_data=validation_data,
            epochs=self.cnn_epochs,
            batch_size=self.cnn_batch_size,
            verbose=verbose_level,
            callbacks=callbacks
        )
        
        # Create feature extractor (remove final classification layer)
        self.feature_extractor_ = keras.Model(
            inputs=self.cnn_model_.input,
            outputs=self.cnn_model_.get_layer('feature_dense').output
        )
        
        if self.verbose > 0:
            print("\n" + "="*60)
            print("STAGE 2: Training XGBoost on CNN Features")
            print("="*60)
        
        # Extract CNN features
        train_features = self.feature_extractor_.predict(X, verbose=0)
        
        if self.verbose > 0:
            print(f"✓ Extracted CNN features: {train_features.shape}")
        
        # Train XGBoost
        if X_val is not None and y_val is not None:
            val_features = self.feature_extractor_.predict(X_val, verbose=0)
            self.xgb_model_.fit(
                train_features, y,
                eval_set=[(val_features, y_val)],
                verbose=self.verbose > 0
            )
        else:
            self.xgb_model_.fit(train_features, y, verbose=False)
        
        self.is_fitted_ = True
        
        if self.verbose > 0:
            print("✓ Ensemble training complete!")
        
        return self  # ✅ CRITICAL: Must return self for sklearn
    
    def predict(self, X):
        """
        Predict class labels.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
        
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted class labels
        """
        if not self.is_fitted_:
            raise ValueError("Model must be trained before prediction. Call fit() first.")
        
        X = np.array(X)
        
        # Extract features with CNN
        features = self.feature_extractor_.predict(X, verbose=0)
        
        # Classify with XGBoost
        predictions = self.xgb_model_.predict(features)
        
        return predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
        
        Returns:
        --------
        y_proba : array, shape (n_samples, n_classes)
            Predicted class probabilities
        """
        if not self.is_fitted_:
            raise ValueError("Model must be trained before prediction. Call fit() first.")
        
        X = np.array(X)
        
        # Extract features with CNN
        features = self.feature_extractor_.predict(X, verbose=0)
        
        # Get probabilities from XGBoost
        probabilities = self.xgb_model_.predict_proba(features)
        
        return probabilities
    
    def get_cnn_features(self, X):
        """
        Extract CNN features for downstream analysis.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        
        Returns:
        --------
        features : array, shape (n_samples, n_cnn_features)
        """
        if not self.is_fitted_:
            raise ValueError("Model must be trained first.")
        
        X = np.array(X)
        return self.feature_extractor_.predict(X, verbose=0)


# ============================================================================
# STEP 4: Training Pipeline (Simple Version)
# ============================================================================

def train_cnn_xgboost_simple():
    """Simple training and testing (no cross-validation)"""
    
    # Load data
    X_train, X_test, y_train, y_test = load_stroke_data()
    
    # Initialize model
    model = CNNXGBoostClassifier(
        input_dim=X_train.shape[1],
        cnn_filters=[64, 128, 256],
        cnn_epochs=30,
        cnn_batch_size=64,
        verbose=1
    )
    
    # Train ensemble
    model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                               target_names=['Control', 'Stroke']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Control', 'Stroke'],
               yticklabels=['Control', 'Stroke'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('CNN+XGBoost Confusion Matrix')
    plt.tight_layout()
    plt.savefig('cnn_xgboost_confusion_matrix.png', dpi=300)
    print("✓ Confusion matrix saved to 'cnn_xgboost_confusion_matrix.png'")
    
    return model


# ============================================================================
# STEP 5: Integration with Phase 1 Framework
# ============================================================================

def integrate_with_phase1_framework():
    """
    Use the CNN+XGBoost model with Phase 1 validation framework.
    This includes 10-fold cross-validation.
    """
    from phase1_framework import ModelValidator, PublicationFigures
    
    # Load data
    X_train, X_test, y_train, y_test = load_stroke_data()
    
    # Initialize Phase 1 validator
    print("\n" + "="*60)
    print("PHASE 1: STATISTICAL VALIDATION")
    print("="*60)
    
    validator = ModelValidator(X_train, y_train, X_test, y_test)
    
    # Initialize sklearn-compatible CNN+XGBoost model
    model = CNNXGBoostClassifier(
        input_dim=X_train.shape[1],
        cnn_filters=[64, 128, 256],
        xgb_params={
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        },
        cnn_epochs=30,
        cnn_batch_size=64,
        verbose=1
    )
    
    # Evaluate with full statistical framework
    # This will now work because model implements get_params() and set_params()
    results = validator.evaluate_model(
        model, 
        model_name="CNN+XGBoost",
        n_folds=10,
        n_bootstrap=1000
    )
    
    # Save results
    validator.save_results(results, "results_cnn_xgboost.json")
    
    # Generate publication figures
    figures = PublicationFigures(validator)
    figures.create_all_figures(output_dir='figures_cnn_xgboost')
    
    # Create performance table
    perf_table = validator.create_performance_table()
    print("\n" + "="*60)
    print("PERFORMANCE TABLE")
    print("="*60)
    print("\n", perf_table.to_string(index=False))
    
    print("\n✓ CNN+XGBoost complete evaluation done!")
    print("✓ Ready to add more models (scGPT, scBERT, etc.)")
    
    return validator, model, results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Run complete CNN+XGBoost pipeline with Phase 1 validation.
    """
    
    print("="*60)
    print("CNN+XGBoost Ensemble for Stroke Classification")
    print("="*60)
    
    # Choose your execution mode:
    
    # Option 1: Simple training and testing (FAST - for initial testing)
    # Uncomment this line to just train and test without cross-validation
    # model = train_cnn_xgboost_simple()
    
    # Option 2: Full Phase 1 validation (RECOMMENDED for publication)
    # This includes 10-fold cross-validation and bootstrap confidence intervals
    validator, model, results = integrate_with_phase1_framework()
    
    print("\n" + "="*60)
    print("✓ ALL DONE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Train remaining models (scGPT, scBERT, scFormer, Geneformer)")
    print("  2. Run statistical comparison: validator.compare_models()")
    print("  3. Generate combined publication figures")
    print("  4. Move to Phase 2: Interpretability analysis")
    print("\nGenerated files:")
    print("  • results_cnn_xgboost.json - Complete evaluation metrics")
    print("  • figures_cnn_xgboost/ - Publication-quality figures")
    print("  • All figures at 300 DPI, publication-ready")