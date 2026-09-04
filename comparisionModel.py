"""
Baseline Models for Comparison
================================
Add Logistic Regression, Random Forest, and XGBoost to compare with CNN+XGBoost.

This script:
1. Trains 3 baseline models
2. Performs 10-fold CV + bootstrap CI for each
3. Compares all 4 models statistically
4. Generates combined publication figures
5. Creates comparison tables
6. SAVES trained models for gene extraction  # ← NEW!

Run after: cnn_xgboost_model.py
"""
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

# Rest of your imports...
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pickle  # ← ADDED: For saving models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Import your framework
from phase1_framework import ModelValidator, PublicationFigures

print("="*70)
print("BASELINE MODELS COMPARISON")
print("="*70)
print("\nThis will train and evaluate:")
print("  1. Logistic Regression (Linear baseline)")
print("  2. Random Forest (Ensemble baseline)")
print("  3. XGBoost (Gradient boosting baseline)")
print("  4. CNN+XGBoost (Your deep learning model - already trained)")
print("\n" + "="*70)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\n[STEP 1/6] Loading Data...")

# Load the preprocessed data
from repo_paths import data_path

train_pca = pd.read_csv(data_path('stroke_pca_train.csv'))
test_pca = pd.read_csv(data_path('stroke_pca_test.csv'))
train_labels = pd.read_csv(data_path('stroke_labels_train.csv'))
test_labels = pd.read_csv(data_path('stroke_labels_test.csv'))

# Prepare features and labels
X_train = train_pca.drop(columns=['cell_id']).values
X_test = test_pca.drop(columns=['cell_id']).values
y_train = (train_labels['condition'] == 'Stroke').astype(int).values
y_test = (test_labels['condition'] == 'Stroke').astype(int).values

print(f"[OK] Train: {X_train.shape[0]} cells, {X_train.shape[1]} features")
print(f"[OK] Test: {X_test.shape[0]} cells, {X_test.shape[1]} features")
print(f"[OK] Class balance - Train: {np.sum(y_train)} Stroke / {len(y_train)-np.sum(y_train)} Control")
print(f"[OK] Class balance - Test: {np.sum(y_test)} Stroke / {len(y_test)-np.sum(y_test)} Control")

# Initialize validator
validator = ModelValidator(X_train, y_train, X_test, y_test)

# ============================================================================
# STEP 2: LOAD CNN+XGBOOST RESULTS (Already trained)
# ============================================================================

print("\n[STEP 2/6] Loading CNN+XGBoost Results...")

try:
    with open('results_cnn_xgboost.json', 'r') as f:
        cnn_xgb_results = json.load(f)
    
    # Add to validator's results
    validator.all_results['CNN+XGBoost'] = cnn_xgb_results
    print("[OK] CNN+XGBoost results loaded")
    print(f"     Test AUC-ROC: {cnn_xgb_results['test_metrics']['roc_auc']:.4f}")
except FileNotFoundError:
    print("[WARNING] CNN+XGBoost results not found. Please run cnn_xgboost_model.py first.")
    print("          Continuing with baseline models only...")

# ============================================================================
# STEP 3: TRAIN BASELINE MODEL 1 - LOGISTIC REGRESSION
# ============================================================================

print("\n[STEP 3/6] Training Logistic Regression...")
print("-" * 70)

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced',  # Handle class imbalance
    solver='lbfgs',
    n_jobs=-1
)

lr_results = validator.evaluate_model(
    lr_model,
    model_name="Logistic Regression",
    n_folds=10,
    n_bootstrap=1000
)

validator.save_results(lr_results, "results_logistic_regression.json")
print("[OK] Logistic Regression complete")

# ============ ADDED: Save trained model ============
with open('model_logistic_regression.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
print("[OK] Model saved: model_logistic_regression.pkl")
# ===================================================

# ============================================================================
# STEP 4: TRAIN BASELINE MODEL 2 - RANDOM FOREST
# ============================================================================

print("\n[STEP 4/6] Training Random Forest...")
print("-" * 70)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    class_weight='balanced',
    n_jobs=-1,
    verbose=0
)

rf_results = validator.evaluate_model(
    rf_model,
    model_name="Random Forest",
    n_folds=10,
    n_bootstrap=1000
)

validator.save_results(rf_results, "results_random_forest.json")
print("[OK] Random Forest complete")

# ============ ADDED: Save trained model ============
with open('model_random_forest.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("[OK] Model saved: model_random_forest.pkl")
# ===================================================

# ============================================================================
# STEP 5: TRAIN BASELINE MODEL 3 - XGBOOST (Standalone)
# ============================================================================

print("\n[STEP 5/6] Training XGBoost (Standalone)...")
print("-" * 70)

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False,
    verbosity=0
)

xgb_results = validator.evaluate_model(
    xgb_model,
    model_name="XGBoost",
    n_folds=10,
    n_bootstrap=1000
)

validator.save_results(xgb_results, "results_xgboost.json")
print("[OK] XGBoost complete")

# ============ ADDED: Save trained model ============
with open('model_xgboost.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
print("[OK] Model saved: model_xgboost.pkl")
# ===================================================

# ============================================================================
# STEP 6: COMPREHENSIVE COMPARISON
# ============================================================================

print("\n[STEP 6/6] Statistical Comparison & Visualization...")
print("=" * 70)

# 6.1: Performance Table
print("\n1. Creating Performance Table...")
perf_table = validator.create_performance_table()
print("\n" + perf_table.to_string(index=False))

# Save table
perf_table.to_csv("performance_table_all_models.csv", index=False)
perf_table.to_latex("performance_table_all_models.tex", index=False, escape=False)
print("\n[OK] Tables saved:")
print("     - performance_table_all_models.csv")
print("     - performance_table_all_models.tex")

# 6.2: Statistical Comparison
print("\n2. Statistical Model Comparison...")
comparison_df = validator.compare_models()

if comparison_df is not None:
    comparison_df.to_csv("statistical_comparison_all_models.csv", index=False)
    print("\n[OK] Statistical comparison saved: statistical_comparison_all_models.csv")

# 6.3: Generate All Publication Figures
print("\n3. Generating Publication Figures...")
figures = PublicationFigures(validator)
figures.create_all_figures(output_dir='figures_all_models')

# ============================================================================
# STEP 7: ADDITIONAL ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("ADDITIONAL ANALYSIS")
print("="*70)

# Extract key metrics for all models
models_summary = []
for model_name, results in validator.all_results.items():
    models_summary.append({
        'Model': model_name,
        'Accuracy': results['test_metrics']['accuracy'],
        'Sensitivity': results['test_metrics']['recall'],
        'Specificity': results['test_metrics']['specificity'],
        'F1-Score': results['test_metrics']['f1'],
        'AUC-ROC': results['test_metrics']['roc_auc']
    })

summary_df = pd.DataFrame(models_summary)
summary_df = summary_df.sort_values('AUC-ROC', ascending=False)

print("\nModel Ranking (by AUC-ROC):")
print(summary_df.to_string(index=False))

# Find best model
best_model = summary_df.iloc[0]['Model']
best_auc = summary_df.iloc[0]['AUC-ROC']

print(f"\n[WINNER] Best Model: {best_model} (AUC-ROC: {best_auc:.4f})")

# ============================================================================
# GENERATE SUMMARY REPORT
# ============================================================================

report = f"""
{'='*70}
BASELINE MODELS COMPARISON - SUMMARY REPORT
{'='*70}

DATASET INFORMATION:
-------------------
Training samples: {X_train.shape[0]} cells
Test samples: {X_test.shape[0]} cells
Features: {X_train.shape[1]} PCs
Class distribution (Train): {np.sum(y_train)} Stroke, {len(y_train)-np.sum(y_train)} Control
Class distribution (Test): {np.sum(y_test)} Stroke, {len(y_test)-np.sum(y_test)} Control

MODELS EVALUATED:
-----------------
1. Logistic Regression (Linear baseline)
2. Random Forest (Ensemble method)
3. XGBoost (Gradient boosting)
4. CNN+XGBoost (Deep learning + ensemble)

BEST PERFORMING MODEL:
---------------------
Model: {best_model}
AUC-ROC: {best_auc:.4f}

PERFORMANCE SUMMARY (Test Set):
-------------------------------
{summary_df.to_string(index=False)}

FILES GENERATED:
---------------
1. Results (JSON):
   - results_logistic_regression.json
   - results_random_forest.json
   - results_xgboost.json
   - results_cnn_xgboost.json (from previous run)

2. Trained Models (PKL):  # ← ADDED
   - model_logistic_regression.pkl
   - model_random_forest.pkl
   - model_xgboost.pkl

3. Tables:
   - performance_table_all_models.csv
   - performance_table_all_models.tex (LaTeX)
   - statistical_comparison_all_models.csv

4. Figures (300 DPI):
   - figures_all_models/fig1_roc_curves.png
   - figures_all_models/fig2_pr_curves.png
   - figures_all_models/fig3_cv_distributions.png
   - figures_all_models/fig4_radar_chart.png

NEXT STEPS:
-----------
1. Review figures in 'figures_all_models/' folder
2. Check statistical comparisons in CSV files
3. Models saved for Phase 2 gene extraction  # ← ADDED
4. Run: python integrated_interpretability_pipeline.py  # ← ADDED

{'='*70}
"""

print(report)

# Save report
with open('comparison_report.txt', 'w') as f:
    f.write(report)

print("\n[OK] Summary report saved: comparison_report.txt")

# ============================================================================
# FINAL VISUALIZATION: Model Comparison Bar Chart
# ============================================================================

print("\n4. Creating Model Comparison Bar Chart...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC-ROC']
colors = plt.cm.Set3(np.linspace(0, 1, len(validator.all_results)))

for idx, metric in enumerate(metrics):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    model_names = list(summary_df['Model'])
    values = list(summary_df[metric])
    
    bars = ax.bar(range(len(model_names)), values, color=colors[:len(model_names)])
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} Comparison')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylim([min(values) - 0.05, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Highlight best model
    best_idx = values.index(max(values))
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)

# Remove empty subplot
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('figures_all_models/model_comparison_bars.png', dpi=300, bbox_inches='tight')
print("[OK] Bar chart saved: figures_all_models/model_comparison_bars.png")

# ============================================================================
# DONE!
# ============================================================================

print("\n" + "="*70)
print("ALL BASELINE MODELS COMPLETE!")
print("="*70)
print(f"\nTotal models trained: {len(validator.all_results)}")
print(f"Best model: {best_model} (AUC-ROC: {best_auc:.4f})")
print("\nAll results saved in:")
print("  - figures_all_models/")
print("  - *.json files")
print("  - *.csv files")
print("  - *.pkl files (trained models)")  # ← ADDED
print("\n" + "="*70)
print("READY FOR PHASE 2: INTERPRETABILITY ANALYSIS")
print("="*70)

# ============================================================================
# QUICK CHECK: Are differences statistically significant?
# ============================================================================

if comparison_df is not None and len(comparison_df) > 0:
    print("\nQUICK ANALYSIS:")
    print("-" * 70)
    
    # Check if CNN+XGBoost significantly outperforms others
    if 'CNN+XGBoost' in validator.all_results:
        cnn_comparisons = comparison_df[
            (comparison_df['Model 1'] == 'CNN+XGBoost') | 
            (comparison_df['Model 2'] == 'CNN+XGBoost')
        ]
        
        if len(cnn_comparisons) > 0:
            print("\nCNN+XGBoost vs Others (Statistical Significance):")
            for _, row in cnn_comparisons.iterrows():
                model1 = row['Model 1']
                model2 = row['Model 2']
                other_model = model2 if model1 == 'CNN+XGBoost' else model1
                mcnemar_sig = row['Significant (McNemar)']
                delong_sig = row['Significant (DeLong)']
                
                if mcnemar_sig == 'Yes' or delong_sig == 'Yes':
                    print(f"  [SIGNIFICANT] vs {other_model}")
                else:
                    print(f"  [NOT SIGNIFICANT] vs {other_model} (statistically equivalent)")

print("\n" + "="*70)
print("Script completed successfully!")
print("="*70)

# ============ ADDED: Final verification ============
print("\n" + "="*70)
print("VERIFYING MODEL FILES")
print("="*70)
import os
model_files = ['model_logistic_regression.pkl', 'model_random_forest.pkl', 'model_xgboost.pkl']
for mf in model_files:
    if os.path.exists(mf):
        size = os.path.getsize(mf) / (1024 * 1024)  # Size in MB
        print(f"✓ {mf} ({size:.2f} MB)")
    else:
        print(f"✗ {mf} NOT FOUND")
# ===================================================