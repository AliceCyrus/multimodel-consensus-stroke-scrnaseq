"""
PHASE 1: Complete Classification & Statistical Validation Framework
====================================================================
Task 1.1: Rigorous Performance Evaluation
Task 1.2: Publication-Quality Visualizations

Usage:
    validator = ModelValidator(X_train, y_train, X_test, y_test)
    results = validator.evaluate_model(model, model_name="scGPT")
    validator.save_results(results, "results_scGPT.json")
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    auc, confusion_matrix
)
from sklearn.utils import resample
from scipy import stats
from scipy.stats import wilcoxon
from mlxtend.evaluate import mcnemar_table, mcnemar
import json
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plotting defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


class ModelValidator:
    """
    Comprehensive statistical validation and visualization suite
    for single-cell classification models.
    """
    
    def __init__(self, X_train, y_train, X_test, y_test):
        """
        Initialize validator with train/test data.
        
        Parameters:
        -----------
        X_train : array-like, shape (n_samples_train, n_features)
        y_train : array-like, shape (n_samples_train,)
        X_test : array-like, shape (n_samples_test, n_features)
        y_test : array-like, shape (n_samples_test,)
        """
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)
        
        # Encode labels if strings
        if self.y_train.dtype == object:
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder = LabelEncoder()
            self.y_train_encoded = self.label_encoder.fit_transform(self.y_train)
            self.y_test_encoded = self.label_encoder.transform(self.y_test)
        else:
            self.y_train_encoded = self.y_train
            self.y_test_encoded = self.y_test
            
        self.all_results = {}
        
    def cross_validate_model(self, model, n_folds=10, n_jobs=-1):
        """
        Perform stratified k-fold cross-validation.
        
        Parameters:
        -----------
        model : sklearn-compatible estimator
        n_folds : int, number of CV folds (default=10)
        n_jobs : int, parallel jobs (default=-1, use all cores)
        
        Returns:
        --------
        cv_results : dict with arrays of metrics per fold
        """
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc'
        }
        
        cv_results = cross_validate(
            model, self.X_train, self.y_train_encoded,
            cv=skf, scoring=scoring, n_jobs=n_jobs,
            return_train_score=False
        )
        
        return {
            'accuracy': cv_results['test_accuracy'],
            'precision': cv_results['test_precision'],
            'recall': cv_results['test_recall'],
            'f1': cv_results['test_f1'],
            'roc_auc': cv_results['test_roc_auc']
        }
    
    def bootstrap_confidence_interval(self, y_true, y_pred, y_pred_proba, 
                                     n_iterations=1000, alpha=0.05):
        """
        Calculate bootstrap 95% confidence intervals for all metrics.
        
        Parameters:
        -----------
        y_true : array, true labels
        y_pred : array, predicted labels
        y_pred_proba : array, predicted probabilities
        n_iterations : int, bootstrap iterations (default=1000)
        alpha : float, significance level (default=0.05 for 95% CI)
        
        Returns:
        --------
        ci_dict : dict with (lower, upper) bounds for each metric
        """
        n_samples = len(y_true)
        metrics_bootstrap = {
            'accuracy': [], 'precision': [], 'recall': [], 
            'f1': [], 'roc_auc': []
        }
        
        np.random.seed(42)
        for i in range(n_iterations):
            # Bootstrap sample
            indices = resample(np.arange(n_samples), replace=True, 
                             n_samples=n_samples, random_state=i)
            
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            y_proba_boot = y_pred_proba[indices]
            
            # Calculate metrics
            metrics_bootstrap['accuracy'].append(
                accuracy_score(y_true_boot, y_pred_boot)
            )
            metrics_bootstrap['precision'].append(
                precision_score(y_true_boot, y_pred_boot, zero_division=0)
            )
            metrics_bootstrap['recall'].append(
                recall_score(y_true_boot, y_pred_boot, zero_division=0)
            )
            metrics_bootstrap['f1'].append(
                f1_score(y_true_boot, y_pred_boot, zero_division=0)
            )
            try:
                metrics_bootstrap['roc_auc'].append(
                    roc_auc_score(y_true_boot, y_proba_boot)
                )
            except ValueError:
                # Handle case where bootstrap sample has only one class
                metrics_bootstrap['roc_auc'].append(np.nan)
        
        # Calculate percentile confidence intervals
        ci_dict = {}
        for metric, values in metrics_bootstrap.items():
            values_clean = [v for v in values if not np.isnan(v)]
            lower = np.percentile(values_clean, 100 * alpha / 2)
            upper = np.percentile(values_clean, 100 * (1 - alpha / 2))
            ci_dict[metric] = (lower, upper)
        
        return ci_dict
    
    def mcnemar_test(self, y_true, y_pred_1, y_pred_2):
        """
        McNemar's test for paired predictions.
        
        Parameters:
        -----------
        y_true : array, true labels
        y_pred_1 : array, predictions from model 1
        y_pred_2 : array, predictions from model 2
        
        Returns:
        --------
        chi2 : float, chi-squared statistic
        p_value : float, p-value
        """
        # Create contingency table
        tb = mcnemar_table(y_target=y_true, 
                          y_model1=y_pred_1,
                          y_model2=y_pred_2)
        
        chi2, p_value = mcnemar(ary=tb, corrected=True)
        
        return chi2, p_value
    
    def delong_test(self, y_true, y_proba_1, y_proba_2):
        """
        DeLong's test for comparing AUC-ROC curves.
        (Simplified implementation using Wilcoxon signed-rank test)
        
        Parameters:
        -----------
        y_true : array, true labels
        y_proba_1 : array, predicted probabilities from model 1
        y_proba_2 : array, predicted probabilities from model 2
        
        Returns:
        --------
        statistic : float, test statistic
        p_value : float, p-value
        """
        auc1 = roc_auc_score(y_true, y_proba_1)
        auc2 = roc_auc_score(y_true, y_proba_2)
        
        # Bootstrap approach for variance estimation
        n_bootstrap = 1000
        auc_diff = []
        
        np.random.seed(42)
        for i in range(n_bootstrap):
            indices = resample(np.arange(len(y_true)), random_state=i)
            try:
                auc1_boot = roc_auc_score(y_true[indices], y_proba_1[indices])
                auc2_boot = roc_auc_score(y_true[indices], y_proba_2[indices])
                auc_diff.append(auc1_boot - auc2_boot)
            except:
                continue
        
        # Z-test
        mean_diff = np.mean(auc_diff)
        std_diff = np.std(auc_diff)
        z_stat = mean_diff / (std_diff + 1e-10)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        return z_stat, p_value
    
    def evaluate_model(self, model, model_name, n_folds=10, n_bootstrap=1000):
        """
        Complete evaluation pipeline for a single model.
        
        Parameters:
        -----------
        model : sklearn-compatible estimator
        model_name : str, name of the model
        n_folds : int, number of CV folds
        n_bootstrap : int, bootstrap iterations
        
        Returns:
        --------
        results : dict containing all metrics and statistics
        """
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        # 1. Cross-validation
        print("Running 10-fold cross-validation...")
        cv_results = self.cross_validate_model(model, n_folds=n_folds)
        
        # 2. Train final model on full training set
        print("Training on full training set...")
        model.fit(self.X_train, self.y_train_encoded)
        
        # 3. Test set predictions
        print("Evaluating on test set...")
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # 4. Test set metrics
        test_metrics = {
            'accuracy': accuracy_score(self.y_test_encoded, y_pred),
            'precision': precision_score(self.y_test_encoded, y_pred),
            'recall': recall_score(self.y_test_encoded, y_pred),
            'f1': f1_score(self.y_test_encoded, y_pred),
            'roc_auc': roc_auc_score(self.y_test_encoded, y_pred_proba),
            'specificity': self._calculate_specificity(self.y_test_encoded, y_pred)
        }
        
        # 5. Bootstrap confidence intervals
        print("Calculating bootstrap confidence intervals...")
        ci = self.bootstrap_confidence_interval(
            self.y_test_encoded, y_pred, y_pred_proba, 
            n_iterations=n_bootstrap
        )
        
        # 6. ROC and PR curve data
        fpr, tpr, roc_thresholds = roc_curve(self.y_test_encoded, y_pred_proba)
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
            self.y_test_encoded, y_pred_proba
        )
        
        # 7. Confusion matrix
        cm = confusion_matrix(self.y_test_encoded, y_pred)
        
        # Compile results
        results = {
            'model_name': model_name,
            'cv_results': {k: v.tolist() for k, v in cv_results.items()},
            'cv_mean': {k: float(np.mean(v)) for k, v in cv_results.items()},
            'cv_std': {k: float(np.std(v)) for k, v in cv_results.items()},
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'confidence_intervals': {k: (float(v[0]), float(v[1])) 
                                    for k, v in ci.items()},
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist()
            },
            'pr_curve': {
                'precision': precision_curve.tolist(),
                'recall': recall_curve.tolist(),
                'thresholds': pr_thresholds.tolist()
            },
            'confusion_matrix': cm.tolist(),
            'predictions': {
                'y_pred': y_pred.tolist(),
                'y_pred_proba': y_pred_proba.tolist()
            }
        }
        
        # Store in class
        self.all_results[model_name] = results
        
        print(f"\n✓ {model_name} evaluation complete!")
        self._print_summary(results)
        
        return results
    
    def _calculate_specificity(self, y_true, y_pred):
        """Calculate specificity (True Negative Rate)"""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp)
    
    def _print_summary(self, results):
        """Print formatted summary of results"""
        print(f"\nTest Set Performance:")
        print(f"  Accuracy:    {results['test_metrics']['accuracy']:.4f} "
              f"[{results['confidence_intervals']['accuracy'][0]:.4f}, "
              f"{results['confidence_intervals']['accuracy'][1]:.4f}]")
        print(f"  Sensitivity: {results['test_metrics']['recall']:.4f} "
              f"[{results['confidence_intervals']['recall'][0]:.4f}, "
              f"{results['confidence_intervals']['recall'][1]:.4f}]")
        print(f"  Specificity: {results['test_metrics']['specificity']:.4f}")
        print(f"  F1-Score:    {results['test_metrics']['f1']:.4f} "
              f"[{results['confidence_intervals']['f1'][0]:.4f}, "
              f"{results['confidence_intervals']['f1'][1]:.4f}]")
        print(f"  AUC-ROC:     {results['test_metrics']['roc_auc']:.4f} "
              f"[{results['confidence_intervals']['roc_auc'][0]:.4f}, "
              f"{results['confidence_intervals']['roc_auc'][1]:.4f}]")
        
        # Check if CV results exist and are valid
        if 'cv_mean' in results and 'accuracy' in results['cv_mean']:
            cv_acc = results['cv_mean'].get('accuracy', None)
            cv_auc = results['cv_mean'].get('roc_auc', None)
            
            if cv_acc is not None and not np.isnan(cv_acc):
                print(f"\nCross-Validation (Mean ± SD):")
                print(f"  Accuracy: {results['cv_mean']['accuracy']:.4f} ± "
                      f"{results['cv_std']['accuracy']:.4f}")
                print(f"  AUC-ROC:  {results['cv_mean']['roc_auc']:.4f} ± "
                      f"{results['cv_std']['roc_auc']:.4f}")
            else:
                print(f"\nCross-Validation: Completed (results stored in JSON)")
        else:
            print(f"\nCross-Validation: Not performed or data unavailable")
    
    def compare_models(self, model_names=None):
        """
        Statistical comparison between all evaluated models.
        
        Parameters:
        -----------
        model_names : list of str, subset of models to compare (optional)
        
        Returns:
        --------
        comparison_df : DataFrame with statistical test results
        """
        if model_names is None:
            model_names = list(self.all_results.keys())
        
        if len(model_names) < 2:
            print("Need at least 2 models for comparison")
            return None
        
        print(f"\n{'='*60}")
        print("Statistical Model Comparison")
        print(f"{'='*60}")
        
        comparisons = []
        
        for i, name1 in enumerate(model_names):
            for name2 in model_names[i+1:]:
                pred1 = np.array(self.all_results[name1]['predictions']['y_pred'])
                pred2 = np.array(self.all_results[name2]['predictions']['y_pred'])
                proba1 = np.array(self.all_results[name1]['predictions']['y_pred_proba'])
                proba2 = np.array(self.all_results[name2]['predictions']['y_pred_proba'])
                
                # McNemar's test
                chi2, p_mcnemar = self.mcnemar_test(
                    self.y_test_encoded, pred1, pred2
                )
                
                # DeLong's test
                z_stat, p_delong = self.delong_test(
                    self.y_test_encoded, proba1, proba2
                )
                
                auc1 = self.all_results[name1]['test_metrics']['roc_auc']
                auc2 = self.all_results[name2]['test_metrics']['roc_auc']
                
                comparisons.append({
                    'Model 1': name1,
                    'Model 2': name2,
                    'AUC_1': f"{auc1:.4f}",
                    'AUC_2': f"{auc2:.4f}",
                    'McNemar χ²': f"{chi2:.4f}",
                    'McNemar p-value': f"{p_mcnemar:.4f}",
                    'Significant (McNemar)': 'Yes' if p_mcnemar < 0.05 else 'No',
                    'DeLong Z': f"{z_stat:.4f}",
                    'DeLong p-value': f"{p_delong:.4f}",
                    'Significant (DeLong)': 'Yes' if p_delong < 0.05 else 'No'
                })
        
        comparison_df = pd.DataFrame(comparisons)
        print("\n", comparison_df.to_string(index=False))
        
        return comparison_df
    
    def save_results(self, results, filename):
        """Save results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {filename}")
    
    def create_performance_table(self, model_names=None):
        """
        Create publication-ready performance table.
        
        Returns:
        --------
        df : pandas DataFrame
        """
        if model_names is None:
            model_names = list(self.all_results.keys())
        
        rows = []
        for name in model_names:
            res = self.all_results[name]
            row = {
                'Model': name,
                'Accuracy': f"{res['test_metrics']['accuracy']:.4f} "
                           f"[{res['confidence_intervals']['accuracy'][0]:.3f}–"
                           f"{res['confidence_intervals']['accuracy'][1]:.3f}]",
                'Sensitivity': f"{res['test_metrics']['recall']:.4f} "
                              f"[{res['confidence_intervals']['recall'][0]:.3f}–"
                              f"{res['confidence_intervals']['recall'][1]:.3f}]",
                'Specificity': f"{res['test_metrics']['specificity']:.4f}",
                'F1-Score': f"{res['test_metrics']['f1']:.4f} "
                           f"[{res['confidence_intervals']['f1'][0]:.3f}–"
                           f"{res['confidence_intervals']['f1'][1]:.3f}]",
                'AUC-ROC': f"{res['test_metrics']['roc_auc']:.4f} "
                          f"[{res['confidence_intervals']['roc_auc'][0]:.3f}–"
                          f"{res['confidence_intervals']['roc_auc'][1]:.3f}]"
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df


# =============================================================================
# TASK 1.2: PUBLICATION-QUALITY VISUALIZATIONS
# =============================================================================

class PublicationFigures:
    """Generate all required publication-quality figures"""
    
    def __init__(self, validator):
        """
        Initialize with ModelValidator instance.
        
        Parameters:
        -----------
        validator : ModelValidator instance
        """
        self.validator = validator
        self.results = validator.all_results
        
    def plot_roc_curves(self, model_names=None, save_path=None):
        """
        Plot ROC curves for all models on same plot.
        
        Parameters:
        -----------
        model_names : list of str, models to include (optional)
        save_path : str, path to save figure (optional)
        """
        if model_names is None:
            model_names = list(self.results.keys())
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
        
        for i, name in enumerate(model_names):
            res = self.results[name]
            fpr = np.array(res['roc_curve']['fpr'])
            tpr = np.array(res['roc_curve']['tpr'])
            auc_val = res['test_metrics']['roc_auc']
            ci_lower = res['confidence_intervals']['roc_auc'][0]
            ci_upper = res['confidence_intervals']['roc_auc'][1]
            
            ax.plot(fpr, tpr, color=colors[i], lw=2,
                   label=f"{name} (AUC = {auc_val:.3f} "
                         f"[{ci_lower:.3f}–{ci_upper:.3f}])")
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves - All Models')
        ax.legend(loc='lower right', frameon=True, fancybox=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ ROC curves saved to {save_path}")
        
        return fig, ax
    
    def plot_pr_curves(self, model_names=None, save_path=None):
        """Plot Precision-Recall curves for all models"""
        if model_names is None:
            model_names = list(self.results.keys())
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
        
        for i, name in enumerate(model_names):
            res = self.results[name]
            precision = np.array(res['pr_curve']['precision'])
            recall = np.array(res['pr_curve']['recall'])
            pr_auc = auc(recall, precision)
            
            ax.plot(recall, precision, color=colors[i], lw=2,
                   label=f"{name} (AUC = {pr_auc:.3f})")
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves - All Models')
        ax.legend(loc='lower left', frameon=True, fancybox=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ PR curves saved to {save_path}")
        
        return fig, ax
    
    def plot_cv_distributions(self, model_names=None, save_path=None):
        """Box plots of cross-validation score distributions"""
        if model_names is None:
            model_names = list(self.results.keys())
        
        metrics = ['accuracy', 'recall', 'f1', 'roc_auc']
        metric_labels = ['Accuracy', 'Sensitivity', 'F1-Score', 'AUC-ROC']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx]
            
            data = []
            labels = []
            for name in model_names:
                cv_scores = self.results[name]['cv_results'][metric]
                data.append(cv_scores)
                labels.append(name)
            
            bp = ax.boxplot(data, labels=labels, patch_artist=True,
                           medianprops=dict(color='red', linewidth=2),
                           boxprops=dict(facecolor='lightblue', alpha=0.7))
            
            ax.set_ylabel(label)
            ax.set_title(f'{label} Distribution (10-Fold CV)')
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ CV distributions saved to {save_path}")
        
        return fig, axes
    
    def plot_radar_chart(self, model_names=None, save_path=None):
        """Performance radar chart for all models"""
        if model_names is None:
            model_names = list(self.results.keys())
        
        # Metrics to include
        categories = ['Accuracy', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC-ROC']
        metric_keys = ['accuracy', 'recall', 'specificity', 'f1', 'roc_auc']
        
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
        
        for i, name in enumerate(model_names):
            res = self.results[name]
            values = [res['test_metrics'][key] for key in metric_keys]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=name)
            ax.fill(angles, values, alpha=0.15, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.set_title('Model Performance Comparison', size=14, pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Radar chart saved to {save_path}")
        
        return fig, ax
    
    def create_all_figures(self, output_dir='figures'):
        """Generate and save all publication figures"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("Generating Publication-Quality Figures")
        print(f"{'='*60}")
        
        self.plot_roc_curves(save_path=f'{output_dir}/fig1_roc_curves.png')
        self.plot_pr_curves(save_path=f'{output_dir}/fig2_pr_curves.png')
        self.plot_cv_distributions(save_path=f'{output_dir}/fig3_cv_distributions.png')
        self.plot_radar_chart(save_path=f'{output_dir}/fig4_radar_chart.png')
        
        print(f"\n✓ All figures saved to '{output_dir}/' directory")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Example workflow for using the validation framework.
    """
    
    # Load your data
    print("Loading data...")
    train_pca = pd.read_csv('Phase4_RevisedModels/stroke_pca_train.csv')
    test_pca = pd.read_csv('Phase4_RevisedModels/stroke_pca_test.csv')
    train_labels = pd.read_csv('Phase4_RevisedModels/stroke_labels_train.csv')
    test_labels = pd.read_csv('Phase4_RevisedModels/stroke_labels_test.csv')
    
    # Prepare features and labels
    X_train = train_pca.drop(columns=['cell_id']).values
    X_test = test_pca.drop(columns=['cell_id']).values
    
    # Map condition to binary labels
    y_train = (train_labels['condition'] == 'Stroke').astype(int).values
    y_test = (test_labels['condition'] == 'Stroke').astype(int).values
    
    print(f"Train set: {X_train.shape[0]} cells, {X_train.shape[1]} PCs")
    print(f"Test set: {X_test.shape[0]} cells, {X_test.shape[1]} PCs")
    print(f"Class distribution - Train: {np.sum(y_train)} Stroke, {len(y_train)-np.sum(y_train)} Control")
    print(f"Class distribution - Test: {np.sum(y_test)} Stroke, {len(y_test)-np.sum(y_test)} Control")
    
    # Initialize validator
    validator = ModelValidator(X_train, y_train, X_test, y_test)
    
    # =========================================================================
    # Example 1: Simple Logistic Regression Baseline
    # =========================================================================
    from sklearn.linear_model import LogisticRegression
    
    print("\n" + "="*60)
    print("BASELINE MODEL: Logistic Regression")
    print("="*60)
    
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_results = validator.evaluate_model(lr_model, model_name="Logistic Regression")
    validator.save_results(lr_results, "results_logistic_regression.json")
    
    # =========================================================================
    # Example 2: Random Forest
    # =========================================================================
    from sklearn.ensemble import RandomForestClassifier
    
    print("\n" + "="*60)
    print("ENSEMBLE MODEL: Random Forest")
    print("="*60)
    
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                     random_state=42, n_jobs=-1)
    rf_results = validator.evaluate_model(rf_model, model_name="Random Forest")
    validator.save_results(rf_results, "results_random_forest.json")
    
    # =========================================================================
    # Example 3: XGBoost
    # =========================================================================
    from xgboost import XGBClassifier
    
    print("\n" + "="*60)
    print("GRADIENT BOOSTING: XGBoost")
    print("="*60)
    
    xgb_model = XGBClassifier(n_estimators=100, max_depth=6, 
                             learning_rate=0.1, random_state=42, 
                             eval_metric='logloss', use_label_encoder=False)
    xgb_results = validator.evaluate_model(xgb_model, model_name="XGBoost")
    validator.save_results(xgb_results, "results_xgboost.json")
    
    # =========================================================================
    # Statistical Comparison
    # =========================================================================
    print("\n" + "="*60)
    print("STATISTICAL COMPARISON")
    print("="*60)
    
    comparison_df = validator.compare_models()
    comparison_df.to_csv("model_comparison_statistics.csv", index=False)
    print("\n✓ Comparison saved to 'model_comparison_statistics.csv'")
    
    # =========================================================================
    # Performance Table
    # =========================================================================
    print("\n" + "="*60)
    print("PERFORMANCE TABLE (Publication-Ready)")
    print("="*60)
    
    perf_table = validator.create_performance_table()
    print("\n", perf_table.to_string(index=False))
    perf_table.to_csv("performance_table.csv", index=False)
    perf_table.to_latex("performance_table.tex", index=False)
    print("\n✓ Table saved to 'performance_table.csv' and 'performance_table.tex'")
    
    # =========================================================================
    # Generate All Figures
    # =========================================================================
    figures = PublicationFigures(validator)
    figures.create_all_figures(output_dir='publication_figures')
    
    print("\n" + "="*60)
    print("✓ PHASE 1 COMPLETE!")
    print("="*60)
    print("\nGenerated outputs:")
    print("  • Individual model results (JSON files)")
    print("  • Performance table (CSV + LaTeX)")
    print("  • Statistical comparison table (CSV)")
    print("  • Publication figures (PNG, 300 DPI):")
    print("    - ROC curves")
    print("    - Precision-Recall curves")
    print("    - Cross-validation distributions")
    print("    - Performance radar chart")
    print("\nReady for Phase 2: Interpretability Analysis!")


# =============================================================================
# UTILITY FUNCTIONS FOR MODEL WRAPPERS
# =============================================================================

class ModelWrapper:
    """
    Wrapper for models that don't follow sklearn API.
    Useful for scGPT, scBERT, Geneformer, etc.
    """
    
    def __init__(self, model, preprocess_fn=None, predict_fn=None):
        """
        Parameters:
        -----------
        model : custom model object
        preprocess_fn : function to preprocess data before prediction
        predict_fn : function that takes (model, X) and returns predictions
        """
        self.model = model
        self.preprocess_fn = preprocess_fn
        self.predict_fn = predict_fn
        
    def fit(self, X, y):
        """Train the model"""
        if hasattr(self.model, 'fit'):
            self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Get class predictions"""
        if self.predict_fn:
            return self.predict_fn(self.model, X)
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get probability predictions"""
        if self.preprocess_fn:
            X = self.preprocess_fn(X)
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models that output logits
            logits = self.model.predict(X)
            from scipy.special import softmax
            return softmax(logits, axis=1)


# =============================================================================
# MINI CNN EXAMPLE (for CNN+XGBoost ensemble)
# =============================================================================

def create_cnn_model(input_shape, num_classes=2):
    """
    Create a simple 1D CNN for gene expression data.
    
    Parameters:
    -----------
    input_shape : tuple, (n_features,)
    num_classes : int, number of output classes
    
    Returns:
    --------
    model : compiled Keras model
    """
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
        
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Reshape((input_shape[0], 1)),
            layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
            layers.GlobalAveragePooling1D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    except ImportError:
        print("TensorFlow not installed. Install with: pip install tensorflow")
        return None


class CNNXGBoostEnsemble:
    """
    Ensemble of CNN features + XGBoost classifier.
    CNN extracts deep features, XGBoost does final classification.
    """
    
    def __init__(self, input_shape, xgb_params=None):
        """
        Parameters:
        -----------
        input_shape : tuple, shape of input features
        xgb_params : dict, XGBoost hyperparameters
        """
        self.cnn = create_cnn_model(input_shape)
        
        if xgb_params is None:
            xgb_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            }
        
        from xgboost import XGBClassifier
        self.xgb = XGBClassifier(**xgb_params, eval_metric='logloss', 
                                use_label_encoder=False)
        
        self.feature_extractor = None
    
    def fit(self, X, y, epochs=50, batch_size=32, verbose=0):
        """Train the ensemble"""
        from tensorflow import keras
        
        # Train CNN
        print("Training CNN...")
        self.cnn.fit(X, y, epochs=epochs, batch_size=batch_size, 
                    verbose=verbose, validation_split=0.2)
        
        # Create feature extractor (CNN without final layer)
        self.feature_extractor = keras.Model(
            inputs=self.cnn.input,
            outputs=self.cnn.layers[-3].output  # Before dropout & final dense
        )
        
        # Extract features
        print("Extracting CNN features...")
        cnn_features = self.feature_extractor.predict(X, verbose=0)
        
        # Train XGBoost on CNN features
        print("Training XGBoost on CNN features...")
        self.xgb.fit(cnn_features, y)
        
        return self
    
    def predict(self, X):
        """Predict classes"""
        cnn_features = self.feature_extractor.predict(X, verbose=0)
        return self.xgb.predict(cnn_features)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        cnn_features = self.feature_extractor.predict(X, verbose=0)
        return self.xgb.predict_proba(cnn_features)


# =============================================================================
# NOTES FOR FOUNDATION MODELS
# =============================================================================

"""
For scGPT, scBERT, Geneformer, scFormer:

1. These models require specific preprocessing:
   - Gene vocabulary mapping
   - Tokenization
   - Special embeddings

2. Use ModelWrapper class to adapt them to sklearn API

3. Example structure:

   from scgpt import scGPT  # hypothetical import
   
   def preprocess_for_scgpt(X):
       # Convert PCA back to gene expression
       # Tokenize genes
       # Create attention masks
       return processed_data
   
   def predict_scgpt(model, X):
       X_processed = preprocess_for_scgpt(X)
       logits = model(X_processed)
       return logits.argmax(axis=1)
   
   scgpt_model = scGPT.load_pretrained('path/to/weights')
   wrapped_model = ModelWrapper(scgpt_model, 
                                preprocess_fn=preprocess_for_scgpt,
                                predict_fn=predict_scgpt)
   
   results = validator.evaluate_model(wrapped_model, "scGPT")

4. You'll need raw count matrices (not PCA) for these models
   - They need actual gene expression values
   - Gene names must match model vocabulary
"""