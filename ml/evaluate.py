"""
Evaluation Module for MRO Anomaly Detection

Comprehensive evaluation with metrics, visualizations, and performance analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import structlog
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Import local modules
from ml.features import MROFeatureEngineer, load_and_engineer_features
from ml.models import HybridAnomalyDetector

logger = structlog.get_logger()


class MROEvaluator:
    """Comprehensive evaluator for MRO anomaly detection models."""
    
    def __init__(self, config: Dict):
        """Initialize evaluator with configuration."""
        self.config = config
        self.feature_engineer = None
        self.model = None
        self.evaluation_results = {}
        
    def load_trained_model(self, artifacts_dir: str):
        """Load trained model and feature engineer."""
        logger.info("Loading trained model", artifacts_dir=artifacts_dir)
        
        # Initialize model
        self.model = HybridAnomalyDetector(self.config)
        
        # Load models
        self.model.load_models(artifacts_dir)
        
        # Load feature engineer (recreate from saved metadata)
        artifacts_path = Path(artifacts_dir)
        feature_metadata_path = artifacts_path / self.config['artifacts']['models']['feature_metadata']
        
        with open(feature_metadata_path, 'r') as f:
            feature_metadata = json.load(f)
        
        # Initialize feature engineer with config
        self.feature_engineer = MROFeatureEngineer(self.config)
        self.feature_engineer._feature_metadata = feature_metadata
        
        # Load the fitted scaler
        scaler_path = artifacts_path / self.config['artifacts']['models']['scaler']
        import pickle
        with open(scaler_path, 'rb') as f:
            self.feature_engineer.scaler = pickle.load(f)
        
        logger.info("Trained model loaded successfully")
    
    def load_test_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Load and transform test data."""
        logger.info("Loading test data", path=data_path)
        
        # Load raw data
        if data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)
        
        # Apply feature engineering (transform only, not fit)
        X, task_index = self.feature_engineer.transform(df)
        
        # Extract target if available
        y = None
        if 'is_anomaly' in df.columns:
            y = df['is_anomaly']
        
        logger.info("Test data loaded", 
                   X_shape=X.shape,
                   task_index_shape=task_index.shape,
                   has_target=y is not None)
        
        return X, task_index, y
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate model performance on test data."""
        logger.info("Evaluating model performance")
        
        # Get predictions
        results = self.model.predict(X_test)
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(y_test, results)
        
        # Store results
        self.evaluation_results = {
            'metrics': metrics,
            'predictions': results,
            'y_test': y_test,
            'test_data': {
                'X_shape': X_test.shape,
                'y_shape': y_test.shape if y_test is not None else None,
                'anomaly_rate': y_test.mean() if y_test is not None else 0.0
            }
        }
        
        logger.info("Model evaluation complete", metrics=metrics)
        
        return metrics
    
    def _calculate_comprehensive_metrics(self, y_true: pd.Series, results: Dict) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        from sklearn.metrics import (
            precision_score, recall_score, f1_score, roc_auc_score,
            confusion_matrix, classification_report, precision_recall_curve,
            roc_curve, average_precision_score
        )
        
        y_pred = results['anomaly_flag']
        scores = results['anomaly_score']
        
        # Basic classification metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC metrics
        try:
            auc_roc = roc_auc_score(y_true, scores)
        except ValueError:
            auc_roc = 0.5
        
        try:
            auc_pr = average_precision_score(y_true, scores)
        except ValueError:
            auc_pr = 0.0
        
        # Confusion matrix
        cm_matrix = confusion_matrix(y_true, y_pred)
        
        # Alert metrics
        alert_rate = y_pred.mean()
        true_anomaly_rate = y_true.mean()
        
        # Model-specific metrics
        if_scores = results['model_scores']['isolation_forest']
        ae_scores = results['model_scores']['autoencoder']
        
        if_auc = roc_auc_score(y_true, if_scores) if len(np.unique(y_true)) > 1 else 0.5
        ae_auc = roc_auc_score(y_true, ae_scores) if len(np.unique(y_true)) > 1 else 0.5
        
        metrics = {
            'classification': {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'alert_rate': float(alert_rate),
                'true_anomaly_rate': float(true_anomaly_rate)
            },
            'auc_metrics': {
                'auc_roc': float(auc_roc),
                'auc_pr': float(auc_pr)
            },
            'model_comparison': {
                'isolation_forest_auc': float(if_auc),
                'autoencoder_auc': float(ae_auc),
                'ensemble_auc': float(auc_roc)
            },
            'confusion_matrix': cm_matrix.tolist(),
            'total_samples': len(y_true),
            'anomalies_detected': int(y_pred.sum()),
            'true_anomalies': int(y_true.sum())
        }
        
        return metrics
    
    def create_visualizations(self, output_dir: str):
        """Create comprehensive evaluation visualizations."""
        logger.info("Creating evaluation visualizations", output_dir=output_dir)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Get data
        metrics = self.evaluation_results['metrics']
        predictions = self.evaluation_results['predictions']
        
        # Use actual test data if available, otherwise create placeholder
        if 'y_test' in self.evaluation_results:
            y_true = self.evaluation_results['y_test']
        else:
            # Create placeholder with same length as predictions
            y_true = pd.Series([0] * len(predictions['anomaly_flag']))
        
        # 1. ROC Curve
        self._plot_roc_curve(predictions['anomaly_score'], y_true, output_path / 'roc_curve.png')
        
        # 2. Precision-Recall Curve
        self._plot_pr_curve(predictions['anomaly_score'], y_true, output_path / 'pr_curve.png')
        
        # 3. Confusion Matrix
        self._plot_confusion_matrix(metrics['confusion_matrix'], output_path / 'confusion_matrix.png')
        
        # 4. Score Distributions
        self._plot_score_distributions(predictions, output_path / 'score_distributions.png')
        
        # 5. Model Comparison
        self._plot_model_comparison(predictions, y_true, output_path / 'model_comparison.png')
        
        # 6. Feature Importance (sample)
        self._plot_feature_importance(predictions, output_path / 'feature_importance.png')
        
        logger.info("Visualizations created successfully")
    
    def _plot_roc_curve(self, scores: np.ndarray, y_true: pd.Series, filepath: Path):
        """Plot ROC curve."""
        from sklearn.metrics import roc_curve, roc_auc_score
        
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'Ensemble (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - MRO Anomaly Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pr_curve(self, scores: np.ndarray, y_true: pd.Series, filepath: Path):
        """Plot Precision-Recall curve."""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label=f'Ensemble (AP = {ap:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - MRO Anomaly Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self, cm: List[List[int]], filepath: Path):
        """Plot confusion matrix."""
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        plt.title('Confusion Matrix - MRO Anomaly Detection')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_score_distributions(self, predictions: Dict, filepath: Path):
        """Plot score distributions."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Ensemble scores
        axes[0, 0].hist(predictions['anomaly_score'], bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title('Ensemble Anomaly Scores')
        axes[0, 0].set_xlabel('Score')
        axes[0, 0].set_ylabel('Frequency')
        
        # Isolation Forest scores
        axes[0, 1].hist(predictions['model_scores']['isolation_forest'], bins=50, alpha=0.7, color='green')
        axes[0, 1].set_title('Isolation Forest Scores')
        axes[0, 1].set_xlabel('Score')
        axes[0, 1].set_ylabel('Frequency')
        
        # Autoencoder scores
        axes[1, 0].hist(predictions['model_scores']['autoencoder'], bins=50, alpha=0.7, color='red')
        axes[1, 0].set_title('Autoencoder Scores')
        axes[1, 0].set_xlabel('Score')
        axes[1, 0].set_ylabel('Frequency')
        
        # Score comparison
        axes[1, 1].boxplot([
            predictions['anomaly_score'],
            predictions['model_scores']['isolation_forest'],
            predictions['model_scores']['autoencoder']
        ], labels=['Ensemble', 'IF', 'AE'])
        axes[1, 1].set_title('Score Distribution Comparison')
        axes[1, 1].set_ylabel('Score')
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_comparison(self, predictions: Dict, y_true: pd.Series, filepath: Path):
        """Plot model comparison."""
        from sklearn.metrics import roc_auc_score
        
        # Calculate AUC for each model
        ensemble_auc = roc_auc_score(y_true, predictions['anomaly_score'])
        if_auc = roc_auc_score(y_true, predictions['model_scores']['isolation_forest'])
        ae_auc = roc_auc_score(y_true, predictions['model_scores']['autoencoder'])
        
        models = ['Ensemble', 'Isolation Forest', 'Autoencoder']
        auc_scores = [ensemble_auc, if_auc, ae_auc]
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(models, auc_scores, color=['blue', 'green', 'red'], alpha=0.7)
        plt.title('Model Performance Comparison (AUC-ROC)')
        plt.ylabel('AUC-ROC Score')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, auc_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self, predictions: Dict, filepath: Path):
        """Plot feature importance (sample from first prediction)."""
        if not predictions['explainability']['isolation_forest']:
            return
        
        # Get top features from first sample
        if_features = predictions['explainability']['isolation_forest'][0]
        ae_features = predictions['explainability']['autoencoder'][0]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Isolation Forest features
        if_names = [f['feature'] for f in if_features[:10]]
        if_values = [abs(f['contribution']) for f in if_features[:10]]
        
        ax1.barh(range(len(if_names)), if_values, color='green', alpha=0.7)
        ax1.set_yticks(range(len(if_names)))
        ax1.set_yticklabels(if_names)
        ax1.set_title('Isolation Forest - Top Features')
        ax1.set_xlabel('|SHAP Value|')
        
        # Autoencoder features
        ae_names = [f['feature'] for f in ae_features[:10]]
        ae_values = [abs(f['contribution']) for f in ae_features[:10]]
        
        ax2.barh(range(len(ae_names)), ae_values, color='red', alpha=0.7)
        ax2.set_yticks(range(len(ae_names)))
        ax2.set_yticklabels(ae_names)
        ax2.set_title('Autoencoder - Top Features')
        ax2.set_xlabel('|Reconstruction Error|')
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_evaluation_report(self, output_dir: str):
        """Save comprehensive evaluation report."""
        logger.info("Saving evaluation report", output_dir=output_dir)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create report
        report = {
            'evaluation_summary': {
                'timestamp': datetime.now().isoformat(),
                'model_info': {
                    'ensemble_weights': {
                        'isolation_forest': self.config['models']['ensemble']['if_weight'],
                        'autoencoder': self.config['models']['ensemble']['ae_weight']
                    },
                    'thresholds': self.config['thresholds']
                },
                'data_info': self.evaluation_results['test_data'],
                'metrics': self.evaluation_results['metrics']
            },
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        report_path = output_path / 'evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save metrics summary
        metrics_path = output_path / 'metrics_summary.txt'
        with open(metrics_path, 'w') as f:
            f.write(self._format_metrics_summary())
        
        logger.info("Evaluation report saved", report_path=str(report_path))
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on evaluation results."""
        metrics = self.evaluation_results['metrics']
        recommendations = []
        
        # Check precision
        if metrics['classification']['precision'] < 0.5:
            recommendations.append("Low precision detected. Consider adjusting thresholds or feature engineering.")
        
        # Check recall
        if metrics['classification']['recall'] < 0.7:
            recommendations.append("Low recall detected. Consider lowering anomaly thresholds.")
        
        # Check alert rate
        if metrics['classification']['alert_rate'] > 0.2:
            recommendations.append("High alert rate detected. Consider increasing thresholds to reduce false positives.")
        
        # Check model balance
        if_auc = metrics['model_comparison']['isolation_forest_auc']
        ae_auc = metrics['model_comparison']['autoencoder_auc']
        if abs(if_auc - ae_auc) > 0.1:
            recommendations.append("Significant performance gap between models. Consider adjusting ensemble weights.")
        
        if not recommendations:
            recommendations.append("Model performance looks good. Consider monitoring for drift over time.")
        
        return recommendations
    
    def _format_metrics_summary(self) -> str:
        """Format metrics summary for text output."""
        metrics = self.evaluation_results['metrics']
        
        summary = f"""
MRO Anomaly Detection - Evaluation Summary
==========================================

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Data Information:
----------------
Total samples: {metrics['total_samples']:,}
True anomalies: {metrics['true_anomalies']:,}
Anomaly rate: {metrics['classification']['true_anomaly_rate']:.1%}

Classification Metrics:
----------------------
Precision: {metrics['classification']['precision']:.3f}
Recall: {metrics['classification']['recall']:.3f}
F1 Score: {metrics['classification']['f1_score']:.3f}
Alert Rate: {metrics['classification']['alert_rate']:.1%}

AUC Metrics:
------------
AUC-ROC: {metrics['auc_metrics']['auc_roc']:.3f}
AUC-PR: {metrics['auc_metrics']['auc_pr']:.3f}

Model Comparison:
----------------
Isolation Forest AUC: {metrics['model_comparison']['isolation_forest_auc']:.3f}
Autoencoder AUC: {metrics['model_comparison']['autoencoder_auc']:.3f}
Ensemble AUC: {metrics['model_comparison']['ensemble_auc']:.3f}

Confusion Matrix:
-----------------
{np.array(metrics['confusion_matrix'])}

Recommendations:
---------------
"""
        
        for rec in self._generate_recommendations():
            summary += f"- {rec}\n"
        
        return summary
    
    def evaluate_pipeline(self, artifacts_dir: str, test_data_path: str, 
                         output_dir: str) -> Dict:
        """Complete evaluation pipeline."""
        logger.info("Starting complete evaluation pipeline")
        
        # Load trained model
        self.load_trained_model(artifacts_dir)
        
        # Load test data
        X_test, task_index, y_test = self.load_test_data(test_data_path)
        
        # Evaluate model
        metrics = self.evaluate_model(X_test, y_test)
        
        # Create visualizations
        self.create_visualizations(output_dir)
        
        # Save evaluation report
        self.save_evaluation_report(output_dir)
        
        logger.info("Evaluation pipeline complete")
        
        return self.evaluation_results


def evaluate_from_config(config_path: str = 'config.yaml', 
                        artifacts_dir: str = None,
                        test_data_path: str = None,
                        output_dir: str = 'evaluation_results'):
    """Evaluate models using configuration file."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set default paths
    if artifacts_dir is None:
        artifacts_dir = config['artifacts']['dir']
    
    if test_data_path is None:
        # Use the same data for testing (in real scenario would be separate)
        data_dir = Path(config['data']['output_dir'])
        parquet_files = list(data_dir.glob('*.parquet'))
        if parquet_files:
            test_data_path = str(max(parquet_files, key=lambda x: x.stat().st_mtime))
        else:
            raise FileNotFoundError("No test data files found.")
    
    # Initialize evaluator
    evaluator = MROEvaluator(config)
    
    # Run evaluation pipeline
    results = evaluator.evaluate_pipeline(artifacts_dir, test_data_path, output_dir)
    
    # Print summary
    metrics = results['metrics']
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)
    print(f"Precision: {metrics['classification']['precision']:.3f}")
    print(f"Recall: {metrics['classification']['recall']:.3f}")
    print(f"F1 Score: {metrics['classification']['f1_score']:.3f}")
    print(f"AUC-ROC: {metrics['auc_metrics']['auc_roc']:.3f}")
    print(f"Alert Rate: {metrics['classification']['alert_rate']:.1%}")
    print(f"\nResults saved to: {output_dir}")
    print("="*50)
    
    return results


if __name__ == "__main__":
    # Test evaluation pipeline
    evaluate_from_config()
