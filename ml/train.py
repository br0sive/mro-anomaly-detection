"""
Training Pipeline for MRO Anomaly Detection

Handles model training with time-series aware splitting and artifact persistence.
"""

import pandas as pd
import numpy as np
import yaml
import structlog
from typing import Dict, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime

# Import local modules
from ml.features import MROFeatureEngineer, load_and_engineer_features
from ml.models import HybridAnomalyDetector

logger = structlog.get_logger()


class MROTrainer:
    """Training pipeline for MRO anomaly detection models."""
    
    def __init__(self, config: Dict):
        """Initialize trainer with configuration."""
        self.config = config
        self.feature_engineer = None
        self.model = None
        self.training_metrics = {}
        
    def load_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Load and engineer features from data."""
        logger.info("Loading and engineering data", path=data_path)
        
        # Load data and engineer features
        X, task_index, y, self.feature_engineer = load_and_engineer_features(data_path, self.config)
        
        logger.info("Data loading complete", 
                   X_shape=X.shape,
                   task_index_shape=task_index.shape,
                   has_target=y is not None)
        
        return X, task_index, y
    
    def time_series_split(self, X: pd.DataFrame, y: pd.Series, 
                         test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Perform time-series aware train-test split."""
        logger.info("Performing time-series split", test_size=test_size)
        
        # Sort by time (assuming data is already sorted by planned_start_ts)
        # For this synthetic data, we'll use index-based split
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx] if y is not None else None
        y_test = y.iloc[split_idx:] if y is not None else None
        
        logger.info("Time-series split complete", 
                   train_size=len(X_train),
                   test_size=len(X_test))
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None):
        """Train the hybrid anomaly detection models."""
        logger.info("Training hybrid anomaly detection models", X_train_shape=X_train.shape)
        
        # Initialize model
        self.model = HybridAnomalyDetector(self.config)
        
        # Train models
        self.model.fit(X_train, y_train)
        
        logger.info("Model training complete")
    
    def evaluate_training(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """Evaluate model performance on training data."""
        logger.info("Evaluating training performance")
        
        # Get predictions
        results = self.model.predict(X_train)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_train, results['anomaly_flag'])
        
        # Store training metrics
        self.training_metrics = metrics
        
        logger.info("Training evaluation complete", metrics=metrics)
        
        return metrics
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict:
        """Calculate evaluation metrics."""
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        # Basic metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC-ROC (using anomaly scores)
        try:
            auc_roc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc_roc = 0.5  # Default when only one class present
        
        # Alert rate
        alert_rate = y_pred.mean()
        
        # Mean alert latency (simulated - in real scenario would use timestamps)
        mean_latency = 0.0  # Placeholder
        
        metrics = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc_roc': float(auc_roc),
            'alert_rate': float(alert_rate),
            'mean_latency': float(mean_latency),
            'total_samples': len(y_true),
            'anomalies_detected': int(y_pred.sum()),
            'true_anomalies': int(y_true.sum())
        }
        
        return metrics
    
    def save_artifacts(self, artifacts_dir: str):
        """Save all training artifacts."""
        logger.info("Saving training artifacts", artifacts_dir=artifacts_dir)
        
        artifacts_path = Path(artifacts_dir)
        artifacts_path.mkdir(parents=True, exist_ok=True)
        
        # Save models
        self.model.save_models(artifacts_dir)
        
        # Save feature engineer (scaler, encoders, etc.)
        scaler_path = artifacts_path / self.config['artifacts']['models']['scaler']
        import pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.feature_engineer.scaler, f)
        
        # Save feature metadata
        feature_metadata = self.feature_engineer.feature_metadata()
        feature_metadata_path = artifacts_path / self.config['artifacts']['models']['feature_metadata']
        with open(feature_metadata_path, 'w') as f:
            json.dump(feature_metadata, f, indent=2)
        
        # Save thresholds
        thresholds = {
            'anomaly_score': self.config['thresholds']['anomaly_score'],
            'reconstruction_error': self.config['thresholds']['reconstruction_error'],
            'alert_cooldown_minutes': self.config['thresholds']['alert_cooldown_minutes'],
            'training_metrics': self.training_metrics,
            'training_timestamp': datetime.now().isoformat()
        }
        thresholds_path = artifacts_path / self.config['artifacts']['models']['thresholds']
        with open(thresholds_path, 'w') as f:
            json.dump(thresholds, f, indent=2)
        
        logger.info("Artifacts saved successfully")
    
    def train_pipeline(self, data_path: str, artifacts_dir: str) -> Dict:
        """Complete training pipeline."""
        logger.info("Starting complete training pipeline")
        
        # Load and engineer data
        X, task_index, y = self.load_data(data_path)
        
        # Time-series split
        X_train, X_test, y_train, y_test = self.time_series_split(X, y, 
                                                                 self.config['evaluation']['test_size'])
        
        # Train models
        self.train_models(X_train, y_train)
        
        # Evaluate training performance
        training_metrics = self.evaluate_training(X_train, y_train)
        
        # Save artifacts
        self.save_artifacts(artifacts_dir)
        
        # Return training summary
        summary = {
            'training_metrics': training_metrics,
            'data_info': {
                'total_samples': len(X),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features': X.shape[1],
                'anomaly_rate': y.mean() if y is not None else 0.0
            },
            'model_info': {
                'isolation_forest_weight': self.config['models']['ensemble']['if_weight'],
                'autoencoder_weight': self.config['models']['ensemble']['ae_weight'],
                'threshold_percentile': self.config['models']['ensemble']['threshold_percentile']
            },
            'artifacts_saved': artifacts_dir,
            'training_timestamp': datetime.now().isoformat()
        }
        
        logger.info("Training pipeline complete", summary=summary)
        
        return summary


def train_from_config(config_path: str = 'config.yaml', data_path: str = None):
    """Train models using configuration file."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set default data path if not provided
    if data_path is None:
        # Find the most recent data file
        data_dir = Path(config['data']['output_dir'])
        parquet_files = list(data_dir.glob('*.parquet'))
        if parquet_files:
            data_path = str(max(parquet_files, key=lambda x: x.stat().st_mtime))
        else:
            raise FileNotFoundError("No data files found. Please generate data first.")
    
    # Initialize trainer
    trainer = MROTrainer(config)
    
    # Run training pipeline
    summary = trainer.train_pipeline(data_path, config['artifacts']['dir'])
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Total samples: {summary['data_info']['total_samples']:,}")
    print(f"Training samples: {summary['data_info']['train_samples']:,}")
    print(f"Test samples: {summary['data_info']['test_samples']:,}")
    print(f"Features: {summary['data_info']['features']}")
    print(f"Anomaly rate: {summary['data_info']['anomaly_rate']:.1%}")
    print("\nTraining Metrics:")
    print(f"  Precision: {summary['training_metrics']['precision']:.3f}")
    print(f"  Recall: {summary['training_metrics']['recall']:.3f}")
    print(f"  F1 Score: {summary['training_metrics']['f1_score']:.3f}")
    print(f"  AUC-ROC: {summary['training_metrics']['auc_roc']:.3f}")
    print(f"  Alert Rate: {summary['training_metrics']['alert_rate']:.1%}")
    print(f"\nArtifacts saved to: {summary['artifacts_saved']}")
    print("="*50)
    
    return summary


if __name__ == "__main__":
    # Test training pipeline
    train_from_config()
