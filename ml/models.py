"""
ML Models for MRO Anomaly Detection

Implements hybrid anomaly detection using Isolation Forest and Autoencoder
with ensemble decision logic and explainability features.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
import shap
import structlog
from typing import Dict, List, Tuple, Optional, Any
import json
import pickle
from pathlib import Path

logger = structlog.get_logger()


class AutoencoderModel:
    """Autoencoder model for anomaly detection."""
    
    def __init__(self, config: Dict):
        """Initialize autoencoder with configuration."""
        self.config = config
        self.ae_config = config['models']['autoencoder']
        self.model = None
        self.threshold = None
        self.feature_names = None
        
    def build_model(self, input_dim: int) -> keras.Model:
        logger.info("Building enhanced autoencoder model", input_dim=input_dim)
        
        # Encoder with improved architecture
        encoder_input = keras.Input(shape=(input_dim,))
        
        # Enhanced encoder with better regularization
        x = encoder_input
        for i, units in enumerate(self.ae_config['hidden_dims']):
            x = keras.layers.Dense(
                units, 
                activation='relu',
                name=f'encoder_dense_{i}'
            )(x)
            x = keras.layers.Dropout(
                self.ae_config['dropout_rate'],
                name=f'encoder_dropout_{i}'
            )(x)
        
        # Bottleneck (smaller latent space for better compression)
        encoded = keras.layers.Dense(
            max(8, self.ae_config['hidden_dims'][-1] // 4),  # Smaller bottleneck
            activation='relu',
            name='bottleneck'
        )(x)
        
        # Decoder (symmetrical to encoder)
        x = encoded
        for i, units in enumerate(reversed(self.ae_config['hidden_dims'])):
            x = keras.layers.Dense(
                units,
                activation='relu',
                name=f'decoder_dense_{i}'
            )(x)
            x = keras.layers.Dropout(
                self.ae_config['dropout_rate'],
                name=f'decoder_dropout_{i}'
            )(x)
        
        # Output layer with linear activation for better reconstruction
        decoded = keras.layers.Dense(
            input_dim,
            activation='linear',  # Back to linear for better gradient flow
            name='output'
        )(x)
        
        # Create model
        model = keras.Model(encoder_input, decoded, name='enhanced_autoencoder')
        
        # Compile model with better optimizer settings
        model.compile(
            optimizer=keras.optimizers.legacy.Adam(  # Use legacy optimizer for M1/M2 Macs
                learning_rate=self.ae_config['learning_rate'],
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            ),
            loss='mse',  # MSE for better gradient flow
            metrics=['mae', 'mse']
        )
        
        logger.info("Enhanced autoencoder model built", 
                   input_dim=input_dim,
                   bottleneck_dim=max(8, self.ae_config['hidden_dims'][-1] // 4))
        
        return model
    
    def fit(self, X: pd.DataFrame, validation_split: float = 0.2):
        """Train the autoencoder."""
        logger.info("Training autoencoder", 
                   X_shape=X.shape,
                   validation_split=validation_split)
        
        # Build model
        self.model = self.build_model(X.shape[1])
        self.feature_names = X.columns.tolist()
        
        # Early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.ae_config['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        
        # Train model
        history = self.model.fit(
            X, X,  # Autoencoder learns to reconstruct input
            epochs=self.ae_config['epochs'],
            batch_size=self.ae_config['batch_size'],
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Calculate reconstruction error threshold
        self._calculate_threshold(X)
        
        logger.info("Autoencoder training complete", 
                   final_loss=history.history['loss'][-1],
                   threshold=self.threshold)
        
        return history
    
    def _calculate_threshold(self, X: pd.DataFrame):
        """Calculate reconstruction error threshold using improved method."""
        # Get reconstruction errors
        reconstructed = self.model.predict(X)
        
        # Use MSE for better anomaly detection
        reconstruction_errors = np.mean((X - reconstructed) ** 2, axis=1)
        
        # Use adaptive threshold based on error distribution
        # Use 90th percentile for more sensitive detection
        percentile = 90  # More sensitive than 95th percentile
        self.threshold = np.percentile(reconstruction_errors, percentile)
        
        # Additional threshold refinement
        mean_error = np.mean(reconstruction_errors)
        std_error = np.std(reconstruction_errors)
        
        # Use statistical threshold as backup
        statistical_threshold = mean_error + 2 * std_error
        
        # Use the more conservative threshold
        self.threshold = max(self.threshold, statistical_threshold)
        
        logger.info("Enhanced reconstruction error threshold calculated", 
                   threshold=self.threshold,
                   percentile=percentile,
                   mean_error=mean_error,
                   std_error=std_error)
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies using enhanced reconstruction error."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Get reconstruction
        reconstructed = self.model.predict(X)
        
        # Calculate enhanced reconstruction error (MSE)
        reconstruction_errors = np.mean((X - reconstructed) ** 2, axis=1)
        
        # Determine anomalies
        anomaly_flags = (reconstruction_errors > self.threshold).astype(int)
        
        return reconstruction_errors, anomaly_flags
    
    def top_features(self, X: pd.DataFrame, top_k: int = 5) -> List[Dict]:
        # Top contributing features for reconstruction error
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Get reconstruction
        reconstructed = self.model.predict(X)
        
        # Calculate feature-wise reconstruction errors
        feature_errors = np.abs(X - reconstructed)
        
        # Get top contributing features for each sample
        top_features = []
        for i in range(len(X)):
            sample_errors = feature_errors.iloc[i]  # Use iloc instead of direct indexing
            feature_contributions = list(zip(self.feature_names, sample_errors))
            feature_contributions.sort(key=lambda x: x[1], reverse=True)
            
            top_k_features = [
                {
                    'feature': feature,
                    'contribution': float(contribution),
                    'signed_contribution': float(contribution)
                }
                for feature, contribution in feature_contributions[:top_k]
            ]
            top_features.append(top_k_features)
        
        return top_features
    
    def save(self, filepath: str):
        """Save the autoencoder model."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Save model
        self.model.save(filepath)
        
        # Save metadata
        metadata = {
            'threshold': self.threshold,
            'feature_names': self.feature_names,
            'config': self.ae_config
        }
        
        metadata_path = filepath.replace('.h5', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Autoencoder model saved", filepath=filepath)
    
    def load(self, filepath: str):
        """Load the autoencoder model."""
        # Load model
        self.model = keras.models.load_model(filepath)
        
        # Load metadata
        metadata_path = filepath.replace('.h5', '_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.threshold = metadata['threshold']
        self.feature_names = metadata['feature_names']
        
        logger.info("Autoencoder model loaded", filepath=filepath)


class IsolationForestModel:
    """Isolation Forest model for anomaly detection."""
    
    def __init__(self, config: Dict):
        """Initialize Isolation Forest with configuration."""
        self.config = config
        self.if_config = config['models']['isolation_forest']
        self.model = None
        self.threshold = None
        self.feature_names = None
        self.explainer = None
        
    def fit(self, X: pd.DataFrame):
        """Train the Isolation Forest model."""
        logger.info("Training Isolation Forest", X_shape=X.shape)
        
        # Initialize enhanced model
        self.model = IsolationForest(
            n_estimators=self.if_config['n_estimators'],
            max_samples=self.if_config['max_samples'],
            contamination=self.if_config['contamination'],
            random_state=self.if_config['random_state'],
            max_features=0.8,  # Use subset of features for better generalization
            bootstrap=True,   # Enable bootstrap sampling
            n_jobs=-1         # Use all available cores
        )
        
        self.feature_names = X.columns.tolist()
        
        # Train model
        self.model.fit(X)
        
        # Calculate anomaly scores and threshold
        anomaly_scores = self.model.score_samples(X)
        self.threshold = np.percentile(anomaly_scores, 
                                     (1 - self.if_config['contamination']) * 100)
        
        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        
        logger.info("Isolation Forest training complete", 
                   threshold=self.threshold)
        
        return self.model
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies using Isolation Forest."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Get anomaly scores (lower = more anomalous)
        anomaly_scores = self.model.score_samples(X)
        
        # Convert to positive scores (higher = more anomalous)
        positive_scores = -anomaly_scores
        
        # Determine anomalies
        anomaly_flags = (positive_scores > self.threshold).astype(int)
        
        return positive_scores, anomaly_flags
    
    def top_features(self, X: pd.DataFrame, top_k: int = 5) -> List[Dict]:
        """Get top contributing features using SHAP."""
        if self.model is None or self.explainer is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # Get top contributing features for each sample
        top_features = []
        for i in range(len(X)):
            sample_shap = shap_values[i]
            feature_contributions = list(zip(self.feature_names, sample_shap))
            feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            top_k_features = [
                {
                    'feature': feature,
                    'contribution': float(contribution),
                    'signed_contribution': float(contribution)
                }
                for feature, contribution in feature_contributions[:top_k]
            ]
            top_features.append(top_k_features)
        
        return top_features
    
    def save(self, filepath: str):
        """Save the Isolation Forest model."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Save model
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save metadata
        metadata = {
            'threshold': self.threshold,
            'feature_names': self.feature_names,
            'config': self.if_config
        }
        
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Isolation Forest model saved", filepath=filepath)
    
    def load(self, filepath: str):
        """Load the Isolation Forest model."""
        # Load model
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load metadata
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.threshold = metadata['threshold']
        self.feature_names = metadata['feature_names']
        
        # Reinitialize explainer
        self.explainer = shap.TreeExplainer(self.model)
        
        logger.info("Isolation Forest model loaded", filepath=filepath)


class HybridAnomalyDetector:
    """Hybrid anomaly detection combining Isolation Forest and Autoencoder."""
    
    def __init__(self, config: Dict):
        """Initialize hybrid detector with configuration."""
        self.config = config
        self.ensemble_config = config['models']['ensemble']
        
        # Initialize models
        self.isolation_forest = IsolationForestModel(config)
        self.autoencoder = AutoencoderModel(config)
        
        # Ensemble weights
        self.if_weight = self.ensemble_config['if_weight']
        self.ae_weight = self.ensemble_config['ae_weight']
        
        # Feature engineer
        self.feature_engineer = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Train both models."""
        logger.info("Training hybrid anomaly detector", X_shape=X.shape)
        
        # Train Isolation Forest
        self.isolation_forest.fit(X)
        
        # Train Autoencoder
        self.autoencoder.fit(X)
        
        logger.info("Hybrid model training complete")
    
    def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Predict anomalies using ensemble approach."""
        logger.info("Making hybrid predictions", X_shape=X.shape)
        
        # Get predictions from both models
        if_scores, if_flags = self.isolation_forest.predict(X)
        ae_scores, ae_flags = self.autoencoder.predict(X)
        
        # Normalize scores to 0-1 range (handle edge cases)
        if if_scores.max() == if_scores.min():
            if_scores_norm = np.zeros_like(if_scores)
        else:
            if_scores_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min())
        
        if ae_scores.max() == ae_scores.min():
            ae_scores_norm = np.zeros_like(ae_scores)
        else:
            ae_scores_norm = (ae_scores - ae_scores.min()) / (ae_scores.max() - ae_scores.min())
        
        # Combine scores
        combined_scores = (self.if_weight * if_scores_norm + 
                          self.ae_weight * ae_scores_norm)
        
        # Enhanced ensemble logic with better score combination
        combined_threshold = self.config['thresholds']['anomaly_score']
        
        # Weighted combination with confidence adjustment
        # Give more weight to the model with higher confidence (larger score difference from threshold)
        if_confidence = np.abs(if_scores_norm - combined_threshold)
        ae_confidence = np.abs(ae_scores_norm - combined_threshold)
        
        # Dynamic weights based on confidence
        total_confidence = if_confidence + ae_confidence + 1e-6
        dynamic_if_weight = if_confidence / total_confidence
        dynamic_ae_weight = ae_confidence / total_confidence
        
        # Combine scores with dynamic weights
        combined_scores = (dynamic_if_weight * if_scores_norm + 
                          dynamic_ae_weight * ae_scores_norm)
        
        # Primary: Use combined score threshold
        combined_flags = (combined_scores > combined_threshold).astype(int)
        
        # Secondary: Model agreement with higher threshold
        # If both models agree on anomaly, use a lower threshold for flagging
        model_agreement = (if_flags & ae_flags).astype(int)
        agreement_threshold = combined_threshold * 0.7  # Lower threshold for agreement
        agreement_flags = (combined_scores > agreement_threshold) & (model_agreement == 1)
        
        # Final flags: primary OR agreement-based flags
        combined_flags = (combined_flags | agreement_flags).astype(int)
        
        # Get explainability
        if_top_features = self.isolation_forest.top_features(X)
        ae_top_features = self.autoencoder.top_features(X)
        
        results = {
            'anomaly_flag': combined_flags,
            'anomaly_score': combined_scores,
            'model_scores': {
                'isolation_forest': if_scores_norm,
                'autoencoder': ae_scores_norm
            },
            'model_flags': {
                'isolation_forest': if_flags,
                'autoencoder': ae_flags
            },
            'explainability': {
                'isolation_forest': if_top_features,
                'autoencoder': ae_top_features
            }
        }
        
        logger.info("Hybrid predictions complete", 
                   anomalies=combined_flags.sum(),
                   anomaly_rate=combined_flags.mean())
        
        return results
    
    def predict_single(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Predict for a single sample with detailed explainability."""
        if len(X) != 1:
            raise ValueError("predict_single expects exactly one sample")
        
        # Get predictions
        results = self.predict(X)
        
        # Extract single sample results
        single_result = {
            'anomaly_flag': int(results['anomaly_flag'][0]),
            'anomaly_score': float(results['anomaly_score'][0]),
            'model_scores': {
                'isolation_forest': float(results['model_scores']['isolation_forest'][0]),
                'autoencoder': float(results['model_scores']['autoencoder'][0])
            },
            'model_flags': {
                'isolation_forest': int(results['model_flags']['isolation_forest'][0]),
                'autoencoder': int(results['model_flags']['autoencoder'][0])
            },
            'explainability': {
                'isolation_forest': results['explainability']['isolation_forest'][0],
                'autoencoder': results['explainability']['autoencoder'][0]
            }
        }
        
        return single_result
    
    def save_models(self, artifacts_dir: str):
        """Save both models."""
        artifacts_path = Path(artifacts_dir)
        artifacts_path.mkdir(parents=True, exist_ok=True)
        
        # Save Isolation Forest
        if_path = artifacts_path / self.config['artifacts']['models']['isolation_forest']
        self.isolation_forest.save(str(if_path))
        
        # Save Autoencoder
        ae_path = artifacts_path / self.config['artifacts']['models']['autoencoder']
        self.autoencoder.save(str(ae_path))
        
        logger.info("Models saved", 
                   isolation_forest=str(if_path),
                   autoencoder=str(ae_path))
    
    def load_models(self, artifacts_dir: str):
        """Load both models."""
        artifacts_path = Path(artifacts_dir)
        
        # Load Isolation Forest
        if_path = artifacts_path / self.config['artifacts']['models']['isolation_forest']
        self.isolation_forest.load(str(if_path))
        
        # Load Autoencoder
        ae_path = artifacts_path / self.config['artifacts']['models']['autoencoder']
        self.autoencoder.load(str(ae_path))
        
        logger.info("Models loaded", 
                   isolation_forest=str(if_path),
                   autoencoder=str(ae_path))


if __name__ == "__main__":
    # Test model functionality
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create sample data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(1000, 50))
    
    # Test hybrid detector
    detector = HybridAnomalyDetector(config)
    detector.fit(X)
    
    # Test predictions
    results = detector.predict(X.head(10))
    print(f"Hybrid predictions complete:")
    print(f"Anomalies: {results['anomaly_flag'].sum()}")
    print(f"Anomaly rate: {results['anomaly_flag'].mean():.3f}")
    print(f"Model scores shape: {len(results['model_scores']['isolation_forest'])}")
