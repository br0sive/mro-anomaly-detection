"""
Feature Engineering for MRO Anomaly Detection

Transforms raw MRO workflow data into ML-ready features with proper scaling
and encoding for anomaly detection models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import Dict, List, Tuple, Optional
import structlog
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
logger = structlog.get_logger()


class MROFeatureEngineer:
    """Feature engineering pipeline for MRO workflow data."""
    
    def __init__(self, config: Dict):
        """Initialize the feature engineer with configuration."""
        self.config = config
        self.features_config = config['features']
        
        # Initialize encoders and scalers
        self.scaler = RobustScaler()
        self.label_encoders = {}
        self.onehot_encoder = None
        self.feature_names = []
        self.categorical_features = []
        self.numerical_features = []
        
        # Store feature metadata
        self._feature_metadata = {}
        
    def parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse timestamps and create duration/gap features."""
        logger.info("Parsing timestamps and creating duration features")
        
        # Convert timestamps to datetime
        timestamp_cols = ['planned_start_ts', 'planned_end_ts', 'actual_start_ts', 'actual_end_ts']
        for col in timestamp_cols:
            df[col] = pd.to_datetime(df[col])
        
        # Calculate duration features
        df['planned_duration_minutes'] = (df['planned_end_ts'] - df['planned_start_ts']).dt.total_seconds() / 60
        df['actual_duration_minutes'] = (df['actual_end_ts'] - df['actual_start_ts']).dt.total_seconds() / 60
        
        # Calculate lag features
        df['start_lag_minutes'] = (df['actual_start_ts'] - df['planned_start_ts']).dt.total_seconds() / 60
        df['end_lag_minutes'] = (df['actual_end_ts'] - df['planned_end_ts']).dt.total_seconds() / 60
        
        # Duration ratios
        df['duration_ratio'] = df['actual_minutes'] / df['planned_minutes']
        df['duration_variance'] = df['actual_minutes'] - df['planned_minutes']
        
        # Extract time-based features
        df['hour_of_day'] = df['planned_start_ts'].dt.hour
        df['day_of_week'] = df['planned_start_ts'].dt.dayofweek
        df['month'] = df['planned_start_ts'].dt.month
        df['day_of_month'] = df['planned_start_ts'].dt.day
        
        return df
    
    def rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating enhanced rolling statistics features")
        
        # Sort by time for rolling calculations
        df = df.sort_values('planned_start_ts')
        
        # Enhanced rolling features by task_name
        task_rolling = df.groupby('task_name').rolling(
            window=f"{self.features_config['rolling_window_days']}D",
            on='planned_start_ts',
            min_periods=1
        ).agg({
            'actual_minutes': ['mean', 'std', 'median', 'min', 'max'],
            'duration_ratio': ['mean', 'std', 'min', 'max'],
            'snag_count': ['mean', 'sum', 'max'],
            'concurrent_tasks': ['mean', 'max', 'std'],
            'snag_severity': ['mean', 'max'],
            'start_lag_minutes': ['mean', 'std', 'max'],
            'end_lag_minutes': ['mean', 'std', 'max']
        }).fillna(method='ffill')
        
        # Flatten column names
        task_rolling.columns = [f"task_{col[0]}_{col[1]}" for col in task_rolling.columns]
        
        # Enhanced rolling features by module_id
        module_rolling = df.groupby('module_id').rolling(
            window=f"{self.features_config['rolling_window_days']}D",
            on='planned_start_ts',
            min_periods=1
        ).agg({
            'actual_minutes': ['mean', 'std', 'median', 'min', 'max'],
            'duration_ratio': ['mean', 'std', 'min', 'max'],
            'snag_count': ['mean', 'sum', 'max'],
            'snag_severity': ['mean', 'max'],
            'concurrent_tasks': ['mean', 'max']
        }).fillna(method='ffill')
        
        # Flatten column names
        module_rolling.columns = [f"module_{col[0]}_{col[1]}" for col in module_rolling.columns]
        
        # Rolling features by technician
        tech_rolling = df.groupby('tech_id').rolling(
            window=f"{self.features_config['rolling_window_days']}D",
            on='planned_start_ts',
            min_periods=1
        ).agg({
            'actual_minutes': ['mean', 'std'],
            'duration_ratio': ['mean', 'std'],
            'snag_count': ['mean', 'sum'],
            'concurrent_tasks': ['mean', 'max']
        }).fillna(method='ffill')
        
        tech_rolling.columns = [f"tech_{col[0]}_{col[1]}" for col in tech_rolling.columns]
        
        # Merge rolling features back to main dataframe
        df = df.reset_index(drop=True)
        task_rolling = task_rolling.reset_index(drop=True)
        module_rolling = module_rolling.reset_index(drop=True)
        tech_rolling = tech_rolling.reset_index(drop=True)
        
        df = pd.concat([df, task_rolling, module_rolling, tech_rolling], axis=1)
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using label encoding and one-hot encoding."""
        logger.info("Encoding categorical features")
        
        # Define categorical features
        categorical_cols = ['task_name', 'tech_grade', 'shift', 'bay_id']
        
        # Label encoding for ordinal features
        ordinal_cols = ['tech_grade']
        for col in ordinal_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f"{col}_encoded"] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        
        # One-hot encoding for nominal features
        nominal_cols = ['shift']
        if nominal_cols:
            onehot_cols = df[nominal_cols].copy()
            onehot_cols = pd.get_dummies(onehot_cols, prefix=nominal_cols)
            df = pd.concat([df, onehot_cols], axis=1)
        
        # Cyclical encoding for weekday
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
        
        # Cyclical encoding for hour of day
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        
        return df
    
    def context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating enhanced context and KPI features")
        
        # Technician load features
        df['tech_grade_ordinal'] = df['tech_grade'].map({'A': 3, 'B': 2, 'C': 1})
        df['tech_overload'] = (df['concurrent_tasks'] >= 3).astype(int)
        df['tech_load_score'] = df['concurrent_tasks'] * df['tech_grade_ordinal']
        df['tech_efficiency'] = df['planned_minutes'] / (df['actual_minutes'] + 1e-6)
        
        # Enhanced snag features
        df['snag_severity_weighted'] = df['snag_count'] * df['snag_severity']
        df['has_snags'] = (df['snag_count'] > 0).astype(int)
        df['high_severity_snags'] = (df['snag_severity'] >= 2).astype(int)
        df['snag_rate'] = df['snag_count'] / (df['actual_minutes'] / 60 + 1e-6)  # Snags per hour
        df['snag_intensity'] = df['snag_count'] * df['snag_severity'] / (df['actual_minutes'] / 60 + 1e-6)
        
        # Enhanced KPI features
        df['tat_efficiency'] = df['tat_hours_engine'] / (df['planned_minutes'] / 60 + 1e-6)
        df['wip_density'] = df['wip_count_shop'] / 20
        df['mttr_efficiency'] = df['mttr_hours_engine'] / (df['planned_minutes'] / 60 + 1e-6)
        df['kpi_stress'] = (df['tat_efficiency'] > 1.5) | (df['wip_density'] > 0.8) | (df['mttr_efficiency'] > 1.2)
        
        # Workload and timing features
        df['is_overtime'] = (df['hour_of_day'] >= 18).astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night_shift'] = (df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 6)
        df['is_peak_hours'] = (df['hour_of_day'] >= 8) & (df['hour_of_day'] <= 17)
        
        # Duration anomaly indicators
        df['duration_outlier'] = (df['duration_ratio'] > 2.0) | (df['duration_ratio'] < 0.3)
        df['extreme_delay'] = (df['start_lag_minutes'] > 60) | (df['end_lag_minutes'] > 120)
        df['schedule_deviation'] = abs(df['start_lag_minutes']) + abs(df['end_lag_minutes'])
        
        # Resource utilization features
        df['bay_utilization'] = df.groupby('bay_id')['concurrent_tasks'].transform('mean')
        df['module_complexity'] = df.groupby('module_id')['actual_minutes'].transform('mean')
        df['task_complexity'] = df.groupby('task_name')['actual_minutes'].transform('mean')
        
        # Temporal patterns
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and order features for ML models."""
        logger.info("Selecting features for ML models")
        
        # Define enhanced feature columns
        numerical_features = [
            # Core duration features
            'planned_minutes', 'actual_minutes', 'planned_duration_minutes', 'actual_duration_minutes',
            'start_lag_minutes', 'end_lag_minutes', 'duration_ratio', 'duration_variance',
            
            # Enhanced rolling features
            'task_actual_minutes_mean', 'task_actual_minutes_std', 'task_actual_minutes_median', 'task_actual_minutes_min', 'task_actual_minutes_max',
            'task_duration_ratio_mean', 'task_duration_ratio_std', 'task_duration_ratio_min', 'task_duration_ratio_max',
            'task_snag_count_mean', 'task_snag_count_sum', 'task_snag_count_max',
            'task_concurrent_tasks_mean', 'task_concurrent_tasks_max', 'task_concurrent_tasks_std',
            'task_snag_severity_mean', 'task_snag_severity_max',
            'task_start_lag_minutes_mean', 'task_start_lag_minutes_std', 'task_start_lag_minutes_max',
            'task_end_lag_minutes_mean', 'task_end_lag_minutes_std', 'task_end_lag_minutes_max',
            
            'module_actual_minutes_mean', 'module_actual_minutes_std', 'module_actual_minutes_median', 'module_actual_minutes_min', 'module_actual_minutes_max',
            'module_duration_ratio_mean', 'module_duration_ratio_std', 'module_duration_ratio_min', 'module_duration_ratio_max',
            'module_snag_count_mean', 'module_snag_count_sum', 'module_snag_count_max',
            'module_snag_severity_mean', 'module_snag_severity_max',
            'module_concurrent_tasks_mean', 'module_concurrent_tasks_max',
            
            'tech_actual_minutes_mean', 'tech_actual_minutes_std',
            'tech_duration_ratio_mean', 'tech_duration_ratio_std',
            'tech_snag_count_mean', 'tech_snag_count_sum',
            'tech_concurrent_tasks_mean', 'tech_concurrent_tasks_max',
            
            # Enhanced context features
            'concurrent_tasks', 'tech_grade_encoded', 'tech_overload', 'tech_load_score', 'tech_efficiency',
            'snag_count', 'snag_severity', 'snag_severity_weighted', 'has_snags', 'high_severity_snags',
            'snag_rate', 'snag_intensity',
            'tat_hours_engine', 'wip_count_shop', 'mttr_hours_engine',
            'tat_efficiency', 'wip_density', 'mttr_efficiency', 'kpi_stress',
            
            # Enhanced timing features
            'hour_of_day', 'day_of_week', 'month', 'day_of_month',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'is_overtime', 'is_weekend', 'is_night_shift', 'is_peak_hours',
            
            # Anomaly indicators
            'duration_outlier', 'extreme_delay', 'schedule_deviation',
            
            # Resource utilization
            'bay_utilization', 'module_complexity', 'task_complexity'
        ]
        
        # Filter to existing columns
        available_features = [col for col in numerical_features if col in df.columns]
        
        # Add one-hot encoded features
        onehot_features = [col for col in df.columns if col.startswith('shift_')]
        available_features.extend(onehot_features)
        
        # Store feature metadata
        self._feature_metadata = {
            'numerical_features': available_features,
            'categorical_features': [],
            'total_features': len(available_features)
        }
        
        # Select features
        feature_df = df[available_features].copy()
        
        # Handle missing values
        feature_df = feature_df.fillna(method='ffill').fillna(0)
        
        logger.info("Feature selection complete", 
                   total_features=len(available_features),
                   shape=feature_df.shape)
        
        return feature_df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features using RobustScaler."""
        logger.info("Scaling features")
        
        if fit:
            scaled_features = self.scaler.fit_transform(df)
        else:
            scaled_features = self.scaler.transform(df)
        
        scaled_df = pd.DataFrame(scaled_features, columns=df.columns, index=df.index)
        
        logger.info("Feature scaling complete", shape=scaled_df.shape)
        return scaled_df
    
    def task_index(self, df: pd.DataFrame) -> pd.DataFrame:
        # Task index for tracking predictions
        task_index = df[['engine_id', 'module_id', 'submodule_id', 'task_id', 'wo_id']].copy()
        return task_index
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series]]:
        """Complete feature engineering pipeline."""
        logger.info("Starting feature engineering pipeline")
        
        # Parse timestamps
        df = self.parse_timestamps(df)
        
        # Create rolling features
        df = self.rolling_features(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Create context features
        df = self.context_features(df)
        
        # Select features
        X = self.select_features(df)
        
        # Scale features
        X = self.scale_features(X, fit=True)
        
        # Create task index
        task_index = self.task_index(df)
        
        # Extract target if available
        y = None
        if 'is_anomaly' in df.columns:
            y = df['is_anomaly']
        
        logger.info("Feature engineering complete", 
                   X_shape=X.shape,
                   task_index_shape=task_index.shape,
                   has_target=y is not None)
        
        return X, task_index, y
    
    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Transform new data using fitted encoders and scalers."""
        logger.info("Transforming new data")
        
        # Parse timestamps
        df = self.parse_timestamps(df)
        
        # Create rolling features
        df = self.rolling_features(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Create context features
        df = self.context_features(df)
        
        # Select features
        X = self.select_features(df)
        
        # Scale features (using fitted scaler)
        X = self.scale_features(X, fit=False)
        
        # Create task index
        task_index = self.task_index(df)
        
        logger.info("Data transformation complete", 
                   X_shape=X.shape,
                   task_index_shape=task_index.shape)
        
        return X, task_index
    
    def feature_names(self) -> List[str]:
        # Feature names list
        return self._feature_metadata.get('numerical_features', [])
    
    def feature_metadata(self) -> Dict:
        # Feature metadata
        return self._feature_metadata.copy()


def load_and_engineer_features(data_path: str, config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load data and apply feature engineering."""
    logger.info("Loading data for feature engineering", path=data_path)
    
    # Load data
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    
    # Initialize feature engineer
    engineer = MROFeatureEngineer(config)
    
    # Apply feature engineering
    X, task_index, y = engineer.fit_transform(df)
    
    return X, task_index, y, engineer


if __name__ == "__main__":
    # Test feature engineering
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load sample data
    X, task_index, y, engineer = load_and_engineer_features('data/out/mro_data.parquet', config)
    
    print(f"Feature engineering complete:")
    print(f"X shape: {X.shape}")
    print(f"Task index shape: {task_index.shape}")
    print(f"Target shape: {y.shape if y is not None else 'None'}")
    print(f"Feature names: {len(engineer.feature_names())}")
