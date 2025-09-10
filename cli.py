"""
Command Line Interface for MRO Anomaly Detection

Provides CLI commands for data generation, training, evaluation, and serving.
"""

import click
import yaml
import structlog
from pathlib import Path
from typing import Optional
import pandas as pd
import json
from datetime import datetime

# Import local modules
from data.generate_synthetic_mro import MRODataGenerator
from ml.train import MROTrainer
from ml.evaluate import MROEvaluator
from ml.features import MROFeatureEngineer
from ml.models import HybridAnomalyDetector

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        click.echo(f"Configuration file not found: {config_path}")
        click.echo("Please ensure config.yaml exists in the current directory.")
        exit(1)
    except yaml.YAMLError as e:
        click.echo(f"Error parsing configuration file: {e}")
        exit(1)


@click.group()
@click.option('--config', default='config.yaml', help='Configuration file path')
@click.option('--verbose', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config: str, verbose: bool):
    """MRO Anomaly Detection CLI"""
    ctx.ensure_object(dict)
    ctx.obj['config'] = load_config(config)
    ctx.obj['verbose'] = verbose
    
    # Set log level
    if verbose:
        structlog.configure(processors=[structlog.processors.ConsoleRenderer()])
    
    click.echo("MRO Anomaly Detection CLI")
    click.echo("=" * 40)


@cli.command()
@click.option('--rows', default=45000, help='Number of rows to generate')
@click.option('--anomaly-rate', default=0.07, help='Anomaly rate (0.0-1.0)')
@click.option('--seed', default=42, help='Random seed')
@click.option('--output-dir', default=None, help='Output directory')
@click.pass_context
def gen_data(ctx, rows: int, anomaly_rate: float, seed: int, output_dir: Optional[str]):
    """Generate synthetic MRO data with controlled anomalies."""
    config = ctx.obj['config']
    
    if output_dir is None:
        output_dir = config['data']['output_dir']
    
    click.echo(f"Generating {rows:,} rows with {anomaly_rate:.1%} anomaly rate...")
    
    try:
        # Initialize generator
        generator = MRODataGenerator(config)
        
        # Generate dataset
        df = generator.generate_dataset(rows, anomaly_rate)
        
        # Save dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mro_data_{timestamp}"
        csv_path, parquet_path = generator.save_dataset(df, output_dir, filename)
        
        # Print summary
        click.echo("\n" + "="*50)
        click.echo("DATA GENERATION COMPLETE")
        click.echo("="*50)
        click.echo(f"Total rows: {len(df):,}")
        click.echo(f"Anomalies: {df['is_anomaly'].sum():,} ({df['is_anomaly'].mean():.1%})")
        click.echo(f"Engines: {df['engine_id'].nunique()}")
        click.echo(f"Modules: {df['module_id'].nunique()}")
        click.echo(f"Technicians: {df['tech_id'].nunique()}")
        click.echo(f"Date range: {df['planned_start_ts'].min()} to {df['planned_start_ts'].max()}")
        click.echo(f"\nFiles saved:")
        click.echo(f"  CSV: {csv_path}")
        click.echo(f"  Parquet: {parquet_path}")
        click.echo("="*50)
        
    except Exception as e:
        click.echo(f"Error generating data: {e}")
        logger.error("Data generation failed", error=str(e))
        exit(1)


@cli.command()
@click.option('--data-path', default=None, help='Path to training data')
@click.option('--artifacts-dir', default=None, help='Artifacts output directory')
@click.pass_context
def train(ctx, data_path: Optional[str], artifacts_dir: Optional[str]):
    """Train anomaly detection models."""
    config = ctx.obj['config']
    
    if artifacts_dir is None:
        artifacts_dir = config['artifacts']['dir']
    
    if data_path is None:
        # Find the most recent data file
        data_dir = Path(config['data']['output_dir'])
        parquet_files = list(data_dir.glob('*.parquet'))
        if parquet_files:
            data_path = str(max(parquet_files, key=lambda x: x.stat().st_mtime))
        else:
            click.echo("No data files found. Please generate data first using 'gen-data'.")
            exit(1)
    
    click.echo(f"Training models with data: {data_path}")
    click.echo(f"Artifacts will be saved to: {artifacts_dir}")
    
    try:
        # Initialize trainer
        trainer = MROTrainer(config)
        
        # Run training pipeline
        summary = trainer.train_pipeline(data_path, artifacts_dir)
        
        # Print summary
        click.echo("\n" + "="*50)
        click.echo("TRAINING COMPLETE")
        click.echo("="*50)
        click.echo(f"Total samples: {summary['data_info']['total_samples']:,}")
        click.echo(f"Training samples: {summary['data_info']['train_samples']:,}")
        click.echo(f"Test samples: {summary['data_info']['test_samples']:,}")
        click.echo(f"Features: {summary['data_info']['features']}")
        click.echo(f"Anomaly rate: {summary['data_info']['anomaly_rate']:.1%}")
        click.echo("\nTraining Metrics:")
        click.echo(f"  Precision: {summary['training_metrics']['precision']:.3f}")
        click.echo(f"  Recall: {summary['training_metrics']['recall']:.3f}")
        click.echo(f"  F1 Score: {summary['training_metrics']['f1_score']:.3f}")
        click.echo(f"  AUC-ROC: {summary['training_metrics']['auc_roc']:.3f}")
        click.echo(f"  Alert Rate: {summary['training_metrics']['alert_rate']:.1%}")
        click.echo(f"\nArtifacts saved to: {summary['artifacts_saved']}")
        click.echo("="*50)
        
    except Exception as e:
        click.echo(f"Error training models: {e}")
        logger.error("Training failed", error=str(e))
        exit(1)


@cli.command()
@click.option('--artifacts-dir', default=None, help='Path to trained model artifacts')
@click.option('--test-data-path', default=None, help='Path to test data')
@click.option('--output-dir', default='evaluation_results', help='Evaluation output directory')
@click.pass_context
def evaluate(ctx, artifacts_dir: Optional[str], test_data_path: Optional[str], output_dir: str):
    """Evaluate trained models."""
    config = ctx.obj['config']
    
    if artifacts_dir is None:
        artifacts_dir = config['artifacts']['dir']
    
    if test_data_path is None:
        # Use the same data for testing (in real scenario would be separate)
        data_dir = Path(config['data']['output_dir'])
        parquet_files = list(data_dir.glob('*.parquet'))
        if parquet_files:
            test_data_path = str(max(parquet_files, key=lambda x: x.stat().st_mtime))
        else:
            click.echo("No test data files found. Please generate data first.")
            exit(1)
    
    click.echo(f"Evaluating models from: {artifacts_dir}")
    click.echo(f"Test data: {test_data_path}")
    click.echo(f"Results will be saved to: {output_dir}")
    
    try:
        # Initialize evaluator
        evaluator = MROEvaluator(config)
        
        # Run evaluation pipeline
        results = evaluator.evaluate_pipeline(artifacts_dir, test_data_path, output_dir)
        
        # Print summary
        metrics = results['metrics']
        click.echo("\n" + "="*50)
        click.echo("EVALUATION COMPLETE")
        click.echo("="*50)
        click.echo(f"Precision: {metrics['classification']['precision']:.3f}")
        click.echo(f"Recall: {metrics['classification']['recall']:.3f}")
        click.echo(f"F1 Score: {metrics['classification']['f1_score']:.3f}")
        click.echo(f"AUC-ROC: {metrics['auc_metrics']['auc_roc']:.3f}")
        click.echo(f"Alert Rate: {metrics['classification']['alert_rate']:.1%}")
        click.echo(f"\nResults saved to: {output_dir}")
        click.echo("="*50)
        
    except Exception as e:
        click.echo(f"Error evaluating models: {e}")
        logger.error("Evaluation failed", error=str(e))
        exit(1)


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8002, help='Port to bind to')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.pass_context
def dashboard(ctx, host: str, port: int, debug: bool):
    """Start the AI Model Analyzer Dashboard."""
    click.echo(f"Starting MRO AI Model Analyzer Dashboard...")
    click.echo(f"Host: {host}")
    click.echo(f"Port: {port}")
    click.echo(f"Debug: {debug}")
    click.echo(f"Dashboard URL: http://{host}:{port}")
    
    try:
        import subprocess
        import sys
        
        # Start the dashboard
        subprocess.run([sys.executable, 'dashboard.py'], check=True)
        
    except subprocess.CalledProcessError as e:
        click.echo(f"Error starting dashboard: {e}")
        logger.error("Dashboard startup failed", error=str(e))
        exit(1)
    except KeyboardInterrupt:
        click.echo("\n👋 Dashboard stopped.")


@cli.command()
@click.option('--host', default=None, help='Host to bind to')
@click.option('--port', default=None, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
@click.pass_context
def serve(ctx, host: Optional[str], port: Optional[int], reload: bool):
    """Start the FastAPI server."""
    config = ctx.obj['config']
    
    if host is None:
        host = config['api']['host']
    if port is None:
        port = config['api']['port']
    
    click.echo(f"Starting MRO Anomaly Detection API server...")
    click.echo(f"Host: {host}")
    click.echo(f"Port: {port}")
    click.echo(f"Auto-reload: {reload}")
    click.echo(f"API Documentation: http://{host}:{port}/docs")
    click.echo(f"Health check: http://{host}:{port}/health")
    
    try:
        import uvicorn
        from service.app import app
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload
        )
        
    except Exception as e:
        click.echo(f"Error starting server: {e}")
        logger.error("Server startup failed", error=str(e))
        exit(1)


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--artifacts-dir', default=None, help='Path to trained model artifacts')
@click.option('--output-file', default=None, help='Output file for results')
@click.pass_context
def score_file(ctx, file_path: str, artifacts_dir: Optional[str], output_file: Optional[str]):
    """Score a file containing task records."""
    config = ctx.obj['config']
    
    if artifacts_dir is None:
        artifacts_dir = config['artifacts']['dir']
    
    click.echo(f"Scoring file: {file_path}")
    click.echo(f"Using models from: {artifacts_dir}")
    
    try:
        # Load data
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)
        
        click.echo(f"Loaded {len(df)} records")
        
        # Initialize feature engineer
        feature_engineer = MROFeatureEngineer(config)
        
        # Load feature metadata
        artifacts_path = Path(artifacts_dir)
        feature_metadata_path = artifacts_path / config['artifacts']['models']['feature_metadata']
        
        with open(feature_metadata_path, 'r') as f:
            feature_metadata = json.load(f)
        
        feature_engineer._feature_metadata = feature_metadata
        
        # Apply feature engineering
        X, task_index = feature_engineer.transform(df)
        
        # Initialize model
        model = HybridAnomalyDetector(config)
        model.load_models(artifacts_dir)
        
        # Get predictions
        results = model.predict(X)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'task_id': task_index['task_id'],
            'engine_id': task_index['engine_id'],
            'anomaly_flag': results['anomaly_flag'],
            'anomaly_score': results['anomaly_score'],
            'if_score': results['model_scores']['isolation_forest'],
            'ae_score': results['model_scores']['autoencoder'],
            'if_flag': results['model_flags']['isolation_forest'],
            'ae_flag': results['model_flags']['autoencoder']
        })
        
        # Save results
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"scoring_results_{timestamp}.csv"
        
        results_df.to_csv(output_file, index=False)
        
        # Print summary
        anomaly_count = results['anomaly_flag'].sum()
        anomaly_rate = anomaly_count / len(df)
        
        click.echo("\n" + "="*50)
        click.echo("SCORING COMPLETE")
        click.echo("="*50)
        click.echo(f"Total records: {len(df):,}")
        click.echo(f"Anomalies detected: {anomaly_count:,}")
        click.echo(f"Anomaly rate: {anomaly_rate:.1%}")
        click.echo(f"Results saved to: {output_file}")
        click.echo("="*50)
        
    except Exception as e:
        click.echo(f"Error scoring file: {e}")
        logger.error("File scoring failed", error=str(e))
        exit(1)


@cli.command()
@click.pass_context
def info(ctx):
    """Show system information and configuration."""
    config = ctx.obj['config']
    
    click.echo("MRO Anomaly Detection System Information")
    click.echo("=" * 50)
    
    # Configuration info
    click.echo("\nConfiguration:")
    click.echo(f"  Data output directory: {config['data']['output_dir']}")
    click.echo(f"  Artifacts directory: {config['artifacts']['dir']}")
    click.echo(f"  API host: {config['api']['host']}")
    click.echo(f"  API port: {config['api']['port']}")
    
    # Model info
    click.echo("\nModel Configuration:")
    click.echo(f"  Isolation Forest weight: {config['models']['ensemble']['if_weight']}")
    click.echo(f"  Autoencoder weight: {config['models']['ensemble']['ae_weight']}")
    click.echo(f"  Threshold percentile: {config['models']['ensemble']['threshold_percentile']}")
    
    # Check for data files
    data_dir = Path(config['data']['output_dir'])
    if data_dir.exists():
        parquet_files = list(data_dir.glob('*.parquet'))
        csv_files = list(data_dir.glob('*.csv'))
        click.echo(f"\nData Files:")
        click.echo(f"  Parquet files: {len(parquet_files)}")
        click.echo(f"  CSV files: {len(csv_files)}")
    
    # Check for artifacts
    artifacts_dir = Path(config['artifacts']['dir'])
    if artifacts_dir.exists():
        model_files = list(artifacts_dir.glob('*.pkl')) + list(artifacts_dir.glob('*.h5'))
        click.echo(f"\nModel Artifacts:")
        click.echo(f"  Model files: {len(model_files)}")
    
    click.echo("\n" + "=" * 50)


if __name__ == "__main__":
    cli()
