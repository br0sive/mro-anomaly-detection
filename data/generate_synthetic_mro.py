"""
Synthetic MRO Data Generator for Aircraft Engine Maintenance

Generates realistic workflow data with controled anomalies for training
and evaluating anomaly detection models.
"""

import pandas as pd
import numpy as np
import yaml
import click
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import structlog
from pathlib import Path
from tqdm import tqdm

logger = structlog.get_logger()


class MRODataGenerator:
    """Generates synthetic MRO workflow data with controled anomalies."""
    
    def __init__(self, config: Dict):
        """Initialize the data generator with configuration."""
        self.config = config
        self.data_config = config['data']
        self.rng = np.random.RandomState(self.data_config['seed'])
        
        # Initialize entity mappings
        self.engines = [f"E{i:03d}" for i in range(1, self.data_config['engines'] + 1)]
        self.modules = [f"M{i:03d}" for i in range(1, len(self.engines) * self.data_config['modules_per_engine'] + 1)]
        self.submodules = [f"S{i:03d}" for i in range(1, len(self.modules) * self.data_config['submodules_per_module'] + 1)]
        self.parts = [f"P{i:03d}" for i in range(1, len(self.submodules) * self.data_config['parts_per_submodule'] + 1)]
        self.technicians = [f"T{i:03d}" for i in range(1, 101)]  # 100 technicians
        self.bays = [f"B{i:03d}" for i in range(1, 21)]  # 20 bays
        
        # Task name mapping
        self.task_names = self.data_config['task_names']
        self.tech_grades = self.data_config['tech_grades']
        
        # Time settings
        self.start_date = datetime.strptime(self.data_config['start_date'], "%Y-%m-%d")
        self.date_range = timedelta(days=self.data_config['date_range_days'])
        
    def generate_serial_numbers(self, count: int) -> List[str]:
        """Generate realistic alphanumeric serial numbers."""
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return [''.join(self.rng.choice(list(chars), size=8)) for _ in range(count)]
    
    def generate_normal_task(self, task_id: int, engine_id: str, module_id: str, 
                           submodule_id: str, part_id: str) -> Dict:
        """Generate a normal task record."""
        # Base planned duration (minutes)
        base_duration = self.rng.choice([60, 90, 120, 180, 240, 360])
        
        # Add some variance (±15-25%)
        variance = self.rng.uniform(0.75, 1.25)
        planned_minutes = int(base_duration * variance)
        
        # Actual duration with some realistic variance
        actual_variance = self.rng.uniform(0.8, 1.3)
        actual_minutes = int(planned_minutes * actual_variance)
        
        # Time scheduling
        task_date = self.start_date + timedelta(
            days=self.rng.randint(0, self.data_config['date_range_days'])
        )
        
        planned_start = task_date + timedelta(hours=self.rng.randint(6, 18))
        planned_end = planned_start + timedelta(minutes=planned_minutes)
        
        # Actual times with some realistic delays
        start_delay = self.rng.randint(0, 30)  # 0-30 min delay
        actual_start = planned_start + timedelta(minutes=start_delay)
        actual_end = actual_start + timedelta(minutes=actual_minutes)
        
        # Technician assignment
        tech_id = self.rng.choice(self.technicians)
        tech_grade = self.rng.choice(self.tech_grades, p=[0.2, 0.5, 0.3])  # A:20%, B:50%, C:30%
        concurrent_tasks = self.rng.randint(1, 4)
        
        # Snags (most tasks have few or no snags)
        snag_count = self.rng.poisson(0.5)  # Poisson distribution
        snag_severity = self.rng.choice([0, 1, 2, 3], p=[0.7, 0.2, 0.08, 0.02])
        
        # KPIs (simulated rolling metrics)
        tat_hours_engine = self.rng.uniform(24, 72)
        wip_count_shop = self.rng.randint(5, 25)
        mttr_hours_engine = self.rng.uniform(4, 12)
        
        # Context
        shift = self.rng.choice(['D', 'N'])
        weekday = task_date.weekday()
        
        return {
            'task_id': f"WO{task_id:06d}",
            'wo_id': f"WO{task_id:06d}",
            'engine_id': engine_id,
            'module_id': module_id,
            'submodule_id': submodule_id,
            'part_id': part_id,
            'serial_no': self.rng.choice(self.generate_serial_numbers(1000)),
            'task_name': self.rng.choice(self.task_names),
            'bay_id': self.rng.choice(self.bays),
            'planned_start_ts': planned_start.isoformat(),
            'planned_end_ts': planned_end.isoformat(),
            'actual_start_ts': actual_start.isoformat(),
            'actual_end_ts': actual_end.isoformat(),
            'planned_minutes': planned_minutes,
            'actual_minutes': actual_minutes,
            'tech_id': tech_id,
            'tech_grade': tech_grade,
            'concurrent_tasks': concurrent_tasks,
            'snag_count': snag_count,
            'snag_severity': snag_severity,
            'tat_hours_engine': tat_hours_engine,
            'wip_count_shop': wip_count_shop,
            'mttr_hours_engine': mttr_hours_engine,
            'shift': shift,
            'weekday': weekday,
            'is_anomaly': 0
        }
    
    def generate_anomalous_task(self, task_id: int, engine_id: str, module_id: str,
                               submodule_id: str, part_id: str, anomaly_type: str) -> Dict:
        """Generate an anomalous task record with more distinctive patterns."""
        base_task = self.generate_normal_task(task_id, engine_id, module_id, submodule_id, part_id)
        
        if anomaly_type == "duration_outlier":
            # Extreme duration outlier (+200-500%)
            multiplier = self.rng.uniform(3.0, 6.0)
            base_task['actual_minutes'] = int(base_task['planned_minutes'] * multiplier)
            base_task['actual_end_ts'] = (
                datetime.fromisoformat(base_task['actual_start_ts']) + 
                timedelta(minutes=base_task['actual_minutes'])
            ).isoformat()
            # Add extreme delays
            base_task['start_lag_minutes'] = self.rng.randint(60, 300)
            base_task['end_lag_minutes'] = self.rng.randint(120, 600)
            
        elif anomaly_type == "technician_overload":
            # Severe technician overload
            base_task['concurrent_tasks'] = self.rng.randint(5, 8)  # More extreme
            base_task['actual_minutes'] = int(base_task['planned_minutes'] * self.rng.uniform(2.0, 4.0))
            base_task['actual_end_ts'] = (
                datetime.fromisoformat(base_task['actual_start_ts']) + 
                timedelta(minutes=base_task['actual_minutes'])
            ).isoformat()
            # Assign lower grade technician to complex task
            base_task['tech_grade'] = self.rng.choice(['C', 'B'], p=[0.7, 0.3])
            
        elif anomaly_type == "snag_spike":
            # Severe snag spike with high severity
            base_task['snag_count'] = self.rng.randint(8, 20)  # More extreme
            base_task['snag_severity'] = self.rng.choice([3, 2], p=[0.6, 0.4])  # Mostly high severity
            base_task['actual_minutes'] = int(base_task['planned_minutes'] * self.rng.uniform(2.5, 5.0))
            # Add severe delays due to snags
            base_task['start_lag_minutes'] = self.rng.randint(30, 180)
            
        elif anomaly_type == "parts_wait":
            # Severe parts shortage causing extreme delays
            delay_hours = self.rng.randint(8, 48)  # More extreme delays
            base_task['actual_start_ts'] = (
                datetime.fromisoformat(base_task['planned_start_ts']) + 
                timedelta(hours=delay_hours)
            ).isoformat()
            base_task['actual_end_ts'] = (
                datetime.fromisoformat(base_task['actual_start_ts']) + 
                timedelta(minutes=base_task['actual_minutes'])
            ).isoformat()
            # High WIP due to parts shortage
            base_task['wip_count_shop'] = self.rng.randint(20, 35)
            
        elif anomaly_type == "submodule_cluster":
            # Multiple tasks under same module severely delayed
            base_task['actual_minutes'] = int(base_task['planned_minutes'] * self.rng.uniform(2.0, 4.0))
            base_task['actual_end_ts'] = (
                datetime.fromisoformat(base_task['actual_start_ts']) + 
                timedelta(minutes=base_task['actual_minutes'])
            ).isoformat()
            # Add module-specific issues
            base_task['snag_count'] = self.rng.randint(3, 8)
            base_task['snag_severity'] = self.rng.choice([2, 3], p=[0.4, 0.6])
            
        elif anomaly_type == "kpi_anomaly":
            # KPI-related anomalies
            base_task['tat_hours_engine'] = self.rng.uniform(100, 200)  # Very high TAT
            base_task['mttr_hours_engine'] = self.rng.uniform(20, 50)   # Very high MTTR
            base_task['wip_count_shop'] = self.rng.randint(25, 40)    # Very high WIP
            base_task['actual_minutes'] = int(base_task['planned_minutes'] * self.rng.uniform(1.8, 3.0))
            
        elif anomaly_type == "temporal_anomaly":
            # Temporal anomalies (off-hours, extreme timing)
            base_task['hour_of_day'] = self.rng.choice([0, 1, 2, 3, 22, 23])  # Off-hours
            base_task['day_of_week'] = self.rng.choice([5, 6])  # Weekend
            base_task['actual_minutes'] = int(base_task['planned_minutes'] * self.rng.uniform(1.5, 2.5))
            # Assign night shift technician
            base_task['shift'] = 'N'
        
        base_task['is_anomaly'] = 1
        return base_task
    
    def generate_dataset(self, num_rows: int, anomaly_rate: float) -> pd.DataFrame:
        """Generate the complete dataset."""
        logger.info("Starting dataset generation", rows=num_rows, anomaly_rate=anomaly_rate)
        
        data = []
        anomaly_types = ["duration_outlier", "technician_overload", "snag_spike", "parts_wait", "submodule_cluster", "kpi_anomaly", "temporal_anomaly"]
        
        num_anomalies = int(num_rows * anomaly_rate)
        num_normal = num_rows - num_anomalies
        
        print(f"\nGenerating {num_rows:,} records ({num_normal:,} normal + {num_anomalies:,} anomalies)...")
        
        # Generate normal tasks with progress bar
        print("Generating normal tasks...")
        for i in tqdm(range(num_normal), desc="Normal tasks", unit="tasks"):
            engine_id = self.rng.choice(self.engines)
            module_id = self.rng.choice(self.modules)
            submodule_id = self.rng.choice(self.submodules)
            part_id = self.rng.choice(self.parts)
            
            task = self.generate_normal_task(i, engine_id, module_id, submodule_id, part_id)
            data.append(task)
        
        # Generate anomalous tasks with progress bar
        if num_anomalies > 0:
            print("Generating anomalous tasks...")
            for i in tqdm(range(num_normal, num_rows), desc="Anomalous tasks", unit="tasks"):
                engine_id = self.rng.choice(self.engines)
                module_id = self.rng.choice(self.modules)
                submodule_id = self.rng.choice(self.submodules)
                part_id = self.rng.choice(self.parts)
                anomaly_type = self.rng.choice(anomaly_types)
                
                task = self.generate_anomalous_task(i, engine_id, module_id, submodule_id, part_id, anomaly_type)
                data.append(task)
        
        # Shuffle the data
        print("Shuffling data...")
        self.rng.shuffle(data)
        
        df = pd.DataFrame(data)
        
        # Add some submodule clustering for anomalies
        if num_anomalies > 0:
            print("Adding clustered anomalies...")
            cluster_size = min(5, num_anomalies // 4)
            for _ in tqdm(range(num_anomalies // cluster_size), desc="Clustering", unit="clusters"):
                cluster_module = self.rng.choice(self.modules)
                cluster_submodule = self.rng.choice(self.submodules)
                
                for j in range(cluster_size):
                    if len(df) < num_rows:
                        task = self.generate_anomalous_task(
                            len(df), 
                            self.rng.choice(self.engines),
                            cluster_module,
                            cluster_submodule,
                            self.rng.choice(self.parts),
                            "submodule_cluster"
                        )
                        data.append(task)
        
        df = pd.DataFrame(data)
        df = df.head(num_rows)  # Ensure exact row count
        
        logger.info("Dataset generation complete", 
                   total_rows=len(df), 
                   anomalies=df['is_anomaly'].sum(),
                   anomaly_rate=df['is_anomaly'].mean())
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, output_dir: str, filename: str = None):
        """Save dataset to CSV and Parquet formats."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mro_data_{timestamp}"
        
        # Save as CSV
        csv_path = output_path / f"{filename}.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Saved CSV dataset", path=str(csv_path), rows=len(df))
        
        # Save as Parquet
        parquet_path = output_path / f"{filename}.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info("Saved Parquet dataset", path=str(parquet_path), rows=len(df))
        
        return csv_path, parquet_path


@click.command()
@click.option('--rows', default=50000, help='Number of rows to generate')
@click.option('--anomaly-rate', default=0.07, help='Anomaly rate (0.0-1.0)')
@click.option('--seed', default=42, help='Random seed')
@click.option('--output-dir', default='data/out', help='Output directory')
@click.option('--config', default='config.yaml', help='Configuration file')
def main(rows: int, anomaly_rate: float, seed: int, output_dir: str, config: str):
    """Generate synthetic MRO data with controlled anomalies."""
    # Load configuration
    with open(config, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Override config with CLI parameters
    config_data['data']['seed'] = seed
    config_data['data']['default_rows'] = rows
    config_data['data']['default_anomaly_rate'] = anomaly_rate
    config_data['data']['output_dir'] = output_dir
    
    # Initialize generator
    generator = MRODataGenerator(config_data)
    
    # Generate dataset
    df = generator.generate_dataset(rows, anomaly_rate)
    
    # Save dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mro_data_{timestamp}"
    csv_path, parquet_path = generator.save_dataset(df, output_dir, filename)
    
    # Print summary
    print(f"\nDataset Generation Complete!")
    print(f"Total rows: {len(df):,}")
    print(f"Anomalies: {df['is_anomaly'].sum():,} ({df['is_anomaly'].mean():.1%})")
    print(f"Engines: {df['engine_id'].nunique()}")
    print(f"Modules: {df['module_id'].nunique()}")
    print(f"Technicians: {df['tech_id'].nunique()}")
    print(f"Date range: {df['planned_start_ts'].min()} to {df['planned_start_ts'].max()}")
    print(f"\nFiles saved:")
    print(f"  CSV: {csv_path}")
    print(f"  Parquet: {parquet_path}")


if __name__ == "__main__":
    main()
