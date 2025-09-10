# MRO Anomaly Detection AI model

by: Sahan Chamuditha Amarasekara Ranasinghe Arachchilage

( 30011917 - MSc Artificial Intelligence | University of South Wales, UK )

## Setup Guide

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate data
# Generate with default 45,000 rows
python cli.py gen-data
# Generate custom ammount
python data/generate_synthetic_mro.py --rows 10000 --anomaly-rate 0.05

# 3. Train models
python cli.py train

# 4. Evaluate model
python cli.py evaluate

# 5. Start dashboard
python dashboard.py
```

Open browser @: http://localhost:8002

## File Structure

```
AI Model/
├── artifacts/                 # Trained models and metadata
│   ├── iforest.pkl          # Isolation Forest model
│   ├── ae.h5                # Autoencoder model
│   ├── scaler.pkl           # Feature scaler
│   └── *.json               # Model metadata files
├── data/
│   ├── synthetic_mro_data.csv # Generated dataset
│   └── generate_synthetic_mro.py
├── evaluation_results/       # Evaluation outputs
│   ├── evaluation_report.json # Performance metrics
│   ├── *.png                 # Visualization charts
│   └── metrics_summary.txt   # Text summary
├── dashboard.py              # Web dashboard
├── service/app.py            # REST API service
├── cli.py                    # Command line interface
└── ml/                       # Machine learning modules
```
