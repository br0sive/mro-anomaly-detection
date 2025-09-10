"""
FastAPI Service for MRO Anomaly Detection

Real-time anomaly detection API with WebSocket streaming and explainability.
"""

import pandas as pd
import numpy as np
import yaml
import structlog
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

# Import local modules
from ml.features import MROFeatureEngineer
from ml.models import HybridAnomalyDetector

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
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

# Global variables for model and feature engineer
model = None
feature_engineer = None
config = None
websocket_connections = []

# Pydantic models for API
class TaskRecord(BaseModel):
    """Task record for anomaly detection."""
    engine_id: str = Field(..., description="Engine ID")
    module_id: str = Field(..., description="Module ID")
    submodule_id: str = Field(..., description="Submodule ID")
    part_id: str = Field(..., description="Part ID")
    task_id: str = Field(..., description="Task ID")
    wo_id: str = Field(..., description="Work Order ID")
    task_name: str = Field(..., description="Task name")
    bay_id: str = Field(..., description="Bay ID")
    planned_start_ts: str = Field(..., description="Planned start timestamp")
    planned_end_ts: str = Field(..., description="Planned end timestamp")
    actual_start_ts: str = Field(..., description="Actual start timestamp")
    actual_end_ts: str = Field(..., description="Actual end timestamp")
    planned_minutes: int = Field(..., description="Planned duration in minutes")
    actual_minutes: int = Field(..., description="Actual duration in minutes")
    tech_id: str = Field(..., description="Technician ID")
    tech_grade: str = Field(..., description="Technician grade")
    concurrent_tasks: int = Field(..., description="Number of concurrent tasks")
    snag_count: int = Field(..., description="Number of snags")
    snag_severity: int = Field(..., description="Snag severity level")
    tat_hours_engine: float = Field(..., description="Turnaround time hours")
    wip_count_shop: int = Field(..., description="Work in progress count")
    mttr_hours_engine: float = Field(..., description="Mean time to repair hours")
    shift: str = Field(..., description="Shift (D/N)")
    weekday: int = Field(..., description="Weekday (0-6)")

class AnomalyResponse(BaseModel):
    """Response model for anomaly detection."""
    anomaly_flag: bool = Field(..., description="Anomaly detected flag")
    anomaly_score: float = Field(..., description="Anomaly score (0-1)")
    model_scores: Dict[str, float] = Field(..., description="Individual model scores")
    model_flags: Dict[str, bool] = Field(..., description="Individual model flags")
    explainability: Dict[str, List[Dict]] = Field(..., description="Feature explanations")
    timestamp: str = Field(..., description="Processing timestamp")

class BatchAnomalyResponse(BaseModel):
    """Response model for batch anomaly detection."""
    results: List[AnomalyResponse] = Field(..., description="List of anomaly results")
    summary: Dict[str, Any] = Field(..., description="Batch summary")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Model loaded status")
    timestamp: str = Field(..., description="Current timestamp")

class VersionResponse(BaseModel):
    """Version information response."""
    version: str = Field(..., description="API version")
    commit_hash: str = Field(..., description="Git commit hash")
    build_date: str = Field(..., description="Build date")

# Authentication
security = HTTPBearer(auto_error=False)

def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> bool:
    """Verify JWT token (stub implementation)."""
    if not config['api']['auth']['enabled']:
        return True
    
    if config['api']['auth']['bypass_dev']:
        return True
    
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing authentication token")
    
    # TODO: Implement proper JWT validation for production
    # For now, accept any token
    return True

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting MRO Anomaly Detection API")
    await load_models()
    yield
    # Shutdown
    logger.info("Shutting down MRO Anomaly Detection API")

# Create FastAPI app
app = FastAPI(
    title="MRO Anomaly Detection API",
    description="Real-time anomaly detection for aircraft engine maintenance workflows",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config['api']['cors']['origins'] if config else ["*"],
    allow_credentials=True,
    allow_methods=config['api']['cors']['methods'] if config else ["*"],
    allow_headers=config['api']['cors']['headers'] if config else ["*"],
)

async def load_models():
    """Load trained models and feature engineer."""
    global model, feature_engineer, config
    
    try:
        # Load configuration
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize model
        model = HybridAnomalyDetector(config)
        
        # Load models
        artifacts_dir = config['artifacts']['dir']
        model.load_models(artifacts_dir)
        
        # Load feature engineer
        feature_engineer = MROFeatureEngineer(config)
        
        # Load feature metadata
        artifacts_path = Path(artifacts_dir)
        feature_metadata_path = artifacts_path / config['artifacts']['models']['feature_metadata']
        
        with open(feature_metadata_path, 'r') as f:
            feature_metadata = json.load(f)
        
        feature_engineer._feature_metadata = feature_metadata
        
        logger.info("Models loaded successfully")
        
    except Exception as e:
        logger.error("Failed to load models", error=str(e))
        raise

def process_single_task(task_record: TaskRecord) -> AnomalyResponse:
    """Process a single task record for anomaly detection."""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([task_record.dict()])
        
        # Apply feature engineering
        X, task_index = feature_engineer.transform(df)
        
        # Get prediction
        result = model.predict_single(X)
        
        # Create response
        response = AnomalyResponse(
            anomaly_flag=bool(result['anomaly_flag']),
            anomaly_score=result['anomaly_score'],
            model_scores=result['model_scores'],
            model_flags={
                'isolation_forest': bool(result['model_flags']['isolation_forest']),
                'autoencoder': bool(result['model_flags']['autoencoder'])
            },
            explainability=result['explainability'],
            timestamp=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        logger.error("Error processing task", error=str(e), task_id=task_record.task_id)
        raise HTTPException(status_code=500, detail=f"Error processing task: {str(e)}")

async def broadcast_anomaly_event(event_data: Dict):
    """Broadcast anomaly event to all connected WebSocket clients."""
    if websocket_connections:
        message = json.dumps(event_data)
        await asyncio.gather(
            *[connection.send_text(message) for connection in websocket_connections],
            return_exceptions=True
        )

# API Endpoints
@app.post("/score", response_model=AnomalyResponse)
async def score_task(
    task: TaskRecord,
    authenticated: bool = Depends(verify_token)
):
    """Score a single task for anomalies."""
    logger.info("Scoring task", task_id=task.task_id)
    
    # Process task
    result = process_single_task(task)
    
    # Broadcast anomaly event if detected
    if result.anomaly_flag:
        await broadcast_anomaly_event({
            "type": "anomaly_detected",
            "task_id": task.task_id,
            "engine_id": task.engine_id,
            "anomaly_score": result.anomaly_score,
            "timestamp": result.timestamp
        })
    
    return result

@app.post("/score/batch", response_model=BatchAnomalyResponse)
async def score_batch(
    tasks: List[TaskRecord],
    authenticated: bool = Depends(verify_token)
):
    """Score multiple tasks for anomalies."""
    logger.info("Scoring batch", batch_size=len(tasks))
    
    results = []
    anomaly_count = 0
    
    for task in tasks:
        result = process_single_task(task)
        results.append(result)
        
        if result.anomaly_flag:
            anomaly_count += 1
    
    # Create summary
    summary = {
        "total_tasks": len(tasks),
        "anomalies_detected": anomaly_count,
        "anomaly_rate": anomaly_count / len(tasks),
        "timestamp": datetime.now().isoformat()
    }
    
    return BatchAnomalyResponse(results=results, summary=summary)

@app.post("/train")
async def retrain_models(
    authenticated: bool = Depends(verify_token)
):
    """Retrain models from data (optional endpoint)."""
    try:
        from ml.train import train_from_config
        
        logger.info("Starting model retraining")
        
        # Train models
        summary = train_from_config()
        
        # Reload models
        await load_models()
        
        return {
            "status": "success",
            "message": "Models retrained successfully",
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Retraining failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat()
    )

@app.get("/version", response_model=VersionResponse)
async def version():
    """Get API version information."""
    return VersionResponse(
        version="1.0.0",
        commit_hash="dev",
        build_date=datetime.now().isoformat()
    )

# WebSocket endpoint
@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time anomaly events."""
    await websocket.accept()
    websocket_connections.append(websocket)
    
    logger.info("WebSocket client connected", total_connections=len(websocket_connections))
    
    try:
        # Send welcome message
        await websocket.send_text(json.dumps({
            "type": "connected",
            "message": "Connected to MRO Anomaly Detection stream",
            "timestamp": datetime.now().isoformat()
        }))
        
        # Keep connection alive
        while True:
            # Wait for client messages (ping/pong)
            data = await websocket.receive_text()
            
            # Echo back for heartbeat
            await websocket.send_text(json.dumps({
                "type": "pong",
                "timestamp": datetime.now().isoformat()
            }))
            
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)
        logger.info("WebSocket client disconnected", total_connections=len(websocket_connections))
    except Exception as e:
        logger.error("WebSocket error", error=str(e))
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)

# Additional utility endpoints
@app.get("/models/info")
async def model_info(
    authenticated: bool = Depends(verify_token)
):
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "ensemble_weights": {
            "isolation_forest": config['models']['ensemble']['if_weight'],
            "autoencoder": config['models']['ensemble']['ae_weight']
        },
        "thresholds": config['thresholds'],
        "feature_count": len(feature_engineer.feature_names()) if feature_engineer else 0,
        "model_type": "hybrid_anomaly_detection"
    }

@app.get("/features")
async def feature_info(
    authenticated: bool = Depends(verify_token)
):
    """Get feature information."""
    if feature_engineer is None:
        raise HTTPException(status_code=503, detail="Feature engineer not loaded")
    
    return {
        "feature_names": feature_engineer.feature_names(),
        "feature_metadata": feature_engineer.feature_metadata(),
        "total_features": len(feature_engineer.feature_names())
    }

if __name__ == "__main__":
    # Load configuration for CLI
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Run server
    uvicorn.run(
        "service.app:app",
        host=config['api']['host'],
        port=config['api']['port'],
        reload=config['api']['debug']
    )
