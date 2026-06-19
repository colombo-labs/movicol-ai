"""Predictions router - congestion forecasting endpoints."""

from fastapi import APIRouter, HTTPException

from app.modules.predictions.schemas import (
    BatchPredictionRequest,
    PredictionRequest,
    PredictionResponse,
)
from app.modules.predictions.service import PredictionService

router = APIRouter()
service = PredictionService()


@router.post("", response_model=PredictionResponse)
async def predict_station(request: PredictionRequest):
    """Predict congestion for a specific station."""
    if not service.is_loaded:
        raise HTTPException(503, "GNN model not loaded. Place gat_best.pt in models/")
    return service.predict(
        station_id=request.station_id,
        day_of_week=request.day_of_week,
        hour=request.hour,
        horizon_minutes=request.horizon_minutes,
    )


@router.post("/batch", response_model=list[PredictionResponse])
async def predict_all_stations(request: BatchPredictionRequest):
    """Predict congestion for all stations."""
    if not service.is_loaded:
        raise HTTPException(503, "GNN model not loaded.")
    return service.predict_all(
        day_of_week=request.day_of_week,
        hour=request.hour,
        horizon_minutes=request.horizon_minutes,
    )
