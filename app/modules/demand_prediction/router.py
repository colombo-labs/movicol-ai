"""Demand prediction router — ST-GAT passenger forecasting."""

from fastapi import APIRouter, HTTPException, Query

from app.modules.demand_prediction.schemas import DemandPredictionResponse
from app.modules.demand_prediction.service import DemandPredictionService

router = APIRouter()
service = DemandPredictionService()


@router.get("", response_model=DemandPredictionResponse)
async def predict_demand(hour: int = Query(..., ge=0, le=23, description="Hour of day")):
    """Predict passenger demand for all TM stations at a given hour."""
    if not service.is_loaded:
        raise HTTPException(
            503, "ST-GAT model not loaded. Place st_gat_transmilenio_optimizado.pth in models/"
        )
    return service.predict_demand(hour)
