"""Siniestralidad prediction router."""

from fastapi import APIRouter, HTTPException, Query

from app.modules.siniestralidad.schemas import (
    RiskByHourResponse,
    SiniestrosStatsResponse,
)
from app.modules.siniestralidad.service import SiniestrosService

router = APIRouter()
service = SiniestrosService()


@router.get(
    "/risk",
    response_model=RiskByHourResponse,
    responses={503: {"description": "Data not loaded"}},
)
async def predict_risk(
    hour: int = Query(..., ge=0, le=23, description="Hour of day"),
    day: int = Query(0, ge=0, le=6, description="Day of week (0=Mon)"),
):
    """Predict accident risk by zone for a given hour and day."""
    if not service.is_loaded:
        raise HTTPException(503, "Siniestralidad data not loaded")
    return service.predict_risk_by_hour(hour, day)


@router.get("/heatmap")
async def get_heatmap():
    """Get heatmap points for accident visualization."""
    if not service.is_loaded:
        raise HTTPException(503, "Siniestralidad data not loaded")
    return service.get_heatmap()


@router.get("/stats", response_model=SiniestrosStatsResponse)
async def get_stats():
    """Get summary statistics for siniestralidad panel."""
    if not service.is_loaded:
        raise HTTPException(503, "Siniestralidad data not loaded")
    return service.get_stats()
