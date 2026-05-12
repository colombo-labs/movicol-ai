"""Prediction request/response schemas."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request body for congestion prediction."""

    station_id: str = Field(..., description="Station identifier")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday)")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    horizon_minutes: int = Field(default=30, description="Prediction horizon: 15, 30, or 60 min")


class PredictionResponse(BaseModel):
    """Response with congestion prediction."""

    station_id: str
    station_name: str
    congestion_level: float = Field(..., ge=0, le=1, description="0=empty, 1=full")
    risk_label: str = Field(..., description="low | medium | high | critical")
    horizon_minutes: int
    confidence: float = Field(..., ge=0, le=1)


class BatchPredictionRequest(BaseModel):
    """Request for multiple station predictions."""

    day_of_week: int = Field(..., ge=0, le=6)
    hour: int = Field(..., ge=0, le=23)
    horizon_minutes: int = Field(default=30)
