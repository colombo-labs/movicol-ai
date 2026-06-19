"""Demand prediction schemas."""

from pydantic import BaseModel, Field


class DemandPredictionRequest(BaseModel):
    """Request for demand prediction."""

    hour: int = Field(..., ge=0, le=23, description="Hour of day")
    interval_minutes: int = Field(default=15, description="Prediction interval (15 min)")


class StationDemand(BaseModel):
    """Predicted demand for a station."""

    station_index: int
    station_name: str
    predicted_passengers: float
    demand_level: str = Field(..., description="low | medium | high | critical")


class DemandPredictionResponse(BaseModel):
    """Response with demand predictions."""

    hour: int
    total_stations: int
    predictions: list[StationDemand]
    model: str = "ST-GAT (Spatial-Temporal Graph Attention Network)"
