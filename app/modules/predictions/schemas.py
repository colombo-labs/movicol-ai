"""Prediction request/response schemas."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request body for congestion prediction."""

    station_id: str = Field(..., description="Station identifier")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday)")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    horizon_minutes: int = Field(default=30, description="Prediction horizon: 15, 30, or 60 min")
    frecuencia_ruta: Optional[int] = Field(default=None, description="Frecuencia base en minutos")
    demanda_actual: Optional[int] = Field(default=None, description="Demanda actual (score 0-100)")


class PredictionResponse(BaseModel):
    """Response with congestion prediction."""

    station_id: str
    station_name: str
    congestion_level: float = Field(..., ge=0, le=1, description="0=empty, 1=full")
    risk_label: str = Field(..., description="low | medium | high | critical")
    horizon_minutes: int
    confidence: float = Field(..., ge=0, le=1)
    tiempo_espera_estimado: Optional[str] = Field(default=None, description="Ej. 5 - 8 minutos")


class BatchPredictionRequest(BaseModel):
    """Request for multiple station predictions."""

    day_of_week: int = Field(..., ge=0, le=6)
    hour: int = Field(..., ge=0, le=23)
    horizon_minutes: int = Field(default=30)
