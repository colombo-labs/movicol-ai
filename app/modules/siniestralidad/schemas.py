"""Siniestralidad prediction schemas."""

from pydantic import BaseModel, Field


class ZoneRisk(BaseModel):
    """Risk prediction for a zone at a given hour."""

    localidad: str
    risk: float = Field(..., ge=0, le=1, description="Risk score 0-1")
    nivel: str = Field(..., description="bajo | moderado | alto | critico")
    total_siniestros: int = 0
    fatales: int = 0
    semaforos: int = 0


class RiskByHourResponse(BaseModel):
    """Risk prediction response for all zones at a given hour."""

    hour: int
    zones: list[ZoneRisk]
    promedio_riesgo: float
    nivel_general: str


class HeatmapPoint(BaseModel):
    """A single heatmap point for siniestralidad."""

    lat: float
    lon: float
    intensity: float


class SiniestrosStatsResponse(BaseModel):
    """Summary statistics for the siniestralidad panel."""

    total_siniestros: int
    total_fallecidos: int
    sectores_criticos: int
    semaforos: int
    localidades_peligrosas: int
    vehiculos_top: list[dict]
    gravedad: dict
