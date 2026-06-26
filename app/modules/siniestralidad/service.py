"""Siniestralidad risk prediction service.

Uses processed accident data from movicol-data to predict risk
by hour and zone (localidad) in Bogota.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config.settings import get_settings
from app.modules.siniestralidad.schemas import (
    HeatmapPoint,
    RiskByHourResponse,
    SiniestrosStatsResponse,
    ZoneRisk,
)

HOUR_RISK_PROFILE = [
    0.15,
    0.10,
    0.08,
    0.07,
    0.08,
    0.12,
    0.25,
    0.45,
    0.55,
    0.50,
    0.42,
    0.40,
    0.48,
    0.45,
    0.42,
    0.45,
    0.52,
    0.62,
    0.65,
    0.55,
    0.45,
    0.35,
    0.28,
    0.20,
]

DAY_FACTORS = {
    0: 1.05,  # Monday
    1: 1.0,
    2: 1.0,
    3: 1.05,
    4: 1.15,  # Friday
    5: 1.10,  # Saturday
    6: 0.75,  # Sunday
}


def _risk_level(risk: float) -> str:
    if risk < 0.25:
        return "bajo"
    if risk < 0.50:
        return "moderado"
    if risk < 0.75:
        return "alto"
    return "critico"


class SiniestrosService:
    """Siniestralidad risk prediction using processed datos.gov.co data."""

    _instance: SiniestrosService | None = None
    _initialized: bool = False

    def __new__(cls) -> SiniestrosService:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._settings = get_settings()
        self._data: dict = {}
        self._load()
        SiniestrosService._initialized = True

    @property
    def is_loaded(self) -> bool:
        return bool(self._data)

    def _load(self) -> None:
        """Load siniestralidad.json from data directory."""
        candidates = [
            Path(self._settings.graph_path).parent.parent
            / "movicol-data"
            / "exports"
            / "siniestralidad.json",
            Path("../movicol-data/exports/siniestralidad.json"),
            Path("data/siniestralidad.json"),
        ]
        for p in candidates:
            resolved = p.resolve()
            if resolved.exists():
                with open(resolved, encoding="utf-8") as f:
                    self._data = json.load(f)
                return

    def predict_risk_by_hour(self, hour: int, day_of_week: int = 0) -> RiskByHourResponse:
        """Predict risk for all localidades at a given hour and day."""
        por_localidad = self._data.get("por_localidad", {})
        risk_by_hour = self._data.get("risk_by_hour", {})

        base_risk = HOUR_RISK_PROFILE[hour]
        day_factor = DAY_FACTORS.get(day_of_week, 1.0)

        zones = []
        for localidad, loc_data in por_localidad.items():
            hour_risk = risk_by_hour.get(str(hour), {}).get(localidad, {})
            local_risk = hour_risk.get("risk", base_risk)
            adjusted_risk = min(1.0, local_risk * day_factor)

            zones.append(
                ZoneRisk(
                    localidad=localidad,
                    risk=round(adjusted_risk, 3),
                    nivel=_risk_level(adjusted_risk),
                    total_siniestros=loc_data.get("total_siniestros", 0),
                    fatales=loc_data.get("fatales", 0),
                    semaforos=loc_data.get("semaforos", 0),
                )
            )

        zones.sort(key=lambda z: z.risk, reverse=True)
        avg_risk = sum(z.risk for z in zones) / len(zones) if zones else 0
        return RiskByHourResponse(
            hour=hour,
            zones=zones,
            promedio_riesgo=round(avg_risk, 3),
            nivel_general=_risk_level(avg_risk),
        )

    def get_heatmap(self) -> list[HeatmapPoint]:
        """Get heatmap points for map rendering."""
        points = self._data.get("heatmap_points", [])
        return [
            HeatmapPoint(lat=p["lat"], lon=p["lon"], intensity=p.get("intensity", 50))
            for p in points
        ]

    def get_stats(self) -> SiniestrosStatsResponse:
        """Get summary statistics for the panel."""
        vehiculos = self._data.get("vehiculos_stats", {})
        por_localidad = self._data.get("por_localidad", {})
        sectores = self._data.get("sectores_criticos", {})
        semaforos = self._data.get("semaforos", {})

        total_fatales = sum(loc.get("fatales", 0) for loc in por_localidad.values())
        peligrosas = sum(1 for loc in por_localidad.values() if loc.get("nivel") == "peligrosa")

        tipo_vehiculo = vehiculos.get("por_tipo_vehiculo", {})
        top_vehiculos = [{"tipo": k, "total": v} for k, v in list(tipo_vehiculo.items())[:5]]

        return SiniestrosStatsResponse(
            total_siniestros=self._data.get("total_siniestros", 0),
            total_fallecidos=total_fatales,
            sectores_criticos=sectores.get("total", 0),
            semaforos=semaforos.get("total", 0),
            localidades_peligrosas=peligrosas,
            vehiculos_top=top_vehiculos,
            gravedad=vehiculos.get("por_gravedad", {}),
        )
