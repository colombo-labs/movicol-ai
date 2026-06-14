"""Demand prediction service."""

from app.modules.demand_prediction.schemas import DemandPredictionResponse, StationDemand
from app.modules.demand_prediction.st_gat_inference import DemandInference


def _demand_level(passengers: float, max_passengers: float) -> str:
    ratio = passengers / max_passengers if max_passengers > 0 else 0
    if ratio < 0.25:
        return "low"
    if ratio < 0.5:
        return "medium"
    if ratio < 0.75:
        return "high"
    return "critical"


class DemandPredictionService:
    """Service for passenger demand predictions using ST-GAT."""

    def __init__(self) -> None:
        self._inference = DemandInference()

    @property
    def is_loaded(self) -> bool:
        return self._inference.is_loaded

    def predict_demand(self, hour: int) -> DemandPredictionResponse:
        """Predict passenger demand for all stations at given hour."""
        preds = self._inference.predict_demand()
        max_val = max(preds.values()) if preds else 1.0
        station_names = self._inference._station_names

        stations = []
        for idx, passengers in sorted(preds.items(), key=lambda x: x[1], reverse=True):
            name = station_names[idx] if idx < len(station_names) else f"Estación {idx}"
            stations.append(StationDemand(
                station_index=idx,
                station_name=name,
                predicted_passengers=round(passengers, 1),
                demand_level=_demand_level(passengers, max_val),
            ))

        return DemandPredictionResponse(
            hour=hour,
            total_stations=len(stations),
            predictions=stations,
        )
