"""Prediction service - GNN inference for station congestion."""

from app.common.congestion import risk_label as _risk_label
from app.common.congestion import time_factor as _time_factor
from app.modules.predictions.gnn_inference import GNNInference
from app.modules.predictions.schemas import PredictionResponse


class PredictionService:
    """Service for congestion predictions using the trained GNN model."""

    def __init__(self) -> None:
        self._gnn = GNNInference()

    @property
    def is_loaded(self) -> bool:
        return self._gnn.is_loaded

    def predict(
        self, station_id: str, day_of_week: int, hour: int, horizon_minutes: int
    ) -> PredictionResponse:
        """Predict congestion for a single station."""
        base_congestion = self._gnn.get_congestion(station_id)

        # Modulate by time of day (peak hours increase congestion)
        time_factor = _time_factor(hour)
        congestion = min(1.0, base_congestion * time_factor)

        return PredictionResponse(
            station_id=station_id,
            station_name=station_id,
            congestion_level=round(congestion, 3),
            risk_label=_risk_label(congestion),
            horizon_minutes=horizon_minutes,
            confidence=0.82 if self._gnn.is_loaded else 0.0,
        )

    def predict_all(
        self, day_of_week: int, hour: int, horizon_minutes: int
    ) -> list[PredictionResponse]:
        """Predict congestion for all stations."""
        all_preds = self._gnn.get_all_predictions()
        time_factor = _time_factor(hour)

        results = []
        for station_id, base in all_preds.items():
            congestion = min(1.0, base * time_factor)
            results.append(
                PredictionResponse(
                    station_id=station_id,
                    station_name=station_id,
                    congestion_level=round(congestion, 3),
                    risk_label=_risk_label(congestion),
                    horizon_minutes=horizon_minutes,
                    confidence=0.82,
                )
            )
        return results
