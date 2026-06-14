"""Prediction service - GNN inference for station congestion."""

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
        time_factor = self._time_factor(hour)
        congestion = min(1.0, base_congestion * time_factor)

        return PredictionResponse(
            station_id=station_id,
            station_name=station_id,
            congestion_level=round(congestion, 3),
            risk_label=self._risk_label(congestion),
            horizon_minutes=horizon_minutes,
            confidence=0.82 if self._gnn.is_loaded else 0.0,
        )

    def predict_all(
        self, day_of_week: int, hour: int, horizon_minutes: int
    ) -> list[PredictionResponse]:
        """Predict congestion for all stations."""
        all_preds = self._gnn.get_all_predictions()
        time_factor = self._time_factor(hour)

        results = []
        for station_id, base in all_preds.items():
            congestion = min(1.0, base * time_factor)
            results.append(
                PredictionResponse(
                    station_id=station_id,
                    station_name=station_id,
                    congestion_level=round(congestion, 3),
                    risk_label=self._risk_label(congestion),
                    horizon_minutes=horizon_minutes,
                    confidence=0.82,
                )
            )
        return results

    @staticmethod
    def _time_factor(hour: int) -> float:
        """Time-of-day multiplier for congestion."""
        factors = {
            0: 0.3,
            1: 0.2,
            2: 0.2,
            3: 0.2,
            4: 0.3,
            5: 0.5,
            6: 0.7,
            7: 0.9,
            8: 1.0,
            9: 0.9,
            10: 0.7,
            11: 0.65,
            12: 0.75,
            13: 0.7,
            14: 0.65,
            15: 0.7,
            16: 0.8,
            17: 0.95,
            18: 1.0,
            19: 0.9,
            20: 0.7,
            21: 0.5,
            22: 0.4,
            23: 0.3,
        }
        return factors.get(hour, 0.7)

    @staticmethod
    def _risk_label(congestion: float) -> str:
        if congestion < 0.3:
            return "low"
        if congestion < 0.6:
            return "medium"
        if congestion < 0.85:
            return "high"
        return "critical"
