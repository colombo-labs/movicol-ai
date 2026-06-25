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
        self,
        station_id: str,
        hour: int,
        horizon_minutes: int,
        frecuencia_ruta: int | None = None,
        demanda_actual: int | None = None,
    ) -> PredictionResponse:
        """Predict congestion for a single station."""
        base_congestion = self._gnn.get_congestion(station_id)

        # Modulate by time of day (peak hours increase congestion)
        time_factor = _time_factor(hour)
        congestion = min(1.0, base_congestion * time_factor)

        # Calcular tiempo de espera si hay datos
        tiempo_espera = None
        if frecuencia_ruta is not None:
            demanda = (demanda_actual or 0) / 100.0
            # Combina demanda y congestión predicha
            factor_retraso = max(demanda, congestion)
            base_min = max(2, int(frecuencia_ruta * 0.5))
            base_max = max(5, int(frecuencia_ruta))
            extra_wait = int(factor_retraso * frecuencia_ruta)

            min_wait = base_min + extra_wait
            max_wait = base_max + extra_wait
            tiempo_espera = f"{min_wait} - {max_wait} min"

        return PredictionResponse(
            station_id=station_id,
            station_name=station_id,
            congestion_level=round(congestion, 3),
            risk_label=_risk_label(congestion),
            horizon_minutes=horizon_minutes,
            confidence=0.82 if self._gnn.is_loaded else 0.0,
            tiempo_espera_estimado=tiempo_espera,
        )

    def predict_all(self, hour: int, horizon_minutes: int) -> list[PredictionResponse]:
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
