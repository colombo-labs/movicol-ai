"""Prediction service - GNN inference logic."""

from pathlib import Path

from app.config.settings import get_settings
from app.modules.predictions.schemas import PredictionResponse


class PredictionService:
    """Service for congestion predictions using the trained GNN model."""

    def __init__(self) -> None:
        self._model = None
        self._settings = get_settings()

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def load_model(self) -> None:
        """Load the trained GNN model from disk."""
        model_path = Path(self._settings.model_path)
        if not model_path.exists():
            return
        # TODO: Load PyTorch Geometric model
        # self._model = torch.load(model_path)

    def predict(
        self, station_id: str, day_of_week: int, hour: int, horizon_minutes: int
    ) -> PredictionResponse:
        """Predict congestion for a single station."""
        # TODO: Implement actual GNN inference
        # Placeholder response
        return PredictionResponse(
            station_id=station_id,
            station_name=f"Station {station_id}",
            congestion_level=0.0,
            risk_label="low",
            horizon_minutes=horizon_minutes,
            confidence=0.0,
        )

    def predict_all(
        self, day_of_week: int, hour: int, horizon_minutes: int
    ) -> list[PredictionResponse]:
        """Predict congestion for all stations."""
        # TODO: Batch inference over all graph nodes
        return []
