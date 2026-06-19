"""Route prediction router — POST /api/v1/predict-route."""

from fastapi import APIRouter

from app.modules.route_prediction.explainer import generate_explanation
from app.modules.route_prediction.schemas import (
    RoutePredictionRequest,
    RoutePredictionResponse,
)
from app.modules.route_prediction.service import RoutePredictionService

router = APIRouter()
service = RoutePredictionService()


@router.post("", response_model=RoutePredictionResponse)
async def predict_route(request: RoutePredictionRequest) -> RoutePredictionResponse:
    """Predict optimal route with congestion risk segments."""
    prediction = await service.predict_route(
        origin=request.origin,
        destination=request.destination,
        departure_time=request.departure_time,
        mode=request.mode,
    )

    # Generate LLM explanation (or template fallback)
    prediction.explanation = generate_explanation(prediction)

    return prediction
