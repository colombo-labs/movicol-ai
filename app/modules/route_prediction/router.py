"""Route prediction router — POST /api/v1/predict-route."""

from fastapi import APIRouter, Query
import httpx

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


@router.post("/alternatives", response_model=list[RoutePredictionResponse])
async def predict_route_alternatives(request: RoutePredictionRequest) -> list[RoutePredictionResponse]:
    """Predict multiple route alternatives (vehicle mode)."""
    results = await service.predict_vehicle_alternatives(
        origin=request.origin,
        destination=request.destination,
        departure_time=request.departure_time,
    )
    for r in results:
        r.explanation = generate_explanation(r)
    return results


@router.get("/safety")
async def get_route_safety(
    ruta: str = Query(..., description="SITP route number"),
    hour: int = Query(default=12, ge=0, le=23, description="Hour of day"),
) -> dict:
    """Get safety score for a specific SITP route based on congestion at its stops."""
    return service.get_route_safety(ruta, hour)


@router.get("/alerts")
async def get_system_alerts() -> dict:
    """Scrape operational alerts from TransMilenio official site."""
    import re
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            resp = await client.get("https://www.transmilenio.gov.co")
            html = resp.text

        # Extract operational alert links
        pattern = r'<a[^>]*href="([^"]*(?:modifica|suspende|demora|cambio|cierre)[^"]*)"[^>]*>(.*?)</a>'
        matches = re.findall(pattern, html, re.I | re.DOTALL)
        alerts = []
        for u, content in matches:
            title = re.sub(r'<[^>]+>', '', content).strip()
            if title and len(title) > 10 and "Concesionario" not in title:
                codes = re.findall(r'\b([A-Z]?\d+[-]?\d*)\b', title)
                alerts.append({"title": title[:100], "url": u, "route_codes": codes})
        alerts = alerts[:5]

        # Count by type
        suspended = sum(1 for a in alerts if "suspende" in a["title"].lower())
        delayed = sum(1 for a in alerts if "demora" in a["title"].lower() or "modifica" in a["title"].lower())
        operating = max(0, 125 - suspended - delayed)  # 125 rutas TM total

        return {
            "operating": operating,
            "delayed": delayed,
            "suspended": suspended,
            "alerts": alerts,
        }
    except Exception:
        return {"operating": 125, "delayed": 0, "suspended": 0, "alerts": []}
