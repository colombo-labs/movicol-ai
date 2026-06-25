"""Route prediction router — POST /api/v1/predict-route."""

from typing import Annotated

import httpx
from fastapi import APIRouter, Query

from app.modules.route_prediction.explainer import generate_explanation
from app.modules.route_prediction.schemas import (
    RoutePredictionRequest,
    RoutePredictionResponse,
)
from app.modules.route_prediction.service import RoutePredictionService

router = APIRouter()
service = RoutePredictionService()


@router.post("")
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


@router.post("/alternatives")
async def predict_route_alternatives(
    request: RoutePredictionRequest,
) -> list[RoutePredictionResponse]:
    """Predict multiple route alternatives (vehicle mode)."""
    mode = request.mode or "vehiculo"
    # Bici/caminando use ORS (single route, no alternatives)
    if mode in ("bicicleta", "caminando"):
        result = await service.predict_route(
            origin=request.origin,
            destination=request.destination,
            departure_time=request.departure_time,
            mode=mode,
        )
        result.explanation = generate_explanation(result)
        return [result]
    # Carro/moto use OSRM with alternatives
    profile_map = {"vehiculo": "driving", "moto": "driving"}
    cost_map = {"vehiculo": 2000, "moto": 1200}
    results = await service.predict_vehicle_alternatives(
        origin=request.origin,
        destination=request.destination,
        departure_time=request.departure_time,
        profile=profile_map.get(mode, "driving"),
        mode_name=mode,
        cost_per_km=cost_map.get(mode, 2000),
    )
    for r in results:
        r.explanation = generate_explanation(r)
    return results

    for r in results:
        r.explanation = generate_explanation(r)
    return results


@router.get("/safety")
async def get_route_safety(
    ruta: Annotated[str, Query(description="SITP route number")],
    hour: Annotated[int, Query(ge=0, le=23, description="Hour of day")] = 12,
) -> dict:
    """Get safety score for a specific SITP route based on congestion at its stops."""
    return service.get_route_safety(ruta, hour)


@router.get("/alerts")
async def get_system_alerts() -> dict:
    """Scrape operational alerts from TransMilenio official site."""
    from html.parser import HTMLParser

    class AlertParser(HTMLParser):
        """Extract alert links from TransMilenio HTML."""

        def __init__(self):
            super().__init__()
            self.alerts: list[dict] = []
            self._current_href = ""
            self._in_link = False
            self._text = ""
            self._keywords = ("modifica", "suspende", "demora", "cambio", "cierre")

        def handle_starttag(self, tag, attrs):
            if tag == "a":
                href = dict(attrs).get("href", "")
                if any(k in href.lower() for k in self._keywords):
                    self._current_href = href
                    self._in_link = True
                    self._text = ""

        def handle_data(self, data):
            if self._in_link:
                self._text += data

        def handle_endtag(self, tag):
            if tag == "a" and self._in_link:
                title = self._text.strip()
                if title and len(title) > 10 and "Concesionario" not in title:
                    import re

                    codes = re.findall(r"\b([A-Z]?\d+-?\d*)\b", title)
                    self.alerts.append(
                        {"title": title[:100], "url": self._current_href, "route_codes": codes}
                    )
                self._in_link = False

    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True, max_redirects=3) as client:
            resp = await client.get("https://www.transmilenio.gov.co")
            html = resp.text

        parser = AlertParser()
        parser.feed(html)
        alerts = parser.alerts[:5]

        # Count by type
        suspended = sum(1 for a in alerts if "suspende" in a["title"].lower())
        delayed = sum(
            1 for a in alerts if "demora" in a["title"].lower() or "modifica" in a["title"].lower()
        )
        operating = max(0, 125 - suspended - delayed)  # 125 rutas TM total

        return {
            "operating": operating,
            "delayed": delayed,
            "suspended": suspended,
            "alerts": alerts,
        }
    except Exception:
        return {"operating": 125, "delayed": 0, "suspended": 0, "alerts": []}
