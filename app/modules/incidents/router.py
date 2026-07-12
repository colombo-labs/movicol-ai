"""Incidents & notifications router."""

from typing import Annotated

from fastapi import APIRouter, Query

from app.modules.incidents.schemas import (
    IncidentCreate,
    IncidentResponse,
    NotificationItem,
)
from app.modules.incidents.service import IncidentsService

router = APIRouter()
service = IncidentsService()


@router.post("/incidents", response_model=IncidentResponse | None)
async def create_incident(incident: IncidentCreate):
    """Report a new incident (demora, lleno, inseguro, cerrado, accidente)."""
    return service.create_incident(incident)


@router.post("/incidents/{incident_id}/vote")
async def vote_incident(incident_id: int):
    """Upvote an incident to confirm it's still active."""
    success = service.vote_incident(incident_id)
    return {"ok": success}


@router.get("/incidents/nearby", response_model=list[IncidentResponse])
async def get_nearby_incidents(
    lat: Annotated[float, Query(description="Latitude")],
    lng: Annotated[float, Query(description="Longitude")],
    radius_km: Annotated[float, Query(le=5.0)] = 1.0,
    hours: Annotated[int, Query(le=24)] = 2,
):
    """Get recent incidents near a location."""
    return service.get_nearby_incidents(lat, lng, radius_km, hours)


@router.get("/notifications", response_model=list[NotificationItem])
async def get_notifications(
    hours: Annotated[int, Query(le=24, description="Hours back")] = 6,
):
    """Get all recent notifications (user incidents + system alerts)."""
    return service.get_notifications(hours=hours)
