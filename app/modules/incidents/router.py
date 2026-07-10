"""Incidents & notifications router."""

from fastapi import APIRouter, Query

from app.modules.incidents.schemas import IncidentCreate, IncidentResponse, NotificationItem
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
    lat: float = Query(...),
    lng: float = Query(...),
    radius_km: float = Query(default=1.0, le=5.0),
    hours: int = Query(default=2, le=24),
):
    """Get recent incidents near a location."""
    return service.get_nearby_incidents(lat, lng, radius_km, hours)


@router.get("/notifications", response_model=list[NotificationItem])
async def get_notifications(
    hours: int = Query(default=6, le=24, description="How many hours back to look"),
):
    """Get all recent notifications (user incidents + system alerts)."""
    return service.get_notifications(hours=hours)
