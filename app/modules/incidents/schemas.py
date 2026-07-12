"""Incidents & Notifications schemas."""

from datetime import datetime

from pydantic import BaseModel, Field


class IncidentCreate(BaseModel):
    """User-reported incident."""

    type: str = Field(..., description="demora | lleno | inseguro | cerrado | accidente")
    lat: float
    lng: float
    route_code: str = Field(default="", description="Affected route code if known")
    description: str = Field(default="", max_length=200)


class IncidentResponse(BaseModel):
    """Stored incident."""

    id: int
    type: str
    lat: float
    lng: float
    route_code: str
    description: str
    created_at: datetime
    source: str = "user_report"
    votes: int = 0


class SystemAlert(BaseModel):
    """System alert from scraping or admin."""

    id: int
    title: str
    type: str  # suspended | delayed | modified
    route_codes: list[str] = []
    source: str = "scraping"  # scraping | admin
    url: str = ""
    created_at: datetime


class NotificationItem(BaseModel):
    """Unified notification for the frontend."""

    id: str
    title: str
    body: str
    type: str  # incident | alert
    severity: str  # info | warning | danger
    lat: float | None = None
    lng: float | None = None
    route_codes: list[str] = []
    created_at: datetime
    source: str
