"""Agent request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AppContext(BaseModel):
    """Rich context from the frontend app state."""

    module: str | None = None  # planificar, rutas, accesibilidad, metricas
    origin: str | None = None  # address or coords
    destination: str | None = None
    origin_coords: list[float] | None = None  # [lat, lon]
    destination_coords: list[float] | None = None
    active_route: str | None = None  # route summary if one is shown
    selected_hour: int | None = None
    transport_mode: str | None = None  # tm, sitp, vehicle, moto
    language: str = "es"


class ChatRequest(BaseModel):
    """Chat message from user."""

    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="default")
    context: AppContext | None = None


class ActionPayload(BaseModel):
    """An action the frontend should execute."""

    type: str  # plan_route, show_station, show_congestion, show_risk, open_module
    data: dict = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """Agent response."""

    response: str
    sources: list[str] = Field(default_factory=list)
    session_id: str
    actions: list[ActionPayload] = Field(default_factory=list)
