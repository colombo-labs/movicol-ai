"""Route prediction request/response schemas — Data Contract v1."""

from pydantic import AliasChoices, BaseModel, Field


class Coordinates(BaseModel):
    """Geographic coordinates."""

    lat: float = Field(..., description="Latitude")
    lng: float = Field(
        ...,
        description="Longitude",
        validation_alias=AliasChoices("lng", "lon"),
        serialization_alias="lng",
    )


class RoutePredictionRequest(BaseModel):
    """Request: predict optimal route between two points."""

    origin: Coordinates
    destination: Coordinates
    departure_time: str = Field(..., description="ISO 8601 datetime")
    mode: str = Field(default="transmilenio", description="transmilenio | sitp | vehiculo")


class RiskSegment(BaseModel):
    """A segment of the route with risk assessment."""

    from_station: str
    to_station: str
    congestion_level: float = Field(..., ge=0, le=1, description="0=free, 1=jammed")
    risk_label: str = Field(..., description="low | medium | high | critical")
    coordinates: list[list[float]] = Field(..., description="[[lat, lng], ...]")


class RoutePredictionResponse(BaseModel):
    """Response: predicted route with risk segments and AI explanation."""

    route_id: str
    total_time_minutes: float
    total_distance_km: float
    cost: str = Field(default="$0", description="Estimated trip cost")
    mode: str = Field(default="transmilenio", description="Transport mode used")
    risk_segments: list[RiskSegment]
    overall_risk: str = Field(..., description="low | medium | high | critical")
    explanation: str = Field(..., description="LLM-generated route explanation")
    stations: list[str] = Field(..., description="Ordered station names in route")
    departure_time: str
