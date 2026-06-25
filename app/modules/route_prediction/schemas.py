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
    mode: str = Field(
        default="transmilenio", description="Transport mode: walk, sitp, transmilenio"
    )


class NavigationStep(BaseModel):
    """A turn-by-turn navigation instruction."""

    instruction: str = Field(..., description="Human-readable instruction")
    street: str = Field(default="", description="Street name")
    distance_m: int = Field(default=0, description="Distance in meters")
    duration_s: int = Field(default=0, description="Duration in seconds")
    maneuver: str = Field(
        default="straight", description="turn type: left, right, straight, depart, arrive"
    )


class RoutePredictionResponse(BaseModel):
    """Response: predicted route with risk segments and AI explanation."""

    route_id: str
    total_time_minutes: float
    total_distance_km: float
    cost: str = Field(default="$0", description="Estimated trip cost")
    mode: str = Field(default="transmilenio", description="Transport mode used")
    risk_segments: list[RiskSegment]
    overall_risk: str = Field(..., description="low | medium | high | critical")
    safety_score: int = Field(
        default=75, ge=0, le=100, description="Road safety score 0-100 (100=safest)"
    )
    explanation: str = Field(..., description="LLM-generated route explanation")
    stations: list[str] = Field(..., description="Ordered station names in route")
    departure_time: str
    route_code: str = Field(
        default="", description="Short route identifier (e.g. '1', 'J74', '18-3')"
    )
    navigation_steps: list[NavigationStep] = Field(
        default=[], description="Turn-by-turn navigation"
    )
    transfers: int = Field(default=0, description="Number of transfers between routes")
    estimated_wait_minutes: float = Field(
        default=0.0, description="Estimated wait time at origin stop (minutes)"
    )
    alternatives: list["RoutePredictionResponse"] = Field(
        default=[], description="2-3 alternative routes"
    )
