"""Graph query schemas."""

from pydantic import BaseModel, Field


class StationResponse(BaseModel):
    """Station node information."""

    id: str
    name: str
    lat: float
    lon: float
    route: str = ""
    degree: int = Field(default=0, description="Number of connections")
    betweenness: float = Field(default=0.0, description="Betweenness centrality")


class NeighborsResponse(BaseModel):
    """Neighbors of a station."""

    station_id: str
    neighbors: list[StationResponse]


class RouteResponse(BaseModel):
    """Route between two stations."""

    origin: str
    destination: str
    path: list[str]
    distance_hops: int
    stations: list[StationResponse]
