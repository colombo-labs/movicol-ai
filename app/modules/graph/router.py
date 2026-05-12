"""Graph router - station and route query endpoints."""

from fastapi import APIRouter, Query

from app.common.exceptions import GraphNotFoundError, StationNotFoundError
from app.modules.graph.schemas import NeighborsResponse, RouteResponse, StationResponse
from app.modules.graph.service import GraphService

router = APIRouter()
service = GraphService()


@router.get("/stats")
async def graph_stats():
    """Get graph statistics (nodes, edges)."""
    return service.stats


@router.get("/stations", response_model=list[StationResponse])
async def list_stations(
    limit: int = Query(default=100, le=500),
    offset: int = Query(default=0, ge=0),
):
    """List stations with pagination."""
    if not service.is_loaded:
        raise GraphNotFoundError()
    return service.get_stations(limit=limit, offset=offset)


@router.get("/stations/{station_id}", response_model=StationResponse)
async def get_station(station_id: str):
    """Get a specific station."""
    if not service.is_loaded:
        raise GraphNotFoundError()
    station = service.get_station(station_id)
    if not station:
        raise StationNotFoundError(station_id)
    return station


@router.get("/stations/{station_id}/neighbors", response_model=NeighborsResponse)
async def get_neighbors(station_id: str):
    """Get neighbors of a station."""
    if not service.is_loaded:
        raise GraphNotFoundError()
    result = service.get_neighbors(station_id)
    if not result:
        raise StationNotFoundError(station_id)
    return result


@router.get("/route", response_model=RouteResponse)
async def find_route(origin: str, destination: str):
    """Find shortest route between two stations."""
    if not service.is_loaded:
        raise GraphNotFoundError()
    route = service.find_route(origin, destination)
    if not route:
        raise StationNotFoundError(f"{origin} -> {destination}")
    return route
