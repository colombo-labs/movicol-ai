"""Graph router - station and route query endpoints."""

from fastapi import APIRouter, Query

from app.common.exceptions import GraphNotFoundError, StationNotFoundError
from app.modules.graph.schemas import NeighborsResponse, RouteResponse, StationResponse
from app.modules.graph.service import GraphService

# Pagination defaults
DEFAULT_STATION_LIMIT = 100
MAX_STATION_LIMIT = 500
DEFAULT_EDGE_LIMIT = 500
MAX_EDGE_LIMIT = 5000
DEFAULT_NEARBY_LIMIT = 10
MAX_NEARBY_LIMIT = 50
DEFAULT_NEARBY_RADIUS = 1.0
MAX_NEARBY_RADIUS = 5.0


router = APIRouter()
service = GraphService()


@router.get("/stats")
async def graph_stats():
    """Get graph statistics (nodes, edges)."""
    return service.stats


@router.get("/analysis")
async def graph_analysis():
    """Get advanced graph analysis: degree distribution, components, density."""
    return service.analysis


@router.get("/heatmap")
async def congestion_heatmap(hour: int = Query(default=8, ge=0, le=23)):
    """Get GNN congestion predictions for all stations at a given hour."""
    return service.get_heatmap(hour)


@router.get("/stations", response_model=list[StationResponse])
async def list_stations(
    limit: int = Query(default=DEFAULT_STATION_LIMIT, le=MAX_STATION_LIMIT),
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


@router.get("/nearby")
async def nearby_stations(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    radius_km: float = Query(default=DEFAULT_NEARBY_RADIUS, le=MAX_NEARBY_RADIUS),
    limit: int = Query(default=DEFAULT_NEARBY_LIMIT, le=MAX_NEARBY_LIMIT),
):
    """Find stations within radius of a point."""
    return service.get_nearby(lat, lon, radius_km, limit)


@router.get("/compare-hours")
async def compare_hours(station_id: str = Query(..., description="Station ID")):
    """Compare congestion levels across all hours for a station."""
    return service.compare_hours(station_id)


@router.get("/edges")
async def get_edges(
    type: str = Query(default="all", description="all | tm | sitp"),
    limit: int = Query(default=DEFAULT_EDGE_LIMIT, le=MAX_EDGE_LIMIT),
):
    """Get graph edges as coordinate pairs for map rendering."""
    return service.get_edges(type, limit)


@router.get("/tm/troncales")
async def get_tm_troncales():
    """Get TransMilenio trunk lines GeoJSON for map rendering."""
    return service.get_tm_troncales()


@router.get("/tm/estaciones")
async def get_tm_estaciones():
    """Get TransMilenio stations GeoJSON for map rendering."""
    return service.get_tm_estaciones()
