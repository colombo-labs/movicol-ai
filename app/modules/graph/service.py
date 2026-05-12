"""Graph service - NetworkX graph queries."""

from pathlib import Path

import networkx as nx

from app.config.settings import get_settings
from app.modules.graph.schemas import NeighborsResponse, RouteResponse, StationResponse


class GraphService:
    """Service for querying the mobility graph."""

    def __init__(self) -> None:
        self._graph: nx.Graph | None = None
        self._settings = get_settings()

    @property
    def is_loaded(self) -> bool:
        """Check if graph is loaded."""
        return self._graph is not None

    def load_graph(self) -> None:
        """Load graph from GraphML file."""
        graph_path = Path(self._settings.graph_path)
        if not graph_path.exists():
            return
        self._graph = nx.read_graphml(graph_path)

    def get_stations(self, limit: int = 100, offset: int = 0) -> list[StationResponse]:
        """Get paginated list of stations."""
        if not self._graph:
            return []
        nodes = list(self._graph.nodes(data=True))[offset : offset + limit]
        return [
            StationResponse(
                id=str(node_id),
                name=data.get("name", ""),
                lat=float(data.get("lat", 0)),
                lon=float(data.get("lon", 0)),
                route=data.get("route", ""),
                degree=self._graph.degree(node_id),
            )
            for node_id, data in nodes
        ]

    def get_station(self, station_id: str) -> StationResponse | None:
        """Get a single station by ID."""
        if not self._graph or station_id not in self._graph:
            return None
        data = self._graph.nodes[station_id]
        return StationResponse(
            id=station_id,
            name=data.get("name", ""),
            lat=float(data.get("lat", 0)),
            lon=float(data.get("lon", 0)),
            route=data.get("route", ""),
            degree=self._graph.degree(station_id),
        )

    def get_neighbors(self, station_id: str) -> NeighborsResponse | None:
        """Get neighbors of a station."""
        if not self._graph or station_id not in self._graph:
            return None
        neighbors = [
            self.get_station(str(n)) for n in self._graph.neighbors(station_id)
        ]
        return NeighborsResponse(
            station_id=station_id,
            neighbors=[n for n in neighbors if n is not None],
        )

    def find_route(self, origin: str, destination: str) -> RouteResponse | None:
        """Find shortest path between two stations."""
        if not self._graph:
            return None
        if origin not in self._graph or destination not in self._graph:
            return None
        try:
            path = nx.shortest_path(self._graph, origin, destination)
        except nx.NetworkXNoPath:
            return None
        stations = [self.get_station(str(n)) for n in path]
        return RouteResponse(
            origin=origin,
            destination=destination,
            path=[str(n) for n in path],
            distance_hops=len(path) - 1,
            stations=[s for s in stations if s is not None],
        )

    @property
    def stats(self) -> dict:
        """Graph statistics."""
        if not self._graph:
            return {"nodes": 0, "edges": 0}
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
        }
