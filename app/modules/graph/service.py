"""Graph service - NetworkX graph queries using static Caracas data."""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from app.config.settings import get_settings
from app.modules.graph.schemas import NeighborsResponse, RouteResponse, StationResponse
from app.modules.route_prediction.graph_data import build_caracas_graph


class GraphService:
    """Service for querying the mobility graph."""

    def __init__(self) -> None:
        self._settings = get_settings()
        # Try to load from file, fallback to static Caracas graph
        self._graph = self._load_or_build()

    def _load_or_build(self) -> nx.Graph:
        """Load graph from file or build static one."""
        graph_path = Path(self._settings.graph_path)
        if graph_path.exists():
            raw = nx.read_graphml(graph_path)
            # Convert MultiGraph to simple Graph
            if isinstance(raw, (nx.MultiGraph, nx.MultiDiGraph)):
                return nx.Graph(raw)
            return raw
        # Fallback: use static Caracas graph (always available)
        return build_caracas_graph().to_undirected()

    @property
    def is_loaded(self) -> bool:
        return self._graph is not None and self._graph.number_of_nodes() > 0

    def get_stations(self, limit: int = 100, offset: int = 0) -> list[StationResponse]:
        """Get paginated list of stations."""
        nodes = list(self._graph.nodes(data=True))[offset : offset + limit]
        return [
            StationResponse(
                id=str(node_id),
                name=data.get("name", "") or str(node_id),
                lat=float(data.get("lat", 0)),
                lon=float(data.get("lon", 0)),
                route=data.get("troncal", "") or data.get("route", ""),
                degree=self._graph.degree(node_id),
            )
            for node_id, data in nodes
        ]

    def get_station(self, station_id: str) -> StationResponse | None:
        if station_id not in self._graph:
            return None
        data = self._graph.nodes[station_id]
        return StationResponse(
            id=station_id,
            name=data.get("name", "") or station_id,
            lat=float(data.get("lat", 0)),
            lon=float(data.get("lon", 0)),
            route=data.get("troncal", "") or data.get("route", ""),
            degree=self._graph.degree(station_id),
        )

    def get_neighbors(self, station_id: str) -> NeighborsResponse | None:
        if station_id not in self._graph:
            return None
        neighbors = [self.get_station(str(n)) for n in self._graph.neighbors(station_id)]
        return NeighborsResponse(
            station_id=station_id,
            neighbors=[n for n in neighbors if n is not None],
        )

    def find_route(self, origin: str, destination: str) -> RouteResponse | None:
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
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
        }

    @property
    def analysis(self) -> dict:
        """Advanced graph analysis."""
        g = self._graph
        n = g.number_of_nodes()
        e = g.number_of_edges()
        degrees = [d for _, d in g.degree()]

        # Degree distribution buckets
        buckets = {"1-2": 0, "3-5": 0, "6-10": 0, "11-20": 0, "21+": 0}
        for d in degrees:
            if d <= 2:
                buckets["1-2"] += 1
            elif d <= 5:
                buckets["3-5"] += 1
            elif d <= 10:
                buckets["6-10"] += 1
            elif d <= 20:
                buckets["11-20"] += 1
            else:
                buckets["21+"] += 1

        # Top hubs
        top_hubs = sorted(g.degree(), key=lambda x: x[1], reverse=True)[:10]
        hubs = [
            {"id": nid, "name": g.nodes[nid].get("nombre", "") or nid, "degree": d}
            for nid, d in top_hubs
        ]

        # Components
        if g.is_directed():
            components = nx.number_weakly_connected_components(g)
            largest = max(nx.weakly_connected_components(g), key=len)
        else:
            components = nx.number_connected_components(g)
            largest = max(nx.connected_components(g), key=len)

        return {
            "nodes": n,
            "edges": e,
            "density": round(nx.density(g), 6),
            "avg_degree": round(sum(degrees) / max(n, 1), 2),
            "max_degree": max(degrees) if degrees else 0,
            "min_degree": min(degrees) if degrees else 0,
            "components": components,
            "largest_component_pct": round(len(largest) / max(n, 1) * 100, 1),
            "degree_distribution": buckets,
            "top_hubs": hubs,
            "node_types": self._count_types(),
        }

    def _count_types(self) -> dict:
        """Count node types (TM vs SITP)."""
        types: dict[str, int] = {}
        for _, data in self._graph.nodes(data=True):
            t = data.get("tipo", "unknown")
            types[t] = types.get(t, 0) + 1
        return types

    def get_heatmap(self, hour: int) -> list[dict]:
        """Get congestion predictions for all stations at a given hour."""
        from app.modules.predictions.gnn_inference import GNNInference

        gnn = GNNInference()
        if not gnn.is_loaded:
            return []

        time_factors = {
            0: 0.3,
            1: 0.2,
            2: 0.2,
            3: 0.2,
            4: 0.3,
            5: 0.5,
            6: 0.7,
            7: 0.9,
            8: 1.0,
            9: 0.9,
            10: 0.7,
            11: 0.65,
            12: 0.75,
            13: 0.7,
            14: 0.65,
            15: 0.7,
            16: 0.8,
            17: 0.95,
            18: 1.0,
            19: 0.9,
            20: 0.7,
            21: 0.5,
            22: 0.4,
            23: 0.3,
        }
        tf = time_factors.get(hour, 0.7)

        results = []
        preds = gnn.get_all_predictions()
        for node_id, base_congestion in preds.items():
            if node_id not in self._graph:
                continue
            data = self._graph.nodes[node_id]
            congestion = min(1.0, base_congestion * tf)
            results.append(
                {
                    "id": node_id,
                    "name": data.get("nombre", "") or node_id,
                    "lat": float(data.get("lat", 0)),
                    "lon": float(data.get("lon", 0)),
                    "congestion": round(congestion, 3),
                    "risk": "low"
                    if congestion < 0.3
                    else "medium"
                    if congestion < 0.6
                    else "high"
                    if congestion < 0.85
                    else "critical",
                }
            )

        return sorted(results, key=lambda x: x["congestion"], reverse=True)

    def get_nearby(self, lat: float, lon: float, radius_km: float, limit: int) -> list[dict]:
        """Find stations within radius of a point."""
        results = []
        for node_id, data in self._graph.nodes(data=True):
            nlat = float(data.get("lat", 0))
            nlon = float(data.get("lon", 0))
            dist = ((nlat - lat) ** 2 + (nlon - lon) ** 2) ** 0.5 * 111
            if dist <= radius_km:
                results.append(
                    {
                        "id": node_id,
                        "name": data.get("nombre", "") or node_id,
                        "lat": nlat,
                        "lon": nlon,
                        "distance_km": round(dist, 3),
                        "degree": self._graph.degree(node_id),
                        "type": data.get("tipo", "unknown"),
                    }
                )
        results.sort(key=lambda x: x["distance_km"])
        return results[:limit]

    def compare_hours(self, station_id: str) -> dict:
        """Compare congestion across all hours for a station."""
        from app.modules.predictions.gnn_inference import GNNInference

        gnn = GNNInference()
        base = gnn.get_congestion(station_id) if gnn.is_loaded else 0.5

        time_factors = {
            0: 0.3,
            1: 0.2,
            2: 0.2,
            3: 0.2,
            4: 0.3,
            5: 0.5,
            6: 0.7,
            7: 0.9,
            8: 1.0,
            9: 0.9,
            10: 0.7,
            11: 0.65,
            12: 0.75,
            13: 0.7,
            14: 0.65,
            15: 0.7,
            16: 0.8,
            17: 0.95,
            18: 1.0,
            19: 0.9,
            20: 0.7,
            21: 0.5,
            22: 0.4,
            23: 0.3,
        }

        hours = []
        best_hour = 0
        best_val = 1.0
        for h in range(24):
            val = min(1.0, base * time_factors[h])
            label = (
                "low"
                if val < 0.3
                else "medium"
                if val < 0.6
                else "high"
                if val < 0.85
                else "critical"
            )
            hours.append({"hour": h, "congestion": round(val, 3), "risk": label})
            if val < best_val:
                best_val = val
                best_hour = h

        name = ""
        if station_id in self._graph:
            name = self._graph.nodes[station_id].get("nombre", "") or station_id

        return {
            "station_id": station_id,
            "station_name": name,
            "base_congestion": round(base, 3),
            "hours": hours,
            "best_hour": best_hour,
            "best_congestion": round(best_val, 3),
            "worst_hour": 8,
            "worst_congestion": round(min(1.0, base * 1.0), 3),
        }

    def get_edges(self, edge_type: str, limit: int) -> list[dict]:
        """Get edges as coordinate pairs for map rendering."""
        edges = []
        count = 0
        for u, v in self._graph.edges():
            if count >= limit:
                break
            u_data = self._graph.nodes[u]
            v_data = self._graph.nodes[v]

            is_tm_edge = "-TM" in u or "-TM" in v or u.startswith("TM_") or v.startswith("TM_")

            if edge_type == "tm" and not is_tm_edge:
                continue
            if edge_type == "sitp" and is_tm_edge:
                continue

            lat1, lon1 = float(u_data.get("lat", 0)), float(u_data.get("lon", 0))
            lat2, lon2 = float(v_data.get("lat", 0)), float(v_data.get("lon", 0))

            if lat1 == 0 or lat2 == 0:
                continue

            edges.append(
                {
                    "from": u_data.get("nombre", "") or u,
                    "to": v_data.get("nombre", "") or v,
                    "coords": [[lat1, lon1], [lat2, lon2]],
                    "type": "tm" if is_tm_edge else "sitp",
                    "distance_km": self._graph.edges[u, v].get("distance_km", 0),
                }
            )
            count += 1

        return edges

    def get_tm_troncales(self) -> dict:
        """Load TransMilenio trunk lines from PostGIS as GeoJSON."""
        import json
        import os

        import psycopg2

        db_url = os.environ.get("DATABASE_URL", "")
        try:
            conn = psycopg2.connect(db_url)
            cur = conn.cursor()
            cur.execute("""
                SELECT json_build_object(
                    'type', 'FeatureCollection',
                    'features', COALESCE(json_agg(
                        json_build_object(
                            'type', 'Feature',
                            'geometry', ST_AsGeoJSON(geom)::json,
                            'properties', propiedades
                        )
                    ), '[]'::json)
                ) FROM tm_troncales;
            """)
            result = cur.fetchone()[0]
            cur.close()
            conn.close()
            return result
        except Exception:
            # Fallback to file
            from pathlib import Path

            path = Path("models/tm_troncales.geojson")
            if path.exists():
                return json.loads(path.read_text())
            return {"type": "FeatureCollection", "features": []}

    def get_tm_estaciones(self) -> dict:
        """Load TransMilenio stations from PostGIS as GeoJSON."""
        import json
        import os

        import psycopg2

        db_url = os.environ.get("DATABASE_URL", "")
        try:
            conn = psycopg2.connect(db_url)
            cur = conn.cursor()
            cur.execute("""
                SELECT json_build_object(
                    'type', 'FeatureCollection',
                    'features', COALESCE(json_agg(
                        json_build_object(
                            'type', 'Feature',
                            'geometry', ST_AsGeoJSON(geom)::json,
                            'properties', propiedades
                        )
                    ), '[]'::json)
                ) FROM tm_estaciones;
            """)
            result = cur.fetchone()[0]
            cur.close()
            conn.close()
            return result
        except Exception:
            # Fallback to file
            from pathlib import Path

            path = Path("models/tm_estaciones.geojson")
            if path.exists():
                return json.loads(path.read_text())
            return {"type": "FeatureCollection", "features": []}
