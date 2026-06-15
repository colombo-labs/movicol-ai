"""Route prediction service — multi-modal: TM, SITP (graph + GNN) and vehiculo (OSRM)."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

import httpx
import networkx as nx

from app.common.congestion import risk_label as _risk_label
from app.common.congestion import time_factor as _time_factor
from app.config.settings import get_settings
from app.modules.demand_prediction.st_gat_inference import DemandInference
from app.modules.predictions.gnn_inference import GNNInference
from app.modules.route_prediction.graph_data import build_caracas_graph
from app.modules.route_prediction.schemas import (
    Coordinates,
    RiskSegment,
    RoutePredictionResponse,
)

OSRM_BASE = "https://router.project-osrm.org"


def _parse_hour(departure_time: str) -> int:
    """Extract hour from ISO departure time string."""
    try:
        dt = datetime.fromisoformat(departure_time.replace("Z", "+00:00"))
        return dt.hour
    except (ValueError, AttributeError):
        return 8


class RoutePredictionService:
    """Route prediction: TM/SITP via graph Dijkstra + GNN, vehiculo via OSRM."""

    def __init__(self) -> None:
        self._gnn = GNNInference()
        self._demand = DemandInference()
        self._graph = self._load_graph()

    def _load_graph(self) -> nx.Graph:
        settings = get_settings()
        graph_path = Path(settings.graph_path)
        if graph_path.exists():
            raw = nx.read_graphml(graph_path)
            g = nx.Graph(raw) if isinstance(raw, (nx.MultiGraph, nx.MultiDiGraph)) else raw
            for u, v in g.edges():
                u_data, v_data = g.nodes[u], g.nodes[v]
                lat1, lon1 = float(u_data.get("lat", 0)), float(u_data.get("lon", 0))
                lat2, lon2 = float(v_data.get("lat", 0)), float(v_data.get("lon", 0))
                dist = ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5 * 111
                g.edges[u, v]["distance_km"] = round(dist, 3)
                g.edges[u, v]["base_time_min"] = round(dist * 2.0, 1)
            return g
        return build_caracas_graph().to_undirected()

    @property
    def graph_size(self) -> dict:
        return {"nodes": self._graph.number_of_nodes(), "edges": self._graph.number_of_edges()}

    def _find_nearest_station(self, coords: Coordinates, tipo_filter: str | None = None) -> str:
        best_id, best_dist = "", float("inf")
        for node_id, data in self._graph.nodes(data=True):
            if tipo_filter:
                node_tipo = data.get("tipo", "")
                if (
                    tipo_filter == "tm"
                    and "tm" not in node_tipo.lower()
                    and "troncal" not in node_tipo.lower()
                ):
                    continue
                if tipo_filter == "sitp" and "tm" in node_tipo.lower():
                    continue
            lat = float(data.get("lat", 0))
            lon = float(data.get("lon", 0))
            d = (lat - coords.lat) ** 2 + (lon - coords.lng) ** 2
            if d < best_dist:
                best_dist = d
                best_id = node_id
        return best_id

    def _get_congestion(self, node_id: str, hour: int) -> float:
        """Combined congestion: GNN base + demand from ST-GAT."""
        base = 0.5
        if self._gnn.is_loaded:
            gnn_val = self._gnn.get_congestion(node_id)
            if abs(gnn_val - 0.5) > 1e-9:
                base = gnn_val

        # Boost congestion with demand prediction if available
        if self._demand.is_loaded:
            try:
                idx = list(self._graph.nodes()).index(node_id)
                demand_score = self._demand.get_station_demand_score(
                    idx % self._demand.station_count
                )
                base = base * 0.6 + demand_score * 0.4  # Blend both models
            except (ValueError, IndexError):
                pass

        return min(1.0, base * _time_factor(hour))

    async def predict_route(
        self,
        origin: Coordinates,
        destination: Coordinates,
        departure_time: str,
        mode: str = "transmilenio",
    ) -> RoutePredictionResponse:
        """Predict route for given mode."""
        if mode == "vehiculo":
            return await self._predict_vehicle(origin, destination, departure_time)
        return self._predict_transit(origin, destination, departure_time, mode)

    @staticmethod
    def _make_segment(
        from_name: str,
        to_name: str,
        congestion: float,
        coordinates: list,
    ) -> RiskSegment:
        """Create a standardized risk segment."""
        return RiskSegment(
            from_station=from_name,
            to_station=to_name,
            congestion_level=round(congestion, 2),
            risk_label=_risk_label(congestion),
            coordinates=coordinates,
        )

    def _build_response(
        self,
        time_min: float,
        distance_km: float,
        cost: str,
        mode: str,
        risk_segments: list,
        stations: list,
        departure_time: str,
    ) -> RoutePredictionResponse:
        """Build standardized route prediction response."""
        avg_c = (
            sum(s.congestion_level for s in risk_segments) / max(len(risk_segments), 1)
            if risk_segments
            else 0.0
        )
        return RoutePredictionResponse(
            route_id=str(uuid.uuid4()),
            total_time_minutes=round(time_min, 1),
            total_distance_km=round(distance_km, 1),
            cost=cost,
            mode=mode,
            risk_segments=risk_segments,
            overall_risk=_risk_label(avg_c),
            explanation="",
            stations=stations,
            departure_time=departure_time,
        )

    async def _predict_vehicle(
        self, origin: Coordinates, destination: Coordinates, departure_time: str
    ) -> RoutePredictionResponse:
        """Vehicle routing via OSRM."""
        hour = _parse_hour(departure_time)

        url = (
            f"{OSRM_BASE}/route/v1/driving/"
            f"{origin.lng},{origin.lat};{destination.lng},{destination.lat}"
            f"?overview=full&geometries=geojson&steps=true"
        )

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url)
                data = resp.json()

            if data.get("code") != "Ok" or not data.get("routes"):
                return self._fallback_vehicle(origin, destination, departure_time, hour)

            route = data["routes"][0]
            duration_min = route["duration"] / 60
            distance_km = route["distance"] / 1000
            coords = route["geometry"]["coordinates"]  # [lng, lat] pairs

            # Build risk segments from route geometry
            congestion = _time_factor(hour) * 0.7  # Base vehicle congestion
            step_size = max(1, len(coords) // 10)
            segments = []
            for i in range(0, len(coords) - step_size, step_size):
                end_i = min(i + step_size, len(coords) - 1)
                seg_coords = [[c[1], c[0]] for c in coords[i : end_i + 1]]
                segments.append(
                    self._make_segment(
                        f"Punto {i // step_size + 1}",
                        f"Punto {i // step_size + 2}",
                        congestion,
                        seg_coords,
                    )
                )

            # Adjust time by congestion
            adjusted_time = duration_min * (1 + congestion * 0.3)

            return self._build_response(
                adjusted_time, distance_km, "$0", "vehiculo", segments, [], departure_time
            )
        except Exception:
            return self._fallback_vehicle(origin, destination, departure_time, hour)

    def _fallback_vehicle(
        self, origin: Coordinates, destination: Coordinates, departure_time: str, hour: int
    ) -> RoutePredictionResponse:
        """Fallback when OSRM is unavailable."""
        dist = (
            (origin.lat - destination.lat) ** 2 + (origin.lng - destination.lng) ** 2
        ) ** 0.5 * 111
        time_min = dist * 3.0 * (1 + _time_factor(hour) * 0.3)
        congestion = _time_factor(hour) * 0.6

        segments = [
            self._make_segment(
                "Origen",
                "Destino",
                congestion,
                [[origin.lat, origin.lng], [destination.lat, destination.lng]],
            )
        ]
        return self._build_response(
            time_min,
            dist,
            "$0",
            "vehiculo",
            segments,
            [],
            departure_time,
        )

    def _find_path(self, origin_id: str, dest_id: str, destination: "Coordinates") -> list:
        """Find shortest path with fallback for disconnected components."""
        try:
            return nx.shortest_path(self._graph, origin_id, dest_id, weight="distance_km")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        try:
            component = nx.node_connected_component(self._graph, origin_id)
            best_dest, best_d = dest_id, float("inf")
            for n in component:
                d = self._graph.nodes[n]
                dist = (float(d.get("lat", 0)) - destination.lat) ** 2 + (
                    float(d.get("lon", 0)) - destination.lng
                ) ** 2
                if dist < best_d:
                    best_d, best_dest = dist, n
            return nx.shortest_path(self._graph, origin_id, best_dest, weight="distance_km")
        except Exception:
            return [origin_id, dest_id]

    def _limit_path(self, path: list, max_display: int = 30) -> list:
        """Limit path nodes for display."""
        if len(path) <= max_display:
            return path
        step = len(path) // max_display
        display = [path[i] for i in range(0, len(path), step)]
        if path[-1] not in display:
            display.append(path[-1])
        return display

    def _predict_transit(
        self, origin: Coordinates, destination: Coordinates, departure_time: str, mode: str
    ) -> RoutePredictionResponse:
        """Transit routing (TM/SITP) via Dijkstra + GNN congestion."""
        hour = _parse_hour(departure_time)

        tipo_map = {"transmilenio": "tm", "sitp": "sitp"}
        tipo_filter = tipo_map.get(mode)
        origin_id = self._find_nearest_station(origin, tipo_filter)
        dest_id = self._find_nearest_station(destination, tipo_filter)

        path = self._find_path(origin_id, dest_id, destination)
        display_path = self._limit_path(path)

        # Build segments with congestion
        risk_segments: list[RiskSegment] = []
        total_distance, total_time = 0.0, 0.0

        for i in range(len(display_path) - 1):
            from_id, to_id = display_path[i], display_path[i + 1]
            from_data = self._graph.nodes.get(from_id, {})
            to_data = self._graph.nodes.get(to_id, {})

            lat1, lon1 = float(from_data.get("lat", 0)), float(from_data.get("lon", 0))
            lat2, lon2 = float(to_data.get("lat", 0)), float(to_data.get("lon", 0))

            edge = self._graph.edges.get((from_id, to_id), {})
            dist = edge.get("distance_km", ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5 * 111)
            base_time = edge.get("base_time_min", dist * 2.0)

            congestion = (
                self._get_congestion(from_id, hour) + self._get_congestion(to_id, hour)
            ) / 2
            adjusted_time = base_time * (1 + congestion * 0.5)

            total_distance += dist
            total_time += adjusted_time

            from_name = from_data.get("nombre", "") or from_data.get("name", "") or from_id
            to_name = to_data.get("nombre", "") or to_data.get("name", "") or to_id

            risk_segments.append(
                self._make_segment(
                    from_name,
                    to_name,
                    congestion,
                    [[lat1, lon1], [lat2, lon2]],
                )
            )

        station_names = [
            self._graph.nodes.get(n, {}).get("nombre", "")
            or self._graph.nodes.get(n, {}).get("name", "")
            or n
            for n in display_path
        ]

        cost = "$2,950" if mode == "transmilenio" else "$2,650"

        return self._build_response(
            total_time, total_distance, cost, mode, risk_segments, station_names, departure_time
        )
