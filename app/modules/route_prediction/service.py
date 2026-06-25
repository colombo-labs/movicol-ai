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
from app.modules.route_prediction.graph_data import build_sitp_graph, build_tm_graph
from app.modules.route_prediction.schemas import (
    Coordinates,
    NavigationStep,
    RiskSegment,
    RoutePredictionResponse,
)


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
        self._graph = self._load_graph()  # fallback graph
        self._tm_graph = build_tm_graph()  # TM-only graph (153 stations, 13 troncales)
        self._sitp_routes = (
            self._load_sitp_route_data()
        )  # {ruta: [{lat,lon,nombre,orden}, ...]}
        self._multimodal_graph = self._build_multimodal_graph()

    def _build_multimodal_graph(self) -> nx.Graph:
        """Combine SITP and TM graphs and add walk edges between nearby stations."""
        g = nx.Graph()
        g.add_nodes_from(self._tm_graph.nodes(data=True))
        g.add_edges_from(self._tm_graph.edges(data=True))

        tm_nodes = list(self._tm_graph.nodes(data=True))
        sitp_nodes_data = []

        for ruta, stops in self._sitp_routes.items():
            for i in range(len(stops)):
                s = stops[i]
                node_id = f"sitp_{s['lat']}_{s['lon']}"
                g.add_node(
                    node_id, lat=s["lat"], lon=s["lon"], name=s["nombre"], type="sitp"
                )
                sitp_nodes_data.append((node_id, s))
                if i > 0:
                    prev = stops[i - 1]
                    prev_id = f"sitp_{prev['lat']}_{prev['lon']}"
                    dist = self._haversine_km(
                        s["lat"], s["lon"], prev["lat"], prev["lon"]
                    )
                    g.add_edge(prev_id, node_id, troncal="SITP", distance_km=dist)

        # Walk edges TM <-> SITP
        for tm_id, tm_d in tm_nodes:
            lat1, lon1 = float(tm_d.get("lat", 0)), float(tm_d.get("lon", 0))
            for sitp_id, sitp_d in sitp_nodes_data:
                lat2, lon2 = sitp_d["lat"], sitp_d["lon"]
                dist = self._haversine_km(lat1, lon1, lat2, lon2)
                if dist <= 0.3:
                    g.add_edge(tm_id, sitp_id, troncal="transbordo", distance_km=dist)
        return g

    def _load_sitp_route_data(self) -> dict:
        """Load SITP route data grouped by route, sorted by orden."""
        import json

        p = (
            Path(__file__).parent.parent.parent.parent
            / "movicol-data"
            / "exports"
            / "backend"
            / "sitp_rutas_paraderos.geojson"
        )
        if not p.exists():
            print(f"Warning: SITP paraderos file not found at {p}")
            return {}
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        by_route: dict = {}
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            ruta = props.get("ruta")
            lat_val = props.get("latitud")
            lon_val = props.get("longitud")
            if not ruta or lat_val is None or lon_val is None:
                continue
            by_route.setdefault(ruta, []).append(
                {
                    "lat": float(lat_val),
                    "lon": float(lon_val),
                    "nombre": props.get("nombre", ""),
                    "orden": props.get("orden", ""),
                }
            )
        # Sort each route by its orden field
        for ruta in by_route:
            by_route[ruta].sort(key=lambda s: s["orden"])
        print(f"SITP routes loaded: {len(by_route)} routes")
        return by_route

    def _load_graph(self) -> nx.Graph:
        settings = get_settings()
        graph_path = Path(settings.graph_path)
        if graph_path.exists():
            raw = nx.read_graphml(graph_path)
            g = (
                nx.Graph(raw)
                if isinstance(raw, (nx.MultiGraph, nx.MultiDiGraph))
                else raw
            )
            for u, v in g.edges():
                u_data, v_data = g.nodes[u], g.nodes[v]
                lat1, lon1 = float(u_data.get("lat", 0)), float(u_data.get("lon", 0))
                lat2, lon2 = float(v_data.get("lat", 0)), float(v_data.get("lon", 0))
                dist = ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5 * 111
                g.edges[u, v]["distance_km"] = round(dist, 3)
                g.edges[u, v]["base_time_min"] = round(dist * 2.0, 1)
            return g
        # Fallback to SITP graph
        print(f"Warning: Graph file {graph_path} not found. Using fallback SITP graph.")
        return build_sitp_graph().to_undirected()

    @property
    def graph_size(self) -> dict:
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
        }

    def _find_nearest_station(
        self, coords: Coordinates, tipo_filter: str | None = None
    ) -> str:
        return self._find_nearest_in(coords, self._graph)

    @staticmethod
    def _find_nearest_in(coords: Coordinates, graph: nx.Graph) -> str:
        best_id, best_dist = "", float("inf")
        for node_id, data in graph.nodes(data=True):
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
        return await self._predict_transit(origin, destination, departure_time, mode)

    @staticmethod
    def _make_segment(
        from_name: str,
        to_name: str,
        congestion: float,
        coordinates: list,
        mode: str = "transmilenio",
    ) -> RiskSegment:
        """Create a standardized risk segment."""
        return RiskSegment(
            from_station=from_name,
            to_station=to_name,
            congestion_level=round(congestion, 2),
            risk_label=_risk_label(congestion),
            coordinates=coordinates,
            mode=mode,
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
        route_code: str = "",
        navigation_steps: list | None = None,
    ) -> RoutePredictionResponse:
        """Build standardized route prediction response."""
        avg_c = (
            sum(s.congestion_level for s in risk_segments) / max(len(risk_segments), 1)
            if risk_segments
            else 0.0
        )
        # Safety score: inversely proportional to congestion + critical segments penalty
        critical_count = sum(1 for s in risk_segments if s.risk_label == "critical")
        high_count = sum(1 for s in risk_segments if s.risk_label == "high")
        safety_base = max(
            0, 100 - int(avg_c * 60) - critical_count * 10 - high_count * 5
        )
        safety_score = max(10, min(100, safety_base))

        return RoutePredictionResponse(
            route_id=str(uuid.uuid4()),
            total_time_minutes=round(time_min, 1),
            total_distance_km=round(distance_km, 1),
            cost=cost,
            mode=mode,
            risk_segments=risk_segments,
            overall_risk=_risk_label(avg_c),
            safety_score=safety_score,
            explanation="",
            stations=stations,
            departure_time=departure_time,
            route_code=route_code,
            navigation_steps=navigation_steps or [],
        )

    async def _predict_vehicle(
        self, origin: Coordinates, destination: Coordinates, departure_time: str
    ) -> RoutePredictionResponse:
        """Vehicle routing via OSRM with street names."""
        hour = _parse_hour(departure_time)

        url = (
            f"{get_settings().osrm_base_url}/route/v1/driving/"
            f"{origin.lng},{origin.lat};{destination.lng},{destination.lat}"
            f"?overview=full&geometries=geojson&steps=true"
        )

        try:
            async with httpx.AsyncClient(
                timeout=15, follow_redirects=True, max_redirects=3
            ) as client:
                resp = await client.get(url)
                data = resp.json()

            if data.get("code") != "Ok" or not data.get("routes"):
                return self._fallback_vehicle(origin, destination, departure_time, hour)

            route = data["routes"][0]
            congestion = _time_factor(hour) * 0.7

            segments, street_names, cost, adjusted_time, distance_km = (
                self._parse_osrm_route(route, congestion)
            )

            # Build navigation steps from OSRM
            steps = route["legs"][0]["steps"]
            nav_steps = self._build_nav_steps(steps)

            return self._build_response(
                adjusted_time,
                distance_km,
                cost,
                "vehiculo",
                segments,
                street_names,
                departure_time,
                navigation_steps=nav_steps,
            )
        except Exception:
            return self._fallback_vehicle(origin, destination, departure_time, hour)

    def _parse_osrm_route(
        self, route: dict, congestion: float
    ) -> tuple[list, list[str], str, float, float]:
        """Parse an OSRM route into segments, street names, cost, time, distance."""
        duration_min = route["duration"] / 60
        distance_km = route["distance"] / 1000
        adjusted_time = duration_min * (1 + congestion * 0.5)

        steps = route["legs"][0]["steps"]
        segments = []
        for i, step in enumerate(steps[:-1]):
            next_step = steps[i + 1]
            from_name = step.get("name") or f"Punto {i + 1}"
            to_name = next_step.get("name") or f"Punto {i + 2}"
            seg_coords = [[c[1], c[0]] for c in step["geometry"]["coordinates"]]
            step_congestion = congestion * (1 + (i % 3) * 0.1)
            segments.append(
                self._make_segment(
                    from_name,
                    to_name,
                    min(1.0, step_congestion),
                    seg_coords,
                    mode="vehiculo",
                )
            )

        street_names = list(
            dict.fromkeys(s.get("name", "") for s in steps if s.get("name"))
        )[:15]

        cost_pesos = round(distance_km * 2000, -2)
        cost = f"${cost_pesos:,.0f}".replace(",", ".")

        return segments, street_names, cost, adjusted_time, distance_km

    @staticmethod
    def _build_nav_steps(steps: list) -> list[NavigationStep]:
        """Build navigation steps from OSRM steps data."""
        maneuver_labels = {
            "turn": {
                "left": "Gira a la izquierda",
                "right": "Gira a la derecha",
                "slight left": "Gira levemente a la izquierda",
                "slight right": "Gira levemente a la derecha",
                "sharp left": "Gira fuerte a la izquierda",
                "sharp right": "Gira fuerte a la derecha",
                "straight": "Continúa recto",
            },
            "depart": {"": "Inicia el recorrido"},
            "arrive": {"": "Has llegado a tu destino"},
            "new name": {"straight": "Continúa por"},
            "merge": {"": "Incorpórate"},
            "roundabout": {"": "Toma la rotonda"},
            "fork": {
                "left": "Toma el desvío izquierdo",
                "right": "Toma el desvío derecho",
            },
        }
        nav_steps = []
        for step in steps:
            m = step.get("maneuver", {})
            mtype = m.get("type", "")
            modifier = m.get("modifier", "")
            street = step.get("name", "")
            labels = maneuver_labels.get(mtype, {})
            label = labels.get(modifier, labels.get("", f"{mtype} {modifier}".strip()))
            instruction = f"{label} en {street}" if street else label
            nav_steps.append(
                NavigationStep(
                    instruction=instruction,
                    street=street,
                    distance_m=round(step.get("distance", 0)),
                    duration_s=round(step.get("duration", 0)),
                    maneuver=modifier or mtype,
                )
            )
        return nav_steps

    async def predict_vehicle_alternatives(
        self, origin: Coordinates, destination: Coordinates, departure_time: str
    ) -> list[RoutePredictionResponse]:
        """Vehicle routing with multiple alternatives via OSRM."""
        hour = _parse_hour(departure_time)
        url = (
            f"{get_settings().osrm_base_url}/route/v1/driving/"
            f"{origin.lng},{origin.lat};{destination.lng},{destination.lat}"
            f"?overview=full&geometries=geojson&steps=true&alternatives=3"
        )
        try:
            async with httpx.AsyncClient(
                timeout=15, follow_redirects=True, max_redirects=3
            ) as client:
                resp = await client.get(url)
                data = resp.json()

            if data.get("code") != "Ok" or not data.get("routes"):
                return [
                    self._fallback_vehicle(origin, destination, departure_time, hour)
                ]

            results = []
            congestion = _time_factor(hour) * 0.7

            for route in data["routes"][:3]:
                segments, street_names, cost, adjusted_time, distance_km = (
                    self._parse_osrm_route(route, congestion)
                )
                results.append(
                    self._build_response(
                        adjusted_time,
                        distance_km,
                        cost,
                        "vehiculo",
                        segments,
                        street_names,
                        departure_time,
                    )
                )

            return (
                results
                if results
                else [self._fallback_vehicle(origin, destination, departure_time, hour)]
            )
        except Exception:
            return [self._fallback_vehicle(origin, destination, departure_time, hour)]

    def _fallback_vehicle(
        self,
        origin: Coordinates,
        destination: Coordinates,
        departure_time: str,
        hour: int,
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
                mode="vehiculo",
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

    def _find_path(
        self, origin_id: str, dest_id: str, destination: "Coordinates"
    ) -> list:
        """Find shortest path with fallback for disconnected components."""
        try:
            return nx.shortest_path(
                self._graph, origin_id, dest_id, weight="distance_km"
            )
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
            return nx.shortest_path(
                self._graph, origin_id, best_dest, weight="distance_km"
            )
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

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        import math

        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def _find_best_sitp_route(
        self, origin: Coordinates, destination: Coordinates, max_walk_km: float = 0.8
    ) -> tuple[str, list, int, int] | None:
        """
        Find the SITP route whose stops best cover both origin and destination.
        Returns (ruta_code, ordered_stops, origin_idx, dest_idx) or None.
        """
        best: tuple | None = None
        best_score = float("inf")

        for ruta, stops in self._sitp_routes.items():
            # Find stop closest to origin
            o_idx, _ = min(
                enumerate(stops),
                key=lambda x: self._haversine_km(
                    x[1]["lat"], x[1]["lon"], origin.lat, origin.lng
                ),
            )
            if (
                self._haversine_km(
                    stops[o_idx]["lat"], stops[o_idx]["lon"], origin.lat, origin.lng
                )
                > max_walk_km
            ):
                continue

            # Find stop closest to destination
            d_idx, _ = min(
                enumerate(stops),
                key=lambda x: self._haversine_km(
                    x[1]["lat"], x[1]["lon"], destination.lat, destination.lng
                ),
            )
            d_km = self._haversine_km(
                stops[d_idx]["lat"],
                stops[d_idx]["lon"],
                destination.lat,
                destination.lng,
            )
            if d_km > max_walk_km:
                continue

            # Route must go FROM origin TO destination (at least 2 stops apart)
            if o_idx == d_idx:
                continue

            # Ensure origin comes before destination in route order
            if o_idx > d_idx:
                o_idx, d_idx = d_idx, o_idx

            n_stops_between = d_idx - o_idx
            if n_stops_between < 1:
                continue

            o_km = self._haversine_km(
                stops[o_idx]["lat"], stops[o_idx]["lon"], origin.lat, origin.lng
            )
            score = o_km + d_km  # minimize total walking
            if score < best_score:
                best_score = score
                best = (ruta, stops, o_idx, d_idx)

        return best

    async def _handle_sitp_mode(
        self,
        origin: Coordinates,
        destination: Coordinates,
        departure_time: str,
        speed_factor: float,
    ) -> RoutePredictionResponse | None:
        result = self._find_best_sitp_route(origin, destination)
        if not result:
            return None

        ruta_code, stops, o_idx, d_idx = result
        sub_stops = stops[o_idx : d_idx + 1]
        max_display = 40
        if len(sub_stops) > max_display:
            step = len(sub_stops) // max_display
            sub_stops = [sub_stops[i] for i in range(0, len(sub_stops), step)]
            if stops[d_idx] not in sub_stops:
                sub_stops.append(stops[d_idx])

        station_names = [s["nombre"] for s in sub_stops]
        risk_segments, total_distance, total_time = await self._build_sitp_segments(
            sub_stops, speed_factor
        )
        return self._build_response(
            total_time,
            total_distance,
            "$3,550",
            "sitp",
            risk_segments,
            station_names,
            departure_time,
            route_code=ruta_code,
        )

    async def _predict_transit(
        self,
        origin: Coordinates,
        destination: Coordinates,
        departure_time: str,
        mode: str,
    ) -> RoutePredictionResponse:
        """Transit routing: TransMilenio, SITP, or Multimodal."""
        hour = _parse_hour(departure_time)
        speed_factor = 1.5 if mode != "sitp" else 2.5

        if mode == "sitp" and self._sitp_routes:
            sitp_resp = await self._handle_sitp_mode(
                origin, destination, departure_time, speed_factor
            )
            if sitp_resp:
                return sitp_resp

        if mode == "multimodal":
            graph = self._multimodal_graph
        else:
            graph = self._tm_graph

        origin_id = self._find_nearest_in(origin, graph)
        dest_id = self._find_nearest_in(destination, graph)

        try:
            path = nx.shortest_path(graph, origin_id, dest_id, weight="distance_km")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path = [origin_id, dest_id]

        max_display = 40 if mode == "multimodal" else 25
        display_path = self._limit_path(path, max_display)
        risk_segments, total_distance, total_time = (
            await self._build_transit_segments_async(
                graph, display_path, speed_factor, hour
            )
        )
        station_names = [
            graph.nodes.get(n, {}).get("name", "")
            or graph.nodes.get(n, {}).get("nombre", "")
            or str(n)
            for n in display_path
        ]
        route_code = self._derive_route_code(graph, display_path)

        has_tm = any(s.mode == "transmilenio" for s in risk_segments)
        has_sitp = any(s.mode == "sitp" for s in risk_segments)
        if has_tm and has_sitp:
            main_mode = "multimodal"
        elif has_sitp:
            main_mode = "sitp"
        else:
            main_mode = "transmilenio"

        return self._build_response(
            total_time,
            total_distance,
            "$3,550" if main_mode != "transmilenio" else "$2,950",
            main_mode,
            risk_segments,
            station_names,
            departure_time,
            route_code=route_code,
        )

    async def _build_sitp_segments(
        self, stops: list, speed_factor: float
    ) -> tuple[list[RiskSegment], float, float]:
        """Build risk segments for SITP from ordered stop list.
        Connects stops with direct lines (not OSRM pedestrian routing).
        """
        risk_segments: list[RiskSegment] = []
        total_distance = 0.0
        total_time = 0.0

        for i in range(len(stops) - 1):
            s1, s2 = stops[i], stops[i + 1]
            dist_km = self._haversine_km(s1["lat"], s1["lon"], s2["lat"], s2["lon"])
            congestion = 0.4  # default for SITP
            adj_time = dist_km * speed_factor * (1 + congestion * 0.5)
            total_distance += dist_km
            total_time += adj_time

            seg_coords = [[s1["lat"], s1["lon"]], [s2["lat"], s2["lon"]]]
            risk_segments.append(
                self._make_segment(
                    s1["nombre"], s2["nombre"], congestion, seg_coords, mode="sitp"
                )
            )

        return risk_segments, total_distance, total_time

    async def _fetch_osrm_transit_geometry(
        self, graph: nx.Graph, path: list
    ) -> list[list]:
        """Fetch exact street geometries passing through path nodes from OSRM."""
        if len(path) < 2:
            return []

        coords = []
        for n in path:
            d = graph.nodes.get(n, {})
            # geojson coordinates are lon, lat
            coords.append(f"{float(d.get('lon', 0))},{float(d.get('lat', 0))}")

        coord_str = ";".join(coords)
        base_url = get_settings().osrm_base_url
        url = f"{base_url}/route/v1/foot/{coord_str}?geometries=geojson&overview=false&steps=true"

        try:
            async with httpx.AsyncClient(
                timeout=15, follow_redirects=True, max_redirects=3
            ) as client:
                resp = await client.get(url)
                data = resp.json()

            if data.get("code") == "Ok" and data.get("routes"):
                legs = data["routes"][0].get("legs", [])
                leg_geometries = []
                for leg in legs:
                    leg_coords = []
                    for step in leg.get("steps", []):
                        # Convert from [lon, lat] (GeoJSON) to [lat, lon] for Leaflet
                        step_coords = [
                            [c[1], c[0]]
                            for c in step.get("geometry", {}).get("coordinates", [])
                        ]
                        if step_coords:
                            leg_coords.extend(step_coords)
                    leg_geometries.append(leg_coords)
                return leg_geometries
        except Exception as e:
            print("OSRM multipoint transit fetch failed:", e)
        return []

    async def _build_transit_segments_async(
        self, graph: nx.Graph, display_path: list, speed_factor: float, hour: int
    ) -> tuple[list[RiskSegment], float, float]:
        """Build risk segments for a transit path, using OSRM multipoint routing."""
        risk_segments: list[RiskSegment] = []
        total_distance, total_time = 0.0, 0.0

        # Obtener los trazos exactos de la calle
        leg_geometries = await self._fetch_osrm_transit_geometry(graph, display_path)
        use_osrm = len(leg_geometries) == (len(display_path) - 1)

        for i in range(len(display_path) - 1):
            from_id, to_id = display_path[i], display_path[i + 1]
            from_data = graph.nodes.get(from_id, {})
            to_data = graph.nodes.get(to_id, {})

            lat1, lon1 = float(from_data.get("lat", 0)), float(from_data.get("lon", 0))
            lat2, lon2 = float(to_data.get("lat", 0)), float(to_data.get("lon", 0))

            edge = graph.edges.get((from_id, to_id), {})
            dist = edge.get(
                "distance_km", ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5 * 111
            )
            base_time = dist * speed_factor

            congestion = (
                self._get_congestion(from_id, hour) + self._get_congestion(to_id, hour)
            ) / 2
            adjusted_time = base_time * (1 + congestion * 0.5)

            total_distance += dist
            total_time += adjusted_time

            from_name = (
                from_data.get("nombre", "") or from_data.get("name", "") or str(from_id)
            )
            to_name = to_data.get("nombre", "") or to_data.get("name", "") or str(to_id)

            segment_coords = (
                leg_geometries[i]
                if (use_osrm and leg_geometries[i])
                else [[lat1, lon1], [lat2, lon2]]
            )

            troncal = edge.get("troncal", "")
            if troncal == "walk" or troncal == "transbordo":
                seg_mode = "walk"
            elif troncal == "SITP":
                seg_mode = "sitp"
            else:
                seg_mode = "transmilenio"

            risk_segments.append(
                self._make_segment(
                    from_name,
                    to_name,
                    congestion,
                    segment_coords,
                    mode=seg_mode,
                )
            )

        return risk_segments, total_distance, total_time

    def _derive_route_code(self, graph: nx.Graph, display_path: list) -> str:
        """Derive TM route code from path troncal data and route matching."""
        from collections import Counter

        from app.modules.route_prediction.graph_data import TM_RUTAS

        troncales_in_path = [
            graph.nodes.get(n, {}).get("troncal", "") for n in display_path
        ]
        troncales_in_path = [t for t in troncales_in_path if t]
        route_code = (
            Counter(troncales_in_path).most_common(1)[0][0] if troncales_in_path else ""
        )

        if not TM_RUTAS or len(display_path) < 2:
            return route_code

        path_coords = [
            (
                float(graph.nodes.get(n, {}).get("lat", 0)),
                float(graph.nodes.get(n, {}).get("lon", 0)),
            )
            for n in display_path[:5]
        ]
        matched = self._match_tm_ruta(path_coords)
        return matched or route_code

    @staticmethod
    def _match_tm_ruta(path_coords: list[tuple[float, float]]) -> str:
        """Match path coordinates to a specific TM route."""
        from app.modules.route_prediction.graph_data import TM_RUTAS

        best_ruta, best_score = "", 0
        for ruta in TM_RUTAS:
            coords = ruta.get("coords", [])
            if not coords:
                continue
            score = sum(
                1
                for plat, plon in path_coords
                if any(
                    abs(plat - clat) < 0.003 and abs(plon - clon) < 0.003
                    for clat, clon in coords[::20]
                )
            )
            if score > best_score:
                best_score = score
                nombre = ruta.get("nombre", "")
                best_ruta = (
                    nombre.split()[0]
                    if nombre
                    else ruta.get("codigo", "").split("-")[0]
                )
        return best_ruta if best_score >= 2 else ""

    def get_route_safety(self, ruta: str, hour: int) -> dict:
        """Calculate safety score for a SITP route based on avg congestion of its stops."""
        # Find nodes that belong to this route (by name or route attribute)
        route_nodes = [
            n
            for n, d in self._graph.nodes(data=True)
            if str(d.get("route", "")) == ruta or ruta in str(d.get("name", ""))
        ]

        if not route_nodes:
            # Fallback: use a sample of SITP nodes
            route_nodes = [
                n for n, d in self._graph.nodes(data=True) if "P_" in str(n)
            ][:20]

        congestions = [self._get_congestion(n, hour) for n in route_nodes]
        avg_congestion = sum(congestions) / max(len(congestions), 1)

        critical = sum(1 for c in congestions if c > 0.75)
        high = sum(1 for c in congestions if 0.5 < c <= 0.75)

        safety_base = max(0, 100 - int(avg_congestion * 60) - critical * 10 - high * 5)
        safety_score = max(10, min(100, safety_base))

        if safety_score >= 70:
            nivel = "segura"
        elif safety_score >= 40:
            nivel = "precaución"
        else:
            nivel = "peligrosa"

        return {
            "ruta": ruta,
            "hour": hour,
            "safety_score": safety_score,
            "nivel": nivel,
            "avg_congestion": round(avg_congestion, 3),
            "paraderos_analizados": len(route_nodes),
            "tramos_criticos": critical,
            "tramos_lentos": high,
        }
