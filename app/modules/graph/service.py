"""Graph service - NetworkX graph queries using static Caracas data."""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from app.config.settings import get_settings
from app.modules.graph.schemas import NeighborsResponse, RouteResponse, StationResponse
from app.modules.route_prediction.graph_data import build_sitp_graph


class GraphService:
    """Service for querying the mobility graph."""

    SITP_SHAPES_PATH = "models/sitp_rutas_shapes.geojson"
    SITP_FREQ_PATH = "models/sitp_rutas_frecuencias.json"

    def __init__(self) -> None:
        self._settings = get_settings()
        # Try to load from file, fallback to static Caracas graph
        self._graph = self._load_or_build()
        # In-memory caches (loaded once on startup)
        self._cache_sitp_paraderos: dict | None = None
        self._cache_sitp_rutas: dict | None = None
        self._cache_sitp_shapes: dict | None = None
        self._cache_tm_rutas: dict | None = None
        self._cache_tm_troncales: dict | None = None
        self._cache_tm_estaciones: dict | None = None
        self._cache_frecuencias: dict | None = None
        self._cache_heatmap: dict[int, list] = {}  # hour -> results

    def _load_or_build(self) -> nx.Graph:
        """Load graph from file or build static one."""
        graph_path = Path(self._settings.graph_path)
        if graph_path.exists():
            raw = nx.read_graphml(graph_path)
            # Convert MultiGraph to simple Graph
            if isinstance(raw, (nx.MultiGraph, nx.MultiDiGraph)):
                return nx.Graph(raw)
            return raw
        # Fallback: use static SITP graph (always available)
        return build_sitp_graph().to_undirected()

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
        """Get congestion predictions for all stations at a given hour (cached per hour)."""
        if hour in self._cache_heatmap:
            return self._cache_heatmap[hour]

        from app.common.congestion import risk_label, time_factor
        from app.modules.predictions.gnn_inference import GNNInference

        gnn = GNNInference()
        if not gnn.is_loaded:
            return []

        tf = time_factor(hour)

        results = []
        preds = gnn.get_all_predictions()
        for node_id, base_congestion in preds.items():
            if node_id not in self._graph:
                continue
            # Solo estaciones/paraderos con demanda relevante
            if base_congestion <= 0.01:
                continue

            data = self._graph.nodes[node_id]
            name = data.get("nombre", "") or data.get("name", "")

            # Filtramos nodos que son simples cruces de calle (sin nombre)
            if not name:
                continue

            congestion = min(1.0, base_congestion * tf)
            results.append(
                {
                    "id": node_id,
                    "name": name or node_id,
                    "lat": float(data.get("lat", 0)),
                    "lon": float(data.get("lon", 0)),
                    "congestion": round(congestion, 3),
                    "risk": risk_label(congestion),
                }
            )

        self._cache_heatmap[hour] = sorted(results, key=lambda x: x["congestion"], reverse=True)
        return self._cache_heatmap[hour]

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
        from app.common.congestion import risk_label, time_factor
        from app.modules.predictions.gnn_inference import GNNInference

        gnn = GNNInference()
        base = gnn.get_congestion(station_id) if gnn.is_loaded else 0.5

        hours = []
        best_hour = 0
        best_val = 1.0
        for h in range(24):
            val = min(1.0, base * time_factor(h))
            hours.append({"hour": h, "congestion": round(val, 3), "risk": risk_label(val)})
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
            "worst_congestion": round(min(1.0, base * time_factor(8)), 3),
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
        """Load TransMilenio trunk lines from PostGIS as GeoJSON (cached)."""
        if self._cache_tm_troncales is not None:
            return self._cache_tm_troncales

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
            self._cache_tm_troncales = result
            return result
        except Exception:
            # Fallback to file
            from pathlib import Path

            path = Path("models/tm_troncales.geojson")
            if path.exists():
                self._cache_tm_troncales = json.loads(path.read_text())
                return self._cache_tm_troncales
            self._cache_tm_troncales = {"type": "FeatureCollection", "features": []}
            return self._cache_tm_troncales

    def get_tm_estaciones(self) -> dict:
        """Load TransMilenio stations from PostGIS as GeoJSON (cached)."""
        if self._cache_tm_estaciones is not None:
            return self._cache_tm_estaciones

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
            self._cache_tm_estaciones = result
            return result
        except Exception:
            # Fallback to file
            from pathlib import Path

            path = Path("models/tm_estaciones.geojson")
            if path.exists():
                self._cache_tm_estaciones = json.loads(path.read_text())
                return self._cache_tm_estaciones
            self._cache_tm_estaciones = {"type": "FeatureCollection", "features": []}
            return self._cache_tm_estaciones

    def get_tm_rutas(self) -> dict:
        """Derive TM routes from troncales + estaciones data (cached)."""
        if self._cache_tm_rutas is not None:
            return self._cache_tm_rutas

        # Load troncales to build id_trazado -> troncal name mapping
        troncales_data = self.get_tm_troncales()
        id_to_troncal: dict[str, str] = {}
        for f in troncales_data.get("features", []):
            props = f.get("properties", {})
            tz_id = props.get("id_trazado_troncal", "")
            name = props.get("troncal", "") or props.get("nombre_trazado_troncal", "")
            if tz_id and name:
                id_to_troncal[tz_id] = name

        # Load estaciones and group by troncal
        estaciones = self.get_tm_estaciones()
        rutas_map: dict[str, list[str]] = {}
        for f in estaciones.get("features", []):
            props = f.get("properties", {})
            # Properties use long names from PostGIS export
            est_name = props.get("transmisig2.tecnica.estacion_troncal.nom_est", "") or props.get(
                "nombre_est", ""
            )
            tz_id = props.get("transmisig2.tecnica.estacion_troncal.id_trazado", "") or props.get(
                "id_trazado", ""
            )
            troncal_name = id_to_troncal.get(tz_id, tz_id)
            if troncal_name and est_name:
                if troncal_name not in rutas_map:
                    rutas_map[troncal_name] = []
                rutas_map[troncal_name].append(est_name)

        rutas = [
            {"nombre": k, "estaciones": v, "tipo": "Troncal"} for k, v in rutas_map.items() if v
        ]
        self._cache_tm_rutas = {"rutas": rutas}
        return self._cache_tm_rutas

    def get_sitp_paraderos(self) -> dict:
        """Load SITP bus stops as GeoJSON (cached in memory)."""
        if self._cache_sitp_paraderos is not None:
            return self._cache_sitp_paraderos

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
                ) FROM sitp_paraderos;
            """)
            result = cur.fetchone()[0]
            cur.close()
            conn.close()
            self._cache_sitp_paraderos = result
            return result
        except Exception:
            from pathlib import Path

            path = Path("models/sitp_paraderos.geojson")
            if path.exists():
                raw = json.loads(path.read_text())
                # Normalize properties to match frontend expectations
                features = []
                for f in raw.get("features", []):
                    props = f.get("properties", {})
                    features.append(
                        {
                            "type": "Feature",
                            "geometry": f.get("geometry"),
                            "id": props.get("OBJECTID", f.get("id")),
                            "properties": {
                                "cenefa": props.get("NTRCODIGO", ""),
                                "nombre": props.get("NTRNOMBRE", ""),
                                "direccion_bandera": props.get("NTRDIRECCION", ""),
                                "objectid": props.get("OBJECTID", ""),
                            },
                        }
                    )
                result = {"type": "FeatureCollection", "features": features}
                self._cache_sitp_paraderos = result
                return result
            self._cache_sitp_paraderos = {"type": "FeatureCollection", "features": []}
            return self._cache_sitp_paraderos

    @staticmethod
    def _coords_to_paraderos(coords: list, limit: int = 50) -> list[dict]:
        """Convert coordinate list to paradero dicts."""
        return [
            {"lat": c[1] if len(c) > 1 else 0, "lon": c[0] if len(c) > 0 else 0, "nombre": ""}
            for c in coords[:limit]
        ]

    def get_sitp_rutas(self) -> dict:
        """Load SITP routes with frequency data (cached in memory)."""
        if self._cache_sitp_rutas is not None:
            return self._cache_sitp_rutas

        import json
        from pathlib import Path

        shapes_path = Path(self.SITP_SHAPES_PATH)
        freq_path = Path(self.SITP_FREQ_PATH)

        frecuencias = {}
        if freq_path.exists():
            frecuencias = json.loads(freq_path.read_text())

        rutas = []
        if shapes_path.exists():
            shapes = json.loads(shapes_path.read_text())
            for f in shapes.get("features", []):
                props = f.get("properties", {})
                ruta_code = props.get("ruta", "")
                coords = f.get("geometry", {}).get("coordinates", [])
                freq_info = frecuencias.get(ruta_code, {})
                rutas.append(
                    {
                        "ruta": ruta_code,
                        "nombre": props.get("nombre", ruta_code),
                        "tipo": freq_info.get("tipo_servicio", "Urbano"),
                        "frecuencia_min": freq_info.get("frecuencia_base_min", 15),
                        "paraderos": self._coords_to_paraderos(coords),
                    }
                )

        self._cache_sitp_rutas = {"rutas": rutas}
        return self._cache_sitp_rutas

    def get_sitp_rutas_shapes(self) -> dict:
        """Load SITP route shapes GeoJSON (cached in memory)."""
        if self._cache_sitp_shapes is not None:
            return self._cache_sitp_shapes

        import json
        from pathlib import Path

        path = Path(self.SITP_SHAPES_PATH)
        if path.exists():
            self._cache_sitp_shapes = json.loads(path.read_text())
        else:
            self._cache_sitp_shapes = {"type": "FeatureCollection", "features": []}
        return self._cache_sitp_shapes

    def _find_nearest_paradero(
        self,
        lat: float,
        lng: float,
    ) -> tuple[str, str, float]:
        """Find the nearest paradero. Returns (nombre, cenefa, dist_km)."""
        import math

        paraderos_data = self.get_sitp_paraderos()
        nearest_name, nearest_cenefa = "", ""
        nearest_dist = float("inf")
        for f in paraderos_data.get("features", []):
            geom = f.get("geometry")
            if not geom or not geom.get("coordinates"):
                continue
            coords = geom["coordinates"]
            dist = math.sqrt((coords[1] - lat) ** 2 + (coords[0] - lng) ** 2) * 111
            if dist < nearest_dist:
                nearest_dist = dist
                props = f.get("properties", {})
                nearest_name = props.get("nombre", "")
                nearest_cenefa = props.get("cenefa", "")
        return nearest_name, nearest_cenefa, nearest_dist

    @staticmethod
    def _min_dist_to_coords(lat: float, lng: float, coords: list) -> float:
        """Get minimum distance (km) from a point to a list of [lon, lat] coords."""
        import math

        min_d = float("inf")
        for c in coords:
            if len(c) < 2:
                continue
            d = math.sqrt((c[1] - lat) ** 2 + (c[0] - lng) ** 2) * 111
            if d < min_d:
                min_d = d
        return min_d

    def get_rutas_cercanas(self, lat: float, lng: float, radius_m: int) -> dict:
        """Find SITP routes with stops within radius (meters) of a point."""
        import json
        from pathlib import Path

        radius_km = radius_m / 1000.0
        shapes_path = Path(self.SITP_SHAPES_PATH)
        freq_path = Path(self.SITP_FREQ_PATH)

        if not shapes_path.exists():
            return {"rutas": []}

        shapes = json.loads(shapes_path.read_text())
        frecuencias = json.loads(freq_path.read_text()) if freq_path.exists() else {}

        nearest_name, nearest_cenefa, nearest_dist = self._find_nearest_paradero(lat, lng)

        results: list[dict] = []
        for f in shapes.get("features", []):
            props = f.get("properties", {})
            ruta_code = props.get("ruta", "")
            coords = f.get("geometry", {}).get("coordinates", [])
            min_dist = self._min_dist_to_coords(lat, lng, coords)

            if min_dist <= radius_km:
                freq_info = frecuencias.get(ruta_code, {})
                results.append(
                    {
                        "ruta": ruta_code,
                        "cenefa": nearest_cenefa if nearest_dist <= radius_km else "",
                        "nombre": props.get("nombre", ruta_code),
                        "tipo": freq_info.get("tipo_servicio", "Urbano"),
                        "frecuencia_min": freq_info.get("frecuencia_base_min", 15),
                        "distanciaMinima": round(min_dist * 1000),
                        "paraderosCercanos": [
                            {
                                "nombre": nearest_name,
                                "distancia": round(nearest_dist * 1000),
                            }
                        ]
                        if nearest_dist <= radius_km
                        else [],
                    }
                )

        results.sort(key=lambda x: x["distanciaMinima"])
        return {"rutas": results[:20]}
