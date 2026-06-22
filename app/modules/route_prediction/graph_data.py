"""Graph data for Bogotá TransMilenio — ALL troncales (153 stations, 13 troncales).

Loads real station data from tm_stations_all.json (generated from official GIS data).
Falls back to hardcoded Caracas if file not found.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

_DATA_FILE = Path(__file__).parent / "tm_stations_all.json"
_RUTAS_FILE = Path(__file__).parent / "tm_rutas_all.json"

# Congestion by hour (system-wide average)
CONGESTION_BY_HOUR: dict[int, float] = {
    0: 0.05, 1: 0.03, 2: 0.02, 3: 0.02, 4: 0.05, 5: 0.15,
    6: 0.35, 7: 0.55, 8: 0.60, 9: 0.45, 10: 0.30, 11: 0.28,
    12: 0.35, 13: 0.32, 14: 0.30, 15: 0.35, 16: 0.45, 17: 0.60,
    18: 0.65, 19: 0.50, 20: 0.30, 21: 0.20, 22: 0.10, 23: 0.07,
}

# Troncal metadata
TRONCALES: dict[str, dict] = {
    "Caracas": {"color": "#E53935", "stations": 17},
    "Autopista Norte": {"color": "#3949AB", "stations": 17},
    "Suba": {"color": "#F06292", "stations": 14},
    "Calle 80": {"color": "#FBC02D", "stations": 13},
    "NQS Central": {"color": "#00897B", "stations": 13},
    "NQS Sur": {"color": "#5E35B1", "stations": 13},
    "Américas": {"color": "#F57C00", "stations": 17},
    "Calle 26": {"color": "#43A047", "stations": 14},
    "Carrera 10": {"color": "#00ACC1", "stations": 10},
    "Caracas Sur": {"color": "#C62828", "stations": 14},
    "Eje Ambiental": {"color": "#6D4C41", "stations": 3},
    "Soacha": {"color": "#546E7A", "stations": 4},
    "Tunal": {"color": "#7B1FA2", "stations": 3},
    "Carrera 7": {"color": "#1565C0", "stations": 1},
}


def _load_stations() -> list[dict]:
    """Load all TM stations from JSON file."""
    if _DATA_FILE.exists():
        with open(_DATA_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def _load_rutas() -> list[dict]:
    """Load all TM routes from JSON file."""
    if _RUTAS_FILE.exists():
        with open(_RUTAS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


# All TM stations (153)
TM_STATIONS: list[dict] = _load_stations()

# All TM routes (126)
TM_RUTAS: list[dict] = _load_rutas()

# Backward compat: Caracas-only subset
CARACAS_STATIONS: list[dict] = [s for s in TM_STATIONS if s["troncal"] == "Caracas"]


def build_tm_graph() -> nx.Graph:
    """Build full TransMilenio graph with all troncales + transfer connections."""
    import math

    def _hav(lat1, lon1, lat2, lon2):
        R = 6371
        dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    G = nx.Graph()
    by_troncal: dict[str, list[dict]] = {}
    for s in TM_STATIONS:
        by_troncal.setdefault(s["troncal"], []).append(s)

    # Add nodes and sequential edges within troncal
    for troncal, stations in by_troncal.items():
        for i, s in enumerate(stations):
            G.add_node(s["id"], name=s["name"], lat=s["lat"], lon=s["lon"], troncal=troncal)
            if i > 0:
                prev = stations[i - 1]
                dist = _hav(prev["lat"], prev["lon"], s["lat"], s["lon"])
                G.add_edge(prev["id"], s["id"], troncal=troncal, distance_km=round(dist, 3))

    # Add transfer edges between troncales (stations < 800m apart)
    nodes = list(G.nodes(data=True))
    for i, (n1, d1) in enumerate(nodes):
        for j, (n2, d2) in enumerate(nodes):
            if i >= j or d1.get("troncal") == d2.get("troncal"):
                continue
            dist = _hav(float(d1["lat"]), float(d1["lon"]), float(d2["lat"]), float(d2["lon"]))
            if dist < 0.8:  # < 800m
                G.add_edge(n1, n2, troncal="transbordo", distance_km=round(dist, 3))

    return G


def build_caracas_graph() -> nx.Graph:
    """Build Caracas-only graph (backward compat)."""
    G = nx.Graph()
    for i, s in enumerate(CARACAS_STATIONS):
        G.add_node(s["id"], name=s["name"], lat=s["lat"], lon=s["lon"], troncal="Caracas")
        if i > 0:
            prev = CARACAS_STATIONS[i - 1]
            G.add_edge(prev["id"], s["id"], troncal="Caracas")
    return G
