"""Static graph data for Bogotá TransMilenio — Troncal Caracas.

Real station coordinates and connections. Used as fallback when
the full graph from movicol-data is not available.
"""

import networkx as nx

# Real TransMilenio stations — Troncal Caracas (North → South)
CARACAS_STATIONS: list[dict] = [
    {
        "id": "portal_norte",
        "name": "Portal Norte",
        "lat": 4.7586,
        "lon": -74.0453,
        "troncal": "Caracas",
    },
    {"id": "toberin", "name": "Toberín", "lat": 4.7459, "lon": -74.0478, "troncal": "Caracas"},
    {
        "id": "cardio_infantil",
        "name": "Cardio Infantil",
        "lat": 4.7330,
        "lon": -74.0500,
        "troncal": "Caracas",
    },
    {"id": "calle_146", "name": "Calle 146", "lat": 4.7200, "lon": -74.0520, "troncal": "Caracas"},
    {"id": "calle_142", "name": "Calle 142", "lat": 4.7130, "lon": -74.0525, "troncal": "Caracas"},
    {"id": "alcala", "name": "Alcalá", "lat": 4.7060, "lon": -74.0528, "troncal": "Caracas"},
    {"id": "calle_127", "name": "Calle 127", "lat": 4.7010, "lon": -74.0530, "troncal": "Caracas"},
    {
        "id": "pepe_sierra",
        "name": "Pepe Sierra",
        "lat": 4.6920,
        "lon": -74.0540,
        "troncal": "Caracas",
    },
    {"id": "calle_106", "name": "Calle 106", "lat": 4.6870, "lon": -74.0545, "troncal": "Caracas"},
    {"id": "calle_100", "name": "Calle 100", "lat": 4.6830, "lon": -74.0550, "troncal": "Caracas"},
    {"id": "virrey", "name": "Virrey", "lat": 4.6730, "lon": -74.0560, "troncal": "Caracas"},
    {"id": "calle_85", "name": "Calle 85", "lat": 4.6680, "lon": -74.0565, "troncal": "Caracas"},
    {"id": "heroes", "name": "Héroes", "lat": 4.6620, "lon": -74.0575, "troncal": "Caracas"},
    {"id": "calle_72", "name": "Calle 72", "lat": 4.6600, "lon": -74.0580, "troncal": "Caracas"},
    {"id": "calle_63", "name": "Calle 63", "lat": 4.6530, "lon": -74.0590, "troncal": "Caracas"},
    {"id": "flores", "name": "Flores", "lat": 4.6480, "lon": -74.0600, "troncal": "Caracas"},
    {"id": "calle_45", "name": "Calle 45", "lat": 4.6430, "lon": -74.0610, "troncal": "Caracas"},
    {"id": "marly", "name": "Marly", "lat": 4.6350, "lon": -74.0630, "troncal": "Caracas"},
    {"id": "calle_34", "name": "Calle 34", "lat": 4.6280, "lon": -74.0650, "troncal": "Caracas"},
    {"id": "calle_26", "name": "Calle 26", "lat": 4.6250, "lon": -74.0700, "troncal": "Caracas"},
    {"id": "calle_22", "name": "Calle 22", "lat": 4.6180, "lon": -74.0720, "troncal": "Caracas"},
    {"id": "calle_19", "name": "Calle 19", "lat": 4.6120, "lon": -74.0730, "troncal": "Caracas"},
    {
        "id": "av_jimenez",
        "name": "Av. Jiménez",
        "lat": 4.6050,
        "lon": -74.0740,
        "troncal": "Caracas",
    },
    {
        "id": "tercer_milenio",
        "name": "Tercer Milenio",
        "lat": 4.6000,
        "lon": -74.0750,
        "troncal": "Caracas",
    },
    {
        "id": "hospitales",
        "name": "Hospitales",
        "lat": 4.5930,
        "lon": -74.0760,
        "troncal": "Caracas",
    },
    {"id": "nari_sur", "name": "Nariño Sur", "lat": 4.5850, "lon": -74.0770, "troncal": "Caracas"},
    {"id": "fucha", "name": "Fucha", "lat": 4.5780, "lon": -74.0780, "troncal": "Caracas"},
    {
        "id": "portal_sur",
        "name": "Portal Sur (Usme)",
        "lat": 4.5750,
        "lon": -74.0800,
        "troncal": "Caracas",
    },
]

# Simulated congestion patterns by hour (0-23) — peak hours have higher values
CONGESTION_BY_HOUR: dict[int, float] = {
    0: 0.05,
    1: 0.03,
    2: 0.02,
    3: 0.02,
    4: 0.05,
    5: 0.10,
    6: 0.30,
    7: 0.50,
    8: 0.55,
    9: 0.45,
    10: 0.30,
    11: 0.25,
    12: 0.35,
    13: 0.30,
    14: 0.25,
    15: 0.30,
    16: 0.40,
    17: 0.55,
    18: 0.60,
    19: 0.45,
    20: 0.30,
    21: 0.20,
    22: 0.10,
    23: 0.05,
}

# Segments with historically higher congestion (bottlenecks)
HIGH_CONGESTION_SEGMENTS: set[tuple[str, str]] = {
    ("calle_127", "pepe_sierra"),
    ("pepe_sierra", "calle_106"),
    ("calle_100", "virrey"),
    ("calle_26", "calle_22"),
    ("calle_45", "marly"),
}


def build_caracas_graph() -> nx.DiGraph:
    """Build a directed graph of the Troncal Caracas corridor."""
    g = nx.DiGraph()

    for station in CARACAS_STATIONS:
        g.add_node(
            station["id"],
            name=station["name"],
            lat=station["lat"],
            lon=station["lon"],
            troncal=station["troncal"],
        )

    # Connect consecutive stations (bidirectional)
    for i in range(len(CARACAS_STATIONS) - 1):
        s1 = CARACAS_STATIONS[i]
        s2 = CARACAS_STATIONS[i + 1]

        # Weight = approximate distance in km (haversine simplified)
        dlat = abs(s1["lat"] - s2["lat"])
        dlon = abs(s1["lon"] - s2["lon"])
        dist_km = ((dlat**2 + dlon**2) ** 0.5) * 111  # rough km

        # Base travel time: ~2 min per km at 30 km/h average
        base_time = dist_km * 2.0

        g.add_edge(
            s1["id"], s2["id"], distance_km=round(dist_km, 2), base_time_min=round(base_time, 1)
        )
        g.add_edge(
            s2["id"], s1["id"], distance_km=round(dist_km, 2), base_time_min=round(base_time, 1)
        )

    return g
