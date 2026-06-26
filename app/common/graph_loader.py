"""Shared graph loading utilities for real-data-first behavior."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
from typing import Optional

_APP_DIR = Path(__file__).resolve().parent.parent
_AI_ROOT = _APP_DIR.parent
_COLOMBO_ROOT = _AI_ROOT.parent

_DEFAULT_GRAPH_CANDIDATES = [
    _COLOMBO_ROOT / "movicol-data" / "data" / "graphs" / "grafo_movilidad_bogota_enriched.graphml",
    _COLOMBO_ROOT / "movicol-data" / "data" / "graphs" / "grafo_integrado_bogota.graphml",
    _COLOMBO_ROOT / "movicol-data" / "data" / "graphs" / "grafo_movilidad_bogota.graphml",
    _COLOMBO_ROOT / "movicol-data" / "graphs" / "grafo_movilidad_bogota_enriched.graphml",
    _COLOMBO_ROOT / "movicol-data" / "graphs" / "grafo_integrado_bogota.graphml",
    _COLOMBO_ROOT / "movicol-data" / "graphs" / "grafo_movilidad_bogota.graphml",
    _AI_ROOT / "models" / "graph_clean.graphml",
    _AI_ROOT / "models" / "graph.graphml",
]


def resolve_graph_path(config_graph_path: Optional[str] = None) -> Optional[Path]:
    """Resolve the best available graph path, prioritizing movicol-data outputs."""
    candidates: list[Path] = []

    if config_graph_path:
        configured = Path(config_graph_path)
        if not configured.is_absolute():
            configured = (_AI_ROOT / configured).resolve()
        candidates.append(configured)

    candidates.extend(_DEFAULT_GRAPH_CANDIDATES)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists() and candidate.is_file():
            return candidate

    return None


def load_graph_from_path(path: Path) -> nx.Graph:
    """Load a graphml file and normalize it to an undirected simple graph."""
    raw = nx.read_graphml(path)
    if isinstance(raw, (nx.MultiGraph, nx.MultiDiGraph)):
        return nx.Graph(raw)
    if raw.is_directed():
        return raw.to_undirected()
    return raw


def ensure_distance_weights(g: nx.Graph) -> None:
    """Ensure edge distance/time weights exist for routing."""
    for u, v in g.edges():
        edge = g.edges[u, v]
        if edge.get("distance_km") is not None and edge.get("base_time_min") is not None:
            continue

        u_data = g.nodes[u]
        v_data = g.nodes[v]
        lat1 = float(u_data.get("lat", 0))
        lon1 = float(u_data.get("lon", 0))
        lat2 = float(v_data.get("lat", 0))
        lon2 = float(v_data.get("lon", 0))
        dist = ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5 * 111
        edge["distance_km"] = round(dist, 3)
        edge["base_time_min"] = round(dist * 2.0, 1)
