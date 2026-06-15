"""Graph cleaning and preparation for ML and routing.

Takes the raw MultiDiGraph and produces:
1. A clean simple Graph (largest component only, shortest edge per pair)
2. Node features for GNN training
3. Edge weights for Dijkstra routing

Usage:
    python -m app.scripts.clean_graph
"""

from pathlib import Path

import networkx as nx
import torch

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
RAW_GRAPH = MODELS_DIR / "graph.graphml"
CLEAN_GRAPH = MODELS_DIR / "graph_clean.graphml"
FEATURES_FILE = MODELS_DIR / "node_features.pt"


def load_raw_graph() -> nx.MultiDiGraph:
    """Load the raw MultiDiGraph."""
    print(f"📥 Loading {RAW_GRAPH}...")
    g = nx.read_graphml(RAW_GRAPH)
    print(f"   Type: {type(g).__name__}")
    print(f"   Nodes: {g.number_of_nodes():,}")
    print(f"   Edges: {g.number_of_edges():,}")
    return g


def extract_largest_component(g: nx.Graph) -> nx.Graph:
    """Extract the largest connected component."""
    if g.is_directed():
        components = sorted(nx.weakly_connected_components(g), key=len, reverse=True)
    else:
        components = sorted(nx.connected_components(g), key=len, reverse=True)

    largest = components[0]
    subgraph = g.subgraph(largest).copy()
    print(
        f"   Largest component: {len(largest):,} nodes"
        f" ({len(largest) / g.number_of_nodes() * 100:.1f}%)"
    )
    print(f"   Removed: {g.number_of_nodes() - len(largest)} isolated nodes")
    return subgraph


def simplify_graph(g: nx.MultiDiGraph) -> nx.Graph:
    """Convert MultiDiGraph to simple undirected Graph, keeping shortest edge per pair."""
    print("\n🔧 Simplifying graph...")
    simple = nx.Graph()

    # Add all nodes with attributes
    for node_id, data in g.nodes(data=True):
        simple.add_node(node_id, **data)

    # For each pair of connected nodes, keep the shortest edge
    seen = set()
    for u, v, data in g.edges(data=True):
        pair = tuple(sorted([u, v]))
        dist = float(data.get("longitud_km", 0))
        if dist == 0:
            # Calculate from coordinates
            u_data = g.nodes[u]
            v_data = g.nodes[v]
            lat1, lon1 = float(u_data.get("lat", 0)), float(u_data.get("lon", 0))
            lat2, lon2 = float(v_data.get("lat", 0)), float(v_data.get("lon", 0))
            dist = ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5 * 111

        if pair not in seen:
            simple.add_edge(
                pair[0], pair[1], distance_km=round(dist, 3), base_time_min=round(dist * 2.0, 1)
            )
            seen.add(pair)
        else:
            # Keep shorter edge
            existing = simple.edges[pair[0], pair[1]].get("distance_km", float("inf"))
            if dist < existing and dist > 0:
                simple.edges[pair[0], pair[1]]["distance_km"] = round(dist, 3)
                simple.edges[pair[0], pair[1]]["base_time_min"] = round(dist * 2.0, 1)

    print(f"   Simple edges: {simple.number_of_edges():,}")
    pct = simple.number_of_edges() / g.number_of_edges() * 100
    print(f"   Compression: {g.number_of_edges():,} → {simple.number_of_edges():,} ({pct:.0f}%)")
    return simple


def compute_centrality(g: nx.Graph) -> nx.Graph:
    """Compute centrality metrics for all nodes."""
    print("\n📊 Computing centrality metrics...")
    n = g.number_of_nodes()

    # Degree centrality (fast)
    print("   Degree centrality...")
    for node_id in g.nodes():
        g.nodes[node_id]["degree"] = g.degree(node_id)

    # Betweenness (approximate for large graphs)
    print("   Betweenness centrality (approximate)...")
    k = min(500, n)  # Sample 500 nodes for approximation
    betweenness = nx.betweenness_centrality(g, k=k)
    for node_id, val in betweenness.items():
        g.nodes[node_id]["betweenness"] = round(val, 6)

    # Closeness (approximate)
    print("   Closeness centrality...")
    # Only compute for largest component to avoid inf
    closeness = nx.closeness_centrality(g)
    for node_id, val in closeness.items():
        g.nodes[node_id]["closeness"] = round(val, 6)

    print(f"   ✅ Centrality computed for {n:,} nodes")
    return g


def extract_features(g: nx.Graph) -> tuple[torch.Tensor, list[str]]:
    """Extract node feature matrix for GNN training."""
    print("\n🧬 Extracting node features...")
    node_ids = list(g.nodes())
    features = []

    for node_id in node_ids:
        data = g.nodes[node_id]
        is_tm = 1.0 if data.get("tipo") == "estacion_tm" else 0.0
        feat = [
            float(data.get("lat", 0)),
            float(data.get("lon", 0)),
            float(data.get("degree", g.degree(node_id))),
            float(data.get("betweenness", 0)),
            float(data.get("closeness", 0)),
            float(data.get("siniestralidad_score", 0)),
            float(data.get("fallecidos_cercanos", 0)),
            is_tm,
        ]
        features.append(feat)

    X = torch.tensor(features, dtype=torch.float32)

    # Compute normalization params
    mean = X.mean(dim=0)
    std = X.std(dim=0)

    print(f"   Feature matrix: {X.shape}")
    print(f"   Mean: {mean.numpy().round(4)}")
    print(f"   Std: {std.numpy().round(4)}")

    return X, node_ids, mean, std


def save_clean_graph(g: nx.Graph) -> None:
    """Save cleaned graph."""
    print(f"\n💾 Saving clean graph to {CLEAN_GRAPH}...")
    nx.write_graphml(g, CLEAN_GRAPH)
    size_mb = CLEAN_GRAPH.stat().st_size / (1024 * 1024)
    print(f"   Size: {size_mb:.1f} MB")


def save_features(
    X: torch.Tensor, node_ids: list[str], mean: torch.Tensor, std: torch.Tensor
) -> None:
    """Save feature matrix and normalization params."""
    print(f"💾 Saving features to {FEATURES_FILE}...")
    torch.save(
        {
            "features": X,
            "node_ids": node_ids,
            "mean": mean,
            "std": std,
            "feature_names": [
                "lat",
                "lon",
                "degree",
                "betweenness",
                "closeness",
                "siniestralidad_score",
                "fallecidos_cercanos",
                "is_tm",
            ],
        },
        FEATURES_FILE,
    )


def main() -> None:
    """Run the full graph cleaning pipeline."""
    print("=" * 60)
    print("🧹 GRAPH CLEANING & FEATURE EXTRACTION")
    print("=" * 60)

    # 1. Load raw
    raw = load_raw_graph()

    # 2. Simplify (MultiDiGraph → Graph)
    simple = simplify_graph(raw)

    # 3. Extract largest component
    print("\n🔗 Extracting largest component...")
    clean = extract_largest_component(simple)

    # 4. Compute centrality
    clean = compute_centrality(clean)

    # 5. Extract features
    X, node_ids, mean, std = extract_features(clean)

    # 6. Save
    save_clean_graph(clean)
    save_features(X, node_ids, mean, std)

    # Summary
    print("\n" + "=" * 60)
    print("✅ DONE")
    print(f"   Clean graph: {clean.number_of_nodes():,} nodes, {clean.number_of_edges():,} edges")
    print(f"   Features: {X.shape[0]} × {X.shape[1]}")
    print(f"   Files: {CLEAN_GRAPH.name}, {FEATURES_FILE.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
