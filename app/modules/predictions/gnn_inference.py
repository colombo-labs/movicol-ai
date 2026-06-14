"""GNN inference module — loads trained GAT model and predicts congestion."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from app.config.settings import get_settings


class GATModel(torch.nn.Module):
    """Graph Attention Network — matches trained architecture."""

    def __init__(self, in_channels: int = 8, hidden: int = 32, heads: int = 4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden, heads=heads)
        self.conv2 = GATConv(hidden * heads, hidden, heads=1)
        self.linear = torch.nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        return torch.sigmoid(self.linear(x)).squeeze(-1)


class GNNInference:
    """Loads trained GAT model and runs inference on the graph.

    Singleton pattern — model is loaded once and predictions cached.
    """

    _instance: GNNInference | None = None
    _initialized: bool = False

    def __new__(cls) -> GNNInference:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._settings = get_settings()
        self._model: GATModel | None = None
        self._predictions: dict[str, float] = {}
        self._load()
        GNNInference._initialized = True

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and len(self._predictions) > 0

    def _load(self) -> None:
        """Load model and run inference."""
        model_path = Path(self._settings.model_path)
        features_path = model_path.parent / "node_features.pt"

        if not model_path.exists():
            return

        try:
            # Load model
            self._model = GATModel(in_channels=8, hidden=32, heads=4)
            state_dict = torch.load(model_path, map_location="cpu")
            self._model.load_state_dict(state_dict)
            self._model.eval()

            # Load pre-computed features
            if features_path.exists():
                feat_data = torch.load(features_path, map_location="cpu")
                X = feat_data["features"]
                node_ids = feat_data["node_ids"]
                mean = feat_data["mean"]
                std = feat_data["std"]
            else:
                # Fallback: load from graph directly
                self._load_from_graph(model_path)
                return

            # Normalize
            X_norm = (X - mean) / (std + 1e-8)

            # Build edge index from clean graph
            graph_path = Path(self._settings.graph_path)
            if graph_path.exists():
                import networkx as nx

                g = nx.read_graphml(graph_path)
                if hasattr(g, "to_undirected"):
                    g = g.to_undirected()
                node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
                src, dst = [], []
                for u, v in g.edges():
                    if u in node_to_idx and v in node_to_idx:
                        src.append(node_to_idx[u])
                        dst.append(node_to_idx[v])
                        src.append(node_to_idx[v])
                        dst.append(node_to_idx[u])
                edge_index = torch.tensor([src, dst], dtype=torch.long)
            else:
                return

            # Run inference
            with torch.no_grad():
                preds = self._model(X_norm, edge_index)

            for i, node_id in enumerate(node_ids):
                self._predictions[node_id] = float(preds[i].item())

        except Exception:
            self._model = None
            self._predictions = {}

    def _load_from_graph(self, model_path: Path) -> None:
        """Fallback: load features directly from graph attributes."""
        import networkx as nx

        graph_path = Path(self._settings.graph_path)
        if not graph_path.exists():
            return

        g = nx.read_graphml(graph_path)
        if hasattr(g, "to_undirected"):
            g = g.to_undirected()

        node_ids = list(g.nodes())
        features = []
        for nid in node_ids:
            data = g.nodes[nid]
            is_tm = 1.0 if data.get("tipo") == "estacion_tm" else 0.0
            features.append(
                [
                    float(data.get("lat", 0)),
                    float(data.get("lon", 0)),
                    float(data.get("degree", g.degree(nid))),
                    float(data.get("betweenness", 0)),
                    float(data.get("closeness", 0)),
                    float(data.get("siniestralidad_score", 0)),
                    float(data.get("fallecidos_cercanos", 0)),
                    is_tm,
                ]
            )

        X = torch.tensor(features, dtype=torch.float32)
        norm_path = model_path.parent / "norm_params.pt"
        if norm_path.exists():
            norm = torch.load(norm_path, map_location="cpu")
            X = (X - norm["mean"]) / (norm["std"] + 1e-8)

        # Edge index
        node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        src, dst = [], []
        for u, v in g.edges():
            if u in node_to_idx and v in node_to_idx:
                src.append(node_to_idx[u])
                dst.append(node_to_idx[v])
                src.append(node_to_idx[v])
                dst.append(node_to_idx[u])
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        with torch.no_grad():
            preds = self._model(X, edge_index)

        for i, nid in enumerate(node_ids):
            self._predictions[nid] = float(preds[i].item())

    def get_congestion(self, station_id: str) -> float:
        """Get predicted congestion for a station (0-1)."""
        return self._predictions.get(station_id, 0.5)

    def get_all_predictions(self) -> dict[str, float]:
        """Get all predictions."""
        return self._predictions.copy()
