"""ST-GAT inference module — loads Danna's trained model for demand prediction."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from app.config.settings import get_settings


class STGATTransmilenio(torch.nn.Module):
    """Spatial-Temporal Graph Attention Network (matches Danna's trained architecture)."""

    def __init__(
        self, in_channels: int = 1, hidden_channels: int = 24, out_channels: int = 1, heads: int = 4
    ):
        super().__init__()
        self.gat = GATConv(in_channels, hidden_channels, heads=heads, edge_dim=2, dropout=0.1)
        self.gru = torch.nn.GRU(hidden_channels * heads, hidden_channels, batch_first=True)
        self.fully_connected = torch.nn.Linear(hidden_channels, out_channels)

    def forward(
        self, x_seq: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        w_size, n_nodes, _ = x_seq.shape
        gat_outputs = []
        for t in range(w_size):
            h_spatial = F.elu(self.gat(x_seq[t], edge_index, edge_attr))
            gat_outputs.append(h_spatial.unsqueeze(0))
        gat_outputs = torch.cat(gat_outputs, dim=0).transpose(0, 1)
        gru_out, _ = self.gru(gat_outputs)
        return self.fully_connected(gru_out[:, -1, :])


class DemandInference:
    """Loads ST-GAT model and predicts passenger demand per station."""

    _instance: DemandInference | None = None
    _initialized: bool = False

    def __new__(cls) -> DemandInference:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._model: STGATTransmilenio | None = None
        self._graph_data: torch.Tensor | None = None
        self._station_names: list[str] = []
        self._load()
        DemandInference._initialized = True

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._graph_data is not None

    @property
    def station_count(self) -> int:
        return len(self._station_names)

    def _load(self) -> None:
        settings = get_settings()
        model_dir = Path(settings.model_path).parent
        st_gat_path = model_dir / "st_gat_transmilenio_optimizado.pth"
        graph_data_path = model_dir / "transmilenio_graph_data.pt"

        if not st_gat_path.exists() or not graph_data_path.exists():
            return

        try:
            self._model = STGATTransmilenio(
                in_channels=1, hidden_channels=24, out_channels=1, heads=4
            )
            state_dict = torch.load(st_gat_path, map_location="cpu", weights_only=False)
            self._model.load_state_dict(state_dict)
            self._model.eval()

            self._graph_data = torch.load(graph_data_path, map_location="cpu", weights_only=False)

            # Try loading station names from CSV
            csv_path = model_dir / "aristas_infraestructura_gat.csv"
            if csv_path.exists():
                import csv

                names = set()
                with open(csv_path) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        names.add(row.get("source_name", ""))
                        names.add(row.get("target_name", ""))
                self._station_names = sorted(n for n in names if n)

        except Exception:
            self._model = None
            self._graph_data = None

    def predict_demand(
        self, current_demand: list[float] | None = None, window_size: int = 4
    ) -> dict[int, float]:
        """Predict next-interval demand for all stations.

        Args:
            current_demand: Optional current demand values per station.
                If None, uses zeros (cold-start prediction).
            window_size: Number of historical intervals (default 4 = 1 hour).

        Returns:
            Dict mapping station_index → predicted_passengers.
        """
        if not self.is_loaded:
            return {}

        num_nodes = self._graph_data.num_nodes
        edge_index = self._graph_data.edge_index
        edge_attr = self._graph_data.edge_attr

        # Build input sequence
        if current_demand and len(current_demand) == num_nodes:
            x_now = torch.tensor(current_demand, dtype=torch.float).unsqueeze(1)
        else:
            x_now = self._graph_data.x

        # Repeat for window (simulating steady-state)
        x_seq = x_now.unsqueeze(0).repeat(window_size, 1, 1)

        with torch.no_grad():
            pred = self._model(x_seq, edge_index, edge_attr)

        results = {}
        for i in range(num_nodes):
            results[i] = max(0.0, float(pred[i, 0].item()))
        return results

    def get_station_demand_score(self, station_idx: int) -> float:
        """Get normalized demand score (0-1) for a station using default prediction."""
        preds = self.predict_demand()
        if station_idx not in preds:
            return 0.5
        max_val = max(preds.values()) if preds else 1.0
        return min(1.0, preds[station_idx] / max_val) if max_val > 0 else 0.0
