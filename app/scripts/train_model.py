"""Train or retrain the GAT model for congestion prediction.

Uses the clean graph and node features to train a Graph Attention Network.
Target: predict congestion level (0-1) per node based on graph structure.

Usage:
    python -m app.scripts.train_model [--epochs 100] [--lr 0.001]
"""

from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
GRAPH_PATH = MODELS_DIR / "graph_clean.graphml"
FEATURES_PATH = MODELS_DIR / "node_features.pt"
MODEL_OUTPUT = MODELS_DIR / "gat_best.pt"
NORM_OUTPUT = MODELS_DIR / "norm_params.pt"


class GATModel(torch.nn.Module):
    """Graph Attention Network for congestion prediction."""

    def __init__(self, in_channels: int = 8, hidden: int = 32, heads: int = 4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden, heads=heads)
        self.conv2 = GATConv(hidden * heads, hidden, heads=1)
        self.linear = torch.nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        return torch.sigmoid(self.linear(x)).squeeze(-1)


def generate_targets(g: nx.Graph, node_ids: list[str]) -> torch.Tensor:
    """Generate synthetic congestion targets based on graph properties.

    In production, these would come from real sensor data.
    For now: congestion ~ f(degree, betweenness, siniestralidad, is_tm)
    """
    targets = []
    for node_id in node_ids:
        data = g.nodes[node_id]
        degree = float(data.get("degree", g.degree(node_id)))
        betweenness = float(data.get("betweenness", 0))
        siniestralidad = float(data.get("siniestralidad_score", 0))
        is_tm = 1.0 if data.get("tipo") == "estacion_tm" else 0.0

        # Synthetic target: higher degree + betweenness + TM = more congestion
        base = 0.3
        base += min(0.3, degree / 15)  # degree contribution
        base += min(0.2, betweenness * 20)  # betweenness contribution
        base += 0.1 * is_tm  # TM stations are busier
        base += min(0.1, siniestralidad * 0.5)  # siniestralidad adds risk

        # Add noise
        noise = np.random.normal(0, 0.05)
        target = max(0.0, min(1.0, base + noise))
        targets.append(target)

    return torch.tensor(targets, dtype=torch.float32)


def build_edge_index(g: nx.Graph, node_ids: list[str]) -> torch.Tensor:
    """Build edge index tensor for PyG."""
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    src, dst = [], []
    for u, v in g.edges():
        if u in node_to_idx and v in node_to_idx:
            src.append(node_to_idx[u])
            dst.append(node_to_idx[v])
            # Undirected: add reverse
            src.append(node_to_idx[v])
            dst.append(node_to_idx[u])
    return torch.tensor([src, dst], dtype=torch.long)


def train(epochs: int = 100, lr: float = 0.001) -> None:
    """Train the GAT model."""
    print("=" * 60)
    print("🧠 GAT MODEL TRAINING")
    print("=" * 60)

    # Load data
    print("\n📥 Loading data...")
    feat_data = torch.load(FEATURES_PATH, map_location="cpu")
    X = feat_data["features"]
    node_ids = feat_data["node_ids"]
    mean = feat_data["mean"]
    std = feat_data["std"]

    g = nx.read_graphml(GRAPH_PATH)
    print(f"   Nodes: {len(node_ids)} | Features: {X.shape[1]}")

    # Normalize features
    X_norm = (X - mean) / (std + 1e-8)

    # Build edge index
    edge_index = build_edge_index(g, node_ids)
    print(f"   Edges: {edge_index.shape[1] // 2}")

    # Generate targets
    print("   Generating synthetic targets...")
    y = generate_targets(g, node_ids)
    print(f"   Target range: [{y.min():.3f}, {y.max():.3f}], mean: {y.mean():.3f}")

    # Train/val split (80/20)
    n = len(node_ids)
    perm = torch.randperm(n)
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[: int(0.8 * n)]] = True
    val_mask = ~train_mask

    # Model
    model = GATModel(in_channels=8, hidden=32, heads=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # Training loop
    print(f"\n🏋️ Training for {epochs} epochs...")
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(X_norm, edge_index)
        loss = F.mse_loss(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_norm, edge_index)
            val_loss = F.mse_loss(val_out[val_mask], y[val_mask])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

        if epoch % 20 == 0 or epoch == 1:
            print(f"   Epoch {epoch:3d} | Train MSE: {loss:.4f} | Val MSE: {val_loss:.4f}")

    # Save best model
    print(f"\n💾 Saving model (best val MSE: {best_val_loss:.4f})...")
    torch.save(best_state, MODEL_OUTPUT)
    torch.save({"mean": mean, "std": std}, NORM_OUTPUT)

    # Final evaluation
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(X_norm, edge_index)
        mse = F.mse_loss(preds, y).item()
        rmse = mse**0.5

    print("\n📊 Final metrics:")
    print(f"   MSE:  {mse:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   Pred range: [{preds.min():.3f}, {preds.max():.3f}]")
    print(f"\n✅ Model saved to {MODEL_OUTPUT}")


if __name__ == "__main__":
    train()
