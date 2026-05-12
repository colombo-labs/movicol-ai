"""GNN Training script - Graph Attention Network for congestion prediction."""

from pathlib import Path

# TODO: Implement full training pipeline
# 1. Load graph from movicol-data (GraphML or PostGIS)
# 2. Build PyTorch Geometric Data object
# 3. Define GAT model architecture
# 4. Train with temporal features (day, hour, holiday)
# 5. Evaluate (MSE, RMSE, MAE)
# 6. Save model to models/gat_model.pt

MODELS_DIR = Path(__file__).parent.parent / "models"


def main() -> None:
    """Run training pipeline."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("🧠 Training GNN model...")
    print("   TODO: Implement training pipeline")
    print("   - Load graph + features")
    print("   - Build PyG Data object")
    print("   - Train GAT model")
    print("   - Save to models/gat_model.pt")


if __name__ == "__main__":
    main()
