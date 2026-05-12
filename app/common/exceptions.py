"""Custom exceptions for the application."""

from fastapi import HTTPException, status


class ModelNotLoadedError(HTTPException):
    """Raised when the GNN model is not available."""

    def __init__(self) -> None:
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GNN model not loaded. Run training first.",
        )


class GraphNotFoundError(HTTPException):
    """Raised when the graph data is not available."""

    def __init__(self) -> None:
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph data not available. Load data first.",
        )


class StationNotFoundError(HTTPException):
    """Raised when a station is not found in the graph."""

    def __init__(self, station_id: str) -> None:
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Station '{station_id}' not found in graph.",
        )
