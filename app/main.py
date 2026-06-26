"""MoviCol AI - FastAPI Application Entry Point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import get_settings
from app.modules.agent.router import router as agent_router
from app.modules.demand_prediction.router import router as demand_router
from app.modules.graph.router import router as graph_router
from app.modules.health.router import router as health_router
from app.modules.predictions.router import router as predictions_router
from app.modules.route_prediction.router import router as route_prediction_router
from app.modules.siniestralidad.router import router as siniestralidad_router

settings = get_settings()


def create_app() -> FastAPI:
    """Application factory."""
    application = FastAPI(
        title="MoviCol AI",
        description="GNN predictions and conversational agent for urban mobility",
        version="0.1.0",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register module routers
    application.include_router(health_router, prefix="/health", tags=["Health"])
    application.include_router(predictions_router, prefix="/predictions", tags=["Predictions"])
    application.include_router(graph_router, prefix="/graph", tags=["Graph"])
    application.include_router(agent_router, prefix="/agent", tags=["Agent"])
    application.include_router(
        route_prediction_router, prefix="/api/v1/predict-route", tags=["Route Prediction"]
    )
    application.include_router(demand_router, prefix="/demand", tags=["Demand Prediction"])
    application.include_router(
        siniestralidad_router, prefix="/siniestralidad", tags=["Siniestralidad"]
    )

    return application


app = create_app()
