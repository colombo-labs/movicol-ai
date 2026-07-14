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
        route_prediction_router, prefix="/route-prediction", tags=["Route Prediction"]
    )
    application.include_router(demand_router, prefix="/demand", tags=["Demand Prediction"])
    application.include_router(
        siniestralidad_router, prefix="/siniestralidad", tags=["Siniestralidad"]
    )

    return application


app = create_app()


@app.on_event("startup")
async def load_sitp_data():
    """Fetch SITP route data and TM troncal geometries from backend after startup."""
    import asyncio

    async def _fetch():
        await asyncio.sleep(15)  # Wait for backend to be ready
        try:
            from app.modules.route_prediction.router import service

            await service._ensure_sitp_loaded()
            await service._load_troncal_geometries()

            # Retry SITP once if first attempt failed (backend may need cache warm-up)
            if not service._sitp_routes:
                print("[Startup] SITP not loaded, retrying in 20s...")
                await asyncio.sleep(20)
                service._sitp_fetch_failed = False
                await service._ensure_sitp_loaded()
        except Exception as e:
            print(f"[Startup] Data fetch failed (non-critical): {e}")

    _task = asyncio.create_task(_fetch())  # noqa: F841
