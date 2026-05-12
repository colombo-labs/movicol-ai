"""Health check router."""

from fastapi import APIRouter

from app.modules.health.service import HealthService

router = APIRouter()
service = HealthService()


@router.get("")
async def health_check():
    """Check service health status."""
    return service.check()
