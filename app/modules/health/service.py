"""Health check service."""


class HealthService:
    """Service to check application health."""

    def check(self) -> dict:
        """Return health status."""
        return {
            "status": "ok",
            "service": "movicol-ai",
            "version": "0.1.0",
        }
