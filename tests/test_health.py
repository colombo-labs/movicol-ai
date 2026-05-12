"""Health endpoint tests."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    """Test health endpoint returns ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "movicol-ai"
