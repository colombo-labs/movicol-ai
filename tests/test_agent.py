"""Tests for the conversational agent."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.modules.agent.service import AgentService


class TestAgentService:
    """Unit tests for rule-based agent."""

    def setup_method(self):
        self.service = AgentService()

    @pytest.mark.anyio
    async def test_station_query(self):
        r = await self.service.chat("Info de Calle 72", "test")
        assert "Calle 72" in r.response
        assert "station_data" in r.sources

    @pytest.mark.anyio
    async def test_congestion_query(self):
        r = await self.service.chat("¿Cómo está el tráfico a las 8am?", "test")
        assert "8:00" in r.response
        assert "%" in r.response

    @pytest.mark.anyio
    async def test_station_list(self):
        r = await self.service.chat("¿Cuántas estaciones tiene?", "test")
        assert "28" in r.response

    @pytest.mark.anyio
    async def test_default_response(self):
        r = await self.service.chat("Hola", "test")
        assert "MoviBot" in r.response

    @pytest.mark.anyio
    async def test_unknown_station(self):
        r = await self.service.chat("Info de Estación Fantasma", "test")
        # Should not crash, returns default
        assert r.response


@pytest.mark.anyio
class TestAgentAPI:
    """Integration tests for the chat endpoint."""

    async def test_chat_endpoint(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/agent/chat",
                json={"message": "¿Cuántas estaciones hay?", "session_id": "test"},
            )
        assert response.status_code == 200
        data = response.json()
        assert "28" in data["response"]
        assert data["session_id"] == "test"

    async def test_chat_validation(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/agent/chat",
                json={"message": "", "session_id": "test"},
            )
        assert response.status_code == 422
