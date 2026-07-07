"""Tests for the AgentService chat method (rule-based mode)."""

import pytest

from app.modules.agent.schemas import AppContext
from app.modules.agent.service import AgentService


@pytest.fixture
def service():
    return AgentService()


class TestAgentServiceChat:
    """Test the full chat flow in rule-based mode."""

    @pytest.mark.asyncio
    async def test_greeting_response(self, service):
        resp = await service.chat("hola", "s1")
        assert "MoviBot" in resp.response
        assert resp.session_id == "s1"

    @pytest.mark.asyncio
    async def test_plan_route_asks_confirmation(self, service):
        resp = await service.chat("ir de usaquen al centro", "s2")
        assert "Quieres" in resp.response or "quieres" in resp.response
        assert len(resp.actions) == 0  # pending, not sent yet

    @pytest.mark.asyncio
    async def test_confirm_executes_action(self, service):
        await service.chat("ir de suba a chapinero", "s3")
        resp = await service.chat("si", "s3")
        assert "Listo" in resp.response or "mapa" in resp.response
        assert len(resp.actions) == 1
        assert resp.actions[0].type == "plan_route"
        assert resp.actions[0].data["origin"] == "suba"
        assert resp.actions[0].data["destination"] == "chapinero"

    @pytest.mark.asyncio
    async def test_deny_cancels_action(self, service):
        await service.chat("ruta a kennedy", "s4")
        resp = await service.chat("no", "s4")
        assert "no hay problema" in resp.response.lower() or "Entendido" in resp.response
        assert len(resp.actions) == 0

    @pytest.mark.asyncio
    async def test_congestion_query(self, service):
        resp = await service.chat("trafico a las 7", "s5")
        assert "7" in resp.response
        assert "%" in resp.response

    @pytest.mark.asyncio
    async def test_congestion_general(self, service):
        resp = await service.chat("como esta el trafico", "s6")
        assert "%" in resp.response

    @pytest.mark.asyncio
    async def test_station_query(self, service):
        resp = await service.chat("estacion heroes", "s7")
        assert "Héroes" in resp.response or "heroes" in resp.response.lower()
        assert "troncal" in resp.response.lower()

    @pytest.mark.asyncio
    async def test_station_asks_to_show_map(self, service):
        resp = await service.chat("estacion calle 72", "s8")
        assert "mapa" in resp.response.lower()

    @pytest.mark.asyncio
    async def test_cost_query(self, service):
        resp = await service.chat("cuanto cuesta el pasaje", "s9")
        assert "3.550" in resp.response
        assert "pesos" in resp.response

    @pytest.mark.asyncio
    async def test_schedule_query(self, service):
        resp = await service.chat("a que hora abre tm", "s10")
        assert "4" in resp.response or "5" in resp.response

    @pytest.mark.asyncio
    async def test_compare_query(self, service):
        resp = await service.chat("que es mejor tm o sitp", "s11")
        assert "TransMilenio" in resp.response
        assert "SITP" in resp.response

    @pytest.mark.asyncio
    async def test_thanks_response(self, service):
        resp = await service.chat("gracias", "s12")
        assert len(resp.response) > 10
        assert resp.response != ""

    @pytest.mark.asyncio
    async def test_help_response(self, service):
        resp = await service.chat("que puedes hacer", "s13")
        assert "Planificar" in resp.response or "planificar" in resp.response.lower()

    @pytest.mark.asyncio
    async def test_context_aware_planificar(self, service):
        ctx = AppContext(module="planificar", origin="Portal Norte")
        resp = await service.chat("que hago", "s14", ctx)
        r = resp.response.lower()
        assert "destino" in r or "origen" in r or "planificador" in r

    @pytest.mark.asyncio
    async def test_context_aware_rutas(self, service):
        ctx = AppContext(module="rutas")
        resp = await service.chat("que hay", "s15", ctx)
        assert "ruta" in resp.response.lower() or "troncal" in resp.response.lower()

    @pytest.mark.asyncio
    async def test_session_history(self, service):
        """Test that history is maintained across messages."""
        await service.chat("hola", "hist1")
        await service.chat("gracias", "hist1")
        # Session should have 4 entries (2 user + 2 assistant)
        history = service._get_session_history("hist1")
        assert len(history) == 4

    @pytest.mark.asyncio
    async def test_clear_session(self, service):
        await service.chat("hola", "clear1")
        service.clear_session("clear1")
        history = service._get_session_history("clear1")
        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_unknown_falls_to_default(self, service):
        resp = await service.chat("asdfghjkl xyz", "s16")
        assert "MoviBot" in resp.response or "ayudo" in resp.response.lower()
