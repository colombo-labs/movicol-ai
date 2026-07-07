"""Tests to maximize coverage of agent module."""

import pytest

from app.modules.agent.intents import (
    detect_intent,
    find_nearby_stations,
    get_greeting_response,
)
from app.modules.agent.schemas import AppContext
from app.modules.agent.service import AgentService
from app.modules.agent.tools import (
    find_route_info,
    find_station,
    get_congestion,
    get_risk_by_zone,
    list_troncales,
    plan_route,
)


class TestToolsFunctions:
    """Test all tool functions directly to cover tools.py."""

    def test_plan_route_tool(self):
        result = plan_route.invoke({"origin": "suba", "destination": "centro", "mode": "tm"})
        assert "ACTION:plan_route" in result
        assert "suba" in result
        assert "centro" in result

    def test_find_station_found(self):
        result = find_station.invoke({"query": "heroes"})
        assert "Héroes" in result or "ACTION:show_station" in result

    def test_find_station_not_found(self):
        result = find_station.invoke({"query": "xyznonexistent"})
        assert "No encontré" in result

    def test_find_route_info_found(self):
        result = find_route_info.invoke({"route_code": "J74"})
        assert "J74" in result or "Ruta" in result

    def test_find_route_info_not_found(self):
        result = find_route_info.invoke({"route_code": "ZZZ999"})
        assert "No encontré" in result

    def test_find_route_info_partial(self):
        result = find_route_info.invoke({"route_code": "J"})
        assert "encontradas" in result or "No encontré" in result

    def test_get_congestion_with_hour(self):
        result = get_congestion.invoke({"hour": 8})
        assert "8" in result
        assert "ACTION:show_congestion" in result

    def test_get_congestion_no_hour(self):
        result = get_congestion.invoke({"hour": None})
        assert "Pico" in result or "pico" in result.lower()

    def test_get_risk_by_zone(self):
        result = get_risk_by_zone.invoke({"hour": 18})
        assert "ACTION:show_risk" in result

    def test_list_troncales(self):
        result = list_troncales.invoke({})
        assert "troncales" in result.lower()
        assert "Caracas" in result


class TestServiceContextBuilding:
    """Test context and system prompt building."""

    @pytest.fixture
    def service(self):
        return AgentService()

    def test_build_context_none(self, service):
        result = service._build_context_prompt(None)
        assert result == ""

    def test_build_context_empty(self, service):
        ctx = AppContext()
        result = service._build_context_prompt(ctx)
        assert result == ""

    def test_build_context_with_module(self, service):
        ctx = AppContext(module="planificar")
        result = service._build_context_prompt(ctx)
        assert "planificar" in result

    def test_build_context_with_origin(self, service):
        ctx = AppContext(origin="Portal Norte")
        result = service._build_context_prompt(ctx)
        assert "Portal Norte" in result

    def test_build_context_with_transport(self, service):
        ctx = AppContext(transport_mode="tm")
        result = service._build_context_prompt(ctx)
        assert "TransMilenio" in result

    def test_build_context_full(self, service):
        ctx = AppContext(
            module="planificar",
            origin="Suba",
            destination="Centro",
            selected_hour=8,
            transport_mode="sitp",
        )
        result = service._build_context_prompt(ctx)
        assert "Suba" in result
        assert "Centro" in result
        assert "8" in result
        assert "SITP" in result

    def test_get_system_context(self, service):
        result = service._get_system_context()
        assert "estaciones" in result
        assert "conexiones" in result


class TestServiceEdgeCases:
    """Test edge cases and branches in service."""

    @pytest.fixture
    def service(self):
        return AgentService()

    @pytest.mark.asyncio
    async def test_troncal_query(self, service):
        resp = await service.chat("troncales de transmilenio", "edge1")
        assert "troncal" in resp.response.lower() or "Caracas" in resp.response

    @pytest.mark.asyncio
    async def test_station_not_found(self, service):
        resp = await service.chat("estacion xyznonexistent", "edge2")
        assert "No encontre" in resp.response or "153" in resp.response

    @pytest.mark.asyncio
    async def test_route_code_query(self, service):
        resp = await service.chat("ruta j74", "edge3")
        assert "J74" in resp.response or "ruta" in resp.response.lower()

    @pytest.mark.asyncio
    async def test_route_code_not_found(self, service):
        resp = await service.chat("ruta ZZZ999", "edge4")
        assert "disponibles" in resp.response or "codigo" in resp.response.lower()

    @pytest.mark.asyncio
    async def test_nearby_without_coords(self, service):
        resp = await service.chat("estaciones cerca", "edge5")
        assert "ubicacion" in resp.response.lower() or "punto" in resp.response.lower()

    @pytest.mark.asyncio
    async def test_nearby_with_coords(self, service):
        ctx = AppContext(origin_coords=[4.695, -74.031])
        resp = await service.chat("estaciones cerca", "edge6", ctx)
        assert "cercanas" in resp.response.lower() or "estacion" in resp.response.lower()

    @pytest.mark.asyncio
    async def test_followup_unknown(self, service):
        # First message creates context
        await service.chat("estacion heroes", "edge7")
        # Follow-up should relate to previous topic
        resp = await service.chat("dime mas", "edge7")
        assert resp.response != ""

    @pytest.mark.asyncio
    async def test_best_time_peak(self, service):
        # This depends on current hour, just verify it responds
        resp = await service.chat("mejor hora para viajar", "edge8")
        r = resp.response.lower()
        assert "hora" in r or "viajar" in r or "congestion" in r

    @pytest.mark.asyncio
    async def test_default_with_rutas_context(self, service):
        ctx = AppContext(module="rutas")
        resp = await service.chat("xyz", "edge9", ctx)
        assert "ruta" in resp.response.lower() or "troncal" in resp.response.lower()

    @pytest.mark.asyncio
    async def test_default_with_metricas_context(self, service):
        ctx = AppContext(module="metricas")
        resp = await service.chat("xyz", "edge10", ctx)
        r = resp.response.lower()
        assert "métrica" in r or "congestion" in r or "metricas" in r

    @pytest.mark.asyncio
    async def test_congestion_specific_hour_low(self, service):
        resp = await service.chat("trafico a las 3", "edge11")
        assert "3" in resp.response
        r = resp.response.lower()
        assert "tranquilo" in r or "baja" in r or "%" in resp.response

    @pytest.mark.asyncio
    async def test_congestion_specific_hour_high(self, service):
        resp = await service.chat("trafico a las 8", "edge12")
        assert "8" in resp.response
        r = resp.response.lower()
        assert "pesado" in r or "moderada" in r or "%" in resp.response


class TestIntentsEdgeCases:
    """Cover remaining lines in intents.py."""

    def test_find_nearby_limit(self):
        results = find_nearby_stations(4.65, -74.08, 1)
        assert len(results) == 1

    def test_find_nearby_zero(self):
        results = find_nearby_stations(4.65, -74.08, 0)
        assert len(results) == 0

    def test_greeting_morning(self):
        # Can't easily control time, just verify function works
        response = get_greeting_response()
        assert "MoviBot" in response

    def test_detect_intent_with_hour_pm(self):
        _, slots = detect_intent("trafico a las 5pm")
        assert slots.get("hour") == 17

    def test_detect_intent_nearby(self):
        intent, _ = detect_intent("estaciones cercanas")
        assert intent == "nearby"


class TestRouterEndpoints:
    """Test router endpoints via TestClient."""

    def test_chat_endpoint(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from app.modules.agent.router import router

        app = FastAPI()
        app.include_router(router, prefix="/agent")
        client = TestClient(app)

        response = client.post(
            "/agent/chat",
            json={
                "message": "hola",
                "session_id": "router-test",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "MoviBot" in data["response"]

    def test_clear_endpoint(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from app.modules.agent.router import router

        app = FastAPI()
        app.include_router(router, prefix="/agent")
        client = TestClient(app)

        response = client.delete("/agent/chat/test-session")
        assert response.status_code == 200
        assert response.json()["status"] == "cleared"
