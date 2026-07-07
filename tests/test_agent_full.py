"""Tests for the agent module — intents, tools, and service."""

from app.modules.agent.intents import (
    detect_intent,
    find_nearby_stations,
    get_comparison_info,
    get_cost_info,
    get_current_congestion_summary,
    get_greeting_response,
    get_schedule_info,
)
from app.modules.agent.schemas import ActionPayload, AppContext, ChatRequest
from app.modules.agent.tools import parse_actions


class TestIntentDetection:
    """Test intent classification from user messages."""

    def test_greeting(self):
        intent, _ = detect_intent("hola")
        assert intent == "greeting"

    def test_greeting_buenas(self):
        intent, _ = detect_intent("buenas tardes")
        assert intent == "greeting"

    def test_plan_route_ir_de_a(self):
        intent, slots = detect_intent("ir de suba a chapinero")
        assert intent == "plan_route"
        assert "suba" in slots["groups"][0]
        assert "chapinero" in slots["groups"][1]

    def test_plan_route_como_llego(self):
        intent, slots = detect_intent("cómo llego de usaquén al centro")
        assert intent == "plan_route"
        assert len(slots["groups"]) == 2

    def test_plan_route_mejor_ruta_a(self):
        intent, slots = detect_intent("mejor ruta a ADL digital lab")
        assert intent == "plan_route"
        assert "adl digital lab" in slots["groups"][0]

    def test_plan_route_del_centro(self):
        intent, slots = detect_intent("del centro a soacha")
        assert intent == "plan_route"
        assert "centro" in slots["groups"][0]
        assert "soacha" in slots["groups"][1]

    def test_plan_route_dame_ruta(self):
        intent, _ = detect_intent("dame ruta a chapinero")
        assert intent == "plan_route"

    def test_congestion(self):
        intent, _ = detect_intent("cómo está el tráfico")
        assert intent == "congestion"

    def test_congestion_hora(self):
        intent, slots = detect_intent("tráfico a las 7")
        assert intent == "congestion"
        assert slots.get("hour") == 7

    def test_safety(self):
        intent, _ = detect_intent("zonas de riesgo")
        assert intent == "safety"

    def test_station(self):
        intent, _ = detect_intent("estación Héroes")
        assert intent == "station"

    def test_cost(self):
        intent, _ = detect_intent("cuánto cuesta el pasaje")
        assert intent == "cost"

    def test_schedule(self):
        intent, _ = detect_intent("a qué hora abre TransMilenio")
        assert intent == "schedule"

    def test_compare(self):
        intent, _ = detect_intent("qué es mejor TM o SITP")
        assert intent == "compare"

    def test_best_time(self):
        intent, _ = detect_intent("mejor hora para viajar")
        assert intent in ("congestion", "best_time")

    def test_thanks(self):
        intent, _ = detect_intent("gracias")
        assert intent == "thanks"

    def test_confirm(self):
        intent, _ = detect_intent("sí dale")
        assert intent == "confirm"

    def test_deny(self):
        intent, _ = detect_intent("no")
        assert intent == "deny"

    def test_help(self):
        intent, _ = detect_intent("qué puedes hacer")
        assert intent == "help"

    def test_unknown(self):
        intent, _ = detect_intent("asdfghjkl")
        assert intent == "unknown"


class TestIntentResponses:
    """Test that response generators return valid content."""

    def test_greeting_has_content(self):
        response = get_greeting_response()
        assert "MoviBot" in response
        assert len(response) > 50

    def test_cost_info_has_price(self):
        response = get_cost_info()
        assert "3.550" in response
        assert "pesos" in response

    def test_schedule_info(self):
        response = get_schedule_info()
        assert "4" in response or "5" in response
        assert "11" in response or "10" in response

    def test_comparison_info(self):
        response = get_comparison_info()
        assert "TransMilenio" in response
        assert "SITP" in response

    def test_congestion_summary(self):
        response = get_current_congestion_summary()
        assert "%" in response
        assert "Ahora" in response


class TestNearbyStations:
    """Test nearby station search."""

    def test_finds_stations(self):
        # Coords near Portal Norte
        results = find_nearby_stations(4.759, -74.045, 3)
        assert len(results) == 3
        assert all("name" in s for s in results)
        assert all("lat" in s for s in results)

    def test_results_sorted_by_distance(self):
        results = find_nearby_stations(4.659, -74.056, 5)
        dists = [r["dist"] for r in results]
        assert dists == sorted(dists)


class TestParseActions:
    """Test action parsing from tool output."""

    def test_parse_plan_route(self):
        output = "ACTION:plan_route|origin=suba|destination=centro|mode=tm"
        actions = parse_actions(output)
        assert len(actions) == 1
        assert actions[0].type == "plan_route"
        assert actions[0].data["origin"] == "suba"
        assert actions[0].data["destination"] == "centro"

    def test_parse_show_station(self):
        output = "ACTION:show_station|name=Heroes|lat=4.668|lon=-74.060"
        actions = parse_actions(output)
        assert len(actions) == 1
        assert actions[0].type == "show_station"
        assert abs(actions[0].data["lat"] - 4.668) < 0.001

    def test_no_actions(self):
        output = "Respuesta normal sin acciones"
        actions = parse_actions(output)
        assert len(actions) == 0

    def test_multiple_actions(self):
        output = "ACTION:show_congestion|hour=7\nTexto\nACTION:show_risk|hour=18"
        actions = parse_actions(output)
        assert len(actions) == 2


class TestSchemas:
    """Test pydantic schemas."""

    def test_chat_request_valid(self):
        req = ChatRequest(message="hola", session_id="test")
        assert req.message == "hola"

    def test_chat_request_with_context(self):
        req = ChatRequest(
            message="ir al centro",
            session_id="s1",
            context=AppContext(module="planificar", origin="Suba"),
        )
        assert req.context.module == "planificar"
        assert req.context.origin == "Suba"

    def test_action_payload(self):
        action = ActionPayload(type="plan_route", data={"origin": "a"})
        assert action.type == "plan_route"
        assert action.data["origin"] == "a"
