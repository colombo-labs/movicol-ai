"""Agent service - LLM conversational agent with graph tools."""

from __future__ import annotations

from app.config.settings import get_settings
from app.modules.agent.schemas import ChatResponse
from app.modules.route_prediction.graph_data import (
    CARACAS_STATIONS,
    CONGESTION_BY_HOUR,
    build_caracas_graph,
)
from app.modules.route_prediction.service import RoutePredictionService

SYSTEM_PROMPT = """Eres MoviBot, un asistente experto en movilidad urbana de Bogotá, Colombia.
Tienes acceso a datos del sistema TransMilenio (Troncal Caracas: 28 estaciones).

Capacidades:
- Información de estaciones (nombre, ubicación, conexiones)
- Predicción de congestión por hora del día
- Recomendaciones de rutas y horarios
- Estadísticas del sistema de transporte

Responde en español, de forma concisa y útil. Si no tienes datos para responder, dilo honestamente.
Cuando menciones congestión, usa porcentajes y etiquetas (baja/media/alta/crítica)."""


class AgentService:
    """Conversational agent with graph knowledge."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._graph = build_caracas_graph()
        self._route_service = RoutePredictionService()
        self._sessions: dict[str, list[dict]] = {}

    def _get_context(self) -> str:
        """Build context string from graph data."""
        return (
            f"Sistema: Troncal Caracas, {self._graph.number_of_nodes()} estaciones, "
            f"{self._graph.number_of_edges() // 2} conexiones.\n"
            f"Estaciones: {', '.join(s['name'] for s in CARACAS_STATIONS)}\n"
            f"Horas pico: 7-9am (congestión ~55%), 5-7pm (congestión ~60%)\n"
            f"Horas valle: 10pm-5am (congestión <10%)"
        )

    def _find_station_info(self, query: str) -> str | None:
        """Find station info matching a query."""
        query_lower = query.lower()
        for station in CARACAS_STATIONS:
            if station["name"].lower() in query_lower or query_lower in station["name"].lower():
                neighbors = list(self._graph.neighbors(station["id"]))
                neighbor_names = [self._graph.nodes[n]["name"] for n in neighbors]
                return (
                    f"Estación: {station['name']}\n"
                    f"Coordenadas: {station['lat']}, {station['lon']}\n"
                    f"Conecta con: {', '.join(neighbor_names)}"
                )
        return None

    def _get_congestion_info(self, hour: int | None = None) -> str:
        """Get congestion info using GNN predictions."""
        from app.modules.predictions.gnn_inference import GNNInference

        gnn = GNNInference()

        if hour is not None and gnn.is_loaded:
            # Use real GNN predictions
            time_factors = {
                0: 0.3,
                1: 0.2,
                2: 0.2,
                3: 0.2,
                4: 0.3,
                5: 0.5,
                6: 0.7,
                7: 0.9,
                8: 1.0,
                9: 0.9,
                10: 0.7,
                11: 0.65,
                12: 0.75,
                13: 0.7,
                14: 0.65,
                15: 0.7,
                16: 0.8,
                17: 0.95,
                18: 1.0,
                19: 0.9,
                20: 0.7,
                21: 0.5,
                22: 0.4,
                23: 0.3,
            }
            tf = time_factors.get(hour, 0.7)
            preds = gnn.get_all_predictions()
            values = [min(1.0, v * tf) for v in preds.values()]
            avg = sum(values) / len(values) if values else 0
            critical = sum(1 for v in values if v >= 0.85)
            label = (
                "baja"
                if avg < 0.3
                else "media"
                if avg < 0.6
                else "alta"
                if avg < 0.85
                else "crítica"
            )
            return (
                f"A las {hour}:00, la congestión promedio es {int(avg * 100)}% ({label}). "
                f"{critical} estaciones en nivel crítico de {len(values)} totales."
            )

        if hour is not None:
            level = CONGESTION_BY_HOUR.get(hour, 0.3)
            label = (
                "baja"
                if level < 0.3
                else "media"
                if level < 0.6
                else "alta"
                if level < 0.85
                else "crítica"
            )
            return f"A las {hour}:00, la congestión promedio es {int(level * 100)}% ({label})."

        return (
            "Pico AM (7-9h): congestión alta (~55-60%).\n"
            "Pico PM (17-19h): congestión alta (~55-60%).\n"
            "Mejor horario: antes de las 6am o después de las 9pm (< 15%)."
        )

    async def chat(self, message: str, session_id: str) -> ChatResponse:
        """Process a chat message."""
        # Try LLM first, fallback to rule-based
        if self._settings.openai_api_key:
            return await self._llm_chat(message, session_id)
        return self._rule_based_chat(message, session_id)

    async def _llm_chat(self, message: str, session_id: str) -> ChatResponse:
        """Chat using LangChain + OpenAI."""
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=self._settings.llm_model,
                api_key=self._settings.openai_api_key,
                temperature=0.4,
                max_tokens=300,
            )

            context = self._get_context()
            station_info = self._find_station_info(message)
            extra = f"\nInfo relevante: {station_info}" if station_info else ""

            # Check for hour-related queries
            import re

            hour_match = re.search(r"(\d{1,2})\s*(am|pm|:00|hrs?)", message.lower())
            if hour_match:
                h = int(hour_match.group(1))
                if "pm" in (hour_match.group(2) or "") and h < 12:
                    h += 12
                extra += f"\n{self._get_congestion_info(h)}"

            messages = [
                SystemMessage(content=f"{SYSTEM_PROMPT}\n\nContexto:\n{context}{extra}"),
                HumanMessage(content=message),
            ]

            response = llm.invoke(messages)
            sources = ["grafo_troncal_caracas"]
            if station_info:
                sources.append("station_data")

            return ChatResponse(
                response=response.content.strip(),
                sources=sources,
                session_id=session_id,
            )
        except Exception:
            # Fallback to rule-based on any LLM error
            return self._rule_based_chat(message, session_id)

    def _rule_based_chat(self, message: str, session_id: str) -> ChatResponse:
        """Fallback rule-based responses when LLM is unavailable."""
        msg_lower = message.lower()
        sources = ["grafo_troncal_caracas"]

        # Station query
        station_info = self._find_station_info(message)
        if station_info:
            sources.append("station_data")
            return ChatResponse(response=station_info, sources=sources, session_id=session_id)

        # Congestion query
        if any(
            w in msg_lower for w in ["congestión", "congestion", "tráfico", "trafico", "hora pico"]
        ):
            import re

            hour_match = re.search(r"(\d{1,2})", message)
            hour = int(hour_match.group(1)) if hour_match else None
            return ChatResponse(
                response=self._get_congestion_info(hour),
                sources=sources,
                session_id=session_id,
            )

        # Station list
        if any(w in msg_lower for w in ["estaciones", "paradas", "cuántas", "cuantas", "lista"]):
            names = [s["name"] for s in CARACAS_STATIONS]
            return ChatResponse(
                response=f"La Troncal Caracas tiene {len(names)} estaciones:\n{', '.join(names)}",
                sources=sources,
                session_id=session_id,
            )

        # Route query
        if any(w in msg_lower for w in ["ruta", "cómo llego", "como llego", "ir de", "ir a"]):
            return ChatResponse(
                response=(
                    "Para predecir una ruta, usa el módulo de Predicción de Rutas en el mapa. "
                    "Haz clic en origen y destino, y te mostraré la congestión por tramo. "
                    "¿Necesitas info de alguna estación específica?"
                ),
                sources=sources,
                session_id=session_id,
            )

        # Default
        return ChatResponse(
            response=(
                "Soy MoviBot 🚌 — tu asistente de movilidad para Bogotá. Puedo ayudarte con:\n"
                "• Info de estaciones de TransMilenio (Troncal Caracas)\n"
                "• Niveles de congestión por hora\n"
                "• Recomendaciones de horarios\n\n"
                "¿Qué necesitas saber?"
            ),
            sources=sources,
            session_id=session_id,
        )
