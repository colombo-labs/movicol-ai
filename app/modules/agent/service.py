"""Agent service - LLM conversational agent with graph tools."""

from __future__ import annotations

from app.config.settings import get_settings
from app.modules.agent.schemas import ChatResponse
from app.modules.route_prediction.graph_data import (
    CONGESTION_BY_HOUR,
    TM_RUTAS,
    TM_STATIONS,
    TRONCALES,
    build_tm_graph,
)
from app.modules.route_prediction.service import RoutePredictionService
from app.modules.siniestralidad.service import SiniestrosService

SYSTEM_PROMPT = """Eres MoviBot, un asistente experto en movilidad urbana de Bogotá, Colombia.
Tienes acceso a datos de TransMilenio (13 troncales, 153 estaciones) y SITP (689 rutas zonales).

Troncales TM: Caracas, Autopista Norte, Suba, Calle 80, NQS Central,
NQS Sur, Américas, Calle 26, Carrera 10, Caracas Sur,
Eje Ambiental, Soacha, Tunal, Carrera 7.

Capacidades:
- Información de estaciones TM y paraderos SITP
- Predicción de congestión por hora (modelo GNN)
- Predicción de riesgo vial por zona y hora (siniestralidad con IA)
- Planificación de rutas A→B con alternativas y trasbordos
- Estadísticas de accidentes por localidad (datos.gov.co)
- Recomendaciones de horarios seguros

IMPORTANTE:
- TransMilenio (TM): buses articulados, estaciones cerradas
- SITP: buses zonales, paraderos abiertos, ~689 rutas
- Siniestralidad: datos reales de 196K+ accidentes en Bogotá

Responde en español, de forma concisa y útil. Si no tienes datos, dilo honestamente.
Usa porcentajes y etiquetas para congestión (baja/media/alta/crítica)
y niveles de riesgo (bajo/moderado/alto/crítico)."""


class AgentService:
    """Conversational agent with graph knowledge."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._graph = build_tm_graph()
        self._route_service = RoutePredictionService()
        self._siniestros_service: SiniestrosService | None = None
        self._sessions: dict[str, list[dict]] = {}
        self._init_siniestros()

    def _init_siniestros(self) -> None:
        try:
            svc = SiniestrosService()
            if svc.is_loaded:
                self._siniestros_service = svc
        except Exception:
            pass

    def _get_context(self) -> str:
        """Build context string from graph data."""
        troncal_info = ", ".join(f"{t} ({d['stations']})" for t, d in TRONCALES.items())
        ctx = (
            f"Sistema TransMilenio: {self._graph.number_of_nodes()} estaciones, "
            f"{self._graph.number_of_edges()} conexiones, 13 troncales.\n"
            f"Troncales: {troncal_info}\n"
            f"SITP: 689 rutas zonales con ~7,290 paraderos.\n"
            f"Horas pico: 7-9am (congestión ~55-60%), 5-7pm (congestión ~60-65%)\n"
            f"Horas valle: 10pm-5am (congestión <10%)"
        )
        if self._siniestros_service:
            stats = self._siniestros_service.get_stats()
            ctx += (
                f"\nSiniestralidad: {stats.total_siniestros:,} accidentes registrados, "
                f"{stats.total_fallecidos:,} fatales, "
                f"{stats.sectores_criticos} sectores críticos, "
                f"{stats.semaforos:,} semáforos."
            )
        return ctx

    def _find_station_info(self, query: str) -> str | None:
        """Find station info matching a query."""
        query_lower = query.lower()
        for station in TM_STATIONS:
            if station["name"].lower() in query_lower or query_lower in station["name"].lower():
                neighbors = list(self._graph.neighbors(station["id"]))
                neighbor_names = [
                    self._graph.nodes[n]["name"] for n in neighbors if n in self._graph.nodes
                ]
                return (
                    f"Estación: {station['name']}\n"
                    f"Troncal: {station['troncal']}\n"
                    f"Coordenadas: {station['lat']}, {station['lon']}\n"
                    f"Conecta con: {', '.join(neighbor_names) if neighbor_names else 'N/A'}"
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

        # Siniestralidad / safety queries
        if any(
            w in msg_lower
            for w in [
                "siniestro",
                "accidente",
                "peligro",
                "riesgo",
                "seguridad",
                "zona peligrosa",
                "mortalidad",
                "insegura",
                "inseguro",
            ]
        ):
            return self._handle_siniestralidad_query(msg_lower, session_id)

        # Station list
        if any(w in msg_lower for w in ["estaciones", "paradas", "cuántas", "cuantas", "lista"]):
            troncal_match = None
            for t in TRONCALES:
                if t.lower() in msg_lower:
                    troncal_match = t
                    break
            if troncal_match:
                names = [s["name"] for s in TM_STATIONS if s["troncal"] == troncal_match]
                return ChatResponse(
                    response=(
                        f"Troncal {troncal_match} tiene {len(names)} estaciones:\n"
                        f"{', '.join(names)}"
                    ),
                    sources=sources,
                    session_id=session_id,
                )
            return ChatResponse(
                response=(
                    f"TransMilenio tiene {len(TM_STATIONS)} estaciones en 13 troncales.\n"
                    "Pregúntame por una troncal específica "
                    "(Caracas, Suba, Calle 80, NQS, Américas, Calle 26, etc.)"
                ),
                sources=sources,
                session_id=session_id,
            )

        # Route query — search in TM_RUTAS
        if any(
            w in msg_lower
            for w in [
                "ruta",
                "cómo llego",
                "como llego",
                "ir de",
                "ir a",
                "j74",
                "f51",
                "g4",
                "l4",
                "m5",
            ]
        ):
            # Try to find specific route code
            found_rutas = [r for r in TM_RUTAS if r["codigo"].lower() in msg_lower]
            if found_rutas:
                r = found_rutas[0]
                return ChatResponse(
                    response=(
                        f"Ruta: {r['codigo']}\n"
                        f"Recorrido: {r['origen']} → {r['destino']}\n"
                        f"Bus: {r['tipo_bus']}\n"
                        f"Horario L-V: {r['horario_lv']}\n"
                        f"Horario Sáb: {r['horario_sab']}\n"
                        f"Estado: {r['estado']}"
                    ),
                    sources=["tm_rutas_troncales"],
                    session_id=session_id,
                )
            return ChatResponse(
                response=(
                    f"TransMilenio tiene {len(TM_RUTAS)} rutas troncales. "
                    "Para planificar tu viaje, usa el módulo Planificar en el mapa. "
                    "¿Quieres info de alguna ruta específica? (ej: J74, F51, G43)"
                ),
                sources=sources,
                session_id=session_id,
            )

        # Default
        return ChatResponse(
            response=(
                "Soy MoviBot 🚌 — tu asistente de movilidad para Bogotá. Puedo ayudarte con:\n"
                "• Info de 153 estaciones TM y 689 rutas SITP\n"
                "• Niveles de congestión por hora (modelo GNN)\n"
                "• Riesgo vial por zona y hora (siniestralidad con IA)\n"
                "• Planificación de rutas con alternativas\n"
                "• Recomendaciones de horarios seguros\n\n"
                "¿Qué necesitas saber?"
            ),
            sources=sources,
            session_id=session_id,
        )

    def _handle_siniestralidad_query(self, msg_lower: str, session_id: str) -> ChatResponse:
        """Handle siniestralidad / safety queries."""
        import re

        sources = ["siniestralidad_datos_gov_co"]

        if not self._siniestros_service:
            return ChatResponse(
                response="No tengo datos de siniestralidad cargados en este momento.",
                sources=sources,
                session_id=session_id,
            )

        stats = self._siniestros_service.get_stats()

        # Check for hour query
        hour_match = re.search(r"(\d{1,2})", msg_lower)
        hour = int(hour_match.group(1)) if hour_match else None

        if hour is not None and 0 <= hour <= 23:
            risk = self._siniestros_service.predict_risk_by_hour(hour)
            top3 = risk.zones[:3]
            zones_text = "\n".join(
                f"  • {z.localidad}: riesgo {z.nivel} ({int(z.risk * 100)}%)" for z in top3
            )
            return ChatResponse(
                response=(
                    f"Riesgo vial a las {hour}:00 — nivel general: "
                    f"{risk.nivel_general} ({int(risk.promedio_riesgo * 100)}%)\n"
                    f"Zonas más riesgosas:\n{zones_text}\n\n"
                    f"Total accidentes registrados: {stats.total_siniestros:,}"
                ),
                sources=sources,
                session_id=session_id,
            )

        # General siniestralidad info
        top_vehiculos = ", ".join(f"{v['tipo']} ({v['total']:,})" for v in stats.vehiculos_top[:3])
        return ChatResponse(
            response=(
                f"Siniestralidad vial en Bogotá:\n"
                f"• Total accidentes: {stats.total_siniestros:,}\n"
                f"• Fatales: {stats.total_fallecidos:,}\n"
                f"• Sectores críticos (ANSV): {stats.sectores_criticos}\n"
                f"• Semáforos: {stats.semaforos:,}\n"
                f"• Localidades peligrosas: {stats.localidades_peligrosas}\n"
                f"• Vehículos más involucrados: {top_vehiculos}\n\n"
                "Pregúntame por una hora específica para ver el riesgo "
                'por zona (ej: "riesgo a las 18").'
            ),
            sources=sources,
            session_id=session_id,
        )
