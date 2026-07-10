"""Agent service - LLM conversational agent with tools, context, and session history."""

from __future__ import annotations

import re
from collections import deque
from datetime import datetime

from app.config.settings import get_settings
from app.modules.agent.intents import (
    detect_intent,
    find_nearby_stations,
    get_comparison_info,
    get_cost_info,
    get_current_congestion_summary,
    get_greeting_response,
    get_schedule_info,
)
from app.modules.agent.schemas import ActionPayload, AppContext, ChatResponse
from app.modules.agent.tools import AGENT_TOOLS, parse_actions
from app.modules.route_prediction.graph_data import (
    CONGESTION_BY_HOUR,
    TM_RUTAS,
    TM_STATIONS,
    TRONCALES,
    build_tm_graph,
)
from app.modules.siniestralidad.service import SiniestrosService

MAX_HISTORY = 20

SYSTEM_PROMPT = """Eres MoviBot, asistente experto en movilidad urbana de Bogota, Colombia.
Tienes acceso a datos en tiempo real de TransMilenio (13 troncales, 153 estaciones) y SITP (689 rutas zonales, 7694 paraderos).

Capacidades:
- Informacion de estaciones TM y paraderos SITP con ubicacion exacta
- Prediccion de congestion por hora usando modelo GNN (Graph Neural Network)
- Prediccion de demanda de pasajeros por estacion (modelo ST-GAT)
- Prediccion de riesgo vial por zona y hora (1998 puntos de siniestralidad)
- Planificacion de rutas A->B con TM, SITP, multimodal, vehiculo, moto, bicicleta
- Transbordos reales entre TM y SITP (conexiones caminables < 300m)
- Estadisticas de accidentes por localidad (datos.gov.co 2024)
- Busqueda de rutas SITP cercanas a una ubicacion
- Informacion de frecuencias y horarios de servicio

CONTEXTO TEMPORAL:
- Hora actual: {hour}:00
- Dia: {day_name}
- Horario TM: L-S 4am-11pm, Dom/Festivo 5am-10pm
- Si es hora pico (6-9am, 5-8pm): advierte tiempos mayores
- Si esta fuera de servicio: sugiere alternativas

REGLAS:
- Responde SIEMPRE en espanol, conciso y util (max 200 palabras)
- Si el usuario pide ir de A a B, USA la herramienta plan_route
- Si pregunta por una estacion, USA find_station
- Si pregunta por congestion/trafico, USA get_congestion
- Si pregunta por seguridad/riesgo/peligro, USA get_risk_by_zone
- Si pregunta por una ruta especifica (codigo), USA find_route_info
- Si pregunta por rutas cercanas/que pasa por aqui, USA find_nearby_routes
- Usa datos reales, NO inventes. Si no sabes, dilo
- Si el usuario da coordenadas o una direccion, usala directamente
- TransMilenio (TM): $3.550, buses articulados, estaciones cerradas, rapido
- SITP: $3.550, buses zonales, paraderos abiertos, mas cobertura
- Cuando sugieras rutas, menciona tiempo estimado y transbordos
- Se proactivo: si detectas hora pico, advierte sin que pregunten"""


class AgentService:
    """Conversational agent with tools, context awareness, and session memory."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._graph = build_tm_graph()
        self._siniestros_service: SiniestrosService | None = None
        self._sessions: dict[str, deque] = {}
        self._pending_actions: dict[str, list[ActionPayload]] = {}
        self._init_siniestros()

    def _init_siniestros(self) -> None:
        try:
            svc = SiniestrosService()
            if svc.is_loaded:
                self._siniestros_service = svc
        except Exception:
            pass

    def _get_session_history(self, session_id: str) -> deque:
        """Get or create session history."""
        if session_id not in self._sessions:
            self._sessions[session_id] = deque(maxlen=MAX_HISTORY * 2)
        return self._sessions[session_id]

    def _build_context_prompt(self, context: AppContext | None) -> str:
        """Build context section from app state."""
        if not context:
            return ""
        parts = []
        if context.module:
            parts.append(f"El usuario esta en el modulo: {context.module}")
        if context.origin:
            parts.append(f"Origen seleccionado: {context.origin}")
        if context.destination:
            parts.append(f"Destino seleccionado: {context.destination}")
        if context.active_route:
            parts.append(f"Ruta activa en mapa: {context.active_route}")
        if context.selected_hour is not None:
            parts.append(f"Hora seleccionada: {context.selected_hour}:00")
        if context.transport_mode:
            modes = {
                "tm": "TransMilenio",
                "sitp": "SITP",
                "vehicle": "Vehiculo",
                "moto": "Moto",
            }
            parts.append(f"Modo: {modes.get(context.transport_mode, context.transport_mode)}")
        if not parts:
            return ""
        return "\n\nCONTEXTO ACTUAL:\n" + "\n".join(f"- {p}" for p in parts)

    def _get_system_context(self) -> str:
        """Build static system context from graph data."""
        ctx = (
            f"Sistema TM: {self._graph.number_of_nodes()} estaciones, "
            f"{self._graph.number_of_edges()} conexiones, {len(TRONCALES)} troncales.\n"
            f"SITP: 689 rutas zonales. Rutas TM: {len(TM_RUTAS)}"
        )
        if self._siniestros_service:
            stats = self._siniestros_service.get_stats()
            ctx += f"\nSiniestralidad: {stats.total_siniestros:,} accidentes registrados."
        return ctx

    async def chat(
        self, message: str, session_id: str, context: AppContext | None = None
    ) -> ChatResponse:
        """Process a chat message with history and context."""
        if self._settings.openai_api_key or self._settings.groq_api_key:
            return await self._llm_chat(message, session_id, context)
        return self._rule_based_chat(message, session_id, context)

    async def _llm_chat(
        self, message: str, session_id: str, context: AppContext | None = None
    ) -> ChatResponse:
        """Chat using LangChain with tools and history."""
        try:
            from langchain.agents import AgentExecutor, create_tool_calling_agent
            from langchain_core.messages import AIMessage, HumanMessage
            from langchain_core.prompts import (
                ChatPromptTemplate,
                MessagesPlaceholder,
            )

            if self._settings.openai_api_key:
                from langchain_openai import ChatOpenAI

                llm = ChatOpenAI(
                    model=self._settings.llm_model,
                    api_key=self._settings.openai_api_key,
                    temperature=0.4,
                    max_tokens=400,
                )
            else:
                from langchain_groq import ChatGroq

                llm = ChatGroq(
                    model=self._settings.groq_model,
                    api_key=self._settings.groq_api_key,
                    temperature=0.4,
                    max_tokens=400,
                )

            system_context = self._get_system_context()
            app_context = self._build_context_prompt(context)
            now = datetime.now()
            day_names = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
            dynamic_prompt = SYSTEM_PROMPT.format(
                hour=now.hour,
                day_name=day_names[now.weekday()],
            )
            full_system = f"{dynamic_prompt}\n\nDATOS:\n{system_context}{app_context}"

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", full_system),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )

            agent = create_tool_calling_agent(llm, AGENT_TOOLS, prompt)
            executor = AgentExecutor(
                agent=agent,
                tools=AGENT_TOOLS,
                verbose=False,
                max_iterations=5,
                handle_parsing_errors=True,
            )

            history = self._get_session_history(session_id)
            chat_history = []
            for msg in history:
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                else:
                    chat_history.append(AIMessage(content=msg["content"]))

            result = await executor.ainvoke(
                {
                    "input": message,
                    "chat_history": chat_history,
                }
            )

            response_text = result["output"]
            actions: list[ActionPayload] = []
            actions.extend(parse_actions(response_text))
            clean_response = "\n".join(
                line for line in response_text.split("\n") if not line.startswith("ACTION:")
            ).strip()

            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": clean_response})

            return ChatResponse(
                response=clean_response,
                sources=["agent_tools"],
                session_id=session_id,
                actions=actions,
            )
        except Exception:
            return self._rule_based_chat(message, session_id, context)

    def _rule_based_chat(
        self, message: str, session_id: str, context: AppContext | None = None
    ) -> ChatResponse:
        """Smart rule-based responses using intent detection."""
        sources = ["movicol_ai"]
        actions: list[ActionPayload] = []

        history = self._get_session_history(session_id)
        history.append({"role": "user", "content": message})

        intent, slots = detect_intent(message)
        hour = slots.get("hour")
        msg_lower = message.lower().strip()

        intent = self._resolve_followup(intent, msg_lower, history)

        def reply(response: str) -> ChatResponse:
            history.append({"role": "assistant", "content": response})
            pending = [a for a in actions if a.type.startswith("pending_")]
            if pending:
                self._pending_actions[session_id] = pending
            return ChatResponse(
                response=response,
                sources=sources,
                session_id=session_id,
                actions=[a for a in actions if not a.type.startswith("pending_")],
            )

        # Dispatch to handler
        handler = getattr(self, f"_intent_{intent}", None)
        if handler:
            return handler(
                reply=reply,
                actions=actions,
                sources=sources,
                message=message,
                session_id=session_id,
                context=context,
                slots=slots,
                hour=hour,
                msg_lower=msg_lower,
            )

        # Unknown — try station search as last resort
        station_info = self._find_station_info(message)
        if station_info:
            sources.append("station_data")
            if station_info.get("action"):
                actions.append(station_info["action"])
            return reply(station_info["text"])

        return reply(self._default_response(context))

    @staticmethod
    def _resolve_followup(intent, msg_lower, history):
        """Detect follow-up intent from conversation context."""
        followup_words = [
            "esa",
            "ese",
            "ahi",
            "mas info",
            "cuentame mas",
            "y esa",
            "que mas",
            "dime mas",
            "otra",
        ]
        is_followup = any(w in msg_lower for w in followup_words) and len(history) >= 2
        if not is_followup or intent != "unknown":
            return intent
        prev = [m for m in history if m["role"] == "assistant"]
        if not prev:
            return intent
        last = prev[-1]["content"].lower()
        if "estacion" in last or "troncal" in last:
            return "station_list"
        if "congestion" in last or "trafico" in last:
            return "congestion"
        if "ruta" in last:
            return "route_code"
        return intent

    def _intent_greeting(self, **kw):
        reply = kw["reply"]
        return reply(get_greeting_response())

    def _intent_confirm(self, **kw):
        reply, actions, sid = kw["reply"], kw["actions"], kw["session_id"]
        if hasattr(self, "_pending_actions") and sid in self._pending_actions:
            pending = self._pending_actions.pop(sid)
            for a in pending:
                real_type = a.type.replace("pending_", "")
                actions.append(ActionPayload(type=real_type, data=a.data))
            return reply("Listo, ya lo hice. Mira el mapa.")
        return reply("No tengo nada pendiente. En que te ayudo?")

    def _intent_deny(self, **kw):
        reply, sid = kw["reply"], kw["session_id"]
        if hasattr(self, "_pending_actions") and sid in self._pending_actions:
            self._pending_actions.pop(sid)
        return reply("Entendido, no hay problema. Que mas necesitas?")

    def _intent_plan_route(self, **kw):
        reply, actions = kw["reply"], kw["actions"]
        context, slots = kw["context"], kw["slots"]
        groups = slots.get("groups", ())
        origin, dest = self._extract_route_points(groups, context)
        if not dest:
            return reply(self._default_response(context))
        actions.append(
            ActionPayload(
                type="pending_plan_route",
                data={"origin": origin, "destination": dest, "mode": "tm"},
            )
        )
        congestion = get_current_congestion_summary()
        tip = self._get_travel_tip()
        return reply(
            f"Para ir de {origin.title()} a {dest.title()}:\n\n"
            f"{congestion}\n{tip}\n"
            f"El pasaje en TransMilenio o SITP cuesta 3.550 pesos.\n\n"
            f"Quieres que te muestre la ruta en el mapa?\n"
            "(Tip: entre mas especifica la direccion, mejor ubico los puntos)"
        )

    @staticmethod
    def _extract_route_points(groups, context):
        """Extract origin and destination from intent groups."""
        if len(groups) >= 2:
            return groups[0].strip(), groups[1].strip()
        if len(groups) == 1:
            origin = context.origin if context and context.origin else "tu ubicacion"
            return origin, groups[0].strip()
        return None, None

    @staticmethod
    def _get_travel_tip() -> str:
        """Get contextual travel tip based on current hour."""
        h = datetime.now().hour
        if 6 <= h <= 9 or 16 <= h <= 19:
            return (
                "Estas en hora pico, te recomiendo TransMilenio "
                "que tiene carril exclusivo y es mas rapido."
            )
        if h >= 23 or h < 4:
            return (
                "OJO: TransMilenio NO opera a esta hora "
                "(horario: 4am a 11pm entre semana, 5am a 10pm domingos). "
                "Tendrias que usar vehiculo particular o esperar al inicio "
                "del servicio."
            )
        if h >= 22:
            return (
                "TransMilenio esta por cerrar (cierra a las 11pm). "
                "Si tu viaje es largo, puede que no alcances. "
                "Considera alternativas."
            )
        return "El trafico esta moderado, TransMilenio o SITP son buenas opciones."

    def _intent_congestion(self, **kw):
        reply, actions, hour = kw["reply"], kw["actions"], kw["hour"]
        if hour is not None:
            response = self._get_congestion_info(hour)
            actions.append(ActionPayload(type="show_congestion", data={"hour": hour}))
        else:
            current_hour = datetime.now().hour
            response = (
                f"{get_current_congestion_summary()}\n\n"
                "Durante el dia, el trafico se comporta asi:\n"
                "En la manana de 7 a 9 es cuando peor se pone, "
                "llega al 55 o 60 por ciento.\n"
                "Al mediodia baja un poco, alrededor del 40%.\n"
                "En la tarde de 5 a 7 vuelve a subir fuerte.\n"
                "Despues de las 10 de la noche ya esta tranquilo.\n\n"
                "Si quieres evitar congestion, viaja antes de las 6am "
                "o despues de las 9pm."
            )
            actions.append(ActionPayload(type="show_congestion", data={"hour": current_hour}))
        return reply(response)

    def _intent_safety(self, **kw):
        reply, actions, hour = kw["reply"], kw["actions"], kw["hour"]
        response = self._handle_siniestralidad(kw["msg_lower"])
        if hour is not None:
            actions.append(ActionPayload(type="show_risk", data={"hour": hour}))
        return reply(response)

    def _intent_station(self, **kw):
        reply, actions, sources = kw["reply"], kw["actions"], kw["sources"]
        station_info = self._find_station_info(kw["message"])
        if station_info:
            sources.append("station_data")
            if station_info.get("action"):
                station_info["action"].type = "pending_show_station"
                actions.append(station_info["action"])
            return reply(station_info["text"] + "\n\nQuieres que te la muestre en el mapa?")
        return reply(
            "No encontre esa estacion. Hay 153 estaciones TM. "
            "Prueba con: Heroes, Portal Norte, Calle 72, Suba, Americas..."
        )

    def _intent_route_code(self, **kw):
        return kw["reply"](self._handle_route_query(kw["msg_lower"]))

    def _intent_station_list(self, **kw):
        return kw["reply"](self._handle_stations_query(kw["msg_lower"]))

    def _intent_troncal(self, **kw):
        reply = kw["reply"]
        lines = [f"- {n}: {d['stations']} estaciones" for n, d in TRONCALES.items()]
        return reply(f"TransMilenio tiene {len(TRONCALES)} troncales:\n" + "\n".join(lines))

    def _intent_cost(self, **kw):
        reply = kw["reply"]
        return reply(get_cost_info())

    def _intent_schedule(self, **kw):
        reply = kw["reply"]
        return reply(get_schedule_info())

    def _intent_compare(self, **kw):
        reply = kw["reply"]
        return reply(get_comparison_info())

    def _intent_best_time(self, **kw):
        reply = kw["reply"]
        h = datetime.now().hour
        if 6 <= h <= 9 or 16 <= h <= 19:
            return reply(
                "Ahora mismo estas en hora pico. "
                "Te recomiendo esperar hasta despues de las 9pm "
                "o madrugar antes de las 6am para viajar mas tranquilo."
            )
        return reply(
            "Ahora es buen momento para viajar, la congestion esta baja. "
            "Evita salir entre 7-9am y 5-7pm que es cuando mas se "
            "congestiona."
        )

    def _intent_nearby(self, **kw):
        reply, actions, ctx = kw["reply"], kw["actions"], kw["context"]
        if ctx and ctx.origin_coords:
            lat, lon = ctx.origin_coords
            nearby = find_nearby_stations(lat, lon, 3)
            names = ", ".join(s["name"] for s in nearby)
            response = f"Las estaciones mas cercanas son: {names}."
            if nearby:
                s = nearby[0]
                actions.append(
                    ActionPayload(
                        type="show_station",
                        data={"name": s["name"], "lat": s["lat"], "lon": s["lon"]},
                    )
                )
            return reply(response)
        return reply("No tengo tu ubicacion. Coloca un punto en el mapa y preguntame de nuevo.")

    def _intent_thanks(self, **kw):
        reply = kw["reply"]
        responses = [
            "De nada, estoy aqui para lo que necesites.",
            "Con gusto. Algo mas en lo que te pueda ayudar?",
            "Para eso estoy. Preguntame lo que quieras.",
        ]
        idx = datetime.now().second % len(responses)
        return reply(responses[idx])

    def _intent_help(self, **kw):
        reply = kw["reply"]
        return reply(
            "Soy MoviBot y puedo ayudarte con lo siguiente:\n\n"
            'Planificar rutas: dime "ir de X a Y"\n'
            'Informacion de estaciones: "estacion Heroes"\n'
            'Congestion: "como esta el trafico" o "trafico a las 7"\n'
            'Seguridad vial: "riesgo a las 18"\n'
            'Rutas TransMilenio: "ruta J74"\n'
            'Costos: "cuanto cuesta el pasaje"\n'
            'Horarios: "a que hora abre TM"\n'
            'Comparar: "que es mejor TM o SITP"\n\n'
            "Tambien puedes hablarme por voz con el boton del microfono."
        )

    def _find_station_info(self, query: str) -> dict | None:
        """Find station info matching a query (fuzzy word match, accent-insensitive)."""
        import unicodedata

        def normalize(text: str) -> str:
            """Remove accents for comparison."""
            nfkd = unicodedata.normalize("NFKD", text.lower())
            return "".join(c for c in nfkd if not unicodedata.combining(c))

        query_norm = normalize(query)
        fillers = {
            "info",
            "estacion",
            "parada",
            "sobre",
            "la",
            "el",
            "de",
            "del",
            "las",
            "los",
            "una",
            "un",
            "por",
            "para",
            "con",
            "como",
            "que",
            "esta",
            "hay",
            "tiene",
        }
        words = [w for w in query_norm.split() if w not in fillers and len(w) > 2]
        for station in TM_STATIONS:
            name_norm = normalize(station["name"])
            if any(w in name_norm for w in words):
                neighbors = list(self._graph.neighbors(station["id"]))
                neighbor_names = [
                    self._graph.nodes[n]["name"] for n in neighbors if n in self._graph.nodes
                ]
                connects = ", ".join(neighbor_names[:5]) if neighbor_names else "ninguna registrada"
                text = (
                    f"Estacion {station['name']}.\n\n"
                    f"Pertenece a la troncal {station['troncal']}.\n"
                    f"Se conecta con: {connects}."
                )
                action = ActionPayload(
                    type="show_station",
                    data={
                        "name": station["name"],
                        "lat": station["lat"],
                        "lon": station["lon"],
                    },
                )
                return {"text": text, "action": action}
        return None

    def _get_congestion_info(self, hour: int | None = None) -> str:
        """Get congestion info in natural language."""
        if hour is not None and 0 <= hour <= 23:
            level = CONGESTION_BY_HOUR.get(hour, 0.3)
            pct = int(level * 100)
            if level < 0.3:
                return (
                    f"A las {hour}:00 el trafico esta tranquilo, "
                    f"solo al {pct}%. Buen momento para moverse."
                )
            elif level < 0.6:
                return (
                    f"A las {hour}:00 hay congestion moderada, "
                    f"alrededor del {pct}%. Se puede viajar con algo de demora."
                )
            elif level < 0.85:
                return (
                    f"A las {hour}:00 el trafico esta pesado, "
                    f"al {pct}%. Te recomiendo usar TransMilenio."
                )
            else:
                return (
                    f"A las {hour}:00 la congestion esta critica, "
                    f"al {pct}%. Evita viajar en carro si puedes."
                )
        return (
            "Ahora mismo el trafico en Bogota esta asi:\n\n"
            "Las horas mas pesadas son de 7 a 9 de la manana "
            "y de 5 a 7 de la tarde, donde llega al 55-60%.\n"
            "Al mediodia baja un poco.\n"
            "Despues de las 10 de la noche esta libre.\n\n"
            "Si puedes, viaja temprano antes de las 6 o tarde despues de las 9."
        )

    def _handle_siniestralidad(self, msg_lower: str) -> str:
        """Handle safety/risk queries."""
        if not self._siniestros_service:
            return "No tengo datos de siniestralidad cargados en este momento."
        stats = self._siniestros_service.get_stats()
        hour_match = re.search(r"(\d{1,2})", msg_lower)
        if hour_match:
            hour = int(hour_match.group(1))
            if 0 <= hour <= 23:
                risk = self._siniestros_service.predict_risk_by_hour(hour)
                top3 = risk.zones[:3]
                zones = "\n".join(
                    f"  - {z.localidad}: {z.nivel} ({int(z.risk * 100)}%)" for z in top3
                )
                return (
                    f"Riesgo vial a las {hour}:00: "
                    f"{risk.nivel_general} ({int(risk.promedio_riesgo * 100)}%)\n"
                    f"Zonas mas riesgosas:\n{zones}"
                )
        return (
            f"Siniestralidad vial en Bogota:\n"
            f"Total accidentes: {stats.total_siniestros:,}\n"
            f"Fatales: {stats.total_fallecidos:,}\n"
            f"Sectores criticos: {stats.sectores_criticos}\n"
            "Pregunta por una hora para ver el riesgo por zona."
        )

    def _handle_stations_query(self, msg_lower: str) -> str:
        """Handle station listing queries."""
        for t in TRONCALES:
            if t.lower() in msg_lower:
                names = [s["name"] for s in TM_STATIONS if s["troncal"] == t]
                return f"Troncal {t} tiene {len(names)} estaciones:\n" + ", ".join(names)
        return (
            f"TransMilenio tiene {len(TM_STATIONS)} estaciones "
            f"en {len(TRONCALES)} troncales. "
            "Preguntame por una troncal: Caracas, Suba, Calle 80, NQS, Americas..."
        )

    def _handle_route_query(self, msg_lower: str) -> str:
        """Handle route queries."""
        found = [r for r in TM_RUTAS if r["codigo"].lower() in msg_lower]
        if found:
            r = found[0]
            return (
                f"Ruta {r['codigo']}:\n"
                f"Recorrido: {r['origen']} a {r['destino']}\n"
                f"Bus: {r['tipo_bus']}\n"
                f"Lunes a viernes: {r['horario_lv']}\n"
                f"Sabados: {r['horario_sab']}\n"
                f"Estado: {r['estado']}"
            )
        return (
            f"Hay {len(TM_RUTAS)} rutas TM disponibles. "
            "Dime un codigo como J74, F51 o G43 para darte la info."
        )

    def _default_response(self, context: AppContext | None = None) -> str:
        """Context-aware default response."""
        if context and context.module == "planificar":
            if context.origin and not context.destination:
                return "Ya tienes un origen. A donde quieres ir? Dime el destino o toca en el mapa."
            if context.origin and context.destination:
                return "Tienes origen y destino listos. Quieres que busque la mejor ruta?"
            return (
                "Estas en el planificador de viajes. "
                "Dime de donde sales y a donde vas, "
                "o preguntame cual es la mejor hora."
            )
        if context and context.module == "rutas":
            return (
                "Estas viendo las rutas. "
                "Preguntame por un codigo como J74 o F51, o por una troncal."
            )
        if context and context.module == "metricas":
            return "Estas en las metricas. Preguntame por la congestion a una hora especifica."
        return (
            "Soy MoviBot, tu asistente de movilidad en Bogota. Puedo:\n\n"
            'Planificar rutas: "ir de Usaquen a Centro"\n'
            'Info de estaciones: "estacion Heroes"\n'
            'Congestion por hora: "trafico a las 7"\n'
            'Riesgo vial: "riesgo a las 18"\n'
            'Info de rutas TM: "ruta J74"\n\n'
            "En que te ayudo?"
        )

    def clear_session(self, session_id: str) -> None:
        """Clear session history."""
        self._sessions.pop(session_id, None)
