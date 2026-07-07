"""Agent tools — functions the LLM can call to interact with the app."""

from __future__ import annotations

from langchain_core.tools import tool

from app.modules.agent.schemas import ActionPayload
from app.modules.route_prediction.graph_data import (
    CONGESTION_BY_HOUR,
    TM_RUTAS,
    TM_STATIONS,
    TRONCALES,
)


@tool
def plan_route(origin: str, destination: str, mode: str = "tm") -> str:
    """Plan a route between two locations. Use this when the user asks to go from A to B.
    Args:
        origin: Starting address or station name
        destination: End address or station name
        mode: Transport mode - 'tm' (TransMilenio), 'sitp', 'vehicle', 'moto'
    """
    return f"ACTION:plan_route|origin={origin}|destination={destination}|mode={mode}"


@tool
def find_station(query: str) -> str:
    """Find a TransMilenio station by name. Use when user asks about a specific station.
    Args:
        query: Station name or partial name to search
    """
    query_lower = query.lower()
    matches = []
    for station in TM_STATIONS:
        if query_lower in station["name"].lower():
            matches.append(station)

    if not matches:
        return f"No encontré estaciones con '{query}'. Hay {len(TM_STATIONS)} estaciones TM."

    results = []
    for s in matches[:5]:
        results.append(f"• {s['name']} — Troncal {s['troncal']} ({s['lat']:.4f}, {s['lon']:.4f})")

    m = matches[0]
    action = f"ACTION:show_station|name={m['name']}|lat={m['lat']}|lon={m['lon']}"
    return "\n".join(results) + "\n" + action


@tool
def find_route_info(route_code: str) -> str:
    """Find information about a specific TransMilenio route by its code (e.g., J74, F51, G43).
    Args:
        route_code: The route code to search for
    """
    code_lower = route_code.lower().strip()
    for r in TM_RUTAS:
        if r["codigo"].lower() == code_lower:
            return (
                f"Ruta {r['codigo']}:\n"
                f"• Recorrido: {r['origen']} → {r['destino']}\n"
                f"• Bus: {r['tipo_bus']}\n"
                f"• Horario L-V: {r['horario_lv']}\n"
                f"• Horario Sáb: {r['horario_sab']}\n"
                f"• Estado: {r['estado']}"
            )
    # Try partial match
    partials = [r for r in TM_RUTAS if code_lower in r["codigo"].lower()]
    if partials:
        return "Rutas encontradas: " + ", ".join(r["codigo"] for r in partials[:10])
    return f"No encontré la ruta '{route_code}'. Hay {len(TM_RUTAS)} rutas TM."


@tool
def get_congestion(hour: int | None = None) -> str:
    """Get traffic congestion levels. Use when user asks about traffic or best times to travel.
    Args:
        hour: Optional specific hour (0-23) to check. If None, returns general overview.
    """
    if hour is not None and 0 <= hour <= 23:
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
        return (
            f"A las {hour}:00, congestión promedio: {int(level * 100)}% ({label}).\n"
            f"ACTION:show_congestion|hour={hour}|level={level}"
        )

    return (
        "Pico AM (7-9h): congestión alta (~55-60%).\n"
        "Pico PM (17-19h): congestión alta (~55-60%).\n"
        "Valle (22-5h): congestión baja (<15%).\n"
        "Mejor horario: antes de 6am o después de 9pm."
    )


@tool
def get_risk_by_zone(hour: int = 8) -> str:
    """Get road accident risk by zone and hour. Use when user asks about safety or danger zones.
    Args:
        hour: Hour of day (0-23) to check risk levels
    """
    return f"ACTION:show_risk|hour={hour}"


@tool
def list_troncales() -> str:
    """List all TransMilenio troncales with station counts. Use when user asks about TM lines."""
    lines = []
    for name, data in TRONCALES.items():
        lines.append(f"• {name}: {data['stations']} estaciones")
    return f"TransMilenio tiene {len(TRONCALES)} troncales:\n" + "\n".join(lines)


# All available tools for the agent
AGENT_TOOLS = [
    plan_route,
    find_station,
    find_route_info,
    get_congestion,
    get_risk_by_zone,
    list_troncales,
]


def parse_actions(tool_output: str) -> list[ActionPayload]:
    """Extract ACTION: directives from tool output."""
    actions = []
    for line in tool_output.split("\n"):
        if line.startswith("ACTION:"):
            parts = line[7:].split("|")
            action_type = parts[0]
            data = {}
            for part in parts[1:]:
                if "=" in part:
                    k, v = part.split("=", 1)
                    # Try to parse numbers
                    try:
                        data[k] = float(v) if "." in v else int(v)
                    except ValueError:
                        data[k] = v
            actions.append(ActionPayload(type=action_type, data=data))
    return actions
