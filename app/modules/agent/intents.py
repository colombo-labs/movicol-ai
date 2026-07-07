"""Smart intent detection and contextual intelligence for MoviBot."""

from __future__ import annotations

import re
from datetime import datetime

from app.modules.route_prediction.graph_data import CONGESTION_BY_HOUR, TM_STATIONS

# Intent patterns with priority
INTENTS = [
    # Greeting
    (
        "greeting",
        [
            r"\b(hola|hey|buenas|qué tal|buenos días|buenas tardes|buenas noches)\b",
        ],
    ),
    # Route planning
    (
        "plan_route",
        [
            r"(?:ir|llegar|viajar|quiero ir)\s+(?:de|desde)\s+(.+?)"
            r"\s+(?:a|al|hasta|hacia)\s+(.+?)(?:\?|$|\.)",
            r"(?:cómo|como)\s+(?:llego|voy|ir)\s+(?:de|desde)\s+(.+?)\s+(?:a|al|hasta|hacia)\s+(.+?)(?:\?|$)",
            r"(?:cómo|como)\s+(?:llego|voy|ir)\s+(?:a|al|hasta)\s+(.+?)(?:\?|$)",
            r"(?:llévame|llevame|planifica|planificar)\s+(?:de|desde)\s+(.+?)\s+(?:a|hasta)\s+(.+?)(?:\?|$)",
            r"(?:llévame|llevame|planifica|planificar)\s+(?:a|hasta)\s+(.+?)(?:\?|$)",
            r"\b(?:de|del)\s+(.+?)\s+(?:a|al|hasta|hacia)\s+(.+?)(?:\s+(?:dame|dime|muestra).*)?(?:\?|$|\.|,)",
            r"\bruta\s+(?:de|del|desde|para)\s+(.+?)\s+(?:a|al|hasta)\s+(.+?)(?:\?|$)",
            r"(?:dame|dime|muestra)\s+(?:la\s+)?ruta\s+(?:de|del|desde)\s+(.+?)\s+(?:a|al|hasta)\s+(.+?)(?:\?|$)",
            r"(?:mejor\s+)?ruta\s+(?:a|al|hasta|hacia|para)\s+(.+?)(?:\?|$)",
            r"(?:quiero|necesito)\s+ir\s+(?:a|al|hasta|hacia)\s+(.+?)(?:\?|$)",
            r"\bir\s+(?:a|al|hasta|hacia)\s+(.+?)(?:\?|$)",
            r"(?:dame|dime)\s+(?:la\s+)?ruta\s+(?:a|al|hasta)\s+(.+?)(?:\?|$)",
        ],
    ),
    # Congestion/Traffic
    (
        "congestion",
        [
            r"\b(congestión|congestion|tráfico|trafico|hora pico|peak|atasco)\b",
            r"\b(mejor hora|peor hora|cuándo viajar|cuando viajar)\b",
        ],
    ),
    # Safety/Risk
    (
        "safety",
        [
            r"\b(siniestro|accidente|peligro|riesgo|seguridad|seguro|insegur|mortalidad)\b",
            r"\b(zona peligrosa|zona segura|cuidado)\b",
        ],
    ),
    # Station info
    (
        "station",
        [
            r"\b(estación|estacion|parada|paradero)\s+(.+)",
            r"\binfo\s+(.+?)(?:\?|$)",
        ],
    ),
    # Route info (specific code)
    (
        "route_code",
        [
            r"\b(ruta|línea|linea)\s+([a-zA-Z]\d{1,3})\b",
            r"\b([a-zA-Z]\d{1,3})\b",  # bare code like J74
        ],
    ),
    # Station list / count
    (
        "station_list",
        [
            r"\b(estaciones|paradas|cuántas|cuantas|lista|todas)\b",
        ],
    ),
    # Troncal info
    (
        "troncal",
        [
            r"\b(troncal|troncales|líneas|lineas)\b",
        ],
    ),
    # Cost/Price
    (
        "cost",
        [
            r"\b(costo|precio|tarifa|pasaje|cuánto cuesta|cuanto cuesta|vale|pagar)\b",
        ],
    ),
    # Schedule/Hours
    (
        "schedule",
        [
            r"\b(horario|hora|abre|cierra|funciona|opera|servicio)\b",
        ],
    ),
    # Comparison
    (
        "compare",
        [
            r"\b(comparar|versus|vs|diferencia|más rápido|más barato)\b",
            r"\b(mejor|peor)\b.*(tm|sitp|transmilenio|bus|carro|moto|vehiculo)",
            r"\b(tm|sitp|transmilenio)\b.*(mejor|peor|vs|versus)",
        ],
    ),
    # Help
    (
        "best_time",
        [
            r"\b(mejor hora|cuándo viajar|cuando viajar|mejor momento|hora recomendada)\b",
        ],
    ),
    (
        "nearby",
        [
            r"\b(cerca|cercana|cercano|proxim|nearest)\b",
        ],
    ),
    (
        "confirm",
        [
            r"^(s[ií]|si|dale|va|hazlo|claro|por favor|porfa|ok|okey|sip|sep|aja|bueno)$",
            r"^(s[ií]|dale|va|hazlo|claro|por favor|ok)\b",
        ],
    ),
    (
        "deny",
        [
            r"^(no|nop|nah|nel|mejor no|no gracias|cancelar)\b",
        ],
    ),
    (
        "thanks",
        [
            r"\b(gracias|thanks|genial|perfecto|listo|entendido|chevere)\b",
        ],
    ),
    (
        "help",
        [
            r"\b(ayuda|help|qué puedes|que puedes|funciones|capacidades)\b",
        ],
    ),
]


def detect_intent(message: str) -> tuple[str, dict]:
    """Detect user intent and extract slots from message."""
    msg_lower = message.lower().strip()
    slots: dict = {}

    for intent_name, patterns in INTENTS:
        for pattern in patterns:
            match = re.search(pattern, msg_lower)
            if match:
                slots["groups"] = match.groups()
                # Extract hour if present
                hour_match = re.search(r"(\d{1,2})\s*(?:am|pm|:00|hrs?|horas?)?", msg_lower)
                if hour_match:
                    h = int(hour_match.group(1))
                    if "pm" in msg_lower and h < 12:
                        h += 12
                    if 0 <= h <= 23:
                        slots["hour"] = h
                return intent_name, slots

    return "unknown", slots


def get_current_congestion_summary() -> str:
    """Get a smart summary based on current time."""
    now = datetime.now()
    hour = now.hour
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

    # Find next good window
    good_hours = [h for h, v in CONGESTION_BY_HOUR.items() if v < 0.3 and h > hour]
    next_good = good_hours[0] if good_hours else None

    summary = f"Ahora ({hour}:00): congestión {label} ({int(level * 100)}%)"
    if level >= 0.5 and next_good:
        summary += f"\nMejora a las {next_good}:00."
    elif level < 0.3:
        summary += "\nBuen momento para viajar."

    return summary


def get_cost_info() -> str:
    """Transport cost information."""
    return (
        "Tarifas actuales de transporte en Bogotá:\n\n"
        "TransMilenio: 3.550 pesos por viaje\n"
        "SITP Zonal: 3.550 pesos por viaje\n"
        "Transbordo TM a SITP: incluido en la tarifa (tienes 95 minutos)\n"
        "Cable aéreo: ya está incluido en la tarifa de TM\n"
        "Vehículo propio: aproximadamente 2.000 pesos por kilómetro"
    )


def get_schedule_info() -> str:
    """TransMilenio schedule information."""
    return (
        "Horarios de TransMilenio:\n\n"
        "Lunes a Sábado: abre a las 4 de la mañana, cierra a las 11 de la noche\n"
        "Domingos y Festivos: abre a las 5 de la mañana, cierra a las 10 de la noche\n\n"
        "En hora pico los buses pasan cada 3 a 5 minutos.\n"
        "En horas valle, cada 8 a 12 minutos.\n"
        "El SITP zonal tiene horarios similares según la ruta."
    )


def get_comparison_info() -> str:
    """Comparison between transport modes."""
    return (
        "Comparación de modos de transporte:\n\n"
        "TransMilenio: 3.550 pesos, el más rápido por carril exclusivo. Opera de 4am a 11pm.\n"
        "SITP Zonal: 3.550 pesos, tiene la cobertura más amplia de la ciudad.\n"
        "Vehículo: flexible pero sufre la congestión, cuesta unos 2.000 pesos por km.\n"
        "Moto: rápida y barata, unos 800 pesos por km, pero con mayor riesgo de accidente.\n\n"
        "En hora pico, TransMilenio es aproximadamente 40% más rápido que el carro."
    )


def find_nearby_stations(lat: float, lon: float, limit: int = 5) -> list[dict]:
    """Find stations closest to given coordinates."""
    stations_with_dist = []
    for s in TM_STATIONS:
        dist = ((s["lat"] - lat) ** 2 + (s["lon"] - lon) ** 2) ** 0.5
        stations_with_dist.append({**s, "dist": dist})
    stations_with_dist.sort(key=lambda x: x["dist"])
    return stations_with_dist[:limit]


def get_greeting_response(hour: int | None = None) -> str:
    """Smart greeting based on time of day."""
    h = hour if hour is not None else datetime.now().hour
    congestion = get_current_congestion_summary()

    if 5 <= h < 12:
        greeting = "¡Buenos días!"
    elif 12 <= h < 18:
        greeting = "¡Buenas tardes!"
    else:
        greeting = "¡Buenas noches!"

    return (
        f"{greeting} Soy MoviBot, tu asistente de movilidad en Bogotá.\n\n"
        f"Estado del tráfico:\n{congestion}\n\n"
        "¿En qué te ayudo? Puedo planificar rutas, darte información de estaciones, "
        "o decirte cuál es la mejor hora para viajar."
    )
