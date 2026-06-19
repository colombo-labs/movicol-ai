"""LLM explanation generator for route predictions."""

from app.config.settings import get_settings
from app.modules.route_prediction.schemas import RoutePredictionResponse

SYSTEM_PROMPT = """Eres un asistente de movilidad urbana para Bogotá, Colombia.
Tu trabajo es explicar predicciones de congestión de rutas de TransMilenio de forma clara y útil.
Responde en español, máximo 3 oraciones. Sé conciso y práctico.
Incluye: resumen del estado de la ruta, el tramo más congestionado, y una recomendación."""

USER_TEMPLATE = """Ruta: {stations}
Hora de salida: {departure_time}
Tiempo estimado: {total_time} min
Distancia: {total_distance} km
Riesgo general: {overall_risk}
Tramos críticos: {critical_segments}

Genera una explicación breve para el usuario."""


def generate_explanation(prediction: RoutePredictionResponse) -> str:
    """Generate LLM explanation for a route prediction.

    Falls back to template-based explanation if OpenAI key is not configured.
    """
    settings = get_settings()

    # Find critical segments
    critical = [
        f"{s.from_station}→{s.to_station} ({int(s.congestion_level * 100)}%)"
        for s in prediction.risk_segments
        if s.congestion_level >= 0.6
    ]
    critical_text = ", ".join(critical) if critical else "Ninguno"

    # If no API key, use template-based explanation
    if not settings.openai_api_key:
        return _template_explanation(prediction, critical_text)

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0.3,
            max_tokens=200,
        )

        user_msg = USER_TEMPLATE.format(
            stations=" → ".join(prediction.stations),
            departure_time=prediction.departure_time,
            total_time=prediction.total_time_minutes,
            total_distance=prediction.total_distance_km,
            overall_risk=prediction.overall_risk,
            critical_segments=critical_text,
        )

        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ]
        )
        return response.content.strip()

    except Exception:
        return _template_explanation(prediction, critical_text)


def _template_explanation(prediction: RoutePredictionResponse, critical_text: str) -> str:
    """Fallback template-based explanation when LLM is unavailable."""
    risk_map = {
        "low": "fluida",
        "medium": "moderada",
        "high": "alta",
        "critical": "crítica",
    }
    risk_desc = risk_map.get(prediction.overall_risk, "moderada")

    msg = (
        f"La ruta presenta congestión {risk_desc}. "
        f"Tiempo estimado: {prediction.total_time_minutes} min "
        f"para {prediction.total_distance_km} km. "
    )

    if critical_text != "Ninguno":
        msg += f"Tramos críticos: {critical_text}. "
        msg += "Considere salir antes o usar una ruta alternativa."
    else:
        msg += "No se detectan tramos críticos en este momento."

    return msg
