"""Shared congestion utilities used across prediction modules."""

from datetime import datetime

HOUR_FACTORS: dict[int, float] = {
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

# Day of week multiplier (0=Monday, 6=Sunday)
DAY_FACTORS: dict[int, float] = {
    0: 1.0,  # Lunes — alta congestión
    1: 1.0,  # Martes
    2: 1.0,  # Miércoles
    3: 1.0,  # Jueves
    4: 1.05,  # Viernes — pico por salida de oficinas
    5: 0.6,  # Sábado — menos tráfico
    6: 0.4,  # Domingo — mínimo
}


def time_factor(hour: int) -> float:
    """Time-of-day + day-of-week multiplier for congestion."""
    day = datetime.now().weekday()
    return HOUR_FACTORS.get(hour, 0.7) * DAY_FACTORS.get(day, 1.0)


def risk_label(congestion: float) -> str:
    """Map congestion level to risk label."""
    if congestion < 0.3:
        return "low"
    if congestion < 0.6:
        return "medium"
    if congestion < 0.85:
        return "high"
    return "critical"
