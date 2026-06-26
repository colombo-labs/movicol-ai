"""Shared congestion utilities used across prediction modules."""
from __future__ import annotations

from datetime import datetime

HOUR_FACTORS: dict[int, float] = {
    0: 0.15,
    1: 0.10,
    2: 0.08,
    3: 0.08,
    4: 0.15,
    5: 0.35,
    6: 0.65,
    7: 0.88,
    8: 1.0,
    9: 0.85,
    10: 0.60,
    11: 0.55,
    12: 0.70,
    13: 0.65,
    14: 0.55,
    15: 0.60,
    16: 0.75,
    17: 0.92,
    18: 1.0,
    19: 0.85,
    20: 0.55,
    21: 0.35,
    22: 0.25,
    23: 0.18,
}

# Day of week multiplier (0=Monday, 6=Sunday)
# Wider spread: weekday peak vs weekend low = ~2.5x difference
DAY_FACTORS: dict[int, float] = {
    0: 1.0,  # Lunes — alta congestión
    1: 0.95,  # Martes
    2: 0.95,  # Miércoles
    3: 0.97,  # Jueves
    4: 1.10,  # Viernes — pico por salida de oficinas
    5: 0.50,  # Sábado — tráfico reducido
    6: 0.30,  # Domingo — mínimo
}


def time_factor(hour: int, day: int | None = None) -> float:
    """Time-of-day + day-of-week multiplier for congestion."""
    if day is None:
        day = datetime.now().weekday()
    return HOUR_FACTORS.get(hour, 0.5) * DAY_FACTORS.get(day, 1.0)


def risk_label(congestion: float) -> str:
    """Map congestion level to risk label."""
    if congestion < 0.3:
        return "low"
    if congestion < 0.6:
        return "medium"
    if congestion < 0.85:
        return "high"
    return "critical"
