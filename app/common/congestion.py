"""Shared congestion utilities used across prediction modules."""

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


def time_factor(hour: int) -> float:
    """Time-of-day multiplier for congestion."""
    return HOUR_FACTORS.get(hour, 0.7)


def risk_label(congestion: float) -> str:
    """Map congestion level to risk label."""
    if congestion < 0.3:
        return "low"
    if congestion < 0.6:
        return "medium"
    if congestion < 0.85:
        return "high"
    return "critical"
