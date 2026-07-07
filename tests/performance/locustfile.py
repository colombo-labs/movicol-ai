"""
Load & Stress Testing — MoviCol AI Service
Run: locust -f tests/performance/locustfile.py --host http://localhost:8000

Load test: 50 users, spawn rate 5/s, run 60s
Stress test: 200 users, spawn rate 20/s, run 120s

CLI (headless):
  locust -f tests/performance/locustfile.py --host http://localhost:8000 \
    --users 50 --spawn-rate 5 --run-time 60s --headless --csv results/load
"""

import random

from locust import HttpUser, between, task


CHAT_MESSAGES = [
    "hola",
    "cómo está el tráfico",
    "ir de usaquen al centro",
    "estación Héroes",
    "cuánto cuesta el pasaje",
    "mejor hora para viajar",
    "ruta J74",
    "riesgo a las 18",
    "qué es mejor TM o SITP",
    "gracias",
    "del centro a soacha",
    "a qué hora abre TransMilenio",
    "estaciones de la troncal caracas",
    "ir de chapinero a kennedy",
    "congestión a las 7",
]

ROUTE_ORIGINS = [
    {"lat": 4.695, "lon": -74.031},  # Usaquén
    {"lat": 4.643, "lon": -74.063},  # Chapinero
    {"lat": 4.598, "lon": -74.076},  # Centro
    {"lat": 4.741, "lon": -74.083},  # Suba
    {"lat": 4.621, "lon": -74.152},  # Kennedy
]

ROUTE_DESTINATIONS = [
    {"lat": 4.598, "lon": -74.076},  # Centro
    {"lat": 4.579, "lon": -74.217},  # Soacha
    {"lat": 4.759, "lon": -74.045},  # Portal Norte
    {"lat": 4.625, "lon": -74.175},  # Portal Américas
    {"lat": 4.571, "lon": -74.131},  # Portal Tunal
]


class ChatUser(HttpUser):
    """Simulates a user chatting with MoviBot."""

    wait_time = between(1, 3)

    @task(5)
    def chat_message(self):
        """Send a random chat message."""
        msg = random.choice(CHAT_MESSAGES)
        self.client.post(
            "/agent/chat",
            json={"message": msg, "session_id": f"load-{self.environment.runner.user_count}"},
        )

    @task(2)
    def chat_with_context(self):
        """Send chat with app context."""
        msg = random.choice(CHAT_MESSAGES)
        self.client.post(
            "/agent/chat",
            json={
                "message": msg,
                "session_id": f"ctx-{random.randint(1, 100)}",
                "context": {
                    "module": random.choice(["planificar", "rutas", "metricas"]),
                    "origin": "Usaquén",
                },
            },
        )

    @task(3)
    def predict_route(self):
        """Request a route prediction."""
        origin = random.choice(ROUTE_ORIGINS)
        dest = random.choice(ROUTE_DESTINATIONS)
        self.client.post(
            "/predictions/route",
            json={
                "origin": origin,
                "destination": dest,
                "departure_time": "2026-07-06T08:00:00",
                "mode": random.choice(["transmilenio", "sitp"]),
            },
        )

    @task(1)
    def health_check(self):
        """Check health endpoint."""
        self.client.get("/health")

    @task(1)
    def get_congestion(self):
        """Get congestion heatmap."""
        hour = random.randint(0, 23)
        self.client.get(f"/predictions/congestion?hour={hour}")


class StressUser(HttpUser):
    """High-frequency stress user — minimal wait."""

    wait_time = between(0.1, 0.5)

    @task
    def rapid_chat(self):
        """Rapid-fire chat messages."""
        self.client.post(
            "/agent/chat",
            json={
                "message": random.choice(CHAT_MESSAGES),
                "session_id": f"stress-{random.randint(1, 1000)}",
            },
        )

    @task
    def rapid_route(self):
        """Rapid-fire route predictions."""
        origin = random.choice(ROUTE_ORIGINS)
        dest = random.choice(ROUTE_DESTINATIONS)
        self.client.post(
            "/predictions/route",
            json={
                "origin": origin,
                "destination": dest,
                "departure_time": "2026-07-06T08:00:00",
                "mode": "transmilenio",
            },
        )
