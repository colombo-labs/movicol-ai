"""Incidents & notifications service — PostgreSQL backed."""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import psycopg2
import psycopg2.extras

from app.modules.incidents.schemas import (
    IncidentCreate,
    IncidentResponse,
    NotificationItem,
)

DB_URL = os.environ.get("DATABASE_URL", "")

# SQL to create tables (idempotent)
INIT_SQL = """
CREATE TABLE IF NOT EXISTS incidents (
    id SERIAL PRIMARY KEY,
    type VARCHAR(50) NOT NULL,
    lat DOUBLE PRECISION NOT NULL,
    lng DOUBLE PRECISION NOT NULL,
    route_code VARCHAR(20) DEFAULT '',
    description TEXT DEFAULT '',
    source VARCHAR(20) DEFAULT 'user_report',
    votes INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS system_alerts (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    type VARCHAR(50) NOT NULL,
    route_codes TEXT[] DEFAULT '{}',
    source VARCHAR(20) DEFAULT 'scraping',
    url TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_incidents_created ON incidents(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_incidents_location ON incidents(lat, lng);
CREATE INDEX IF NOT EXISTS idx_system_alerts_created ON system_alerts(created_at DESC);
"""


class IncidentsService:
    """Service for managing incidents and notifications."""

    _initialized: bool = False

    def __init__(self) -> None:
        if not IncidentsService._initialized:
            self._ensure_tables()
            IncidentsService._initialized = True

    def _get_conn(self):
        """Get a database connection."""
        return psycopg2.connect(DB_URL)

    def _ensure_tables(self) -> None:
        """Create tables if they don't exist."""
        if not DB_URL:
            return
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute(INIT_SQL)
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"Warning: Could not initialize incidents tables: {e}")

    def create_incident(self, incident: IncidentCreate) -> IncidentResponse | None:
        """Store a user-reported incident."""
        if not DB_URL:
            return None
        try:
            conn = self._get_conn()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                """
                INSERT INTO incidents (type, lat, lng, route_code, description, source)
                VALUES (%s, %s, %s, %s, %s, 'user_report')
                RETURNING id, type, lat, lng, route_code, description, source, votes, created_at
                """,
                (
                    incident.type,
                    incident.lat,
                    incident.lng,
                    incident.route_code,
                    incident.description,
                ),
            )
            row = cur.fetchone()
            conn.commit()
            cur.close()
            conn.close()
            return IncidentResponse(**row) if row else None
        except Exception as e:
            print(f"Error creating incident: {e}")
            return None

    def vote_incident(self, incident_id: int) -> bool:
        """Upvote an existing incident (confirms it's still active)."""
        if not DB_URL:
            return False
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute("UPDATE incidents SET votes = votes + 1 WHERE id = %s", (incident_id,))
            conn.commit()
            affected = cur.rowcount
            cur.close()
            conn.close()
            return affected > 0
        except Exception:
            return False

    def save_system_alerts(self, alerts: list[dict]) -> int:
        """Save scraped alerts to DB (deduplicates by title+date)."""
        if not DB_URL or not alerts:
            return 0
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            saved = 0
            today = datetime.now().date()
            for alert in alerts:
                title = alert.get("title", "")
                if not title:
                    continue
                # Dedup: don't insert if same title exists today
                cur.execute(
                    "SELECT 1 FROM system_alerts WHERE title = %s AND created_at::date = %s",
                    (title, today),
                )
                if cur.fetchone():
                    continue
                route_codes = alert.get("route_codes", [])
                cur.execute(
                    """
                    INSERT INTO system_alerts (title, type, route_codes, source, url)
                    VALUES (%s, %s, %s, 'scraping', %s)
                    """,
                    (
                        title,
                        self._classify_alert(title),
                        route_codes,
                        alert.get("url", ""),
                    ),
                )
                saved += 1
            conn.commit()
            cur.close()
            conn.close()
            return saved
        except Exception as e:
            print(f"Error saving system alerts: {e}")
            return 0

    def get_notifications(
        self, hours: int = 6, lat: float | None = None, lng: float | None = None
    ) -> list[NotificationItem]:
        """Get recent notifications (incidents + system alerts)."""
        if not DB_URL:
            return []
        try:
            conn = self._get_conn()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            since = datetime.now() - timedelta(hours=hours)
            notifications: list[NotificationItem] = []

            # Recent incidents
            cur.execute(
                """
                SELECT id, type, lat, lng, route_code, description, source, votes, created_at
                FROM incidents
                WHERE created_at > %s
                ORDER BY created_at DESC
                LIMIT 30
                """,
                (since,),
            )
            for row in cur.fetchall():
                notifications.append(
                    NotificationItem(
                        id=f"incident-{row['id']}",
                        title=self._incident_title(row["type"]),
                        body=row["description"]
                        or f"Reportado cerca de ruta {row['route_code'] or '?'}",
                        type="incident",
                        severity=self._incident_severity(row["type"]),
                        lat=row["lat"],
                        lng=row["lng"],
                        route_codes=[row["route_code"]] if row["route_code"] else [],
                        created_at=row["created_at"],
                        source="user_report",
                    )
                )

            # Recent system alerts
            cur.execute(
                """
                SELECT id, title, type, route_codes, source, url, created_at
                FROM system_alerts
                WHERE created_at > %s
                ORDER BY created_at DESC
                LIMIT 20
                """,
                (since,),
            )
            for row in cur.fetchall():
                notifications.append(
                    NotificationItem(
                        id=f"alert-{row['id']}",
                        title=row["title"],
                        body=f"Rutas afectadas: {', '.join(row['route_codes'] or [])}",
                        type="alert",
                        severity="danger" if row["type"] == "suspended" else "warning",
                        route_codes=row["route_codes"] or [],
                        created_at=row["created_at"],
                        source="scraping",
                    )
                )

            cur.close()
            conn.close()

            # Sort by most recent
            notifications.sort(key=lambda n: n.created_at, reverse=True)
            return notifications[:30]
        except Exception as e:
            print(f"Error fetching notifications: {e}")
            return []

    def get_nearby_incidents(
        self, lat: float, lng: float, radius_km: float = 1.0, hours: int = 2
    ) -> list[IncidentResponse]:
        """Get recent incidents near a location."""
        if not DB_URL:
            return []
        try:
            conn = self._get_conn()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            since = datetime.now() - timedelta(hours=hours)
            # Approximate distance filter (1 degree ≈ 111 km)
            delta = radius_km / 111.0
            cur.execute(
                """
                SELECT id, type, lat, lng, route_code, description, source, votes, created_at
                FROM incidents
                WHERE created_at > %s
                  AND lat BETWEEN %s AND %s
                  AND lng BETWEEN %s AND %s
                ORDER BY votes DESC, created_at DESC
                LIMIT 20
                """,
                (since, lat - delta, lat + delta, lng - delta, lng + delta),
            )
            rows = cur.fetchall()
            cur.close()
            conn.close()
            return [IncidentResponse(**row) for row in rows]
        except Exception:
            return []

    @staticmethod
    def _classify_alert(title: str) -> str:
        """Classify alert type from title text."""
        t = title.lower()
        if "suspende" in t or "suspend" in t:
            return "suspended"
        if "demora" in t or "retraso" in t:
            return "delayed"
        if "modifica" in t or "cambio" in t:
            return "modified"
        return "info"

    @staticmethod
    def _incident_title(incident_type: str) -> str:
        """Human-readable title for incident type."""
        titles = {
            "demora": "⏱ Demora reportada",
            "lleno": "🚏 Bus lleno",
            "inseguro": "⚠️ Zona insegura",
            "cerrado": "🚫 Estación/paradero cerrado",
            "accidente": "🚨 Accidente en la vía",
        }
        return titles.get(incident_type, f"📍 Incidente: {incident_type}")

    @staticmethod
    def _incident_severity(incident_type: str) -> str:
        """Map incident type to severity."""
        if incident_type in ("accidente", "cerrado"):
            return "danger"
        if incident_type in ("demora", "inseguro"):
            return "warning"
        return "info"
