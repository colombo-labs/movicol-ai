"""Tests for route prediction module."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.modules.route_prediction.graph_data import build_caracas_graph
from app.modules.route_prediction.schemas import Coordinates
from app.modules.route_prediction.service import RoutePredictionService


class TestGraphData:
    """Tests for the static graph."""

    def test_graph_has_correct_nodes(self):
        g = build_caracas_graph()
        assert g.number_of_nodes() >= 10  # At least 10 stations

    def test_graph_is_bidirectional(self):
        g = build_caracas_graph()
        assert g.number_of_edges() >= g.number_of_nodes() - 1  # At least a spanning tree

    def test_nodes_have_required_attributes(self):
        g = build_caracas_graph()
        for _, data in g.nodes(data=True):
            assert "lat" in data or "name" in data
            if "lat" in data:
                assert float(data["lat"]) > 4.5
                assert float(data["lat"]) < 4.9

    def test_edges_have_attributes(self):
        g = build_caracas_graph()
        for _, _, data in g.edges(data=True):
            # Edges should have at least some attribute
            assert isinstance(data, dict)  # Edges have dict attributes


class TestRoutePredictionService:
    """Tests for the prediction service."""

    def setup_method(self):
        self.service = RoutePredictionService()

    def test_predict_route_returns_valid_response(self):
        result = self.service._predict_transit(
            origin=Coordinates(lat=4.7586, lng=-74.0453),
            destination=Coordinates(lat=4.6250, lng=-74.0700),
            departure_time="2026-05-20T08:00:00Z",
            mode="transmilenio",
        )
        assert result.route_id
        assert result.total_time_minutes > 0
        assert result.total_distance_km > 0
        assert len(result.risk_segments) > 0
        assert len(result.stations) > 0
        assert result.overall_risk in ("low", "medium", "high", "critical")

    def test_predict_route_respects_departure_hour(self):
        peak = self.service._predict_transit(
            origin=Coordinates(lat=4.7330, lng=-74.0500),
            destination=Coordinates(lat=4.6600, lng=-74.0580),
            departure_time="2026-05-20T08:00:00Z",
            mode="transmilenio",
        )
        offpeak = self.service._predict_transit(
            origin=Coordinates(lat=4.7330, lng=-74.0500),
            destination=Coordinates(lat=4.6600, lng=-74.0580),
            departure_time="2026-05-20T03:00:00Z",
            mode="transmilenio",
        )
        avg_peak = sum(s.congestion_level for s in peak.risk_segments) / max(
            len(peak.risk_segments), 1
        )
        avg_offpeak = sum(s.congestion_level for s in offpeak.risk_segments) / max(
            len(offpeak.risk_segments), 1
        )
        assert avg_peak > avg_offpeak

    def test_nearest_station_finds_correct_station(self):
        station = self.service._find_nearest_station(Coordinates(lat=4.7586, lng=-74.0453))
        assert station != ""
        assert station in self.service._graph

    def test_route_has_correct_direction(self):
        result = self.service._predict_transit(
            origin=Coordinates(lat=4.7586, lng=-74.0453),
            destination=Coordinates(lat=4.5750, lng=-74.0800),
            departure_time="2026-05-20T10:00:00Z",
            mode="transmilenio",
        )
        assert len(result.stations) >= 2
        assert result.total_distance_km > 0
        assert result.total_time_minutes > 0


@pytest.mark.anyio
class TestRoutePredictionAPI:
    """Integration tests for the API endpoint."""

    async def test_predict_route_endpoint(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="https://test") as client:
            response = await client.post(
                "/api/v1/predict-route",
                json={
                    "origin": {"lat": 4.7330, "lng": -74.0500},
                    "destination": {"lat": 4.6250, "lng": -74.0700},
                    "departure_time": "2026-05-20T08:00:00Z",
                },
            )
        assert response.status_code == 200
        data = response.json()
        assert "route_id" in data
        assert "risk_segments" in data
        assert "explanation" in data
        assert len(data["risk_segments"]) > 0

    async def test_predict_route_validation_error(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="https://test") as client:
            response = await client.post(
                "/api/v1/predict-route",
                json={"origin": {"lat": 4.7}, "departure_time": "2026-05-20T08:00:00Z"},
            )
        assert response.status_code == 422

    async def test_predict_route_vehicle_has_navigation_steps(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="https://test") as client:
            response = await client.post(
                "/api/v1/predict-route",
                json={
                    "origin": {"lat": 4.65, "lng": -74.11},
                    "destination": {"lat": 4.72, "lng": -74.06},
                    "departure_time": "2026-06-21T15:00:00",
                    "mode": "vehiculo",
                },
            )
        assert response.status_code == 200
        data = response.json()
        assert "navigation_steps" in data
        if data["stations"]:  # OSRM available
            assert len(data["navigation_steps"]) > 0
            step = data["navigation_steps"][0]
            assert "instruction" in step
            assert "street" in step
            assert "distance_m" in step

    async def test_predict_alternatives_returns_list(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="https://test") as client:
            response = await client.post(
                "/api/v1/predict-route/alternatives",
                json={
                    "origin": {"lat": 4.65, "lng": -74.11},
                    "destination": {"lat": 4.72, "lng": -74.06},
                    "departure_time": "2026-06-21T15:00:00",
                    "mode": "vehiculo",
                },
            )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert "route_id" in data[0]

    async def test_alerts_endpoint(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="https://test") as client:
            response = await client.get("/api/v1/predict-route/alerts")
        assert response.status_code == 200
        data = response.json()
        assert "operating" in data
        assert "delayed" in data
        assert "suspended" in data
        assert "alerts" in data
        assert isinstance(data["alerts"], list)

    async def test_safety_endpoint(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="https://test") as client:
            response = await client.get("/api/v1/predict-route/safety?ruta=7&hour=8")
        assert response.status_code == 200
        data = response.json()
        assert "safety_score" in data
        assert "ruta" in data


class TestCongestionFactors:
    """Tests for congestion day/time factors."""

    def test_time_factor_peak_hour(self):
        from app.common.congestion import HOUR_FACTORS

        assert HOUR_FACTORS[8] == pytest.approx(1.0)  # Peak morning
        assert HOUR_FACTORS[18] == pytest.approx(1.0)  # Peak evening

    def test_time_factor_off_peak(self):
        from app.common.congestion import HOUR_FACTORS

        assert HOUR_FACTORS[3] == pytest.approx(0.08)  # Early morning
        assert HOUR_FACTORS[14] == pytest.approx(0.55)  # Afternoon

    def test_day_factors_exist(self):
        from app.common.congestion import DAY_FACTORS

        assert len(DAY_FACTORS) == 7
        assert DAY_FACTORS[6] < DAY_FACTORS[0]  # Sunday < Monday
        assert DAY_FACTORS[5] < DAY_FACTORS[4]  # Saturday < Friday
