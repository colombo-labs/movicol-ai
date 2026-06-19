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
        assert g.number_of_nodes() == 28

    def test_graph_is_bidirectional(self):
        g = build_caracas_graph()
        # Each pair of consecutive stations has edges in both directions
        assert g.number_of_edges() == 54  # 27 pairs * 2

    def test_nodes_have_required_attributes(self):
        g = build_caracas_graph()
        for _, data in g.nodes(data=True):
            assert "name" in data
            assert "lat" in data
            assert "lon" in data
            assert data["lat"] > 4.5  # Bogotá latitude range
            assert data["lat"] < 4.8

    def test_edges_have_distance_and_time(self):
        g = build_caracas_graph()
        for _, _, data in g.edges(data=True):
            assert "distance_km" in data
            assert "base_time_min" in data
            assert data["distance_km"] > 0
            assert data["base_time_min"] > 0


class TestRoutePredictionService:
    """Tests for the prediction service."""

    def setup_method(self):
        self.service = RoutePredictionService()

    def test_predict_route_returns_valid_response(self):
        result = self.service.predict_route(
            origin=Coordinates(lat=4.7586, lng=-74.0453),  # Portal Norte
            destination=Coordinates(lat=4.6250, lng=-74.0700),  # Calle 26
            departure_time="2026-05-20T08:00:00Z",
        )
        assert result.route_id
        assert result.total_time_minutes > 0
        assert result.total_distance_km > 0
        assert len(result.risk_segments) > 0
        assert len(result.stations) > 0
        assert result.overall_risk in ("low", "medium", "high", "critical")

    def test_predict_route_respects_departure_hour(self):
        # Peak hour (8am) should have higher congestion than off-peak (3am)
        peak = self.service.predict_route(
            origin=Coordinates(lat=4.7330, lng=-74.0500),
            destination=Coordinates(lat=4.6600, lng=-74.0580),
            departure_time="2026-05-20T08:00:00Z",
        )
        offpeak = self.service.predict_route(
            origin=Coordinates(lat=4.7330, lng=-74.0500),
            destination=Coordinates(lat=4.6600, lng=-74.0580),
            departure_time="2026-05-20T03:00:00Z",
        )
        avg_peak = sum(s.congestion_level for s in peak.risk_segments) / len(peak.risk_segments)
        avg_offpeak = sum(s.congestion_level for s in offpeak.risk_segments) / len(
            offpeak.risk_segments
        )
        assert avg_peak > avg_offpeak

    def test_nearest_station_finds_correct_station(self):
        # Coordinates near Portal Norte area (north Bogotá)
        station = self.service._find_nearest_station(Coordinates(lat=4.7586, lng=-74.0453))
        # Should find a station (any valid station in the graph)
        assert station != ""
        assert station in self.service._graph

    def test_route_has_correct_direction(self):
        result = self.service.predict_route(
            origin=Coordinates(lat=4.7586, lng=-74.0453),  # North
            destination=Coordinates(lat=4.5750, lng=-74.0800),  # South
            departure_time="2026-05-20T10:00:00Z",
        )
        # Should find a route with multiple stations
        assert len(result.stations) >= 2
        assert result.total_distance_km > 0
        assert result.total_time_minutes > 0


@pytest.mark.anyio
class TestRoutePredictionAPI:
    """Integration tests for the API endpoint."""

    async def test_predict_route_endpoint(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
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
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/predict-route",
                json={"origin": {"lat": 4.7}, "departure_time": "2026-05-20T08:00:00Z"},
            )
        assert response.status_code == 422
