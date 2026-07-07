"""Tests for RoutePredictionService."""

import pytest

from app.modules.route_prediction.schemas import Coordinates
from app.modules.route_prediction.service import RoutePredictionService


@pytest.fixture
def service():
    return RoutePredictionService()


class TestRoutePredictionService:
    """Test route prediction functionality."""

    def test_service_initializes(self, service):
        assert service is not None

    def test_has_tm_graph(self, service):
        assert service._tm_graph is not None
        assert service._tm_graph.number_of_nodes() > 0

    def test_find_nearest_station(self, service):
        """Should find nearest station to coords."""
        # Coords near Portal Norte
        nearest = service._find_nearest_in(
            Coordinates(lat=4.759, lon=-74.045),
            service._tm_graph,
        )
        assert nearest is not None

    def test_find_nearest_station_center(self, service):
        """Should find station near centro."""
        nearest = service._find_nearest_in(
            Coordinates(lat=4.598, lon=-74.076),
            service._tm_graph,
        )
        assert nearest is not None

    def test_count_transfers(self, service):
        """Transfers count with empty segments."""
        transfers = service._count_transfers([])
        assert transfers == 0

    def test_estimate_wait_minutes_tm(self, service):
        wait = service._estimate_wait_minutes("transmilenio")
        assert 2 <= wait <= 10

    def test_estimate_wait_minutes_sitp(self, service):
        wait = service._estimate_wait_minutes("sitp")
        assert 5 <= wait <= 20

    @pytest.mark.asyncio
    async def test_predict_route_tm(self, service):
        """Should predict a TransMilenio route."""
        origin = Coordinates(lat=4.695, lon=-74.031)  # Usaquén area
        dest = Coordinates(lat=4.598, lon=-74.076)  # Centro area

        result = await service.predict_route(
            origin=origin,
            destination=dest,
            departure_time="2026-07-06T08:00:00",
            mode="transmilenio",
        )

        assert result is not None
        assert result.route_id != ""
        assert result.total_time_minutes > 0
        assert result.total_distance_km > 0
        assert result.mode == "transmilenio"
        assert result.cost == "$3.550"

    @pytest.mark.asyncio
    async def test_predict_route_has_stations(self, service):
        """TM route should have station list."""
        origin = Coordinates(lat=4.759, lon=-74.045)
        dest = Coordinates(lat=4.598, lon=-74.076)

        result = await service.predict_route(
            origin=origin,
            destination=dest,
            departure_time="2026-07-06T08:00:00",
            mode="transmilenio",
        )

        assert len(result.stations) >= 2

    @pytest.mark.asyncio
    async def test_predict_route_has_risk(self, service):
        """Route should have risk assessment."""
        origin = Coordinates(lat=4.695, lon=-74.031)
        dest = Coordinates(lat=4.598, lon=-74.076)

        result = await service.predict_route(
            origin=origin,
            destination=dest,
            departure_time="2026-07-06T08:00:00",
            mode="transmilenio",
        )

        assert result.overall_risk in ("low", "medium", "moderate", "high", "critical")
        assert 0 <= result.safety_score <= 100

    @pytest.mark.asyncio
    async def test_predict_route_explanation_has_schedule_warning(self, service):
        """Late night route should warn about schedule."""
        origin = Coordinates(lat=4.695, lon=-74.031)
        dest = Coordinates(lat=4.598, lon=-74.076)

        result = await service.predict_route(
            origin=origin,
            destination=dest,
            departure_time="2026-07-06T00:30:00",  # 12:30am — out of service
            mode="transmilenio",
        )

        assert "no opera" in result.explanation.lower() or "AVISO" in result.explanation
