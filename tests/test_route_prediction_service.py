"""Tests for RoutePredictionService."""

from unittest.mock import AsyncMock, Mock

import networkx as nx
import pytest

from app.modules.route_prediction import graph_data
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

    @pytest.mark.asyncio
    async def test_transit_geometry_uses_complete_path(self):
        service = RoutePredictionService.__new__(RoutePredictionService)
        graph = nx.path_graph(50)
        for node in graph.nodes:
            graph.nodes[node].update(
                name=f"Station {node}",
                lat=4.6 + node * 0.001,
                lon=-74.1,
                troncal="Caracas",
            )
        for first, second in graph.edges:
            graph.edges[first, second].update(distance_km=0.1, troncal="Caracas")

        service._tm_graph = graph
        service._sitp_routes = {}
        service._find_nearest_in = Mock(side_effect=[0, 49])
        service._build_transit_segments_async = AsyncMock(return_value=([], 4.9, 12.0))
        service._derive_route_code = Mock(return_value="A60")

        result = await service._predict_transit(
            Coordinates(lat=4.6, lon=-74.1),
            Coordinates(lat=4.649, lon=-74.1),
            "2026-07-06T08:00:00",
            "transmilenio",
        )

        geometry_path = service._build_transit_segments_async.await_args.args[1]
        route_code_path = service._derive_route_code.call_args.args[1]
        assert geometry_path == list(range(50))
        assert route_code_path == list(range(50))
        assert len(result.stations) < len(geometry_path)
        assert result.stations[0] == "Station 0"
        assert result.stations[-1] == "Station 49"

    def test_match_tm_route_respects_direction(self, monkeypatch):
        forward = {
            "codigo": "F23",
            "coords": [[4.6, -74.1], [4.61, -74.09], [4.62, -74.08]],
        }
        reverse = {
            "codigo": "J23",
            "coords": list(reversed(forward["coords"])),
        }
        monkeypatch.setattr(graph_data, "TM_RUTAS", [forward, reverse])

        assert RoutePredictionService._match_tm_ruta(forward["coords"]) == "F23"
        assert RoutePredictionService._match_tm_ruta(reverse["coords"]) == "J23"
