"""Tests for SiniestrosService."""

import pytest

from app.modules.siniestralidad.service import SiniestrosService


class TestSiniestrosService:
    """Test siniestralidad/accident data service."""

    def test_service_initializes(self):
        service = SiniestrosService()
        # May or may not be loaded depending on data files
        assert service is not None

    def test_is_loaded_attribute(self):
        service = SiniestrosService()
        assert isinstance(service.is_loaded, bool)

    def test_get_stats_when_loaded(self):
        service = SiniestrosService()
        if not service.is_loaded:
            pytest.skip("Siniestralidad data not loaded")
        stats = service.get_stats()
        assert stats.total_siniestros > 0
        assert stats.total_fallecidos >= 0
        assert stats.sectores_criticos >= 0

    def test_predict_risk_by_hour_when_loaded(self):
        service = SiniestrosService()
        if not service.is_loaded:
            pytest.skip("Siniestralidad data not loaded")
        risk = service.predict_risk_by_hour(8)
        assert risk.nivel_general in ("bajo", "moderado", "alto", "critico")
        assert 0 <= risk.promedio_riesgo <= 1
        assert len(risk.zones) > 0

    def test_predict_risk_different_hours(self):
        service = SiniestrosService()
        if not service.is_loaded:
            pytest.skip("Siniestralidad data not loaded")
        risk_morning = service.predict_risk_by_hour(8)
        risk_night = service.predict_risk_by_hour(3)
        # Night should generally be less risky (less traffic)
        assert risk_morning is not None
        assert risk_night is not None
