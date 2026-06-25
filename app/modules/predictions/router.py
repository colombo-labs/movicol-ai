"""Predictions router - congestion forecasting endpoints."""

import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.modules.predictions.schemas import (
    BatchPredictionRequest,
    PredictionRequest,
    PredictionResponse,
)
from app.modules.predictions.service import PredictionService

router = APIRouter()
service = PredictionService()


@router.post("", response_model=PredictionResponse)
async def predict_station(request: PredictionRequest):
    """Predict congestion for a specific station."""
    if not service.is_loaded:
        raise HTTPException(503, "GNN model not loaded. Place gat_best.pt in models/")
    return service.predict(
        station_id=request.station_id,
        hour=request.hour,
        horizon_minutes=request.horizon_minutes,
        frecuencia_ruta=request.frecuencia_ruta,
        demanda_actual=request.demanda_actual,
    )


@router.post("/batch", response_model=list[PredictionResponse])
async def predict_all_stations(request: BatchPredictionRequest):
    """Predict congestion for all stations."""
    if not service.is_loaded:
        raise HTTPException(503, "GNN model not loaded.")
    return service.predict_all(
        hour=request.hour,
        horizon_minutes=request.horizon_minutes,
    )


def _find_paradero(features: list, id_str: str) -> dict | None:
    for feat in features:
        props = feat.get("properties", {})
        if (
            str(feat.get("id")) == id_str
            or str(props.get("objectid")) == id_str
            or str(props.get("cenefa")) == id_str
        ):
            return feat
    return None


def _get_rutas_que_pasan(features: list, id_str: str) -> list[str]:
    rutas_que_pasan = []
    for feat in features:
        props = feat.get("properties", {})
        if str(props.get("cenefa")) == id_str or str(props.get("objectid")) == id_str:
            ruta = props.get("ruta")
            if ruta and ruta not in rutas_que_pasan:
                rutas_que_pasan.append(ruta)
    return rutas_que_pasan


def _build_paradero_response(
    paradero_id: str, paradero: dict, rutas_que_pasan: list, frecuencias: dict
) -> dict:
    demanda_actual = paradero.get("properties", {}).get("demanda_score", 0)
    hora_actual = datetime.now().hour

    rutas_info = []
    for r in rutas_que_pasan:
        frecuencia_base = frecuencias.get(r, {}).get("frecuencia_base_min", 15)
        pred = service.predict(
            station_id=f"SITP_{paradero_id}",
            hour=hora_actual,
            horizon_minutes=30,
            frecuencia_ruta=frecuencia_base,
            demanda_actual=demanda_actual,
        )
        rutas_info.append(
            {
                "ruta": r,
                "frecuencia_estimada_min": frecuencia_base,
                "tiempo_espera_predicho": pred.tiempo_espera_estimado
                or f"{frecuencia_base} min",
                "congestion_esperada": pred.risk_label,
            }
        )

    nivel_demanda = "Baja"
    if demanda_actual > 70:
        nivel_demanda = "Alta"
    elif demanda_actual > 30:
        nivel_demanda = "Media"

    return {
        "id": paradero_id,
        "nombre": paradero.get("properties", {}).get("nombre", ""),
        "demanda_actual_score": demanda_actual,
        "nivel_demanda": nivel_demanda,
        "rutas": rutas_info,
    }


@router.get("/sitp/paradero/{id}/info", responses={404: {"description": "Not found"}})
def get_paradero_info(id: str):
    """Get consolidated paradero info including predicted wait time."""
    data_dir = (
        Path(__file__).parent.parent.parent.parent.parent / "movicol-data" / "exports"
    )
    paraderos_file = data_dir / "sitp_paraderos.geojson"
    rutas_file = data_dir / "sitp_rutas_paraderos.geojson"
    frecuencias_file = data_dir / "sitp_rutas_frecuencias.json"

    if not paraderos_file.exists() or not rutas_file.exists():
        raise HTTPException(404, "SITP data files not found in movicol-data")

    with open(paraderos_file, "r", encoding="utf-8") as f:
        paraderos_data = json.load(f)

    paradero = _find_paradero(paraderos_data.get("features", []), id)
    if not paradero:
        raise HTTPException(404, "Paradero no encontrado")

    with open(rutas_file, "r", encoding="utf-8") as f:
        rutas_data = json.load(f)

    frecuencias = {}
    if frecuencias_file.exists():
        with open(frecuencias_file, "r", encoding="utf-8") as f:
            frecuencias = json.load(f)

    rutas_que_pasan = _get_rutas_que_pasan(rutas_data.get("features", []), id)

    return _build_paradero_response(id, paradero, rutas_que_pasan, frecuencias)
