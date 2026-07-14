# MoviCol AI Service

Servicio de IA para predicción de rutas y congestión —
FastAPI + NetworkX + PyTorch.

## Stack

- FastAPI + Python 3.9+
- NetworkX (grafos de movilidad)
- PyTorch (GNN + ST-GAT)
- OSRM (routing vehicular externo)
- httpx (HTTP async)

## Setup

```bash
pip install -e .      # o pip install -r requirements.txt
uvicorn app.main:app --port 8000 --reload
pytest tests/ -v      # 17 tests
```

## Endpoints

| Método | Endpoint                              | Descripción                      |
| ------ | ------------------------------------- | -------------------------------- |
| POST   | `/api/v1/predict-route`               | Predicción de ruta (TM/SITP/Veh) |
| POST   | `/api/v1/predict-route/alternatives`  | Alternativas vehiculares (OSRM)  |
| GET    | `/api/v1/predict-route/alerts`        | Alertas de TransMilenio          |
| GET    | `/api/v1/predict-route/safety`        | Safety score por ruta SITP       |
| GET    | `/graph/stations`                     | Estaciones del grafo             |
| GET    | `/graph/stats`                        | Nodos y edges                    |
| GET    | `/graph/heatmap`                      | Congestión por estación          |
| POST   | `/predictions`                        | Predicción GNN individual        |
| POST   | `/predictions/batch`                  | Predicciones batch               |
| GET    | `/demand/predict`                     | Predicción demanda ST-GAT        |
| POST   | `/agent/chat`                         | Agente conversacional            |
| GET    | `/health`                             | Health check                     |

## Modelos

| Modelo | Archivo | Descripción |
| ------ | ------- | ----------- |
| GNN | `gat_best.pt` | Predicción congestión por nodo |
| ST-GAT | `st_gat_model.pt` | Predicción demanda espacio-temporal |
| Grafo | `grafo_movilidad_bogota_*.graphml` | Grafo SITP (7290 nodos) |
| TM | `tm_stations_all.json` | 153 estaciones, 125 rutas TM |

## Congestión

Fórmula:

```text
congestion = (GNN_base * 0.6 + ST_GAT * 0.4) * hour * day
```

- `hour_factor`: 0.2 (madrugada) → 1.0 (hora pico)
- `day_factor`: L-J=1.0, Vie=1.05, Sáb=0.6, Dom=0.4

## Routing Vehicular

- Usa OSRM público (`router.project-osrm.org`) via HTTP
- Retorna geometría real, pasos de navegación, calles
- `alternatives=true` para rutas alternativas
- Fallback euclidiano si OSRM no disponible

## Env

```env
GRAPH_PATH=data/grafo_movilidad_bogota_enriched.graphml
MODEL_PATH=models/gat_best.pt
```
