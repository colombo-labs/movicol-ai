![CI](https://github.com/colombo-labs/movicol-ai/actions/workflows/ci.yml/badge.svg)
[![Quality Gate](https://sonarcloud.io/api/project_badges/measure?project=colombo-labs_movicol-ai&metric=alert_status)](https://sonarcloud.io/dashboard?id=colombo-labs_movicol-ai)

# 🧠 MoviCol AI

Microservicio de IA para predicción de congestión y agente conversacional.

## Stack

- **Python** 3.11+
- **FastAPI** 0.111 + **Uvicorn** 0.29
- **PyTorch Geometric** 2.5 (GNN)
- **NetworkX** 3.2 (grafos)
- **LangChain** 0.2 + **OpenAI** (agente conversacional)
- **SQLAlchemy** 2.0 + **GeoAlchemy2** 0.14 (PostGIS)
- **Pydantic** 2.7 (validación)
- **Ruff** 0.4 (linter)
- **Pytest** 8 (testing)

## Arquitectura

Modular SOLID — cada módulo con router, service y schemas (patrón NestJS en Python).

```
app/
├── main.py                      # App factory + routers
├── config/settings.py           # Pydantic settings (env vars)
├── common/
│   ├── exceptions.py            # Custom HTTP exceptions
│   └── middleware/
└── modules/
    ├── health/                  # GET /health
    ├── predictions/             # POST /predictions
    │   ├── router.py / service.py / schemas.py
    ├── graph/                   # GET /graph/*
    │   ├── router.py / service.py / schemas.py
    └── agent/                   # POST /agent/chat
        ├── router.py / service.py / schemas.py
```

## Quick Start

```bash
make install    # pip install -e ".[dev]"
make dev        # uvicorn en http://localhost:8000
```

## Scripts

| Comando | Descripción |
|---------|-------------|
| `make dev` | Dev server (hot reload) |
| `make train` | Entrenar modelo GNN |
| `make test` | Pytest |
| `make lint` | Ruff check + format |

## API Docs

Swagger UI: `http://localhost:8000/docs`

## Modelo

- **Tipo:** GAT (Graph Attention Network)
- **Framework:** PyTorch Geometric
- **Métricas:** MSE 0.18 | RMSE 0.42
- **Archivos:** `models/gat_best.pt` + `models/graph.graphml`

## Docker

```bash
docker compose -f docker-compose.dev.yml up -d
```

## Requisitos

- Python 3.11+
- PostGIS (para datos del grafo)
- OpenAI API key (para agente)
