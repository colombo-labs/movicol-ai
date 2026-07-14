# Tests — MoviCol AI Service

## Estructura

```text
tests/
├── test_agent_full.py          → Intents, tools, schemas (34 tests)
├── test_agent_service.py       → Chat service flow completo (19 tests)
├── test_route_prediction_service.py → Route prediction TM (11 tests)
├── test_siniestralidad.py      → Siniestralidad/riesgo vial (5 tests)
├── test_health.py              → Health endpoint (1 test)
└── performance/
    ├── locustfile.py           → Load & stress test definitions
    ├── run_tests.sh            → Runner script
    └── results/                → CSV reports (gitignored)
```

## Correr

```bash
# Unitarios + integración
.venv/bin/python -m pytest tests/ -v

# Solo un módulo
.venv/bin/python -m pytest tests/test_agent_full.py -v

# Con coverage
.venv/bin/python -m pytest tests/ --cov=app --cov-report=html

# Load test (requiere servicio corriendo en :8000)
./tests/performance/run_tests.sh load

# Stress test
./tests/performance/run_tests.sh stress

# Load + Stress
./tests/performance/run_tests.sh both

# Locust con UI (browser en :8089)
.venv/bin/locust -f tests/performance/locustfile.py --host http://localhost:8000
```

## Convenciones

- Prefijo `test_` en archivos y funciones
- Clases `Test*` para agrupar tests relacionados
- `@pytest.fixture` para setup compartido
- `@pytest.mark.asyncio` para tests async
- `pytest.skip()` cuando faltan datos opcionales
- Tests no dependen de LLM (todo rule-based)
