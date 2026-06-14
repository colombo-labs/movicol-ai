FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install core dependencies only (no torch-geometric for lighter image)
RUN pip install --no-cache-dir \
    fastapi>=0.111.0 \
    uvicorn[standard]>=0.29.0 \
    pydantic>=2.7.0 \
    pydantic-settings>=2.2.0 \
    networkx>=3.2 \
    torch>=2.2.0 \
    langchain>=0.2.0 \
    langchain-openai>=0.1.0 \
    python-dotenv>=1.0.0 \
    psycopg2-binary>=2.9.0

# Install torch-geometric separately (needs special index)
RUN pip install --no-cache-dir torch-geometric -f https://data.pyg.org/whl/torch-2.2.0+cpu.html || true

COPY app/ ./app/
COPY models/ ./models/
COPY scripts/ ./scripts/

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
