# =============================================================================
# Dockerfile multi-stage para PriceIQ backend (FastAPI + recommender ML)
#
# Stage 1 (builder): instala dependencias en venv → cachea capa pesada
# Stage 2 (runtime): copia solo lo necesario → imagen final más pequeña
#
# Build:
#   docker build -t priceiq-backend .
#
# Run local con datos mount-eados desde el host:
#   docker run -p 8000:8000 \
#       -e CORS_ORIGINS="http://localhost:3000" \
#       -e PRICEIQ_DATA_DIR=/app/data \
#       -e PRICEIQ_MODELS_DIR=/app/models \
#       -v $(pwd)/data:/app/data:ro \
#       -v $(pwd)/models:/app/models:ro \
#       priceiq-backend
#
# En producción (Azure Container Apps), data y models se mountan desde Azure Files
# o se descargan al arrancar desde Blob Storage.
# =============================================================================

ARG PYTHON_VERSION=3.11-slim

# -----------------------------------------------------------------------------
# Stage 1 — Builder
# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Dependencias del sistema para LightGBM, XGBoost, CatBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# venv aislado en /opt/venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Instalar deps Python primero (capa cacheada si requirements no cambia)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install "fastapi>=0.110" "uvicorn[standard]>=0.29"

# -----------------------------------------------------------------------------
# Stage 2 — Runtime
# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000

# libgomp1 es runtime dep de los GBM
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Usuario no-root para mejor seguridad
RUN useradd --create-home --uid 1001 appuser

# Copiamos el venv del builder
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Código de la app
COPY --chown=appuser:appuser recommender.py /app/
COPY --chown=appuser:appuser backend /app/backend

USER appuser

# Healthcheck que Container Apps puede usar
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fs http://localhost:${PORT}/api/health || exit 1

EXPOSE 8000

# 2 workers para aprovechar Container Apps; ajustable vía env
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT} --workers 2"]
