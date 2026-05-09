# syntax=docker/dockerfile:1
# Multi-stage build. Build context must be the repo root.
# Build:  docker build -t energyguard/fairness-audit:dev .
# Run:    docker run -p 9006:9006 energyguard/fairness-audit:dev

# ── Stage 1: install Python dependencies ──────────────────────────────────────
FROM python:3.11-slim AS build

WORKDIR /build

# Install uv for fast dep resolution
RUN pip install --no-cache-dir uv

# Copy only what pip needs to resolve deps (layer-cache friendly)
COPY pyproject.toml README.md ./
COPY fairness-backend/ ./fairness-backend/

# Install runtime deps only (no dev extras)
RUN uv pip install --system .

# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from build stage
COPY --from=build /usr/local/lib/python3.11/site-packages \
                  /usr/local/lib/python3.11/site-packages
COPY --from=build /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Job output and temp upload directories
RUN mkdir -p /app/runs /app/tmp_uploads

VOLUME /app/runs

ENV PYTHONUNBUFFERED=1
EXPOSE 9006

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9006"]
