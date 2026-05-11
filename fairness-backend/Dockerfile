# syntax=docker/dockerfile:1
# Fairness-backend service Dockerfile.
# Build context must be the REPO ROOT (not this directory):
#   docker build -f fairness-backend/Dockerfile .
#
# For most purposes, use the root Dockerfile and docker-compose.yml instead.

FROM python:3.11-slim AS build
WORKDIR /build
RUN pip install --no-cache-dir uv
COPY pyproject.toml README.md ./
COPY fairness-backend/ ./fairness-backend/
RUN uv pip install --system .

FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=build /usr/local/lib/python3.11/site-packages \
                  /usr/local/lib/python3.11/site-packages
COPY --from=build /usr/local/bin/uvicorn /usr/local/bin/uvicorn
RUN groupadd --system --gid 1001 fairness \
 && useradd  --system --uid 1001 --gid fairness --create-home fairness \
 && mkdir -p /app/runs /app/tmp_uploads \
 && chown -R fairness:fairness /app
VOLUME /app/runs
ENV PYTHONUNBUFFERED=1
EXPOSE 9006
USER fairness
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9006"]
