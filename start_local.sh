#!/usr/bin/env bash
set -euo pipefail

echo "Starting EnergyGuard Fairness Audit local stack..."
docker compose up --build -d

echo ""
echo "Services:"
echo "  Fairness Audit API : http://localhost:8080"
echo "  API docs (Swagger) : http://localhost:8080/docs"
echo "  MLflow UI          : http://localhost:5001"
echo ""
echo "Stop with: docker compose down"
