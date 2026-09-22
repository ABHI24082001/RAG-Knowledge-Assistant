#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# One worker is intentional: embedded/local Qdrant cannot safely be opened by
# multiple independent Gunicorn worker processes.
exec .venv/bin/gunicorn app.main:app \
  --worker-class uvicorn.workers.UvicornWorker \
  --workers 1 \
  --bind "0.0.0.0:${PORT:-8000}" \
  --timeout 180 \
  --access-logfile - \
  --error-logfile - \
  --log-level "${LOG_LEVEL:-info}"
