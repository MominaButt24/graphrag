#!/usr/bin/env bash
set -Eeuo pipefail

cd /opt/graphrag

# Load .env into the environment for both services.
if [[ ! -f .env ]]; then
    echo "[entrypoint] ERROR: /opt/graphrag/.env not found"
    exit 1
fi

eval "$(uv run --active python -c '
from dotenv import dotenv_values
import shlex

for key, value in dotenv_values("/opt/graphrag/.env").items():
    if value is not None:
        print(f"export {key}={shlex.quote(value)}")
')"

# Gradio calls FastAPI through Ray Serve.
export API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:8002/api/v1}"

# Fail early if required configuration is missing.
if [[ -z "${MILVUS_URI:-}" ]]; then
    echo "[entrypoint] ERROR: MILVUS_URI is missing from .env"
    exit 1
fi

if [[ -z "${MILVUS_TOKEN:-}" ]]; then
    echo "[entrypoint] ERROR: MILVUS_TOKEN is missing from .env"
    exit 1
fi

API_PID=""
GRADIO_PID=""

cleanup() {
    echo "[entrypoint] Stopping services..."
    [[ -z "$GRADIO_PID" ]] || kill "$GRADIO_PID" 2>/dev/null || true
    [[ -z "$API_PID" ]] || kill "$API_PID" 2>/dev/null || true
    [[ -z "$GRADIO_PID" ]] || wait "$GRADIO_PID" 2>/dev/null || true
    [[ -z "$API_PID" ]] || wait "$API_PID" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

echo "[entrypoint] Milvus configuration loaded."
echo "[entrypoint] Starting Ray Serve / FastAPI on port 8002..."
uv run --active serve run serve_config.yaml &
API_PID=$!

echo "[entrypoint] Starting Gradio on port 7860..."
uv run --active python gradio_app.py &
GRADIO_PID=$!

echo "[entrypoint] Ray Serve PID: $API_PID"
echo "[entrypoint] Gradio PID: $GRADIO_PID"

set +e
wait -n "$API_PID" "$GRADIO_PID"
STATUS=$?
set -e

echo "[entrypoint] A service exited with status $STATUS"
exit "$STATUS"

# #!/bin/bash
# cd /opt/graphrag

# uv run --active serve run serve_config.yaml &
# uv run --active python gradio_app.py &

# wait -n