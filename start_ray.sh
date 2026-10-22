#!/usr/bin/env bash
set -euo pipefail
cd /workspace

uv run --active python - <<'PY'
import os
import yaml
from dotenv import dotenv_values

env = {
    key: value
    for key, value in dotenv_values("/workspace/.env").items()
    if value is not None
}

# Keep only environment variables needed by the application.
with open("/tmp/graphrag-runtime-env.yaml", "w") as f:
    yaml.safe_dump({"env_vars": env}, f, sort_keys=False)

print("Generated Ray runtime environment from /workspace/.env")
print("LLM_PROVIDER:", env.get("LLM_PROVIDER"))
print("LLM_MODEL:", env.get("LLM_MODEL"))
print("LLM_BASE_URL:", env.get("LLM_BASE_URL"))
PY

uv run --active serve run \
  --runtime-env /tmp/graphrag-runtime-env.yaml \
  serve_config.yaml
