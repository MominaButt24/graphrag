#!/bin/bash
set -e
cd /workspace

echo ""
echo "=================================================="
echo " GraphRAG Pipeline is starting via PM2..."
echo " FastAPI docs:  http://localhost:8000/docs"
echo " Gradio UI:     http://localhost:7860"
echo " Chainlit UI:   http://localhost:8001"
echo "=================================================="
echo ""

pm2 start ecosystem.config.js

while true; do
  sleep 5
  online_count=$(pm2 jlist | grep -o '"status":"online"' | wc -l)
  if [ "$online_count" -eq 0 ]; then
    echo "All PM2 processes stopped. Shutting down container..."
    break
  fi
done