#!/usr/bin/env bash
# ============================================================================
# Memory-constrained swap: stop one vLLM backend and start another.
# Useful when you need a large model that won't fit alongside the others.
#
# Usage: scripts/swap.sh <stop-alias> <start-alias>
#   e.g. scripts/swap.sh qwen-main gemma-fast
# ============================================================================
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

[ $# -eq 2 ] || { echo "Usage: $0 <stop-alias> <start-alias>"; exit 2; }
STOP="$1"; START="$2"

declare -A SVC=(
  [qwen-main]=vllm-qwen-main
  [qwen-vision]=vllm-qwen-vision
  [gemma-fast]=vllm-gemma-fast
)
declare -A PORT=(
  [qwen-main]=8001 [qwen-vision]=8002 [gemma-fast]=8003
)

[ -n "${SVC[$STOP]:-}"  ] || { echo "Unknown alias: $STOP"; exit 2; }
[ -n "${SVC[$START]:-}" ] || { echo "Unknown alias: $START"; exit 2; }

echo "→ Stopping ${SVC[$STOP]}"
docker compose stop "${SVC[$STOP]}"

echo "→ Starting ${SVC[$START]}"
docker compose up -d --no-deps "${SVC[$START]}"

echo "→ Waiting for /health on port ${PORT[$START]}"
for i in $(seq 1 60); do
  if curl -fsS --max-time 3 "http://127.0.0.1:${PORT[$START]}/health" >/dev/null 2>&1; then
    echo "✓ $START healthy"
    exit 0
  fi
  sleep 5
done
echo "✗ $START did not become healthy in 5 min" >&2
exit 1
