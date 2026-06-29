#!/usr/bin/env bash
# ============================================================================
# Rolling update for one or all vLLM backends.
# Pulls the image referenced in config/image-versions.env, then recreates the
# specified service(s), waiting for /health to return 200 before moving on.
#
# Usage:
#   scripts/update-rolling.sh                    # all three backends
#   scripts/update-rolling.sh qwen-main          # just one
# ============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

[ -f .env ] && set -a && . .env && set +a
[ -f config/image-versions.env ] && set -a && . config/image-versions.env && set +a

declare -A PORTS=(
  [qwen-main]=8001
  [qwen-vision]=8002
  [gemma-fast]=8003
)
declare -A SERVICES=(
  [qwen-main]=vllm-qwen-main
  [qwen-vision]=vllm-qwen-vision
  [gemma-fast]=vllm-gemma-fast
)

TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(qwen-main qwen-vision gemma-fast)

echo "→ Pulling image: $VLLM_IMAGE"
docker pull "$VLLM_IMAGE"

for alias in "${TARGETS[@]}"; do
  svc="${SERVICES[$alias]:-}"
  port="${PORTS[$alias]:-}"
  if [ -z "$svc" ]; then
    echo "✗ Unknown backend alias: $alias" >&2
    continue
  fi

  echo ""
  echo "── Updating $svc ($alias :$port) ──"
  docker compose up -d --no-deps --force-recreate "$svc"

  echo "  Waiting for /health ..."
  for i in $(seq 1 60); do
    if curl -fsS --max-time 3 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "  ✓ $svc healthy after ${i}*5s"
      break
    fi
    sleep 5
    if [ "$i" -eq 60 ]; then
      echo "  ✗ $svc failed to become healthy in 5 min — aborting" >&2
      exit 1
    fi
  done
done

echo ""
echo "✓ Rolling update complete"
