#!/usr/bin/env bash
# ============================================================================
# Quick throughput / latency benchmark for a single backend.
# Saves results under docs/benchmarks/.
#
# Usage:
#   scripts/benchmark.sh qwen-main          # production port (8001)
#   scripts/benchmark.sh qwen-main-canary   # canary port (8011)
# ============================================================================
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT_DIR/docs/benchmarks"

ALIAS="${1:?usage: $0 <alias>}"
case "$ALIAS" in
  qwen-main)         PORT=8001 ;;
  qwen-vision)       PORT=8002 ;;
  gemma-fast)        PORT=8003 ;;
  qwen-main-canary)  PORT=8011 ;;
  qwen-vision-canary)PORT=8012 ;;
  gemma-fast-canary) PORT=8013 ;;
  *) echo "Unknown alias: $ALIAS"; exit 2 ;;
esac

STAMP=$(date +%Y%m%d_%H%M%S)
OUT="$ROOT_DIR/docs/benchmarks/${STAMP}-${ALIAS}.txt"

# Determine the served model name from /v1/models
MODEL=$(curl -s "http://127.0.0.1:${PORT}/v1/models" | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])")

PROMPT="Write a detailed 500-word essay about the history of computing in Hong Kong."

{
  echo "# Benchmark: $ALIAS on port $PORT"
  echo "# Model: $MODEL"
  echo "# When:  $STAMP"
  echo ""
  echo "## Single-stream decode (max_tokens=512)"
  /usr/bin/time -f "elapsed=%es cpu=%P maxrss=%MkB" \
    curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":512,\"temperature\":0.2}" \
    | python3 -c "import sys,json,time; d=json.load(sys.stdin); u=d['usage']; print(f\"prompt_tokens={u['prompt_tokens']} completion_tokens={u['completion_tokens']} total={u['total_tokens']}\")"
  echo ""
  echo "## Concurrent decode (8 streams in parallel, max_tokens=256)"
  start=$(date +%s.%N)
  for i in {1..8}; do
    curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT (variation $i)\"}],\"max_tokens\":256,\"temperature\":0.2}" \
      | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['completion_tokens'])" &
  done
  wait
  end=$(date +%s.%N)
  echo "wall_seconds=$(echo "$end-$start" | bc)"
} | tee "$OUT"

echo ""
echo "✓ Wrote $OUT"
