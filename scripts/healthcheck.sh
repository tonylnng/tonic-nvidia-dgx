#!/usr/bin/env bash
# ============================================================================
# Health-check all services. Exits non-zero if anything is unhealthy.
# Designed to be safe to run from cron (no colour codes if not a TTY).
# ============================================================================
set -uo pipefail

if [ -t 1 ]; then GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
else GREEN=''; RED=''; NC=''; fi

declare -A ENDPOINTS=(
  [litellm]="http://127.0.0.1:4000/health/liveliness"
  [qwen-main]="http://127.0.0.1:8001/health"
  [qwen-vision]="http://127.0.0.1:8002/health"
  [gemma-fast]="http://127.0.0.1:8003/health"
)

fail=0
for svc in "${!ENDPOINTS[@]}"; do
  url="${ENDPOINTS[$svc]}"
  if curl -fsS --max-time 5 "$url" >/dev/null 2>&1; then
    printf "${GREEN}[OK]${NC}   %-12s %s\n" "$svc" "$url"
  else
    printf "${RED}[FAIL]${NC} %-12s %s\n" "$svc" "$url"
    fail=$((fail+1))
  fi
done

# Unified memory snapshot
echo ""
echo "── Unified memory ──"
free -h | awk 'NR==1 || NR==2'

# GPU compute (ignore memory column — meaningless on GB10)
echo ""
echo "── GPU compute ──"
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw,clocks.gr \
  --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"

exit "$fail"
