#!/usr/bin/env bash
# ============================================================================
# Continuous GB10 monitor — prints unified memory + GPU compute every 2 s.
# Replaces the usual `watch nvidia-smi` because nvidia-smi reports N/A on GB10.
# ============================================================================
INTERVAL="${1:-2}"

while true; do
  clear
  echo "=== Tonic NVIDIA DGX Spark (GB10) — $(date '+%F %T %Z') ==="
  echo ""
  echo "── Unified Memory (LPDDR5x 128 GB) ──"
  free -h | awk 'NR<=2'
  echo ""
  echo "── GPU compute / clocks / power / temp ──"
  nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.gr,clocks.mem \
    --format=csv 2>/dev/null
  echo ""
  echo "── Top processes by memory ──"
  ps -e -o pid,rss,pcpu,comm --sort=-rss --no-headers | awk '$2>50000 {printf "%-8s %8.1fMB %5.1f%%  %s\n",$1,$2/1024,$3,$4}' | head -8
  echo ""
  echo "── Container status ──"
  docker ps --filter "name=tonic-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null | head -10
  sleep "$INTERVAL"
done
