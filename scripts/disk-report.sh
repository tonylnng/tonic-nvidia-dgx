#!/usr/bin/env bash
# Disk usage report — warns if /data > 80%.
set -euo pipefail
THRESHOLD="${1:-80}"
USAGE=$(df -P /data | awk 'NR==2 {gsub("%","",$5); print $5}')
echo "/data is at ${USAGE}% (threshold ${THRESHOLD}%)"
df -h /data ~ 2>/dev/null
echo ""
echo "── Huggingface cache ──"
du -sh /data/huggingface/hub/* 2>/dev/null | sort -h | tail -10
if [ "$USAGE" -ge "$THRESHOLD" ]; then
  echo "WARN: /data above ${THRESHOLD}%" >&2
  exit 1
fi
