#!/usr/bin/env bash
# ============================================================================
# Weekly maintenance — Mon 07:00 HKT from cron.
# Idempotent and safe.
# ============================================================================
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
echo "=== Weekly maintenance $(date) ==="

echo "→ Health-check"
./scripts/healthcheck.sh || true

echo "→ Disk report"
./scripts/disk-report.sh 80 || true

echo "→ Re-pull pinned images (no-op if digests unchanged)"
docker compose pull

echo "→ Prune dangling images"
docker image prune -f

echo "→ Prune stopped containers"
docker container prune -f

echo "→ Rotate /var/log/tonic logs older than 30 days"
sudo find /var/log/tonic -type f -name '*.log' -mtime +30 -delete 2>/dev/null || true

echo "✓ Weekly maintenance done"
