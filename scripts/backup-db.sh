#!/usr/bin/env bash
# ============================================================================
# PostgreSQL backup for LiteLLM. Keeps last 30 local backups.
# Cron: 0 2 * * * /path/to/scripts/backup-db.sh
# ============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BACKUP_DIR="${ROOT_DIR}/backups"
ENV_FILE="${ROOT_DIR}/.env"

mkdir -p "$BACKUP_DIR"
[ -f "$ENV_FILE" ] && set -a && . "$ENV_FILE" && set +a

STAMP=$(date +%Y%m%d_%H%M%S)
OUT="${BACKUP_DIR}/litellm_${STAMP}.pgdump"

echo "→ Dumping ${POSTGRES_DB} to ${OUT}"
docker exec tonic-postgres pg_dump \
  -U "$POSTGRES_USER" \
  -d "$POSTGRES_DB" \
  -Fc \
  > "$OUT"

# Retention — keep 30 most recent
ls -1t "$BACKUP_DIR"/*.pgdump 2>/dev/null | tail -n +31 | xargs -r rm -f

# Optional: rsync off-box
if [ -n "${OFFSITE_BACKUP_TARGET:-}" ]; then
  echo "→ rsync to $OFFSITE_BACKUP_TARGET"
  rsync -az "$OUT" "$OFFSITE_BACKUP_TARGET/"
fi

echo "✓ Backup complete: $OUT ($(du -h "$OUT" | cut -f1))"
