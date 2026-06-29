#!/usr/bin/env bash
# ============================================================================
# Restore LiteLLM PostgreSQL DB from a pg_dump file.
# Usage: scripts/restore-db.sh backups/litellm_YYYYMMDD_HHMMSS.pgdump
# ============================================================================
set -euo pipefail

[ $# -eq 1 ] || { echo "Usage: $0 <pgdump-file>"; exit 2; }
FILE="$1"
[ -f "$FILE" ] || { echo "Not found: $FILE"; exit 2; }

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$ROOT_DIR/.env" ] && set -a && . "$ROOT_DIR/.env" && set +a

read -rp "This will DROP and recreate ${POSTGRES_DB}. Continue? (yes/NO): " ans
[ "$ans" = "yes" ] || { echo "aborted"; exit 1; }

echo "→ Stopping LiteLLM"
docker compose stop litellm

echo "→ Restoring $FILE"
docker exec -i tonic-postgres pg_restore \
  -U "$POSTGRES_USER" \
  -d "$POSTGRES_DB" \
  --clean --if-exists --no-owner --no-privileges \
  < "$FILE"

echo "→ Starting LiteLLM"
docker compose start litellm

echo "✓ Restore complete"
