#!/usr/bin/env bash
# ============================================================================
# Roll a single backend back to the previous image digest stored in
# config/image-versions.env.bak. Run BEFORE deciding to promote a new digest:
#   cp config/image-versions.env config/image-versions.env.bak
# ============================================================================
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

[ -f config/image-versions.env.bak ] || { echo "No backup: config/image-versions.env.bak"; exit 1; }
ALIAS="${1:-}"
[ -n "$ALIAS" ] || { echo "Usage: $0 <qwen-main|qwen-vision|gemma-fast>"; exit 2; }

cp config/image-versions.env.bak config/image-versions.env
./scripts/update-rolling.sh "$ALIAS"
echo "✓ Rolled back $ALIAS"
