#!/usr/bin/env bash
# ============================================================================
# One-shot: prepare /data on the DGX Spark for the Tonic stack.
# Creates the subdirectories every service mounts, sets sensible ownership
# and permissions, and does a quick NVMe-vs-slow-disk sanity check.
#
# Safe to re-run — all operations are idempotent.
#
# Usage:  sudo scripts/init-data-dirs.sh
#         (needs sudo the first time to chown /data itself)
# ============================================================================
set -euo pipefail

TARGET_USER="${SUDO_USER:-$USER}"
TARGET_GROUP="$(id -gn "$TARGET_USER")"

bold() { printf "\033[1m%s\033[0m\n" "$*"; }
warn() { printf "\033[33m%s\033[0m\n" "$*"; }
fail() { printf "\033[31m%s\033[0m\n" "$*" >&2; exit 1; }

bold "→ Verifying /data exists"
[ -d /data ] || fail "/data does not exist. Create it first (sudo mkdir /data) and re-run."

bold "→ Taking ownership of /data as $TARGET_USER:$TARGET_GROUP"
if [ "$(stat -c '%U' /data)" != "$TARGET_USER" ]; then
  if [ "$EUID" -ne 0 ]; then
    fail "Need sudo to chown /data. Re-run with: sudo $0"
  fi
  chown "$TARGET_USER:$TARGET_GROUP" /data
fi
chmod 755 /data

bold "→ Creating subdirectories"
declare -A DIRS=(
  [huggingface]=755
  [huggingface/manual]=755
  [postgres]=700
  [redis]=700
  [litellm-logs]=755
  [backups]=755
  [huggingface/hub/_archive]=755
)
for d in "${!DIRS[@]}"; do
  path="/data/$d"
  mkdir -p "$path"
  chown "$TARGET_USER:$TARGET_GROUP" "$path"
  chmod "${DIRS[$d]}" "$path"
  printf "  %-40s (%s)\n" "$path" "${DIRS[$d]}"
done

bold "→ Verifying we didn't land on a slow disk"
DEV=$(df -P /data | awk 'NR==2 {print $1}')
ROTA=$(lsblk -no ROTA "$DEV" 2>/dev/null | head -1 || echo "?")
SIZE_H=$(df -Ph /data | awk 'NR==2 {print $2}')
FREE_H=$(df -Ph /data | awk 'NR==2 {print $4}')
echo "  Device : $DEV"
echo "  Size   : $SIZE_H total, $FREE_H free"
case "$ROTA" in
  0) echo "  Type   : SSD/NVMe ✓" ;;
  1) warn "  Type   : HDD ✗  — model loads will be slow. Move /data to the NVMe." ;;
  *) warn "  Type   : unknown — check lsblk manually" ;;
esac

FREE_G=$(df -P /data | awk 'NR==2 {print int($4/1024/1024)}')
if [ "$FREE_G" -lt 100 ]; then
  warn "  Only ${FREE_G} GiB free — models are big. 500+ GiB free recommended."
fi

bold "→ Final layout"
ls -la /data

echo ""
bold "✓ /data ready."
echo "  Next step: rsync manually-downloaded model files into /data/huggingface/manual/<model-name>/"
echo "  Then edit docker-compose.yml to point --model /models/<model-name> and run: docker compose up -d"
