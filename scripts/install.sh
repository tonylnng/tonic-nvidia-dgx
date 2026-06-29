#!/usr/bin/env bash
# ============================================================================
# One-shot installer / re-runner for the Tonic DGX Spark stack.
# Idempotent — safe to re-run.
# ============================================================================
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

bold() { printf "\033[1m%s\033[0m\n" "$*"; }

bold "1/7 Verify host"
[ "$(uname -m)" = "aarch64" ] || { echo "ERROR: not aarch64 (got $(uname -m))"; exit 1; }
command -v docker >/dev/null || { echo "ERROR: docker not installed"; exit 1; }
docker compose version >/dev/null || { echo "ERROR: docker compose v2 not installed"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "ERROR: NVIDIA driver not installed"; exit 1; }
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
echo "  compute_cap=$CC (expected 12.1)"
[ "$CC" = "12.1" ] || echo "  WARN: not 12.1 — confirm this is a DGX Spark GB10"

bold "2/7 Verify .env"
[ -f .env ] || { echo "Missing .env — copy from .env.example and fill in"; exit 1; }
set -a; . .env; set +a
[ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN unset in .env"; exit 1; }
[ "${LITELLM_MASTER_KEY:-sk-master-CHANGE-ME-32chars-min}" != "sk-master-CHANGE-ME-32chars-min" ] \
  || { echo "Please change LITELLM_MASTER_KEY in .env"; exit 1; }

bold "3/7 Prepare /data directories"
sudo mkdir -p /data/huggingface /data/postgres /data/redis /data/litellm-logs
sudo chown -R "$(id -u):$(id -g)" /data/huggingface /data/litellm-logs

bold "4/7 Load pinned image versions"
set -a; . config/image-versions.env; set +a
echo "  VLLM_IMAGE   = $VLLM_IMAGE"
echo "  LITELLM_IMAGE= $LITELLM_IMAGE"

bold "5/7 Pull images"
docker pull "$VLLM_IMAGE"
docker pull "$LITELLM_IMAGE"
docker pull "$POSTGRES_IMAGE"
docker pull "$REDIS_IMAGE"

bold "6/7 Bring up the stack"
docker compose --env-file .env --env-file config/image-versions.env up -d

bold "7/7 Wait & health-check"
sleep 10
./scripts/healthcheck.sh || {
  echo "Some services not healthy yet — model first-load can take 60–180 s on GB10."
  echo "Re-run ./scripts/healthcheck.sh after a minute."
}

echo ""
echo "Done. LiteLLM UI: http://127.0.0.1:4000/ui  (login with LITELLM_MASTER_KEY)"
