# Changelog

All notable changes are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [v2026.06.30] — 2026-06-30

### Major
- **Re-targeted entire stack to NVIDIA DGX Spark (GB10 Grace Blackwell)**. The original draft assumed an 8-GPU H100/H200 datacenter node; that was incorrect for the actual hardware. Every command, image tag, and tuning parameter has been re-validated for ARM64 + SM 12.1 + 128 GB unified LPDDR5x.

### Changed
- vLLM image switched from `vllm/vllm-openai:latest` (x86 + SM 10) to `hellohal2064/vllm-dgx-spark-gb10:v0.17.1` (ARM64 + SM 12.1 + CUDA 13).
- `tensor-parallel-size` lowered from 4/8 to 1 across all backends (single integrated GPU).
- `gpu-memory-utilization` lowered from 0.90 → 0.70/0.20/0.30 (shared unified-memory pool).
- Backend B replaced: `Qwen3-VL-235B` (needs ≥320 GB VRAM, infeasible on Spark) → `nvidia/Qwen2.5-VL-7B-Instruct-FP4`.
- All vLLM services now use FP8 KV cache + chunked prefill + scheduler look-ahead.
- All listening ports bound to `127.0.0.1`; external access routed exclusively through Tailscale.

### Added
- `config/image-versions.env` for pinned digests.
- `docker-compose.canary.yml` overlay for canary deploys.
- `scripts/` — install, healthcheck, backup/restore, rolling update, rollback, benchmark, GB10 monitor, model swap.
- `docs/VERSIONING.md`, `docs/PERFORMANCE_TUNING.md`, `docs/MAINTENANCE.md`, `docs/TROUBLESHOOTING.md`, `docs/TAILSCALE_ACL.md`.
- LiteLLM Redis prompt cache + usage-based routing v2 + per-key budget defaults.

### Removed
- All `nvidia-smi`-based memory monitoring guidance (reports `N/A` on unified memory). Replaced with `free -h` and `dcgmi`.
- `--swap-space` recommendations (swap on unified memory is a footgun).

### Migration notes
If you deployed the original `.md` guide as-is:
1. Stop the old stack: `docker compose down`.
2. Pull the new images from `config/image-versions.env`.
3. Re-download the Qwen2.5-VL-7B-FP4 weights (the 235B model from the old guide will not run).
4. Apply the new `gpu-memory-utilization` budgets.
5. Bring the new stack up with `./scripts/install.sh`.
