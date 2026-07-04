# Changelog

All notable changes are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [v2026.07.05] — 2026-07-05

### Changed
- **Default `qwen-main` backend swapped from Qwen3.5-122B-A10B-FP8 → `Qwen/Qwen3-30B-A3B-FP8`.** The 122B MoE is ~66 GB on disk and pushes the 119 GiB unified pool near saturation once the vision + fast backends are co-resident; the 30B MoE is ~7 GB with ~3B active params and comfortably co-hosts all three services with headroom for KV cache and concurrent requests.
- `gpu-memory-utilization` budget for `qwen-main` retuned from `0.70` → `0.30` (default). Total footprint across the three backends is now `0.30 + 0.20 + 0.30 = 0.80`, leaving ~24 GiB of unified memory free for CPU/system.
- `--max-num-seqs` for `qwen-main` raised from 8 → 64 (the smaller model frees enough KV headroom to serve many more concurrent requests).
- `HF_TOKEN` in `.env.example` reworded — it is now optional and only consulted if a service *falls back* to pulling from Hugging Face. Manual upload is the recommended path (see below).

### Added
- **`docs/MANUAL_MODEL_UPLOAD.md`** — full workflow for hosts that cannot reach `huggingface.co` directly. Download the model files from the HF web UI to any ARM Mac / laptop, then `rsync` the folder to `/data/huggingface/manual/<Model-Folder>` on the DGX Spark. Includes a per-model download-size cheat sheet, verify step, and troubleshooting table.
- **`scripts/init-data-dirs.sh`** — creates `/data/huggingface`, `/data/huggingface/manual`, `/data/huggingface/hub/_archive`, `/data/postgres`, `/data/redis`, `/data/litellm-logs`, `/data/backups` with the correct ownership and mode (data dirs `755`, DB dirs `700`). Detects NVMe vs HDD via `lsblk ROTA` and warns if `/data` is on rotational storage. Idempotent — safe to re-run.
- **`scripts/verify-model.sh <path>`** — sanity-checks a manually uploaded model directory: required files present (`config.json`, tokenizer, at least one weight shard), safetensors index cross-reference, zero-byte guard, total size report. Run this after every upload.
- **`docker-compose.yml`** — new bind mount `${MANUAL_MODELS_DIR:-/data/huggingface/manual}:/models:ro` on `x-vllm-common`, so every vLLM backend sees uploaded models under `/models/<Folder>`.
- **Env-var overrides** for the three model paths in `.env.example`: `QWEN_MAIN_MODEL`, `QWEN_VISION_MODEL`, `GEMMA_FAST_MODEL`, plus `QWEN_MAIN_MEM_UTIL` and `MANUAL_MODELS_DIR`. You no longer need to edit `docker-compose.yml` to swap the main model — set the variable in `.env` and `docker compose up -d qwen-main`.

### Migration notes
If you deployed a previous `v2026.06.30` build:
1. Pull the repo: `git pull && cp .env .env.bak`.
2. Merge any new keys from `.env.example` into your `.env` (or start from the new template and re-inject secrets).
3. Run `./scripts/init-data-dirs.sh` (sudo once).
4. If you still want the 122B backend, download it per `docs/MANUAL_MODEL_UPLOAD.md`, then in `.env` set `QWEN_MAIN_MODEL=/models/Qwen3.5-122B-A10B-FP8` and `QWEN_MAIN_MEM_UTIL=0.70` — the compose defaults will otherwise pick the 30B model.
5. `docker compose --env-file .env --env-file config/image-versions.env up -d`.

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
