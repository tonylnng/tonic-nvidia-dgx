# vLLM Setup on NVIDIA DGX Spark (GB10)

This document covers the vLLM backend configuration, memory tuning, and operational notes learned from production use.

---

## Hardware

| Component | Details |
|---|---|
| Device | NVIDIA DGX Spark (GB10) |
| Architecture | ARM64 (aarch64) |
| GPU | NVIDIA GB10 — **unified memory** (CPU + GPU share the same pool) |
| Total Memory | 121 GiB unified |
| OS | Ubuntu Linux |

> ⚠️ **Unified Memory:** Unlike discrete GPUs, the GB10 has no separate VRAM. CPU RAM *is* GPU memory. All memory usage (OS, Docker, model weights, KV cache) competes for the same 121 GiB pool.

---

## Docker Images

| Service | Image | Notes |
|---|---|---|
| vllm-qwen-main | `hellohal2064/vllm-dgx-spark-gb10:latest` | GB10-optimised; uses **env vars** not `--model` CLI flags; supports up to Gemma3 |
| vllm-qwen-vision | `hellohal2064/vllm-dgx-spark-gb10:latest` | Same image; optional profile — not auto-started |
| vllm-gemma4-31b | `vllm/vllm-openai:cu129-nightly-aarch64` | Required for Gemma4ForConditionalGeneration (hellohal2064 only supports up to Gemma3) |
| LiteLLM | `ghcr.io/berriai/litellm:main-stable` | API gateway |

> ⚠️ **Image version locking:** `hellohal2064:latest` is pinned in `config/image-versions.env` via `VLLM_IMAGE`. After UAT, replace the tag with a `@sha256:` digest. The `vllm-gemma4-31b` service references `vllm/vllm-openai:cu129-nightly-aarch64` directly — **not** `${VLLM_IMAGE}`.

---

## Active Model: Qwen3-30B-A3B

**Model path:** `/data/huggingface/manual/Qwen3-30B-A3B` → mounted as `/models/Qwen3-30B-A3B` inside the container  
**Served name:** `qwen-main`  
**Container:** `tonic-vllm-qwen-main` → port `127.0.0.1:8001`

### Key Parameters

| Parameter | Value | Source |
|---|---|---|
| `GPU_MEMORY_UTIL` | `0.60` | `.env` → `QWEN_MAIN_MEM_UTIL` |
| `MAX_MODEL_LEN` | `32768` | hard-coded in compose |
| `kv_cache_dtype` | `fp8` | `--kv-cache-dtype fp8` |
| `max_num_batched_tokens` | `8192` | `--max-num-batched-tokens` |
| `max_num_seqs` | `64` | `--max-num-seqs` |
| `ATTENTION_BACKEND` | `FLASHINFER` | env var |
| `TOOL_CALL_PARSER` | `hermes` | `--tool-call-parser` |
| `REASONING_PARSER` | `qwen3` | `--reasoning-parser` |
| `ENABLE_THINKING` | `true` | env var |

> ⚠️ **hellohal2064 image quirk:** This image's entrypoint reads `MODEL_PATH`, `GPU_MEMORY_UTIL`, `MAX_MODEL_LEN`, `ATTENTION_BACKEND`, `TOOL_CALL_PARSER`, `REASONING_PARSER`, and `ENABLE_THINKING` from **env vars**. Do not pass `--model` as the first CLI argument — use `MODEL_PATH` in the environment block instead.

### Memory Budget at 0.60

| Component | GiB |
|---|---|
| vLLM budget (model weights + KV cache) | ~73 |
| OS + Docker + page cache overhead | ~48 |
| **Total** | **121** |

### Startup Time

| Model | Shards | Load Time |
|---|---|---|
| Qwen3-30B-A3B | 16 × ~3.6 GiB | ~6–7 min |

> GPU clock jumps to **~2444 MHz** once model inference begins (idle: ~300 MHz). Container health probe shows `(unhealthy)` or `(health: starting)` for the first 7–10 minutes while shards load — this is normal. Confirm with `docker logs tonic-vllm-qwen-main --tail 10`.

---

## Optional Model: Gemma4-31B-IT

**Model path:** `/data/huggingface/manual/gemma-4-31B-it` → mounted as `/models/gemma-4-31B-it`  
**Served name:** `gemma4-31b`  
**Container:** `tonic-vllm-gemma4-31b` → port `127.0.0.1:8003`

> ⚠️ **gemma4-31b is NOT in an optional Docker Compose profile.** Unlike `vllm-qwen-vision`, it is a regular service. It is simply not started by default because the GB10 cannot run two vLLM instances simultaneously. Start it explicitly (see [Running Only One vLLM at a Time](#running-only-one-vllm-at-a-time)).

### Key Parameters

| Parameter | Value | Source |
|---|---|---|
| `--gpu-memory-utilization` | `0.35` | `.env` → `GEMMA4_31B_MEM_UTIL` |
| `--max-model-len` | `32768` | hard-coded in compose |
| `--max-num-batched-tokens` | `4096` | hard-coded in compose |
| `--max-num-seqs` | `32` | hard-coded in compose |
| `--enforce-eager` | set | disables CUDA graph capture |

### ⚠️ Known Issue: FP8 Quantization Crash

| Item | Status |
|---|---|
| `--quantization fp8` + `--kv-cache-dtype fp8` | ❌ Crashes after model load with `cutlass_gemm_caller Error Internal` |
| `--enforce-eager` workaround | ❌ Does **not** prevent the CUTLASS FP8 kernel crash |
| BF16 (remove both FP8 flags) | ⏳ Not yet tested — would use ~110 GiB, likely requires `GEMMA4_31B_MEM_UTIL=0.90`+ |

**Root cause:** The `cutlass_scaled_mm` CUTLASS FP8 kernel in `cu129-nightly-aarch64` is incompatible with the GB10 (SM 12.1) as of 2026-07-05. The crash occurs after model load, during KV cache warmup — not during loading.

**Pending fix:** Remove `--quantization fp8` and `--kv-cache-dtype fp8` from `docker-compose.yml` to run in native BF16.

---

## Optional Model: Qwen2.5-VL (Vision)

**Container:** `tonic-vllm-qwen-vision` → port `127.0.0.1:8002`  
**Profile:** `optional` — must be started explicitly with `--profile optional`  
**Status:** ⏳ Model not yet downloaded. Pending: download `Qwen2.5-VL-7B-Instruct-FP4` to `/data/huggingface/manual/`.

---

## Running Only One vLLM at a Time

The GB10 **cannot run two vLLM instances simultaneously.** Model weights alone consume 50–110 GiB depending on the model; two instances together cause OOM hangs.

```bash
# Swap: stop Qwen, start Gemma4
docker stop tonic-vllm-qwen-main
docker compose --env-file .env --env-file config/image-versions.env up -d vllm-gemma4-31b

# Swap back: stop Gemma4, start Qwen
docker stop tonic-vllm-gemma4-31b
docker compose --env-file .env --env-file config/image-versions.env up -d vllm-qwen-main
```

For vision (optional profile):
```bash
docker stop tonic-vllm-qwen-main
docker compose --env-file .env --env-file config/image-versions.env --profile optional up -d vllm-qwen-vision
```

---

## LiteLLM Gateway

**Container:** `tonic-litellm` → port `100.125.136.113:4000` (Tailscale only)  
**Config:** `config/litellm_config.yaml`

### ⚠️ Health Check Shows `(unhealthy)` — Expected Behaviour

LiteLLM's `/health/liveliness` endpoint can return non-200 even when the proxy is routing requests successfully. This happens when a backend in `model_list` is unreachable (e.g. Gemma4 not running) or during startup. **The gateway is functional if `curl http://127.0.0.1:4000/health` returns a JSON body.**

```bash
# Verify gateway is alive
curl -s http://127.0.0.1:4000/health | jq .

# List available models
curl -s -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  http://127.0.0.1:4000/v1/models | jq '.data[].id'
```

### `store_model_in_db: false`

This is intentional. Keeping it `false` prevents LiteLLM from writing model configs to the DB at startup, which avoids failures on a fresh DB where no model rows exist yet.

---

## Health Checks

```bash
# vLLM health (qwen-main)
curl http://127.0.0.1:8001/health

# vLLM health (gemma4, when running)
curl http://127.0.0.1:8003/health

# Container status
docker ps | grep tonic

# Tail logs
docker logs tonic-vllm-qwen-main --tail 20 -f
```

> Container status meanings:
> - `(healthy)` — model loaded and accepting requests
> - `(health: starting)` — still loading shards (normal for first 7–10 min)
> - `(unhealthy)` — either still loading or crashed; check logs

---

## `gpu_memory_utilization` Tuning Guide

Critical parameter for unified-memory stability. Values below are **as a fraction of the full 121 GiB pool**.

| Value | vLLM Budget | OS Headroom | Use Case |
|---|---|---|---|
| `0.35` | ~42 GiB | ~79 GiB | Gemma4 (in `.env`), leaves room for OS + future concurrent models |
| `0.55` | ~67 GiB | ~54 GiB | Conservative Qwen default (compose default if `QWEN_MAIN_MEM_UTIL` unset) |
| `0.60` | ~73 GiB | ~48 GiB | **Qwen production default** (current `.env` setting) |
| `0.85` | ~103 GiB | ~18 GiB | Recovery / memory-fragmentation clearing only — risky |

**If vLLM fails with `No available memory for the cache blocks`:**
1. Increase `gpu_memory_utilization` to push the kernel to evict page cache, **or**
2. Restart the server (`docker restart tonic-vllm-qwen-main`) to clear residual memory, then retry at `0.60`

**Changing the value:** Update `.env` → `QWEN_MAIN_MEM_UTIL=<new>` or `GEMMA4_31B_MEM_UTIL=<new>`, then `docker compose up -d --force-recreate vllm-qwen-main`.

---

## Model Storage

Models are mounted read-only from the host into the container:

```
/data/huggingface/manual/        (host)  →  /models/  (container, :ro)
├── Qwen3-30B-A3B/               ✅ running
└── gemma-4-31B-it/              ✅ downloaded, pending FP8 fix
```

Phase 2 models (pending download):
```
├── Qwen2.5-VL-7B-Instruct-FP4/  ⏳ not yet downloaded (for qwen-vision)
```

See `docs/MANUAL_MODEL_UPLOAD.md` for download and placement instructions.

---

## .env Reference (production values)

| Variable | Value | Notes |
|---|---|---|
| `QWEN_MAIN_MODEL` | `/models/Qwen3-30B-A3B` | Model path inside container |
| `QWEN_MAIN_MEM_UTIL` | `0.60` | gpu_memory_utilization for qwen-main |
| `GEMMA4_31B_MODEL` | `/models/gemma-4-31B-it` | Model path inside container |
| `GEMMA4_31B_MEM_UTIL` | `0.35` | gpu_memory_utilization for gemma4-31b |
| `MANUAL_MODELS_DIR` | `/data/huggingface/manual` | Host path → `/models` mount |
| `HF_CACHE_DIR` | `/data/huggingface` | HF cache root |

---

*Last updated: 2026-07-05*
