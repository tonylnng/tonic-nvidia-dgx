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
| Qwen-main | `hellohal2064/vllm-dgx-spark-gb10:latest` | GB10-optimised, supports up to Gemma3 |
| Gemma4 | `vllm/vllm-openai:cu129-nightly-aarch64` | Required for Gemma4ForConditionalGeneration |
| LiteLLM | `ghcr.io/berriai/litellm:main-stable` | API gateway |

---

## Active Model: Qwen3-30B-A3B

**Model:** `Qwen/Qwen3-30B-A3B` (MoE, 3B activated parameters)
**Served name:** `qwen-main`

### Key Parameters

| Parameter | Value | Notes |
|---|---|---|
| `gpu_memory_utilization` | `0.60` | 60% of 121 GiB = ~73 GiB for vLLM (model + KV cache) |
| `max_model_len` | `32768` | Max context window |
| `kv_cache_dtype` | `fp8` | Reduces KV cache footprint |
| `max_num_batched_tokens` | `8192` | Chunked prefill budget |
| `max_num_seqs` | `64` | Max concurrent sequences |
| `attention_backend` | `FLASHINFER` | Fastest on GB10 |
| `reasoning_parser` | `qwen3` | Enables thinking mode |
| `enable_thinking` | `true` | Extended reasoning |

### Memory Budget at 0.60

| Component | GiB |
|---|---|
| vLLM budget (model + KV cache) | ~73 |
| OS + Docker overhead | ~48 |
| **Total** | **121** |

---

## Running Only One vLLM at a Time

**The GB10 cannot run two vLLM instances simultaneously.** Both models together exceed available memory and cause OOM hangs.

Before starting a new model, always stop the current one:

```bash
# Stop Qwen, start Gemma4
docker stop tonic-vllm-qwen-main
docker compose --env-file .env --env-file config/image-versions.env up -d vllm-gemma4-31b

# Stop Gemma4, start Qwen
docker stop tonic-vllm-gemma4-31b
docker compose --env-file .env --env-file config/image-versions.env up -d vllm-qwen-main
```

> Optional-profile containers (Gemma4, Qwen Vision) use `profiles: [optional]` and do **not** auto-start with `docker compose up`.

---

## Health Check

```bash
# Quick API check
curl http://127.0.0.1:8001/health

# Container status
docker ps | grep qwen
# (healthy) = ready | (health: starting) = loading | (unhealthy) = loading or crashed
```

> **Note:** `unhealthy` during the first 7–10 minutes is normal — the model is still loading shards. Check logs to distinguish loading from a real crash:

```bash
docker logs tonic-vllm-qwen-main --tail 10
```

---

## `gpu_memory_utilization` Tuning Guide

This is the most critical parameter for stable operation on unified memory hardware.

| Value | vLLM Budget | Use Case | Risk |
|---|---|---|---|
| `0.38` | ~46 GiB | Minimal footprint | OOM on KV cache if system has residual page cache |
| `0.60` | ~73 GiB | **Recommended default** | Stable, generous KV cache, ~48 GiB OS headroom |
| `0.85` | ~103 GiB | Recovery after memory fragmentation | Only ~9 GiB OS headroom |

**If vLLM fails with `No available memory for the cache blocks`:**
1. Increase `gpu_memory_utilization` to force the kernel to evict page cache, **or**
2. Restart the server to clear residual memory, then use `0.60`

---

## Gemma4-31B Known Issue

| Item | Status |
|---|---|
| FP8 quantization (`--quantization fp8`) | ❌ Crashes — `cutlass_gemm_caller Error Internal` on GB10 |
| `--enforce-eager` workaround | ❌ Does not prevent the CUTLASS FP8 kernel crash |
| BF16 (no quantization) | ⏳ Not yet tested — would use ~110 GiB |

**Root cause:** The CUTLASS FP8 kernel (`cutlass_scaled_mm`) in `cu129-nightly-aarch64` is incompatible with GB10 as of 2026-07-05. The crash occurs after model load, during KV cache warmup.

**Pending fix:** Remove `--quantization fp8` and `--kv-cache-dtype fp8` to run in native BF16.

---

## Startup Time Reference

| Model | Shards | Load Time |
|---|---|---|
| Qwen3-30B-A3B | 16 × ~3.6 GiB | ~6–7 min |
| Gemma4-31B-IT | 2 × ~29 GiB | ~7 min (blocked by CUTLASS bug) |

---

## Model Storage

Models are mounted read-only from the host:

```
/data/huggingface/manual/
├── Qwen3-30B-A3B/
└── gemma-4-31B-it/
```

---

*Last updated: 2026-07-05*
