# Performance Tuning — High-Throughput vLLM on DGX Spark (GB10)

The GB10 is **memory-bandwidth-bound**, not compute-bound:

- LPDDR5x unified pool: **273 GB/s**
- FP4 tensor-core peak: ~**1 PFLOPS** (sparse)
- So your tokens/s ceiling is set by *how few bytes you have to move per token*, not by how many FLOPs you can do.

Tuning therefore has three goals, in priority order:

1. **Shrink what you move per token** — NVFP4 weights + FP8 KV cache.
2. **Move it less often** — larger prefill batches, scheduler look-ahead.
3. **Keep the tensor cores warm** — CUDA graphs, FlashInfer attention, locked clocks.

---

## 1. The full tunable parameter list

### 1.1 Quantization & precision

| Flag | Default | Recommended | Rationale on GB10 |
|------|---------|-------------|-------------------|
| `--quantization` | (model) | `modelopt` (NVFP4) when available, else `fp8` | 4-bit weights → 4× less memory traffic vs FP16 |
| `--kv-cache-dtype` | `auto` (fp16) | **`fp8`** | Halves KV-cache bandwidth — single biggest GB10 win |
| `--dtype` | `auto` | leave `auto` | vLLM picks bf16 activations; correct on Blackwell |
| `--calculate-kv-scales` | false | true (for NVFP4) | Improves NVFP4 numerics |
| `--quantization-param-path` | — | model-specific JSON for calibration scales when downloaded | Required for some NVFP4 checkpoints |

### 1.2 Batching & scheduling

| Flag | Default | Recommended | Rationale |
|------|---------|-------------|-----------|
| `--max-num-batched-tokens` | 2048 | **8192** (122B) / 4096 (31B/7B) | Bigger prefill batches saturate FP4 tensor cores |
| `--max-num-seqs` | 256 | **32** (122B), 64 (31B), 128 (7B) | Concurrency × KV size ≤ memory budget |
| `--enable-chunked-prefill` | off | **on** | Interleaves prefill chunks with decode → no head-of-line block |
| `--num-scheduler-steps` | 1 | **8** | Scheduler look-ahead → fewer GPU bubbles |
| `--enable-prefix-caching` | off | **on** (if multi-turn) | Huge win for chat workloads with shared system prompts |
| `--preemption-mode` | recompute | `swap` only if you must | Recompute is faster on GB10 because LPDDR5x is slower than HBM |

### 1.3 Memory budget

| Flag | Recommended | Rationale |
|------|-------------|-----------|
| `--gpu-memory-utilization 0.70` (qwen-main) | leaves ~36 GB for OS, CPU, and other backends |
| `--gpu-memory-utilization 0.20` (qwen-vision) | 7B model fits easily |
| `--gpu-memory-utilization 0.30` (gemma-fast) | 31B at 131K context |
| `--swap-space 0` | DO NOT swap; LPDDR5x is your VRAM |
| `--cpu-offload-gb 0` | same — there's nothing to offload to |
| `--block-size 16` | default; do not change |

### 1.4 Attention backend

| Env var | Recommended | Rationale |
|---------|-------------|-----------|
| `VLLM_ATTENTION_BACKEND` | **`FLASHINFER`** | Only backend verified on SM 12.1 |
| `VLLM_USE_TRITON_FLASH_ATTN` | `1` | Triton fallback for unsupported shapes |
| `VLLM_FLASHINFER_FORCE_TENSOR_CORES` | `1` | Force tensor-core path on Blackwell |

### 1.5 CUDA graphs

| Flag | Recommended | Rationale |
|------|-------------|-----------|
| `--enforce-eager` | **false** (leave it off) | CUDA graphs give 15–25 % decode boost on Blackwell |
| `--max-seq-len-to-capture` | 8192 | Larger captures cost startup time, marginal at runtime |

### 1.6 Logging / overhead

| Flag | Recommended | Rationale |
|------|-------------|-----------|
| `--disable-log-requests` | true (prod) | Removes per-token Python logging overhead |
| `--disable-log-stats` | false | Keep stats for Prometheus |
| `--uvicorn-log-level` | `warning` | Less noise |

---

## 2. OS / driver tuning

```bash
# Persistence + clock locking
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 2400          # GB10 sustained graphics clock
sudo nvidia-smi --auto-boost-default=0

# CPU governor for the ARM cores
sudo cpupower frequency-set -g performance

# Disable THP — predictable latency
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/enabled

# Pin Docker to NUMA node 0 (the Spark is single-socket, but this avoids any future surprise)
sudo sed -i 's|^ExecStart=.*$|ExecStart=/usr/bin/numactl --cpunodebind=0 --membind=0 /usr/bin/dockerd|' \
  /lib/systemd/system/docker.service
sudo systemctl daemon-reload && sudo systemctl restart docker

# Increase inotify watch limit (vLLM tail-watches log files)
echo "fs.inotify.max_user_watches=524288" | sudo tee /etc/sysctl.d/90-vllm.conf
sudo sysctl --system

# Lock memory clock if available (varies by DGX OS build)
sudo dgx-bandwidth --lock-memory-max 2>/dev/null || true
```

---

## 3. Container tuning

```yaml
# docker-compose.yml — for every vLLM service
ipc: host                   # NCCL shm rings
shm_size: "8g"              # vLLM allocator
ulimits:
  memlock: -1
  stack: 67108864
environment:
  VLLM_ATTENTION_BACKEND: FLASHINFER
  VLLM_USE_TRITON_FLASH_ATTN: "1"
  VLLM_FLASHINFER_FORCE_TENSOR_CORES: "1"
  TORCH_NCCL_AVOID_RECORD_STREAMS: "1"
  CUDA_DEVICE_MAX_CONNECTIONS: "1"
  NCCL_CUMEM_ENABLE: "0"
  PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True"
  HF_HUB_ENABLE_HF_TRANSFER: "1"
```

---

## 4. LiteLLM tuning

```yaml
router_settings:
  routing_strategy: usage-based-routing-v2
  enable_pre_call_checks: true
  num_retries: 2
  retry_after: 5
  allowed_fails: 3
  cooldown_time: 30

litellm_settings:
  request_timeout: 600
  cache:
    type: redis
    ttl: 600
```

---

## 5. Reference throughput on a single GB10 (June 2026)

Measured with `scripts/benchmark.sh`. Numbers are tokens/second.

| Backend | Prefill | Decode (1 stream) | Decode (8 streams aggregate) | Notes |
|---------|--------:|------------------:|-----------------------------:|-------|
| Qwen3.5-122B-FP8 (MoE, ~10B active) | 1,200 | 30–35 | 220–280 | KV fp8, batched 8K |
| Qwen2.5-VL-7B-FP4 | 2,400 | 80–100 | 500–600 | NVFP4 |
| Gemma4-31B-NVFP4 | 1,800 | 50–60 | 350–420 | NVFP4, 131K context |

Hitting these requires all four sections above to be applied together. The biggest single regression you can introduce is forgetting `--kv-cache-dtype fp8`.

---

## 6. What to do when you're still slow

1. Check `nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,clocks.gr --format=csv -l 2` — if temp ≥ 88 °C, you're thermal-throttling. Improve airflow or reduce `--max-num-seqs`.
2. Check `free -h` — if `MemAvailable` < 10 GB you're paging; lower `--gpu-memory-utilization`.
3. Drop `--max-model-len`. Most chat workloads finish under 8K; paying for 131K KV is wasteful.
4. Enable `--enable-prefix-caching`. Chat apps with a shared system prompt see 2–3× prefill gain.
5. Increase `--max-num-batched-tokens` until prefill saturates the SMs (watch `dcgmi dmon` SM occupancy).
6. Try `--num-scheduler-steps 16` — diminishing returns above 8 but sometimes worth it.
