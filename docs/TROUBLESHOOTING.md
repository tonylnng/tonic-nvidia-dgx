# Troubleshooting

| Symptom | Root cause | Fix |
|---------|-----------|-----|
| `RuntimeError: CUDA error: no kernel image is available for execution on the device` | Image built for SM 10.0 / 12.0, not 12.1 (GB10) | Use the GB10-specific vLLM image; verify `nvidia-smi --query-gpu=compute_cap` returns `12.1` |
| `nvidia-smi` shows `N/A` for memory | Expected on unified memory | Use `free -h` for real memory state |
| vLLM exits with `torch.cuda.OutOfMemoryError` despite `free -h` showing plenty free | `--gpu-memory-utilization` is a hard cap inside vLLM, not a query of the pool | Raise the value cautiously, or lower `--max-model-len` / `--max-num-seqs` |
| LiteLLM returns 502 Bad Gateway | Backend not yet healthy after start | First model load takes 60–180 s; check `docker logs tonic-vllm-qwen-main` |
| LiteLLM returns 401 Unauthorized | Wrong / expired virtual key | Mint a new key via `/key/generate` or the UI |
| LiteLLM returns 429 Too Many Requests | RPM/TPM cap on the key | Raise the key's limit in the UI |
| Throughput collapses after ~10 min | GB10 thermal throttle | Inspect `nvidia-smi --query-gpu=temperature.gpu,clocks.gr`; improve airflow, lower `--max-num-seqs` |
| `Triton kernel launch failed` | Triton cache permission or stale | `rm -rf ~/.triton/cache` inside the container, restart |
| Container restarts every few minutes | OOM-killed by host (not by vLLM); check `dmesg | tail` | Lower `--gpu-memory-utilization` across the board; the sum across active backends should be ≤ 0.85 |
| Slow image pull | Coming from Docker Hub | Use the GHCR mirror in `docs/VERSIONING.md` §6 |
| Postgres password is wrong after restore | Restore did not refresh `DATABASE_URL` | Re-set the password explicitly: `docker exec -it tonic-postgres psql -U postgres -c "ALTER USER litellm WITH PASSWORD '<from .env>';"` |
| Tool calling not working | Wrong `--tool-call-parser` | `hermes` for Qwen3.5; `gemma4` for Gemma4; remove flag for vision model |
| Vision request returns text-only | Wrong content schema | Use `[{"type":"text",...},{"type":"image_url","image_url":{"url":...}}]` |
| Long generations time out | LiteLLM default request timeout | Bump `litellm_settings.request_timeout: 600` |
| `host header missing` in LiteLLM logs | Health probe via Docker network without HTTP Host | Use the `/health/liveliness` endpoint, not `/health` |
| Re-deploy resets virtual keys | `STORE_MODEL_IN_DB` not set | Set `STORE_MODEL_IN_DB=True` in `.env` and restart |
| Models redownload every restart | HF cache not mounted | Confirm `/data/huggingface:/root/.cache/huggingface` volume |
| vLLM hangs at startup with "loading safetensors" | Slow read from the SD-card / USB disk | Models must live on the internal 4 TB NVMe, not external |
| Streaming response stalls midway | `stream_timeout` shorter than total generation | Raise `stream_timeout: 600` |
