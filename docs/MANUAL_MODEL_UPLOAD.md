# Manual Model Upload — HF Web Download → DGX Spark

Use this workflow when the DGX Spark cannot download models directly from Hugging Face (blocked, slow, or gated tokens don't work), but you have HF web access from another machine (e.g. your Mac).

The stack is designed for this: `docker-compose.yml` mounts `/data/huggingface/manual` on the host as **`/models`** inside every vLLM container, and every backend's `--model` argument defaults to a path under `/models/`.

---

## 1. Prepare `/data` on the Spark (once)

```bash
# Assumes you've already run `sudo mkdir /data` (or /data is a mount point).
git clone https://github.com/tonylnng/tonic-nvidia-dgx.git ~/tonic-nvidia-dgx
cd ~/tonic-nvidia-dgx
sudo scripts/init-data-dirs.sh
```

The script creates `/data/huggingface/manual/` and sets ownership to your user so `rsync` won't need sudo.

---

## 2. Download the model files from the HF web UI

For each model, open its page (example: [`Qwen/Qwen3-30B-A3B-FP8`](https://huggingface.co/Qwen/Qwen3-30B-A3B-FP8)) → **Files and versions** tab → download **every file** in the root of the tree.

**Do NOT skip any of these:**

| Category | Files |
|----------|-------|
| Config | `config.json`, `generation_config.json` |
| Tokenizer | `tokenizer.json` **and/or** (`vocab.json` + `merges.txt`), `tokenizer_config.json`, `special_tokens_map.json` |
| Weights | **all** `model-*.safetensors` shards (may be 1–60 files) |
| Weight index | `model.safetensors.index.json` (critical for multi-shard models) |
| Code | any `*.py` files (needed when the model uses `--trust-remote-code`) |
| Chat / MM | `chat_template.jinja`, `preprocessor_config.json`, `processor_config.json` if present |

Place them all in one folder on your laptop, named exactly as the last path segment of the HF repo:

```
~/Downloads/Qwen3-30B-A3B-FP8/
├── config.json
├── generation_config.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
└── model.safetensors.index.json
```

> **Tip:** the HF web UI has a "Download all" button on some model pages, but for multi-shard models it's often easier to right-click each file → "Save link as", or use a browser download manager that supports concurrent downloads (e.g. FDM, DownThemAll).

---

## 3. Upload to the Spark over Tailscale

```bash
# From your laptop — Tailscale hostname of the Spark
rsync -avhP --info=progress2 \
  ~/Downloads/Qwen3-30B-A3B-FP8/ \
  tony@spark.<your-tailnet>.ts.net:/data/huggingface/manual/Qwen3-30B-A3B-FP8/
```

`rsync -P` resumes on network interruption. If the connection drops mid-transfer, just re-run the same command.

**Alternative:** if you don't have SSH-over-Tailscale set up, `tailscale file cp` also works:
```bash
tailscale file cp ~/Downloads/Qwen3-30B-A3B-FP8 spark:
# Then on the Spark: tailscale file get /data/huggingface/manual/
```

---

## 4. Verify the upload is complete

Run the integrity check on the Spark:

```bash
cd ~/tonic-nvidia-dgx
scripts/verify-model.sh /data/huggingface/manual/Qwen3-30B-A3B-FP8
```

Expected output ends with:

```
✓ Model directory looks good.
  Point vLLM at: /models/Qwen3-30B-A3B-FP8
```

If any file is flagged as missing, re-download that specific file from HF and rsync it into the same folder.

---

## 5. Wire it into the stack

The default `docker-compose.yml` already expects the following folder names under `/data/huggingface/manual/`:

| Backend | Default path (host) | Alias |
|---------|--------------------|-------|
| `vllm-qwen-main` | `/data/huggingface/manual/Qwen3-30B-A3B-FP8/` | `qwen-main` |
| `vllm-qwen-vision` | `/data/huggingface/manual/Qwen2.5-VL-7B-Instruct-FP4/` | `qwen-vision` |
| `vllm-gemma-fast` | `/data/huggingface/manual/Gemma-4-31B-IT-NVFP4/` | `gemma-fast` |

If your folder names match, no config change is needed. Otherwise, override in `.env`:

```env
# Example: use the larger Qwen3.5-122B as qwen-main
QWEN_MAIN_MODEL=/models/Qwen3.5-122B-A10B-FP8
QWEN_MAIN_MEM_UTIL=0.70

# Or point at a completely different folder name
QWEN_MAIN_MODEL=/models/my-custom-qwen-build
```

---

## 6. Bring the stack up

```bash
cd ~/tonic-nvidia-dgx
docker compose up -d
docker compose logs -f vllm-qwen-main
```

First load takes 30–90 s from NVMe (much faster than an HF download). When you see `Uvicorn running on http://0.0.0.0:8000`, test with:

```bash
curl http://localhost:8001/v1/models | jq
scripts/healthcheck.sh
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `OSError: <path>/config.json not found` | Path wrong or folder empty | Re-check folder name against `QWEN_MAIN_MODEL` |
| `RuntimeError: model.safetensors.index.json refers to missing file 'model-00003-of-00004.safetensors'` | Skipped a shard during HF download | Download the missing shard, rsync it in, re-run `verify-model.sh` |
| Container starts fine but returns garbage tokens | Missing `tokenizer.json` or wrong tokenizer files | Re-download tokenizer files from HF and rsync |
| `Permission denied` when container reads `/models/...` | Host folder isn't world-readable | On host: `chmod -R a+r /data/huggingface/manual/<model>` |
| First inference is very slow | Model files on slow disk | `df /data` — confirm it's the internal NVMe, not USB or SD card |
| Model loads but uses way too much memory | Wrong `--gpu-memory-utilization` for this model size | Adjust `QWEN_MAIN_MEM_UTIL` in `.env` (0.20 for 8B, 0.30 for 30B, 0.70 for 122B) |

---

## Recommended download-size cheat sheet

| Model | Precision | Approx download | Fits alongside others? |
|-------|-----------|----------------:|-----------------------|
| Qwen3-30B-A3B-FP8 (MoE, ~3B active) | FP8 | ~7 GB | Yes, easily |
| Qwen2.5-VL-7B-Instruct-FP4 | FP4 | ~6 GB | Yes |
| Gemma-4-31B-IT-NVFP4 | NVFP4 | ~22 GB | Yes |
| Qwen3-32B-FP8 (dense) | FP8 | ~32 GB | Yes, but slower decode |
| Qwen3.5-122B-A10B-FP8 (MoE) | FP8 | ~85 GB | Alone or with vision only |
| Qwen3-Coder-30B-A3B | BF16 | ~60 GB | Yes |
| GPT-OSS-120B (MoE) | MXFP4 | ~65 GB | Alone or with vision only |
