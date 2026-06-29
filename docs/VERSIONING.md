# Version Control Strategy — vLLM Images & Stack

This document codifies how we pin, promote, and roll back container images for the Tonic NVIDIA DGX Spark (GB10) stack. The goal is **reproducible production deployments with sub-minute rollback**.

---

## 1. Why we don't use `:latest`

| Risk | Reality on GB10 |
|------|-----------------|
| Tag drift | vLLM publishes `:latest` daily; same tag, different bytes. |
| Architecture breakage | Most upstream vLLM images target x86 + SM 10. GB10 (aarch64 + SM 12.1) needs a special build that only some maintainers update. |
| Known-bad versions | vLLM 0.15.x is broken on SM 12.1. `latest` silently moves you into and out of broken zones. |
| Quantization-kernel ABI | NVFP4 / FP8 kernels change shape between minor versions; weights cached on disk can become incompatible. |

Conclusion: pin to **immutable digests**, never tags.

---

## 2. The three layers of identity

For every container image we track:

| Layer | Example | Stored in | Mutability |
|-------|---------|-----------|------------|
| **Tag** | `hellohal2064/vllm-dgx-spark-gb10:v0.17.1` | code review notes | mutable |
| **Digest** | `sha256:abc123…` | `config/image-versions.env` | immutable |
| **Channel** | `stable` / `canary` / `rollback` | git branch | promoted by CI/PR |

`docker-compose.yml` references **only the digest**.

```bash
# Resolve a tag to its digest:
docker buildx imagetools inspect hellohal2064/vllm-dgx-spark-gb10:v0.17.1 \
  --format '{{json .Manifest}}' | jq -r '.digest'
# → sha256:abc123...
```

Then in `config/image-versions.env`:

```env
VLLM_IMAGE=hellohal2064/vllm-dgx-spark-gb10@sha256:abc123...
```

---

## 3. Branching model

```
main            ← production; protected; deploys to Spark on tag push
develop         ← integration; receives PRs from feature/*, release/*
release/<ver>   ← bump candidates undergoing UAT
hotfix/<ver>    ← emergency patches; merged with `--no-ff` into main and develop
feature/<topic> ← config-only or doc-only changes
```

Tagging:

- Stack releases use calendar versioning: `vYYYY.MM.DD` (e.g. `v2026.06.30`).
- vLLM-only bumps use a separate prefix: `vllm-2026.06.30`.
- LiteLLM-only bumps: `litellm-2026.06.30`.

---

## 4. Promotion gate

1. **Open** `release/vllm-0.18.0`.
2. **Bump** only the relevant digest in `config/image-versions.env`. Keep the previous value in `config/image-versions.env.bak`.
3. **Canary deploy**:
   ```bash
   VLLM_IMAGE_CANARY=hellohal2064/vllm-dgx-spark-gb10:v0.18.0 \
     docker compose -f docker-compose.yml -f docker-compose.canary.yml up -d
   ```
4. **Smoke test** — call `qwen-main-canary` (port 8011) through LiteLLM with a canary virtual key.
5. **Benchmark** — `scripts/benchmark.sh qwen-main-canary` and diff against `docs/benchmarks/baseline-qwen-main.txt`.
6. **Soak** — leave canary running for ≥1 hour under representative load.
7. **Acceptance criteria**:
   - p50 decode tokens/s within ±5 % of baseline.
   - p95 latency not more than +10 %.
   - Zero CUDA errors / OOM / Triton kernel-launch failures.
   - Tool-call parser still works (Hermes for Qwen, gemma4 for Gemma).
8. **Promote** — merge `release/*` into `main`, tag, and run `scripts/update-rolling.sh`.
9. **Tear down canary** — `docker compose -f docker-compose.canary.yml down`.

---

## 5. Rollback

Pre-conditions: `config/image-versions.env.bak` was saved before the bump.

```bash
./scripts/rollback.sh qwen-main         # rolls back one backend
./scripts/rollback.sh                   # rolls back all three
```

Rollback takes one model-load cycle (60–180 s on GB10) per backend, and is performed one backend at a time so the other two keep serving.

---

## 6. Local registry mirror

To insulate against upstream image disappearance and to speed re-pulls on the Spark:

```bash
# Tag and push to private GHCR
docker tag hellohal2064/vllm-dgx-spark-gb10@sha256:abc... \
  ghcr.io/tonylnng/vllm-dgx-spark-gb10:v0.17.1
docker push ghcr.io/tonylnng/vllm-dgx-spark-gb10:v0.17.1
```

Keep the **last 3 versions** in GHCR. Older ones can be deleted to save quota.

---

## 7. Audit & evidence

Every deployment is reproducible from:

- The git commit (`main` HEAD at deploy time)
- `config/image-versions.env` (digests at that commit)
- `docs/benchmarks/<stamp>-<alias>.txt` (UAT evidence)

This triad is enough to recreate any past state, even after Hugging Face revisions move or upstream Docker tags are deleted.
