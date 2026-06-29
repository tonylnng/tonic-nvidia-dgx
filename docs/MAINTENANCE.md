# Maintenance Routine — Tonic NVIDIA DGX Spark

A predictable maintenance cadence keeps the cluster healthy and surfaces issues *before* they become incidents.

## Cadence overview

| Frequency | Owner | Action | Automation |
|-----------|-------|--------|-----------|
| **Continuous** | Docker | Service restart on failure (`restart: unless-stopped`) | Built-in |
| **Every 30 s** | LiteLLM | Backend health probe | Built-in |
| **Every 1 h** | cron | `scripts/healthcheck.sh` → log + alert | crontab |
| **Daily 02:00 HKT** | cron | `scripts/backup-db.sh` | crontab |
| **Weekly Mon 07:00 HKT** | cron | Disk report, image GC, integration smoke test | crontab |
| **Monthly** | manual / PR | vLLM image bump (see §3) | semi-auto |
| **Monthly** | manual / PR | Model weight refresh (see §4) | semi-auto |
| **Quarterly** | manual | Driver / DGX OS update, key rotation | manual |
| **Annual** | manual | Disaster-recovery drill (full restore from backup) | manual |

---

## 1. crontab installation

```cron
# /etc/cron.d/tonic-dgx
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
MAILTO=tonylnng@gmail.com

# Hourly health-check (silent if all OK; alert otherwise)
0 *  * * *  tony  /home/tony/tonic-nvidia-dgx/scripts/healthcheck.sh >> /var/log/tonic/healthcheck.log 2>&1 || echo "Tonic DGX healthcheck failed at $(date)" | mail -s "[Tonic DGX] healthcheck FAIL" tonylnng@gmail.com

# Daily DB backup
0 2  * * *  tony  /home/tony/tonic-nvidia-dgx/scripts/backup-db.sh >> /var/log/tonic/backup.log 2>&1

# Weekly maintenance window — Monday 07:00 HKT
0 7  * * 1  tony  /home/tony/tonic-nvidia-dgx/scripts/weekly-maintenance.sh >> /var/log/tonic/weekly.log 2>&1
```

`scripts/weekly-maintenance.sh` runs the disk report, prunes dangling Docker images, and re-pulls images that have moved.

---

## 2. Daily checklist (human, 5 min)

- [ ] `./scripts/healthcheck.sh` returns 0
- [ ] `docker compose ps` — all services Up / healthy
- [ ] LiteLLM UI → **Usage** — no unexpected spend anomalies
- [ ] `free -h` — `MemAvailable` ≥ 10 GB

---

## 3. Monthly — vLLM image refresh

The full version-control flow is in [VERSIONING.md](VERSIONING.md). Quick view:

```bash
# 1. Make sure you have a backup of the current pin
cp config/image-versions.env config/image-versions.env.bak

# 2. On a release/* branch, bump VLLM_IMAGE in config/image-versions.env to the new digest

# 3. Canary deploy
VLLM_IMAGE_CANARY=hellohal2064/vllm-dgx-spark-gb10:v0.18.0 \
  docker compose -f docker-compose.yml -f docker-compose.canary.yml up -d

# 4. Benchmark
./scripts/benchmark.sh qwen-main-canary
./scripts/benchmark.sh gemma-fast-canary
./scripts/benchmark.sh qwen-vision-canary

# 5. If pass → promote
./scripts/update-rolling.sh

# 6. If fail → rollback
./scripts/rollback.sh qwen-main
```

---

## 4. Monthly — Model weight refresh

Some model authors push silent re-uploads (same name, new bytes — bug fixes, tokenizer tweaks). To pick those up safely:

```bash
# 1. Pre-flight
./scripts/healthcheck.sh
./scripts/backup-db.sh

# 2. Archive the current weights so you can roll back
mkdir -p /data/huggingface/hub/_archive
for d in models--Qwen--Qwen3.5-122B-A10B-FP8 \
         models--nvidia--Qwen2.5-VL-7B-Instruct-FP4 \
         models--nvidia--Gemma-4-31B-IT-NVFP4 ; do
  cp -al /data/huggingface/hub/$d /data/huggingface/hub/_archive/${d}.$(date +%Y%m%d)
done

# 3. Re-download latest revisions
huggingface-cli download Qwen/Qwen3.5-122B-A10B-FP8 --cache-dir /data/huggingface
huggingface-cli download nvidia/Qwen2.5-VL-7B-Instruct-FP4 --cache-dir /data/huggingface
huggingface-cli download nvidia/Gemma-4-31B-IT-NVFP4 --cache-dir /data/huggingface

# 4. Restart each backend (one at a time)
./scripts/update-rolling.sh

# 5. Bench + diff
./scripts/benchmark.sh qwen-main
diff docs/benchmarks/baseline-qwen-main.txt docs/benchmarks/$(date +%Y%m%d_*)-qwen-main.txt

# 6. If regression, restore archived weights and restart:
rm -rf /data/huggingface/hub/models--Qwen--Qwen3.5-122B-A10B-FP8
mv /data/huggingface/hub/_archive/models--Qwen--Qwen3.5-122B-A10B-FP8.YYYYMMDD \
   /data/huggingface/hub/models--Qwen--Qwen3.5-122B-A10B-FP8
docker compose restart vllm-qwen-main
```

> **HF revision pinning** — to make this fully reproducible, pin the model revision in `docker-compose.yml`:
>   `--model Qwen/Qwen3.5-122B-A10B-FP8 --revision abc123def`

---

## 5. Quarterly — Driver / DGX OS

1. Schedule a maintenance window (≥30 min).
2. `./scripts/backup-db.sh && tar -czf /data/backups/config-$(date +%F).tgz config/ docker-compose*.yml .env`
3. `sudo apt update && sudo apt full-upgrade`
4. Reboot the Spark.
5. `nvidia-smi --query-gpu=driver_version,compute_cap --format=csv` — confirm.
6. `docker compose up -d`
7. `./scripts/healthcheck.sh`
8. Bump pinned image digests if any new GB10 fixes landed.

## 6. Quarterly — Key rotation

- LiteLLM `LITELLM_MASTER_KEY` → generate new value, update `.env`, `docker compose restart litellm`.
- Postgres password → ALTER USER + update `DATABASE_URL`.
- HF token → revoke old, issue new, update `.env`, restart all vLLM services.

## 7. Annual — DR drill

On a non-production day:

1. `docker compose down`
2. `docker volume rm $(docker volume ls -q | grep tonic)` (or relabel `/data/postgres`)
3. Restore from latest backup: `./scripts/restore-db.sh backups/litellm_LATEST.pgdump`
4. `docker compose up -d`
5. Re-test API keys still validate.
6. Document elapsed time → target < 30 min end-to-end.
