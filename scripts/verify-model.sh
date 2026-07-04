#!/usr/bin/env bash
# ============================================================================
# Verify a manually-downloaded HF model directory is complete and well-formed.
# Checks:
#   1. Required files present (config.json, tokenizer, generation_config)
#   2. safetensors shard index references only files that actually exist
#   3. No zero-byte files
#   4. Total size looks sane
#
# Usage:  scripts/verify-model.sh /data/huggingface/manual/Qwen3-30B-A3B-FP8
# ============================================================================
set -uo pipefail

DIR="${1:-}"
[ -n "$DIR" ] || { echo "Usage: $0 <model-dir>"; exit 2; }
[ -d "$DIR" ] || { echo "Not a directory: $DIR"; exit 2; }

bold() { printf "\033[1m%s\033[0m\n" "$*"; }
ok()   { printf "  \033[32m✓\033[0m %s\n" "$*"; }
warn() { printf "  \033[33m!\033[0m %s\n" "$*"; }
bad()  { printf "  \033[31m✗\033[0m %s\n" "$*"; FAIL=1; }

FAIL=0
cd "$DIR"

bold "→ Checking $DIR"

# --- 1. Required files ---
REQUIRED=(config.json tokenizer_config.json)
for f in "${REQUIRED[@]}"; do
  [ -f "$f" ] && ok "$f present" || bad "$f MISSING"
done

# tokenizer.json OR (vocab.json + merges.txt) is acceptable
if [ -f tokenizer.json ]; then
  ok "tokenizer.json present"
elif [ -f vocab.json ] && [ -f merges.txt ]; then
  ok "vocab.json + merges.txt present (BPE fallback)"
else
  bad "no tokenizer files (need tokenizer.json OR vocab.json + merges.txt)"
fi

[ -f generation_config.json ] && ok "generation_config.json present" || warn "generation_config.json missing (usually recoverable, but worth checking)"

# --- 2. Weight files ---
SAFETENSORS_COUNT=$(ls *.safetensors 2>/dev/null | wc -l)
BIN_COUNT=$(ls pytorch_model*.bin 2>/dev/null | wc -l)

if [ "$SAFETENSORS_COUNT" -gt 0 ]; then
  ok "$SAFETENSORS_COUNT safetensors shard(s) present"
  if [ -f model.safetensors.index.json ]; then
    ok "model.safetensors.index.json present"
    # Cross-check every referenced shard exists
    python3 - <<PYEOF
import json, os, sys
idx = json.load(open("model.safetensors.index.json"))
refs = set(idx["weight_map"].values())
missing = [f for f in refs if not os.path.exists(f)]
if missing:
    print(f"  \033[31m✗\033[0m index references {len(missing)} MISSING file(s): {missing[:5]}{'...' if len(missing)>5 else ''}")
    sys.exit(1)
print(f"  \033[32m✓\033[0m index maps to {len(refs)} shard(s), all present")
PYEOF
    [ $? -ne 0 ] && FAIL=1
  elif [ "$SAFETENSORS_COUNT" -eq 1 ]; then
    ok "single-shard model (no index needed)"
  else
    bad "multi-shard model but no model.safetensors.index.json — download is incomplete"
  fi
elif [ "$BIN_COUNT" -gt 0 ]; then
  warn "using legacy pytorch_model*.bin — vLLM prefers safetensors"
  ok "$BIN_COUNT bin shard(s) present"
else
  bad "no weight files found (*.safetensors or pytorch_model*.bin)"
fi

# --- 3. Zero-byte guard ---
EMPTY=$(find . -maxdepth 1 -type f -size 0 -printf '%f\n' 2>/dev/null)
if [ -n "$EMPTY" ]; then
  bad "zero-byte files (aborted downloads):"
  echo "$EMPTY" | sed 's/^/    /'
else
  ok "no zero-byte files"
fi

# --- 4. Size sanity ---
SIZE_H=$(du -sh . 2>/dev/null | cut -f1)
ok "total size: $SIZE_H"

echo ""
if [ "$FAIL" -eq 0 ]; then
  bold "✓ Model directory looks good."
  echo "   Point vLLM at: /models/$(basename "$DIR")   (if /data/huggingface/manual is mounted as /models)"
  exit 0
else
  bold "✗ Model directory has problems — re-download the missing files."
  exit 1
fi
