#!/usr/bin/env bash
# Serve a Cosmos Reason model locally over an OpenAI-compatible API.
#
#   ./docker/serve.sh                                  # Cosmos-Reason2-2B (smallest; sm_120 check)
#   ./docker/serve.sh nvidia/Cosmos-Reason1-7B         # ungated, 16.6 GB
#   GPU_UTIL=0.85 MAX_LEN=4096 ./docker/serve.sh nvidia/Cosmos-Reason2-8B
#
# Served under the alias `cosmos`, so the client always passes
# --vllm-model cosmos regardless of which checkpoint is loaded.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-vln-annotator:latest}"
MODEL="${1:-nvidia/Cosmos-Reason2-2B}"

# Fraction of the card vLLM may use for weights + KV cache.
# Beware: too low does not degrade, it fails outright with
#   "No available memory for the cache blocks"
# 0.55 on a 32 GB card is ~18 GB, and Cosmos-Reason1-7B weights alone are
# 15.45 GiB -- nothing left for the cache. Lower this only for co-residency,
# and keep (total x GPU_UTIL) at least ~6 GB above the weight size.
GPU_UTIL="${GPU_UTIL:-0.85}"
# The remote Gemma serves max_model_len=4096. Match it when A/B-ing backends,
# otherwise the two runs do not see the same prompt budget.
MAX_LEN="${MAX_LEN:-8192}"
PORT="${PORT:-8100}"

# Keep the ~16 GB of weights out of the container layer.
HF_CACHE="${HF_CACHE:-/media/TrainDataset/hf_cache}"
mkdir -p "$HF_CACHE"

# -it only when a TTY is actually attached, so this works from scripts and CI.
TTY=(-i)
[[ -t 0 && -t 1 ]] && TTY=(-it)

ENVFILE=()
[[ -f "$REPO/keys.env" ]] && ENVFILE=(--env-file "$REPO/keys.env")

echo "serving $MODEL  as 'cosmos'  port=$PORT  gpu_util=$GPU_UTIL  max_len=$MAX_LEN"
echo "hf cache: $HF_CACHE"

exec docker run --rm "${TTY[@]}" \
  --gpus all --network host --ipc host \
  "${ENVFILE[@]}" \
  -v "$HF_CACHE":/hf \
  "$IMAGE" \
  vllm serve "$MODEL" \
    --served-model-name cosmos \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_LEN" \
    --limit-mm-per-prompt '{"image":3}' \
    --port "$PORT"
