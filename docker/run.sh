#!/usr/bin/env bash
# Run the annotator (labeling). No GPU is used or requested.
#
#   ./docker/run.sh --help
#   ./docker/run.sh --gt-path /data/... --frames-dir /data/... --midpoints-dir /data/... \
#                   --vllm-url http://10.77.32.231:8000/v1
#
# Host /media/TrainDataset is mounted read-only at /data.
# The repo is bind-mounted over /app, so source edits need no rebuild.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-vln-annotator:latest}"
DATA="${DATA:-/media/TrainDataset}"

ENVFILE=()
if [[ -f "$REPO/keys.env" ]]; then
  ENVFILE=(--env-file "$REPO/keys.env")
else
  echo "note: $REPO/keys.env not found — API-key backends will fail." >&2
  echo "      cp keys.env.example keys.env  and fill it in." >&2
fi

mkdir -p "$REPO/outputs"

# --network host: reach both the remote vLLM and a local serve.sh on localhost.
exec docker run --rm -it \
  --network host \
  "${ENVFILE[@]}" \
  -v "$REPO":/app \
  -v "$DATA":/data:ro \
  -v "$REPO/outputs":/app/outputs \
  -w /app \
  "$IMAGE" \
  python3 annotate_dataset.py "$@"
