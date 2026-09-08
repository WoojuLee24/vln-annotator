#!/usr/bin/env bash
# Build the vln-annotator image.
#   ./docker/build.sh              # -> vln-annotator:latest
#   IMAGE=foo:dev ./docker/build.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-vln-annotator:latest}"

echo "Building $IMAGE from $REPO (base pull is ~9.7 GB on first run)"
docker build -f "$REPO/docker/Dockerfile" -t "$IMAGE" "$REPO"
echo
docker images --format '{{.Repository}}:{{.Tag}}  {{.Size}}' | grep -F "${IMAGE%%:*}" || true
