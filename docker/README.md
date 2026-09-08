# Docker

One image, two roles.

| role | command | GPU |
|---|---|---|
| **labeling** (client) | `./docker/run.sh …` | not used — no `--gpus` flag is passed |
| **serving** (server) | `./docker/serve.sh <model>` | required |

The annotator only needs `openai` + `numpy`: it talks to an OpenAI-compatible
endpoint over HTTP and never loads a model in-process. So the two roles could
live in separate images. They don't, because one image is less to maintain —
and the client still runs GPU-free inside the vLLM image, since nothing in
that code path imports torch.

The cost is disk: the base `vllm/vllm-openai:v0.28.0` is ~9.7 GB.

## Quick start

```bash
cp keys.env.example keys.env      # fill in HF_TOKEN / API keys as needed
./docker/build.sh

# 1. does the image work at all?
./docker/run.sh --help

# 2. label against the already-running remote vLLM (Gemma 31B) — no GPU, no download
./docker/run.sh \
  --gt-path       /data/<...>/noeun_gt.json.gz \
  --frames-dir    /data/<...>/annotator_export/frames \
  --midpoints-dir /data/<...>/annotator_export/midpoints \
  --vllm-url      http://10.77.32.231:8000/v1 \
  --vllm-model    cyankiwi/gemma-4-31B-it-AWQ-4bit \
  --n-episodes 1

# 3. serve Cosmos locally, then point the client at it
./docker/serve.sh nvidia/Cosmos-Reason2-2B        # smallest — verifies sm_120 first
curl -s http://localhost:8100/v1/models | head -c 200

./docker/run.sh ... --vllm-url http://localhost:8100/v1 --vllm-model cosmos
```

`serve.sh` always publishes the alias **`cosmos`**, so the client passes
`--vllm-model cosmos` no matter which checkpoint is loaded.

## Mounts

| host | container | why |
|---|---|---|
| repo root | `/app` | source is bind-mounted, so edits need no rebuild |
| `/media/TrainDataset` (`$DATA`) | `/data` (ro) | datasets and the gs_vlnpe export |
| `repo/outputs` | `/app/outputs` | annotated `.json.gz` + phase checkpoints |
| `/media/TrainDataset/hf_cache` (`$HF_CACHE`) | `/hf` | model weights stay off the container layer |

Both scripts use `--network host` so the client can reach the remote vLLM and
a local `serve.sh` on `localhost` without extra plumbing.

## Environment knobs

| var | default | notes |
|---|---|---|
| `IMAGE` | `vln-annotator:latest` | |
| `DATA` | `/media/TrainDataset` | `run.sh` |
| `HF_CACHE` | `/media/TrainDataset/hf_cache` | `serve.sh` |
| `GPU_UTIL` | `0.60` | `serve.sh`. Leaves room for a co-resident Isaac Sim job; raise to ~0.85 when the card is free |
| `MAX_LEN` | `8192` | `serve.sh`. **Set `4096` when A/B-ing against the remote Gemma** — it serves `max_model_len=4096`, and unequal prompt budgets make the comparison meaningless |
| `PORT` | `8100` | `serve.sh` |

`--limit-mm-per-prompt image:3` is fixed in `serve.sh` to match
`llm_backend.build_vision_message`, which already truncates to
`image_paths[:3]`. Matching it server-side turns a silent overflow into an
explicit error.

## Model sizing on one 32 GB card

| model | arch | gated | bf16 | fits |
|---|---|:--:|---|---|
| `nvidia/Cosmos-Reason1-7B` | `qwen2_5_vl` | no | 16.6 GB | yes |
| `nvidia/Cosmos-Reason2-2B` | `qwen3_vl` | auto | ~5 GB | yes — cheapest smoke test |
| `nvidia/Cosmos-Reason2-8B` | `qwen3_vl` | auto | ~17 GB | yes |
| `nvidia/Cosmos-Reason2-32B` | `qwen3_vl` | auto | ~64 GB | no (bf16); would need 4-bit |

`gated=auto` means accepting the terms on the model page grants access
immediately — there is no review queue.

## Unverified

**Whether `vllm/vllm-openai:v0.28.0` ships sm_120 kernels.** This host is an
RTX 5090 (compute capability 12.0, Blackwell). The base image is a CUDA 12.9
build so it should, but that has not been run yet — start with
`./docker/serve.sh nvidia/Cosmos-Reason2-2B` (~5 GB) so the answer costs a
short download rather than 17 GB. If it fails, in order: the `cu129-nightly`
tag, then forcing `--dtype bfloat16`, then a source build. Labeling against
the remote endpoint is unaffected either way.
