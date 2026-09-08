# Backends

One pipeline, five providers. The annotator never loads a model in-process —
it speaks HTTP to whatever `--backend` selects, so switching providers is a
flag, not a code change.

| `--backend` | endpoint | image envelope | key |
|---|---|---|---|
| `vllm` (default) | `--vllm-url` | OpenAI `image_url` + data URI | none (`EMPTY`) |
| `openai` | `https://api.openai.com/v1` | same | `OPENAI_API_KEY` |
| `gemini` | Google's OpenAI-compatible surface | same | `GEMINI_API_KEY` |
| `anthropic` | Anthropic Messages API | `source.base64` — **differs** | `ANTHROPIC_API_KEY` |
| `dry` | none | none | none |

Keys come from `keys.env` (gitignored; `cp keys.env.example keys.env`).
`docker/run.sh` passes it with `--env-file`, so nothing is baked into the image.

---

## 1. Dry run — start here, costs nothing

Makes **zero** network calls. Writes every prompt and the image paths it would
have sent to `outputs/dry_run/prompts.jsonl`.

```bash
./docker/run.sh \
  --gt-path       /data/.../noeun_gt.json.gz \
  --frames-dir    /data/.../annotator_export/frames \
  --midpoints-dir /data/.../annotator_export/midpoints \
  --dry-run --n-episodes 1
```

Then read the manifest before spending anything:

```bash
python3 - <<'PY'
import json
rows = [json.loads(l) for l in open('outputs/dry_run/prompts.jsonl')]
print('calls          :', len(rows))
print('images         :', sum(r['n_images'] for r in rows))
print('missing images :', sum(len(r['missing_images']) for r in rows))
print('prompt chars   : median', sorted(r['prompt_chars'] for r in rows)[len(rows)//2])
PY
```

`missing_images` is the payoff: a frame the exporter failed to write shows up
here for free, instead of as a silently degraded instruction after 200 paid
vision calls.

## 2. Remote vLLM (Gemma 31B) — already running, no GPU, no download

```bash
./docker/run.sh ... \
  --vllm-url   http://10.77.32.231:8000/v1 \
  --vllm-model cyankiwi/gemma-4-31B-it-AWQ-4bit
```

Serves `max_model_len=4096`. Keep that in mind when comparing against a local
model — see §3.

## 3. Local Cosmos Reason

Terminal A (serves under the alias `cosmos`):

```bash
./docker/serve.sh nvidia/Cosmos-Reason2-2B          # ~5 GB — verifies sm_120 cheaply
./docker/serve.sh nvidia/Cosmos-Reason1-7B          # ungated, 16.6 GB
MAX_LEN=4096 ./docker/serve.sh nvidia/Cosmos-Reason2-8B   # match Gemma for a fair A/B
```

Terminal B:

```bash
curl -s http://localhost:8100/v1/models | head -c 200
./docker/run.sh ... --vllm-url http://localhost:8100/v1 --vllm-model cosmos
```

`Cosmos-Reason2-*` is `gated=auto`: accepting the terms on the model page
grants access immediately, then `HF_TOKEN` in `keys.env` is enough.

## 4. Commercial APIs — cap the run

Vision calls dominate the bill, and a full 20-episode run is roughly **200
vision calls plus 20 instruction calls**. Use these for a handful of episodes
and let vLLM do the volume.

```bash
./docker/run.sh ... --backend openai    --vllm-model gpt-4o        --n-episodes 1 --max-calls 20
./docker/run.sh ... --backend gemini    --vllm-model gemini-2.5-pro --n-episodes 1 --max-calls 20
./docker/run.sh ... --backend anthropic --vllm-model claude-sonnet-5 --n-episodes 1 --max-calls 20
```

`--max-calls` is checked **before the first request**, so an over-budget run
sends nothing and costs nothing:

```
CallBudgetExceeded: 217 calls requested but --max-calls=20. Nothing was sent.
```

Without a cap on a metered backend the banner prints a warning. Current
per-token prices belong on the provider's pricing page, not in this file —
each run prints `calls=` and `images=` so you can multiply.

## 5. Comparing backends

Give each run its own output name and checkpoint dir, or the resumable
checkpoints from one provider will be reused by the next:

```bash
for B in "vllm cosmos" "vllm cyankiwi/gemma-4-31B-it-AWQ-4bit"; do
  set -- $B
  ./docker/run.sh ... --backend "$1" --vllm-model "$2" \
    --output-name      "noeun_$(echo "$2" | tr '/:' '__').json.gz" \
    --checkpoints-dir  "outputs/checkpoints/$(echo "$2" | tr '/:' '__')"
done
```

**Hold the prompt budget equal.** The remote Gemma is capped at 4096 tokens;
serving a local model at 8192 lets it see more and makes the comparison
measure the wrong thing. Pass `MAX_LEN=4096` to `serve.sh` when A/B-ing.

And keep the deterministic template generator in the comparison as the floor.
A model that cannot beat a fixed grammar is not adding anything, and without
that row there is no way to tell.
