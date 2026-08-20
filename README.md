# VLN Auto-Annotator

Automatically generate GT-compatible navigation instructions for Vision-Language Navigation (VLN) datasets.
Given a set of episode paths (3D waypoints) and rendered scene images, the annotator produces natural-language
instructions that match the statistical distribution of human-annotated R2R data.

```
GT-match score: 0.951+   (target ≥ 0.90)
avg_turns/ep:   0.59      (GT = 0.59 ✓)
hallway%:       19.2%     (GT = 20.4%)
through_the%:   29.4%     (GT = 27.2%)
```

---

## How It Works

```
  ┌──────────────────────────────────────────────────────────────┐
  │                     INPUTS                                    │
  │  ┌─────────────────┐   ┌──────────────────┐                  │
  │  │  Episode paths   │   │  Rendered frames  │                  │
  │  │  (3D waypoints,  │   │  (per-viewpoint   │                  │
  │  │   GT JSON.gz)    │   │   JPEG/PNG imgs)  │                  │
  │  └────────┬────────┘   └────────┬─────────┘                  │
  └───────────│────────────────────│────────────────────────────┘
              │                    │
              ▼                    ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  PHASE 1 — Visual Feature Extraction (VLM, async)            │
  │                                                              │
  │  1a. Frame descriptions  — scene context at start + turns    │
  │  1b. Midpoint objects    — most prominent object ahead       │
  │  1c. Turn-side landmarks — object visible in turn direction  │
  │                                                              │
  │  All 3 sub-phases write checkpoints → resume on failure      │
  └──────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  PATH ANALYSIS — Motion Primitive Extraction                  │
  │                                                              │
  │  Converts 3D waypoints → structured motion primitives:       │
  │    [straight 4m] [left_turn 87°] [right_turn 45°] [stop]    │
  │                                                              │
  │  Detects: turns (≥30°), long straights (>2m), elevations     │
  └──────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  ROUTE BUILDER — Route Description Construction              │
  │                                                              │
  │  Merges motion primitives + visual features into a route     │
  │  notation string that the LLM reads as its instruction:      │
  │                                                              │
  │  "straight 4m → [thru: arched doorway] →                    │
  │   turn left at [grey pillar] → [pass: tall bookshelf] →     │
  │   turn right → Goal: the white wall"                        │
  │                                                              │
  │  Route notation encodes:                                     │
  │   [thru: X]        → LLM writes "through the X"             │
  │   [pass: X]        → LLM writes "walk past the X"           │
  │   turn left at [X] → LLM writes "turn left at the X"        │
  │   → [X ahead]      → LLM writes "walk to the X" (brief)     │
  └──────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  PHASE 2 — LLM Instruction Generation (async, checkpoint)    │
  │                                                              │
  │  LLM (Gemma 4-31B AWQ) generates natural-language            │
  │  instruction from route + few-shot examples (same building)  │
  │  + word-budget + stop/wait phrase                            │
  │                                                              │
  │  Input:  "Walk → turn left at [grey pillar] → Goal: door"   │
  │  Output: "Walk forward and turn left at the grey pillar.     │
  │           Stop near the white door."                         │
  └──────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  POST-PROCESSING — Surgical Distribution Fixes               │
  │                                                              │
  │  Three targeted fixes applied after LLM generation:          │
  │  1. doorway_fix   — remove hallucinated "through the doorway"│
  │  2. walk_past_fix — remove hallucinated "walk past the X"   │
  │  3. hallway_fix   — remove mid-route hallway phrases         │
  │                                                              │
  │  Plus: loop removal, preamble strip, 15+ regex normalizers   │
  └──────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  DATASET ASSEMBLY — GT-Compatible Output                     │
  │                                                              │
  │  Tokenizes instructions with GT vocabulary (word2idx_dict)   │
  │  Copies all structural fields verbatim from source episode   │
  │  Writes val_unseen_annotated.json.gz (Habitat/Isaac ready)   │
  └──────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install

```bash
git clone <repo>
cd vln_annotator
pip install -e .
pip install openai   # async LLM client
```

### 2. Start a vLLM inference server

The annotator uses an OpenAI-compatible endpoint. Any vLLM-served multimodal model works.
Tested with `cyankiwi/gemma-4-31B-it-AWQ-4bit`:

```bash
vllm serve cyankiwi/gemma-4-31B-it-AWQ-4bit \
  --tensor-parallel-size 2 \
  --max-model-len 4096
```

### 3. Prepare inputs

You need three things:
1. **GT JSON.gz** — original dataset file (provides episode paths + vocabulary)
2. **Rendered frames** — images at start viewpoint + turn viewpoints + midpoints
3. **Midpoints metadata** — `midpoints.json` per episode

Directory layout expected:

```
rendered_frames/
  episode_000001/
    start.jpg
    turn_1.jpg
    turn_2.jpg
    turn_side_turn_1.jpg     ← object visible in turn direction
    turn_side_turn_2.jpg
midpoints/
  episode_000001/
    midpoints.json           ← {"frames": [{"label": "turn_1", "path": "img.jpg"}]}
    img.jpg
```

### 4. Run

```bash
python annotate_dataset.py \
  --gt-path /data/val_unseen_patched.json.gz \
  --frames-dir /data/rendered_frames \
  --midpoints-dir /data/midpoints \
  --output-name val_unseen_annotated.json.gz
```

Or in Python:

```python
import asyncio
from pathlib import Path
from vln_annotator import AnnotatorConfig, run

cfg = AnnotatorConfig(
    vllm_base_url="http://localhost:8000/v1",
    vllm_model="cyankiwi/gemma-4-31B-it-AWQ-4bit",
    output_name="val_unseen_annotated.json.gz",
)

asyncio.run(run(
    gt_path="/data/val_unseen_patched.json.gz",
    rendered_frames_dir=Path("/data/rendered_frames"),
    midpoints_dir=Path("/data/midpoints"),
    cfg=cfg,
))
```

---

## Configuration

All parameters are in `AnnotatorConfig` (see `vln_annotator/config.py`):

| Parameter | Default | Description |
|---|---|---|
| `vllm_base_url` | `http://10.77.32.231:8000/v1` | vLLM server URL |
| `vllm_model` | `cyankiwi/gemma-4-31B-it-AWQ-4bit` | Model to use |
| `temperature` | `0.3` | LLM sampling temperature |
| `max_new_tokens` | `256` | Max tokens per instruction |
| `concurrency_vision` | `8` | Parallel Phase 1 vision calls |
| `concurrency_instruct` | `12` | Parallel Phase 2 LLM calls |
| `p_stop` | `0.494` | P(stop ending) — calibrated to GT 50.8% |
| `p_wait` | `0.305` | P(wait ending) — calibrated to GT 30.5% |
| `p_anchor_episode` | `0.231` | P(episode gets anchored turn) — calibrated to GT 16.5% |
| `p_turn_unanchored` | `0.42` | Keep-rate for unanchored turns — calibrated to GT 0.59 turns/ep |
| `p_pass_marker` | `0.50` | P(midpoint becomes `[pass:]` marker) |
| `p_thru_marker` | `0.85` | P(midpoint becomes `[thru:]` marker) |

### Resuming interrupted runs

Each phase writes a checkpoint automatically. Re-run the same command and
completed phases are skipped:

```bash
# Phase 1 already done — skip vision API calls:
python annotate_dataset.py ... --skip-phase1

# Phase 2 already done — only reassemble output:
python annotate_dataset.py ... --skip-phase1 --skip-phase2
```

---

## Output Format

The output `.json.gz` is fully compatible with Habitat-Lab and Isaac-Lab eval configs:

```json
{
  "episodes": [
    {
      "episode_id": 1234,
      "scene_id": "mp3d/17DRP5sb8fy/17DRP5sb8fy",
      "start_position": [1.2, 0.0, -3.4],
      "start_rotation": [0, 0, 0, 1],
      "reference_path": [[1.2, 0.0, -3.4], ...],
      "goals": [...],
      "instruction": {
        "instruction_text": "Walk forward and turn left at the grey pillar. Stop near the white door.",
        "instruction_tokens": [42, 17, 8, ...]
      },
      "info": {"geodesic_distance": 7.34}
    }
  ],
  "instruction_vocab": { ... }
}
```

---

## Statistics

After generation the annotator prints a comparison against GT distribution:

```
Metric                    Generated       GT
-----------------------------------------------
  avg_words                    25.8     26.8
  stop_pct                     51.4     50.8
  wait_pct                     30.5     30.5
  anchor_pct                   15.0     16.5
  avg_turns_per_ep             0.59     0.59   ← exact match
  eps_with_turn_pct            48.8     41.3
  through_the_pct              29.4     27.2
  walk_past_the_pct            14.4      8.7
  hallway_pct                  19.2     20.4
  ...

  dup_stop:               0   (target: 0)
  GT-match score:     0.951   (target ≥ 0.90)
```

The **GT-match score** is `1.0 - (anchor_err×2 + stop_err + wait_err)` where each
`err = |generated_rate - gt_rate|`. A score ≥ 0.90 indicates good calibration.

---

## Project Structure

```
vln_annotator/
├── vln_annotator/
│   ├── __init__.py          — package exports
│   ├── config.py            — AnnotatorConfig (all hyperparameters)
│   ├── path_analysis.py     — 3D waypoints → motion primitives
│   ├── route_builder.py     — primitives + vision → route notation string
│   ├── vision.py            — Phase 1 (a/b/c) VLM frame description
│   ├── llm_backend.py       — async OpenAI-compatible LLM client
│   ├── instruction_gen.py   — Phase 2 generation + cleaning + statistics
│   ├── post_process.py      — surgical post-processing (3 targeted fixes)
│   ├── dataset_assembler.py — VLNTokenizer + GT-format episode assembly
│   └── pipeline.py          — end-to-end orchestration
├── annotate_dataset.py      — CLI entry point
├── docs/
│   ├── ARCHITECTURE.md      — detailed technical walkthrough
│   └── METHOD.md            — method design decisions
├── examples/                — example scripts
└── pyproject.toml
```

---

## Citation

This annotator was developed as part of the ChronoNav / InternNav VLN research project.
If you use the annotated datasets or this code in your research, please cite accordingly.
