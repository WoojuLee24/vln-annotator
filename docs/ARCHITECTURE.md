# Architecture

Technical walkthrough of the VLN Auto-Annotator pipeline.

---

## Module Map

```
annotate_dataset.py          ← CLI
    └─ pipeline.run()
         ├─ vision.describe_frames()          Phase 1a
         ├─ vision.classify_midpoints()       Phase 1b
         ├─ vision.describe_turn_sides()      Phase 1c
         │
         ├─ path_analysis.analyze_path()      Path decomposition
         ├─ route_builder.build_route()       Route notation
         ├─ instruction_gen.build_instruction_prompt()
         │
         ├─ instruction_gen.generate_instructions()  Phase 2
         ├─ instruction_gen.postprocess_instruction()
         │
         └─ dataset_assembler.assemble_dataset()     Output
```

---

## Phase 1 — Visual Feature Extraction

### 1a. Frame Descriptions

For each episode, the Habitat renderer produces one JPEG per key viewpoint:
`start.jpg`, `turn_1.jpg`, `turn_2.jpg`, `turn_3.jpg`.

The VLM is asked:
> "Describe this indoor scene for a navigation instruction. In 1-2 sentences cover: the room type,
> dominant objects, and the most prominent landmark visible. Be concrete and specific (colors, materials)."

Output is stored in a checkpoint dict: `{episode_id: {"start": "...", "turn_1": "..."}}`

### 1b. Midpoint Classification

At each intermediate waypoint the renderer captures an image looking forward.
A `midpoints.json` file lists these frames with labels `turn_1`, `turn_2`, etc.

The VLM is asked:
> "Identify the single most prominent furniture item or object visible directly ahead.
> Reply with a noun phrase only (2-5 words). If only walls or floors are visible, reply: 'open space'."

Raw responses are cleaned with `_clean_midpoint()`:
- Strip article prefixes (`a`, `an`, `the`)
- Reject generic responses (`open space`, `wall`, `hallway`)
- Limit to 2-6 words

### 1c. Turn-Side Descriptions

For each turn viewpoint, a separate image captures the object visible in the turn direction.
The VLM identifies this object as a potential turn anchor:

> "Look at the object or landmark visible in the direction the robot is about to turn.
> Give a 2-4 word noun phrase identifying this object."

This becomes the `[landmark]` bracket in the route notation.

---

## Path Analysis

`path_analysis.analyze_path()` converts a 3D waypoint list into motion primitives:

**Input:**
```python
reference_path = [[x1,y1,z1], [x2,y2,z2], ...]
start_rotation = [qx, qy, qz, qw]  # Habitat Y-up quaternion
```

**Algorithm:**
1. Compute XZ-plane heading at each waypoint pair using `atan2(dx, -dz)`
2. At each step, compute signed angular change from previous heading
3. Threshold at 30°: turns ≥ 30° emit a `left_turn` or `right_turn` primitive
4. Accumulated straight distance emits a `straight` primitive (only if > 2m)
5. Y-axis changes > 0.3m emit an `elevation` primitive

**Output primitives:**
```python
[
  {"type": "straight", "distance_m": 4.2},
  {"type": "left_turn", "angle_deg": 87.0, "sharp": True},
  {"type": "straight", "distance_m": 2.1},
  {"type": "right_turn", "angle_deg": 45.0},
  {"type": "stop"},
]
```

---

## Route Builder

`route_builder.build_route()` converts primitives + visual features into a route string:

```
"straight 4m → [thru: arched doorway] → turn left at [grey pillar]
 → [pass: tall bookshelf] → turn right → Goal: the white wall"
```

**Encoding conventions:**
| Route token | LLM behaviour |
|---|---|
| `turn left at [X]` | "turn left at the X" |
| `turn right at [X] → [Y ahead]` | "turn right at the X. Walk to the Y." |
| `[thru: X]` | "through the X" |
| `[pass: X]` | "walk past the X" |
| `turn left` (bare) | "turn left" or "turn left and walk forward" |

### Calibrated sampling probabilities

Several probabilities control how often each route feature appears.
These are calibrated against GT statistics:

| Param | Value | Target metric | GT value |
|---|---|---|---|
| `P_TURN_UNANCHORED` | 0.42 | avg_turns/ep | 0.59 |
| `P_ANCHOR_EPISODE` | 0.231 | anchor_rate | 16.5% |
| `P_STOP` | 0.494 | stop_rate | 50.8% |
| `P_WAIT` | 0.305 | wait_rate | 30.5% |
| `P_PASS_MARKER` | 0.50 | walk_past_pct | ~8.7% |
| `P_THRU_MARKER` | 0.85 | through_the_pct | ~27.2% |

---

## Phase 2 — LLM Instruction Generation

The full prompt has four sections:

```
{SYSTEM_PROMPT}   ← rules: room names, turn anchors, style, endings

Same-building examples:   ← top-K structurally-similar GT instructions
  1. "Walk forward..."
  2. "Turn left..."

Write ONE instruction for:
  Route: straight 4m → turn left at [grey pillar] → ...
  Start: hallway (starting in a bright hallway with brown doors)
  Goal: the white double doors

Route has 1 bracketed landmark — use it naturally.
LENGTH: Write 27-35 words total.
End: "Stop near the white double doors."
Start with: "Walk"

Walk
```

The LLM receives the `Start with: "Walk"\n\nWalk` suffix so it continues
directly into the instruction body without preamble.

### Few-Shot Examples

`ScenePathIndex` ranks candidate GT episodes from the same building by
structural path similarity (turn edit-distance + distance + waypoint count).
Top-8 are included as examples. This ensures context-appropriate vocabulary.

---

## Post-Processing Chain

After LLM generation, `postprocess_instruction()` applies these steps:

1. **Strip preamble** — remove "Instruction:", "Here is", etc.
2. **Enforce start verb** — ensure instruction begins with the chosen verb
3. **Remove route artifacts** — strip `[brackets]` and `→` arrows
4. **Phrase normalisation** (15 regex substitutions):
   - `passing through the X` → `through the X`
   - `passing the X` → `walk past the X`
   - `go past the X` → `walk past the X`
   - material adjectives stripped
   - generic walk-past objects → `walk forward`
   - `through the hallway` → `down the hallway`
   - `through the [room]` → `into the [room]`
   - `Proceed [forward/to/past]` → `Walk [forward/to/past]`
   - `continue [through/into/along]` → `walk [through/into/along]`
   - duplicate motion phrases collapsed
   - fragment sentences merged with previous sentence
5. **Loop removal** — detect repeated 5-gram, truncate at first repeat
6. **Surgical fixes** (3 targeted, via `post_process.py`):
   - `doorway_fix` — remove `through the doorway` when no arch/door midpoint exists
   - `walk_past_fix` — remove `walk past the X` when no furniture midpoint exists
   - `hallway_fix` — remove mid-route hallway phrases when goal ≠ hallway
7. **Quality check** — reject if < 5 words, > 80 words, or still looping
8. **Enforce stop phrase** — remove any LLM-generated stop, append chosen stop/wait

---

## GT-Match Score

The quality metric used to calibrate annotator versions:

```
anchor_err = |anchor_rate - 0.165|   # anchor_rate = % episodes with turn anchor
stop_err   = |stop_rate   - 0.508|
wait_err   = |wait_rate   - 0.305|

GT-match = 1.0 - (anchor_err×2 + stop_err + wait_err)
```

Target: ≥ 0.90. Best achieved: 0.965 (v63).

The anchor_err is double-weighted because it is the hardest to calibrate
(LLM compliance is ~84%, requiring P_ANCHOR_EPISODE = 0.231 to hit 16.5% target).

---

## Anti-Patterns Learned

These approaches were tried and failed — avoid repeating them:

### Blanket prohibitions cause over-correction
Using "NEVER write hallway" collapsed hallway frequency from 26% to 6.6% (v61).
GT is 20.4%. **Use specific-example prohibitions instead** ("Do NOT write
'walk through the hallway', 'walk down the hallway'…") + surgical post-processing.

### P_PASS_MARKER / P_THRU_MARKER do NOT control LLM output frequency
The LLM generates "walk past the X" / "through the X" from visual context,
independent of whether a `[pass:]` / `[thru:]` marker was present in the route.
Changing these probabilities only affects the route notation, not the final instruction.
Use surgical post-processing to control output frequency.

### LLM variance on fresh generation
Re-running Phase 2 with identical prompts introduces ±0.003–0.010 GT-match variance
due to LLM non-determinism. Reuse checkpoints when parameters haven't changed.
