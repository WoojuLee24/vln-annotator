# Method: VLN Auto-Annotator Design

## Problem

Vision-Language Navigation (VLN) models require large annotated datasets where each
training episode has a natural-language navigation instruction paired with a 3D path.
Human annotation is expensive and slow. We want to automatically generate new instructions
for additional paths (e.g. augmented or novel scenes) that match the statistical
distribution of human-annotated R2R instructions.

---

## Key Insight: Route Notation as Structured Intermediary

The core idea is that an LLM cannot reliably write navigation instructions directly from:
- Raw 3D waypoints (it can't understand `[1.2, 0.0, -3.4]`)
- Image descriptions alone (no path structure)

Instead, we construct a **route notation string** that is:
1. **Grounded** in path geometry (actual turns, distances)
2. **Enriched** with visual features (landmarks, objects)
3. **Compact** enough for an LLM to follow as instructions

```
"straight 4m → [thru: arched doorway] → turn left at [grey pillar]
 → [pass: tall bookshelf] → turn right → Goal: the white wall"
```

The LLM's job is to expand this terse notation into fluent English.
This separation of concerns is what makes the system reliable and controllable.

---

## Design Decisions

### Decision 1: Vision is used for landmarks, not for descriptions

We extract visual features at three levels:
- **Frame descriptions** (Phase 1a): rich scene context for the start + turn viewpoints
- **Midpoint objects** (Phase 1b): the single most prominent object directly ahead
- **Turn-side landmarks** (Phase 1c): the object visible in the turn direction

Only the turn-side landmark (1c) becomes a turn anchor. The frame descriptions (1a)
are background context only — they help the LLM write appropriate vocabulary but
are NOT shown as explicit turn instructions to avoid hallucination.

### Decision 2: Probabilistic sampling calibrated to GT distribution

Human annotators only explicitly mention a turn (e.g. "turn left at the grey pillar")
in ~16.5% of episodes. They write far more implicit turns ("walk toward the kitchen").
We calibrate several probabilities to match GT statistics exactly:

- `P_ANCHOR_EPISODE = 0.231`: sets anchor rate = 16.5%
- `P_TURN_UNANCHORED = 0.42`: sets avg_turns/ep = 0.59 (exact GT match)
- `P_STOP = 0.494`: sets stop phrase rate = 50.8%
- `P_WAIT = 0.305`: sets wait phrase rate = 30.5%

The LLM's natural over-anchoring behaviour is corrected by under-sampling in the route,
and its systematic over-stop behaviour is corrected by reducing P_STOP by 1.4pp.

### Decision 3: Surgical post-processing over blanket rules

Blanket prohibitions ("NEVER write hallway") cause severe over-correction.
Instead, we apply targeted post-processing only to specific hallucination patterns:

1. **doorway_fix**: removes `through the doorway` only when NO architectural passage
   exists in the midpoint data (the LLM hallucinated it from visual similarity)
2. **walk_past_fix**: removes `walk past the X` only when NO furniture midpoint exists
3. **hallway_fix**: removes mid-route hallway motion phrases only when the goal is NOT a hallway

These are applied AFTER all other cleaning, touching only the specific problematic phrase.

### Decision 4: Few-shot examples from the same building

Including 8 same-building GT instructions as few-shot examples dramatically improves:
- Vocabulary appropriateness (furniture names match what's visible in that scene)
- Style consistency (short vs. long instructions, stop phrase style)
- Structural similarity (similar paths get similar instructions)

We rank candidates by path structural similarity (turn edit-distance + distance + waypoints).

### Decision 5: Word budget enforced in the prompt

Human R2R instructions average 26.8 words. Our baseline LLM writes 22-23 words.
We add explicit word budgets calibrated by path length:
- 4 waypoints: 13-21 words
- 5 waypoints: 24-32 words
- 6 waypoints: 27-35 words
- 7+ waypoints: 31-40 words

This brings avg_words from 22.1 to 25.8 (closing the gap from 4.7w to 1.0w).

---

## Calibration Process

Each parameter is calibrated independently by measuring its effect on the target metric:

```
P_TURN_UNANCHORED calibration:
  v44: P=0.30 → avg_turns=0.71 (too many)
  v60: P=0.45 → avg_turns=0.63 (still too many)
  v62: P=0.42 → avg_turns=0.59 ✓ (exact GT match)

P_ANCHOR_EPISODE calibration:
  Target anchor_rate = 16.5% (GT)
  LLM compliance ≈ 84.3% (only ~84% of anchored episodes get "turn at X" in output)
  So route_anchor_rate needs to be: 16.5% / 84.3% ≈ 19.6%
  Since 44% of episodes have turns: P = 19.6% / 44% ≈ 0.231

P_STOP calibration:
  LLM systematically stops 1.4pp above the configured P_STOP
  So: P_STOP = 0.508 - 0.014 = 0.494
```

---

## GT-Match Score Definition

The primary quality metric (single number to optimize):

```
anchor_err = |anchor_rate - 0.165|   (double-weighted: hardest to calibrate)
stop_err   = |stop_rate   - 0.508|
wait_err   = |wait_rate   - 0.305|

GT-match = 1.0 - (anchor_err*2 + stop_err + wait_err)
```

This aggregates three key distributional differences into one score.
A score of 1.0 means perfect match on all three distributions.
Target ≥ 0.90. Best achieved: 0.965 (v63, pending evaluation).

---

## Evolution Summary (v1 → v63)

| Version | Key change | GT-match |
|---|---|---|
| v1-v10 | Basic pipeline, text landmarks only | ~0.85 |
| v19 | Vision frames added (Phase 1a) | ~0.90 |
| v22 | Midpoint objects added (Phase 1b) | 0.92 |
| v24 | Turn-side landmarks added (Phase 1c) | 0.94 |
| v29 | Surgical [thru:]/[pass:] notation | 0.945 |
| v36 | Proceed/continue normalisation | 0.950 |
| v44 | Probabilistic turn sampling | 0.945 |
| v51 | Word budget calibration (+4w all paths) | 0.948 |
| v54 | Surgical doorway fix | 0.950 |
| v58 | Surgical walk_past fix | 0.950 |
| v59 | Surgical hallway fix | 0.952 |
| v62 | P_TURN_UNANCHORED=0.42 (exact turns) | 0.951 |
| v63 | P_ANCHOR=0.231, P_STOP=0.494 calibrated | ~0.965 (est.) |
