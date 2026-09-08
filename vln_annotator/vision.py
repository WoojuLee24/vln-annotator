"""
VLN Auto-Annotator — Vision Extraction (Phase 1)

Three vision sub-phases:
  1a. per_frame_describe  — rich scene description at start + turn viewpoints
  1b. midpoint_classify   — object/landmark at each intermediate waypoint
  1c. turn_side_describe  — object visible to the turn direction at each turn
"""
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .domains import get_domain, quoted
from .llm_backend import batch_call_async, image_to_base64


# ── Prompts ───────────────────────────────────────────────────────────────────
#
# ORIGINAL PROMPTS — kept verbatim, do not edit.
# These are the exact strings the published calibration (GT-match 0.951,
# avg_turns/ep 0.59, hallway 19.2%) was measured against. New prompts go
# BELOW as separate functions, never by editing these, so that "what changed"
# is always one diff away and the original can be re-run as a control.

_FRAME_PROMPT = (
    "Describe this indoor scene for a navigation instruction. "
    "In 1-2 sentences cover: the room type, dominant objects, and the most "
    "prominent landmark visible. Be concrete and specific (colors, materials). "
    "Example: 'Starting in a bright hallway with brown wooden double doors on "
    "the right. The polished stone floor leads forward.'"
)

_MIDPOINT_PROMPT = (
    "A robot navigating indoors has just stepped to this position. "
    "Identify the single most prominent furniture item or object visible directly ahead. "
    "Reply with a noun phrase only (2-5 words), e.g.: 'dark wooden dining table', "
    "'grey upholstered sofa', 'white marble fireplace', 'tall wooden bookshelf'. "
    "If only walls or floors are visible with no distinct furniture, reply: 'open space'. "
    "Reply with ONLY the noun phrase, nothing else."
)

_TURN_SIDE_PROMPT = (
    "Look at the object or landmark visible in the direction the robot is about to turn. "
    "Give a 2-4 word noun phrase identifying this object (e.g. 'grey stone pillar', "
    "'brown wooden cabinet', 'white kitchen counter'). "
    "If nothing distinctive is visible, reply: 'nothing'. "
    "Reply with ONLY the noun phrase."
)

# ── Domain-aware prompts — ADDED ──────────────────────────────────────────────
#
# Same wording as the originals above, with two things made variable:
#   1. the example vocabulary (originals hardcode house furniture, which gave a
#      subway station no word it could use) -> comes from domains.py
#   2. the turn direction (the original asked for "the direction the robot is
#      about to turn" without ever saying which side that was)
#
# _assert_house_parity() below guarantees domain "house" reproduces the
# originals character for character, so the calibration still describes it.

def _frame_prompt(dom) -> str:
    return (
        f"Describe this {dom.setting} scene for a navigation instruction. "
        "In 1-2 sentences cover: the room type, dominant objects, and the most "
        "prominent landmark visible. Be concrete and specific (colors, materials). "
        f"Example: '{dom.frame_example}'"
    )


def _midpoint_prompt(dom) -> str:
    return (
        f"A robot navigating {dom.navigating} has just stepped to this position. "
        "Identify the single most prominent furniture item or object visible directly ahead. "
        f"Reply with a noun phrase only (2-5 words), e.g.: {quoted(dom.midpoint_examples)}. "
        "If only walls or floors are visible with no distinct furniture, reply: 'open space'. "
        "Reply with ONLY the noun phrase, nothing else."
    )


def _turn_side_prompt(dom, direction: str = None) -> str:
    known = direction in ("left", "right")
    where = (f"on the {direction}-hand side of this view" if known
             else "in the direction the robot is about to turn")
    lead = f"The robot is about to turn {direction}. " if known else ""
    return lead + (
        f"Look at the object or landmark visible {where}. "
        f"Give a 2-4 word noun phrase identifying this object (e.g. {quoted(dom.landmark_examples)}). "
        "If nothing distinctive is visible, reply: 'nothing'. "
        "Reply with ONLY the noun phrase."
    )


def _assert_house_parity() -> None:
    """
    Fail loudly at import if the "house" profile has drifted from the originals.

    Silent drift is the real hazard here: the run would still succeed and the
    output would still look plausible, while no longer being the thing the
    calibration numbers describe.
    """
    from .domains import HOUSE
    for name, original, rebuilt in (
        ("_FRAME_PROMPT", _FRAME_PROMPT, _frame_prompt(HOUSE)),
        ("_MIDPOINT_PROMPT", _MIDPOINT_PROMPT, _midpoint_prompt(HOUSE)),
        ("_TURN_SIDE_PROMPT", _TURN_SIDE_PROMPT, _turn_side_prompt(HOUSE, None)),
    ):
        if original != rebuilt:
            raise RuntimeError(
                f"domain 'house' no longer reproduces {name}. "
                f"Either revert the change or move it into a new domain profile.\n"
                f"  original: {original!r}\n  rebuilt : {rebuilt!r}"
            )


_assert_house_parity()


_REJECT_MID_RE = re.compile(
    r'^(open\s+space|plain|featureless|empty|nothing|no\s+furniture|a\s+room|the\s+room|'
    r'floor|ceiling|wall|walls|corridor|hallway without|bare\s)',
    re.IGNORECASE,
)


# ── Helper cleaners ───────────────────────────────────────────────────────────

def _clean_midpoint(raw: str) -> Optional[str]:
    s = raw.strip().strip('"\'')
    for prefix in ("The object is", "I see", "I can see", "The most prominent",
                   "Based on", "In this image", "The item is", "There is"):
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):].strip(" :-")
    s = re.sub(r'^(a|an|the)\s+', '', s, flags=re.IGNORECASE).strip()
    s = re.split(r'[,;]', s)[0].strip()
    words = s.split()
    if len(words) < 2 or len(words) > 8:
        return None
    s = ' '.join(words[:6])
    return None if _REJECT_MID_RE.match(s) else s


def _clean_turn_side(raw: str) -> Optional[str]:
    s = raw.strip().strip('"\'').lower()
    if s in ('nothing', 'none', 'unclear', 'empty', 'n/a'):
        return None
    s = re.sub(r'^(a|an|the)\s+', '', raw.strip(), flags=re.IGNORECASE).strip()
    words = s.split()
    return ' '.join(words[:5]) if 2 <= len(words) <= 8 else None

# ── Phase 1a: Per-frame vision ────────────────────────────────────────────────

async def describe_frames(
    episodes: List[Dict],
    rendered_frames_dir: Path,
    *,
    checkpoint: Dict,
    base_url: str,
    model: str,
    api_key: str = "EMPTY",
    concurrency: int = 8,
    backend: str = "vllm",
    max_calls: "int | None" = None,
    dry_run_dir: "Path | None" = None,
    domain: str = "house",
    turn_directions: "Dict[str, Dict[str, str]] | None" = None,
    max_tokens: int = 80,
) -> Dict:
    """
    For each episode, describe the visual context at the start viewpoint and
    each turn viewpoint (up to turn_1, turn_2, turn_3).

    Returns checkpoint dict: {episode_id: {"start": str, "turn_1": str, ...}}
    """
    dom = get_domain(domain)
    tasks = []
    for ep in episodes:
        eid = str(ep["episode_id"])
        if eid in checkpoint:
            continue
        ep_dir = rendered_frames_dir / f"episode_{int(eid):06d}"
        for label in ("start", "turn_1", "turn_2", "turn_3"):
            for ext in ("jpg", "jpeg", "png"):
                img = ep_dir / f"{label}.{ext}"
                if img.exists():
                    tasks.append({"id": f"{eid}:{label}", "prompt": _frame_prompt(dom),
                                  "images": [img]})
                    break

    if not tasks:
        print(f"Phase 1a: all {len(checkpoint)} episodes already in checkpoint")
        return checkpoint

    print(f"Phase 1a: {len(tasks)} frame tasks ({len(checkpoint)} already done)")
    results = await batch_call_async(
        tasks, base_url=base_url, model=model, api_key=api_key,
        max_tokens=max_tokens, concurrency=concurrency,
        backend=backend, max_calls=max_calls, dry_run_dir=dry_run_dir,
    )

    merged = dict(checkpoint)
    for key, text in results.items():
        eid, label = key.split(":", 1)
        merged.setdefault(eid, {})[label] = text

    return merged


# ── Phase 1b: Midpoint classification ────────────────────────────────────────

async def classify_midpoints(
    episodes: List[Dict],
    midpoints_dir: Path,
    *,
    checkpoint: Dict,
    base_url: str,
    model: str,
    api_key: str = "EMPTY",
    concurrency: int = 8,
    backend: str = "vllm",
    max_calls: "int | None" = None,
    dry_run_dir: "Path | None" = None,
    domain: str = "house",
    turn_directions: "Dict[str, Dict[str, str]] | None" = None,
) -> Dict:
    """
    For each intermediate waypoint, identify the most prominent object ahead.

    Reads midpoints.json from midpoints_dir/episode_XXXXXX/ (produced by the
    Habitat renderer). Returns checkpoint dict: {episode_id: {mid_1: str, ...}}
    """
    dom = get_domain(domain)
    tasks = []
    for ep in episodes:
        eid = str(ep["episode_id"])
        if eid in checkpoint:
            continue
        ep_dir = midpoints_dir / f"episode_{int(eid):06d}"
        meta_file = ep_dir / "midpoints.json"
        if not meta_file.exists():
            continue
        try:
            meta = json.load(open(meta_file))
        except Exception:
            continue
        for frame in meta.get("frames", []):
            img_path = ep_dir / frame["path"]
            if img_path.exists():
                tasks.append({"id": f"{eid}:{frame['label']}", "prompt": _midpoint_prompt(dom),
                              "images": [img_path]})

    if not tasks:
        print(f"Phase 1b: all {len(checkpoint)} episodes in checkpoint")
        return checkpoint

    print(f"Phase 1b: {len(tasks)} midpoint tasks ({len(checkpoint)} already done)")
    results = await batch_call_async(
        tasks, base_url=base_url, model=model, api_key=api_key,
        max_tokens=30, concurrency=concurrency,
        backend=backend, max_calls=max_calls, dry_run_dir=dry_run_dir,
    )

    merged = dict(checkpoint)
    for key, raw in results.items():
        eid, label = key.split(":", 1)
        desc = _clean_midpoint(raw)
        merged.setdefault(eid, {})[label] = desc

    return merged


# ── Phase 1c: Turn-side descriptions ─────────────────────────────────────────

async def describe_turn_sides(
    episodes: List[Dict],
    rendered_frames_dir: Path,
    *,
    checkpoint: Dict,
    base_url: str,
    model: str,
    api_key: str = "EMPTY",
    concurrency: int = 8,
    backend: str = "vllm",
    max_calls: "int | None" = None,
    dry_run_dir: "Path | None" = None,
    domain: str = "house",
    turn_directions: "Dict[str, Dict[str, str]] | None" = None,
) -> Dict:
    """
    At each turn viewpoint, describe the object visible in the turn direction.
    This becomes the turn anchor landmark.

    Returns checkpoint dict: {episode_id: {turn_1: str, turn_2: str, ...}}
    """
    dom = get_domain(domain)
    tasks = []
    for ep in episodes:
        eid = str(ep["episode_id"])
        if eid in checkpoint:
            continue
        ep_dir = rendered_frames_dir / f"episode_{int(eid):06d}"
        for label in ("turn_1", "turn_2", "turn_3"):
            for ext in ("jpg", "jpeg", "png"):
                img = ep_dir / f"turn_side_{label}.{ext}"
                if img.exists():
                    tasks.append({"id": f"{eid}:{label}", "prompt": _turn_side_prompt(dom, (turn_directions or {}).get(eid, {}).get(label)),
                                  "images": [img]})
                    break

    if not tasks:
        print(f"Phase 1c: all {len(checkpoint)} episodes in checkpoint")
        return checkpoint

    print(f"Phase 1c: {len(tasks)} turn-side tasks ({len(checkpoint)} already done)")
    results = await batch_call_async(
        tasks, base_url=base_url, model=model, api_key=api_key,
        max_tokens=20, concurrency=concurrency,
        backend=backend, max_calls=max_calls, dry_run_dir=dry_run_dir,
    )

    merged = dict(checkpoint)
    for key, raw in results.items():
        eid, label = key.split(":", 1)
        desc = _clean_turn_side(raw)
        merged.setdefault(eid, {})[label] = desc

    return merged
