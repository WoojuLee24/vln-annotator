"""
VLN Auto-Annotator — Instruction Generation (Phase 2)

Converts the route-description string (from route_builder) into natural
language VLN navigation instructions via an LLM, then post-processes
the output for formatting, loop removal, and distribution calibration.
"""
import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from .config import AnnotatorConfig
from .llm_backend import batch_call_async
from .post_process import apply_all_fixes
from .route_builder import INSTRUCTION_SYSTEM_PROMPT


# ── Regex helpers (calibrated from v36-v62 analysis) ─────────────────────────

_TURN_ANCHOR_RE = re.compile(
    r'\bturn\s+(?:left|right)\s+(?:at|past|toward|near|by)\s+the\b', re.IGNORECASE
)
_STOP_RE = re.compile(r'\bstop\b', re.IGNORECASE)
_WAIT_RE = re.compile(r'\bwait\b', re.IGNORECASE)
_MATERIAL_ADJ_RE = re.compile(
    r'\b(wooden|wood(?:en)?|hardwood|oak|pine|walnut|maple|cedar|mahogany|'
    r'marble|stone|granite|slate|concrete|brick|'
    r'carpeted|carpet|tiled|tile|vinyl|laminate|'
    r'leather|fabric|upholstered|velvet|plush|linen)\s+',
    re.IGNORECASE,
)
_GENERIC_WPT_RE = re.compile(
    r'\bwalk\s+past\s+the\s+'
    r'(?:wall|walls|floor|ceiling|table|tables|chair|chairs|window|windows|'
    r'furniture|area|space|room|corner|side|left|right)\b',
    re.IGNORECASE,
)
_THROUGH_HALLWAY_RE = re.compile(
    r'\bthrough\s+the\s+(hallway|corridor)\b', re.IGNORECASE
)
_THROUGH_ROOM_RE = re.compile(
    r'\bthrough\s+the\s+(kitchen|bedroom|bathroom|living\s+room|dining\s+room|office|study|lounge)\b',
    re.IGNORECASE,
)
_PROCEED_DEST_RE = re.compile(
    r'\bProceed\s+(to|toward|past|through|into)\s+the\b', re.IGNORECASE
)
_PROCEED_MOTION_RE = re.compile(
    r'\bProceed\s+(?:forward|straight|ahead)[.,]?\s*', re.IGNORECASE
)
_CONTINUE_MOTION_RE = re.compile(
    r'\bContinue\s+(through|down|into|along|forward)\s+the\b', re.IGNORECASE
)
_CONTINUE_FRAG_RE = re.compile(
    r'\b(Continue|continue)\s+forward[.,]?\s*', re.IGNORECASE
)
_DUP_MOTION_RE = re.compile(
    r'\b(walk forward|walk straight|walk ahead)\b[.,]?\s+\1\b', re.IGNORECASE
)
_MID_STOP_RE = re.compile(
    r',?\s*\band\s+(?:stop|wait)\b[^.!?]*', re.IGNORECASE
)
_FRAGMENT_RE = re.compile(
    r'^(?:into|through|via|past|around|along|toward|towards|across|over|under|'
    r'onto|off|away from|out of|from|between|behind|beside|within|upon)\s',
    re.IGNORECASE,
)
_TAKE_FRAG_RE = re.compile(r'\bTake\s+a[.,]?\s*$', re.IGNORECASE)
_HEAD_FRAG_RE = re.compile(r'\bHead[.,]?\s*$', re.IGNORECASE)
_TRUNCATED_ATTR_RE = re.compile(
    r'\s+(?:holding|featuring|with|beside|near|containing)\s+'
    r'(?:a\s+|an\s+|the\s+|two\s+|some\s+|several\s+)?'
    r'(?:small|large|dark|light|brown|white|black|grey|gray|decorative|wooden|'
    r'glass|metal|marble|round|square|rectangular|narrow|tall|short|colorful)\s*[.!?]?$',
    re.IGNORECASE,
)

_PREAMBLES = [
    "Instruction:", "Navigation:", "Answer:", "Sure,", "Certainly,",
    "Of course,", "Here is", "Here's", "Result:", "Walk:", "The instruction:",
]


# ── Stop phrase generator ─────────────────────────────────────────────────────

def strip_material_adj(text: str) -> str:
    return _MATERIAL_ADJ_RE.sub("", text).strip()


def choose_stop(goal_lm: str, goal_room: str, episode_id: int, cfg: AnnotatorConfig) -> Tuple[str, str]:
    """
    Randomly assign a stop/wait/none ending for an episode (seeded by episode_id).
    Returns (stop_phrase, stop_type) where stop_type is 'stop', 'wait', or 'none'.
    """
    rng = random.Random(episode_id ^ 0xA3B7)
    r = rng.random()

    lm_words = goal_lm.split()[:8]
    lm = " ".join(lm_words)
    lm = re.sub(r'\s+(a|an|the|to|in|on|by|at|of|leading)\s*$', '', lm, flags=re.IGNORECASE).strip()
    if not lm.startswith("the "):
        lm = f"the {lm}"
    if goal_lm.strip() in ("", "the destination", "destination"):
        lm = "your destination"

    if r < cfg.p_stop:
        patterns = [f"Stop near {lm}.", f"Stop at {lm}.", f"Stop in front of {lm}."]
        if goal_room and not goal_lm:
            patterns = [f"Stop in the {goal_room}.", f"Stop at the entrance."]
        return rng.choice(patterns), "stop"
    elif r < cfg.p_stop + cfg.p_wait:
        return rng.choice([f"Wait near {lm}.", f"Wait by {lm}.", "Wait there."]), "wait"
    else:
        return "", "none"


def gt_start_verb(gt_instruction: str) -> str:
    """Extract the first word of a GT instruction to use as start verb."""
    tok = gt_instruction.strip().split()
    if not tok:
        return "Walk"
    w = tok[0].rstrip(".,!?;:")
    return (w[0].upper() + w[1:].lower()) if w else "Walk"


# ── Instruction cleaning ──────────────────────────────────────────────────────

def clean_instruction(raw: str, start_verb: str) -> str:
    """
    Clean LLM output to produce a well-formed navigation instruction.
    Applies ~15 targeted fixes accumulated from v29-v62 analysis.
    """
    raw = raw.strip()
    if raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1].strip()

    # Strip LLM preamble
    for pre in _PREAMBLES:
        if raw.lower().startswith(pre.lower()):
            raw = raw[len(pre):].lstrip(" :\n").strip()

    # Ensure starts with the required verb
    sv_l = start_verb.lower()
    if raw.lower().startswith(sv_l + " " + sv_l):
        raw = raw[len(sv_l):].lstrip()
    if not raw.lower().startswith(sv_l):
        raw = start_verb + " " + (raw[0].lower() + raw[1:] if raw else "to the destination.")

    # Remove route notation artifacts from final text
    raw = re.sub(r'\[([^\]]+)\]', r'\1', raw)
    raw = re.sub(r'\s*→\s*', ' ', raw)
    raw = re.sub(r'\s+', ' ', raw).strip()

    # Normalise phrasing
    raw = re.sub(r'\bpass(?:ing|ed)?\s+through\s+the\b', 'through the', raw, flags=re.IGNORECASE)
    raw = re.sub(r'\bpass(?:ing|ed)?\s+the\b', 'walk past the', raw, flags=re.IGNORECASE)
    raw = re.sub(r'\bgo\s+past\s+the\b', 'walk past the', raw, flags=re.IGNORECASE)
    raw = _MATERIAL_ADJ_RE.sub('', raw)
    raw = _GENERIC_WPT_RE.sub('walk forward', raw)
    raw = _THROUGH_HALLWAY_RE.sub(lambda m: f"down the {m.group(1).lower()}", raw)
    raw = _THROUGH_ROOM_RE.sub(lambda m: f"into the {m.group(1).lower()}", raw)
    raw = _PROCEED_DEST_RE.sub(
        lambda m: f"Walk {m.group(1).lower()} the", raw)
    raw = _PROCEED_MOTION_RE.sub('', raw)
    raw = _CONTINUE_MOTION_RE.sub(
        lambda m: f"walk {m.group(1).lower()} the", raw)
    raw = _CONTINUE_FRAG_RE.sub(
        lambda m: 'Walk forward' if m.group(0)[0] == 'C' else 'walk forward', raw)
    raw = _DUP_MOTION_RE.sub(lambda m: m.group(1), raw)
    raw = _TAKE_FRAG_RE.sub('', raw)
    raw = _HEAD_FRAG_RE.sub('', raw)
    raw = re.sub(r'\s+', ' ', raw).strip()

    # Sentence-level cleanup + fragment merging (v40-v41)
    sents = re.split(r'(?<=[.!?])\s+', raw)
    merged = []
    for s in sents:
        s = _MID_STOP_RE.sub('', s)
        s = re.sub(r'\s+', ' ', s).strip()
        if s and s[-1] not in '.!?':
            s += '.'
        if not s:
            continue
        is_frag = bool(_FRAGMENT_RE.match(s))
        if is_frag and merged:
            prev = merged[-1].rstrip('.!?').rstrip()
            merged[-1] = prev + ' ' + s.lstrip()
        elif not is_frag:
            if re.match(r'^(?:Stop|Wait|Halt)\b', s, re.IGNORECASE):
                s = _TRUNCATED_ATTR_RE.sub('.', s)
                if s and s[-1] not in '.!?':
                    s += '.'
            merged.append(s)

    result = " ".join(s for s in merged[:3] if s and s not in ('.',)).strip()
    if result and result[-1] not in ".!?":
        result += "."
    return result


def remove_loops(text: str, stop_phrase: str = "") -> str:
    """Detect repeated 5-gram loops in the generated text and truncate."""
    words = text.split()
    if len(words) < 12:
        return text
    seen: Dict[str, int] = {}
    for i in range(len(words) - 4):
        ng = ' '.join(words[i:i + 5]).lower()
        if ng in seen:
            truncated = ' '.join(words[:seen[ng]]).rstrip('.,;:')
            if stop_phrase:
                truncated = truncated + '. ' + stop_phrase
            elif not truncated.rstrip().endswith(('.', '!', '?')):
                truncated += '.'
            return truncated
        seen[ng] = i
    return text


def quality_ok(text: str) -> Tuple[bool, str]:
    words = text.split()
    if len(words) < 5:
        return False, "too_short"
    if len(words) > 80:
        return False, "too_long"
    ngrams: set = set()
    for i in range(len(words) - 4):
        ng = ' '.join(words[i:i + 5]).lower()
        if ng in ngrams:
            return False, "loop_detected"
        ngrams.add(ng)
    return True, "ok"


def enforce_stop_phrase(text: str, phrase: str, stop_type: str) -> str:
    """Ensure the stop/wait phrase appears exactly once at the end."""
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    cleaned = []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if re.match(r'^(Stop|Wait|Halt|Stand)\b', s, re.IGNORECASE):
            continue
        s = re.sub(
            r'(?:(?:[,;]\s*(?:(?:and|then|to|continue|until|once|when)\s+)*(?:stop|wait|halt)s?[^.!?]*)'
            r'|(?:\s+(?:(?:and|then|to|continue|until|once|when)\s+)+(?:\w+\s+)?(?:stop|wait|halt)s?[^.!?]*))'
            r'[.!?]?$', '.', s, flags=re.IGNORECASE,
        )
        s = re.sub(r'\.{2,}', '.', s).strip()
        if s and s not in ('.', '!', '?'):
            cleaned.append(s)
    text = ' '.join(cleaned).strip()
    text = re.sub(r'\.{2,}', '.', text).strip()
    if stop_type == "none":
        return (text.rstrip('. ') + '.') if text else '.'
    return text.rstrip('. ').rstrip('.') + '. ' + phrase


# ── Prompt builder ────────────────────────────────────────────────────────────

def _word_budget(n_waypoints: int) -> str:
    """Length guidance calibrated to match GT word-count distribution by path length."""
    if n_waypoints <= 4:
        return " LENGTH: Write 13-21 words total. Very short path — be brief."
    elif n_waypoints == 5:
        return " LENGTH: Write 24-32 words total."
    elif n_waypoints == 6:
        return " LENGTH: Write 27-35 words total."
    else:
        return " LENGTH: Write 31-40 words total."


def build_instruction_prompt(
    route: str,
    start_verb: str,
    stop_phrase: str,
    stop_type: str,
    goal_description: str,
    goal_room: str,
    start_room: str,
    start_context: str,
    n_waypoints: int,
    n_bracketed_turns: int,
    n_ahead_markers: int,
    n_midpoint_markers: int,
    similar_examples: Optional[List[str]] = None,
) -> str:
    """
    Build the full LLM prompt for Phase 2 instruction generation.

    Args:
        route:               Route string from route_builder.build_route()
        start_verb:          First word to use (e.g. 'Walk', 'Turn', 'Exit')
        stop_phrase:         End phrase (e.g. 'Stop near the white door.')
        stop_type:           'stop', 'wait', or 'none'
        goal_description:    Full visual description of the goal location
        goal_room:           Room name of the goal (e.g. 'kitchen')
        start_room:          Room name at start
        start_context:       Brief visual context at start
        n_waypoints:         Number of path waypoints (used for word budget)
        n_bracketed_turns:   Number of [landmark] turns in route
        n_ahead_markers:     Number of → [X ahead] markers in route
        n_midpoint_markers:  Number of [thru:]/[pass:] markers in route
        similar_examples:    Up to 8 same-building example instructions
    """
    ex_block = "\n".join(
        f"  {i + 1}. \"{ex}\"" for i, ex in enumerate(similar_examples or [])
    )

    # Anchor note describes what the route contains for the LLM
    if n_bracketed_turns > 0:
        anchor_note = f"Route has {n_bracketed_turns} bracketed landmark(s) — use them naturally."
    else:
        anchor_note = "Route has NO bracketed landmarks — write ONLY direction words for all turns."

    if n_ahead_markers > 0:
        anchor_note += (
            f" Route also has {n_ahead_markers} [X ahead] marker(s) — after turning at"
            f" [landmark], add a brief 'walk toward/past X' phrase (3-5 words max)."
        )
    if n_midpoint_markers == 1:
        anchor_note += (
            " Route has 1 midpoint marker — [thru:] = 'through the X',"
            " [pass:] = 'walk past the X'. Write it naturally in 3-4 words."
        )
    elif n_midpoint_markers > 1:
        anchor_note += (
            f" Route has {n_midpoint_markers} midpoint markers — [thru:] = 'through the X',"
            f" [pass:] = 'walk past the X'. Write each in 3-4 words."
        )

    anchor_note += _word_budget(n_waypoints)

    ending_req = (
        f'End: "{stop_phrase}"'
        if stop_type in ("stop", "wait")
        else "End naturally — no explicit stop/wait"
    )

    prompt = (
        f"{INSTRUCTION_SYSTEM_PROMPT}\n\n"
        f"Same-building examples:\n{ex_block}\n\n"
        f"Write ONE instruction for:\n"
        f"  Route: {route}\n"
        f"  Start: {start_room or 'room'}"
        + (f" ({start_context})" if start_context else "")
        + f"\n  Goal: {goal_description}"
        + (f" in {goal_room}" if goal_room else "")
        + f"\n\n{anchor_note}\n{ending_req}"
        + f"\nStart with: \"{start_verb}\"\n\n{start_verb}"
    )
    return prompt


# ── Phase 2 batch generation ──────────────────────────────────────────────────

async def generate_instructions(
    tasks: List[Dict],
    cfg: AnnotatorConfig,
) -> Dict[str, str]:
    """
    Run Phase 2 LLM instruction generation on a list of tasks.

    Each task must have:
        "id":      episode identifier (str)
        "prompt":  full LLM prompt string

    Returns {id: generated_text}.
    """
    llm_tasks = [{"id": t["id"], "prompt": t["prompt"]} for t in tasks]
    return await batch_call_async(
        llm_tasks,
        base_url=cfg.vllm_base_url,
        model=cfg.vllm_model,
        api_key=cfg.vllm_api_key,
        max_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        concurrency=cfg.concurrency_instruct,
    )


# ── Post-processing ───────────────────────────────────────────────────────────

def postprocess_instruction(
    raw: str,
    start_verb: str,
    stop_phrase: str,
    stop_type: str,
    episode_midpoints: Optional[Dict] = None,
) -> str:
    """
    Full post-processing pipeline for one LLM-generated instruction:
    1. clean_instruction  — normalise phrasing, strip artifacts
    2. remove_loops       — detect and truncate repeated 5-grams
    3. apply_all_fixes    — surgical distribution fixes (doorway/walk_past/hallway)
    4. quality_ok         — validate; fallback if too short/long/looped
    5. enforce_stop_phrase — ensure ending is exactly as chosen
    """
    text = clean_instruction(raw, start_verb)
    text = remove_loops(text, stop_phrase)
    text, _ = apply_all_fixes(text, stop_phrase=stop_phrase, episode_midpoints=episode_midpoints)

    ok, _ = quality_ok(text)
    if not ok:
        text = f"{start_verb} to the destination."
        if stop_phrase:
            text = text.rstrip('.') + '. ' + stop_phrase

    text = enforce_stop_phrase(text, stop_phrase, stop_type)
    return text


# ── Statistics reporter ───────────────────────────────────────────────────────

def compute_stats(texts: List[str], gt_texts: Optional[List[str]] = None) -> Dict:
    """Compute distribution statistics on a list of generated instructions."""
    n = len(texts)
    if n == 0:
        return {}

    stop_n = sum(1 for t in texts if _STOP_RE.search(t) and not _WAIT_RE.search(t))
    wait_n = sum(1 for t in texts if _WAIT_RE.search(t))
    neither_n = n - stop_n - wait_n
    anchor_n = sum(1 for t in texts if _TURN_ANCHOR_RE.search(t))

    turn_counts = [len(re.findall(r'\bturn\s+(?:left|right)\b', t.lower())) for t in texts]
    avg_turns = sum(turn_counts) / n
    eps_with_turn = sum(1 for c in turn_counts if c > 0)

    avg_words = sum(len(t.split()) for t in texts) / n

    through_n = sum(1 for t in texts if re.search(r'\bthrough\s+the\b', t, re.I))
    walkpast_n = sum(1 for t in texts if re.search(r'\bwalk\s+past\s+the\b', t, re.I))
    hallway_n = sum(1 for t in texts if re.search(r'\bhallway\b', t, re.I))
    toward_n = sum(1 for t in texts if re.search(r'\btoward\s+the\b', t, re.I))
    continue_n = sum(1 for t in texts if re.search(r'\bcontinue\b', t, re.I))
    dup_stop = sum(1 for t in texts if len(re.findall(r'\b(stop|wait|halt)\b', t, re.I)) >= 2)

    anchor_rate = anchor_n / n
    stop_rate = stop_n / n
    wait_rate = wait_n / n
    anchor_err = abs(anchor_rate - 0.165)
    stop_err = abs(stop_rate - 0.508)
    wait_err = abs(wait_rate - 0.305)
    gt_match = 1.0 - (anchor_err * 2 + stop_err + wait_err)

    stats = {
        "n": n,
        "avg_words": round(avg_words, 1),
        "stop_pct": round(100 * stop_rate, 1),
        "wait_pct": round(100 * wait_rate, 1),
        "neither_pct": round(100 * neither_n / n, 1),
        "anchor_pct": round(100 * anchor_rate, 1),
        "avg_turns_per_ep": round(avg_turns, 2),
        "eps_with_turn_pct": round(100 * eps_with_turn / n, 1),
        "through_the_pct": round(100 * through_n / n, 1),
        "walk_past_the_pct": round(100 * walkpast_n / n, 1),
        "hallway_pct": round(100 * hallway_n / n, 1),
        "toward_the_pct": round(100 * toward_n / n, 1),
        "continue_pct": round(100 * continue_n / n, 1),
        "dup_stop": dup_stop,
        "gt_match_score": round(gt_match, 3),
    }
    return stats


def print_stats(stats: Dict) -> None:
    """Print a formatted stats table with GT reference values."""
    gt = {
        "avg_words": 26.8, "stop_pct": 50.8, "wait_pct": 30.5,
        "neither_pct": 18.7, "anchor_pct": 16.5, "avg_turns_per_ep": 0.59,
        "eps_with_turn_pct": 41.3, "through_the_pct": 27.2,
        "walk_past_the_pct": 8.7, "hallway_pct": 20.4, "toward_the_pct": 3.8,
        "continue_pct": 10.4,
    }
    print(f"\n{'Metric':<25} {'Generated':>12} {'GT':>8}")
    print("-" * 47)
    for k, v in stats.items():
        if k in ("n", "dup_stop", "gt_match_score"):
            continue
        gt_v = gt.get(k, "—")
        gt_str = f"{gt_v}" if isinstance(gt_v, str) else f"{gt_v}"
        print(f"  {k:<23} {v:>12} {gt_str:>8}")
    print(f"\n  dup_stop:           {stats.get('dup_stop', '?'):>6}  (target: 0)")
    print(f"  GT-match score:     {stats.get('gt_match_score', '?'):>6}  (target ≥ 0.90)")
