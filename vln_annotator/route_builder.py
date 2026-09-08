"""
VLN Auto-Annotator — Route Builder (Phase 2 preparation)

Converts path motion primitives + vision descriptions into a structured
route-description string that serves as the LLM instruction prompt.

The route notation encodes:
  - Turns with landmark anchors:  "turn left at [grey pillar]"
  - Architectural passages:       "[thru: arched doorway]"
  - Furniture passes:             "[pass: wooden bookshelf]"
  - Ahead objects after turns:    "turn right → [sofa ahead]"
  - Bare turns (no context):      "turn left"
  - Straight segments (long):     "straight 5m"
  - Elevation:                    "up stairs" / "down stairs"

Route example:
  "straight 4m → [thru: wooden doorframe] → turn left at [grey pillar]
   → [pass: tall bookshelf] → turn right → Goal: the white wall"
"""
import random
import re
from typing import Dict, List, Optional


# ── Midpoint classification ───────────────────────────────────────────────────

_THRU_WORDS = re.compile(
    r'\b(door(?:way|frame|s)?|arch(?:way|ed)?|entry|entrance|opening|'
    r'threshold|gate|passage|portal|frame|gap|archway)\b',
    re.IGNORECASE,
)
_ROOM_WORDS = re.compile(
    r'\b(kitchen|bedroom|bathroom|living room|dining room|hallway|corridor|'
    r'lounge|foyer|garage|office|study|library|nursery|laundry|closet)\b',
    re.IGNORECASE,
)
_MATERIAL_ADJ_RE = re.compile(
    r'\b(wooden|wood(?:en)?|hardwood|oak|pine|walnut|maple|cedar|mahogany|'
    r'marble|stone|granite|slate|concrete|brick|'
    r'carpeted|carpet|tiled|tile|vinyl|laminate|'
    r'leather|fabric|upholstered|velvet|plush|linen)\s+',
    re.IGNORECASE,
)
_GENERIC_PASS_RE = re.compile(
    r'\b(wall|walls|floor|ceiling|panel|panels|trim|window(?!\s+seat)|'
    r'area|space|room|lobby|hallway|corridor)\b',
    re.IGNORECASE,
)


def classify_pass_action(desc: str) -> str:
    """
    Classify a midpoint object description as 'thru' or 'pass'.
    - 'thru': doorways, arches, room transitions → LLM writes "through the X"
    - 'pass': furniture, objects → LLM writes "walk past the X"
    """
    if not desc:
        return "pass"
    if _THRU_WORDS.search(desc):
        return "thru"
    if _ROOM_WORDS.search(desc):
        return "thru"
    return "pass"


def strip_material_adjectives(desc: str) -> str:
    return _MATERIAL_ADJ_RE.sub("", desc).strip() if desc else desc


def is_generic_pass_object(desc: str) -> bool:
    return bool(_GENERIC_PASS_RE.search(desc)) if desc else True


# ── Room transition helpers (v64) ────────────────────────────────────────────

# Room whitelist. Historically hardcoded to house rooms, which meant a transit
# station had no word that could pass -- "library" reached an instruction from
# here, not from the scene. Now sourced from the active domain profile;
# set_domain() is called once per run. Default keeps the original set.
from .domains import get_domain as _get_domain

_GOOD_RM_WORDS = set(_get_domain("house").room_words)


def set_domain(name: str) -> None:
    """Swap the room whitelist for the named scene domain (see domains.py)."""
    global _GOOD_RM_WORDS
    _GOOD_RM_WORDS = set(_get_domain(name).room_words)
_BAD_RM_WORDS = {
    "hallway", "corridor", "stairwell", "staircase", "attic", "basement",
    "foyer", "lobby", "entry", "area", "room", "space", "passage",
    "landing", "gallery",
}


def _good_room_name(rm: str) -> str:
    """Return canonical good room name for turn marker, or '' if suppressed."""
    rm_low = rm.lower().strip()
    for bad in _BAD_RM_WORDS:
        if bad in rm_low:
            return ""
    for good in _GOOD_RM_WORDS:
        if good in rm_low:
            return good
    return ""


# ── Route string builder ──────────────────────────────────────────────────────

def build_route(
    primitives: List[Dict],
    selected_turns: List[Dict],
    *,
    midpoints: Optional[Dict[str, str]] = None,
    p_pass_marker: float = 0.50,
    p_thru_marker: float = 0.85,
    p_turn_unanchored: float = 0.42,
) -> str:
    """
    Convert path motion primitives + turn metadata into a route-description string.

    Args:
        primitives:         Output of path_analysis.analyze_path() — list of segment dicts.
        selected_turns:     Per-turn metadata with landmark, room, and ahead-object fields.
        midpoints:          {turn_label: description} — per-midpoint visual description.
                            turn_label = "turn_1", "turn_2", ... or "goal".
        p_pass_marker:      Probability of including a [pass:] marker (GT calibrated).
        p_thru_marker:      Probability of including a [thru:] marker (GT calibrated).
        p_turn_unanchored:  Keep-rate for turns with no landmark/room context (GT calibrated).

    Returns:
        Human-readable route string, e.g.:
        "straight 4m → [thru: wooden doorframe] → turn left at [grey pillar] → ..."
    """
    mids = midpoints or {}
    parts: List[str] = []
    turn_idx = 0

    for p in primitives:
        ptype = p["type"]

        if ptype == "straight":
            dist = p.get("distance_m", 0)
            if dist > 2.0:
                parts.append(f"straight {dist:.0f}m")

        elif ptype in ("left_turn", "right_turn"):
            direction = "left" if ptype == "left_turn" else "right"
            td = selected_turns[turn_idx] if turn_idx < len(selected_turns) else {}
            lm = td.get("landmark", "")
            rm = td.get("room", "")
            rt = td.get("room_trans", "")
            ahead = td.get("ahead", "")
            angle = p.get("angle_deg", 90)

            # Inject midpoint marker before the turn (if one exists for this segment)
            turn_label = f"turn_{turn_idx + 1}"
            if mids.get(turn_label):
                mid_desc = strip_material_adjectives(mids[turn_label])
                action = classify_pass_action(mid_desc)
                if action == "thru":
                    if random.random() < p_thru_marker:
                        parts.append(f"[thru: {mid_desc}]")
                elif not is_generic_pass_object(mid_desc):
                    if random.random() < p_pass_marker:
                        parts.append(f"[pass: {mid_desc}]")

            # Skip very small unanchored turns (< 70°)
            if not lm and not rm and angle < 70:
                turn_idx += 1
                continue

            # Probabilistic sampling of unanchored turns to match GT turn frequency
            if not lm and not rm and random.random() > p_turn_unanchored:
                turn_idx += 1
                continue

            # Build the turn phrase based on what context is available
            if lm and rt and "doorway" in rt.lower():
                base = f"turn {direction} past [{lm}] into [{rm}]"
                parts.append(f"{base} → [{ahead} ahead]" if ahead else base)
            elif lm:
                base = f"turn {direction} at [{lm}]"
                parts.append(f"{base} → [{ahead} ahead]" if ahead else base)
            elif rm:
                # Room-only turns: allow good room names (kitchen, bedroom, living room, etc.)
                # Suppress hallway/corridor/stairwell to prevent generic inflation (v64 fix).
                if random.random() > p_turn_unanchored:
                    turn_idx += 1
                    continue
                good_rm = _good_room_name(rm)
                if good_rm:
                    # "→ enter [room]" notation → LLM writes "turn X and enter the room"
                    # Avoids TURN_ANCHOR_RE match (which uses "at/into/past the X" patterns)
                    parts.append(f"turn {direction} → enter [{good_rm}]")
                else:
                    parts.append(f"turn {direction}")  # suppress hallway/corridor/generic
            else:
                parts.append(f"turn {direction}")

            turn_idx += 1

        elif ptype == "elevation":
            direction = "up" if p.get("direction", "up") == "up" else "down"
            parts.append(f"{direction} stairs")

    return " → ".join(parts) or "walk forward"


# ── LLM system prompt (calibrated over 60+ annotator versions) ───────────────

INSTRUCTION_SYSTEM_PROMPT = """\
You write concise R2R navigation instructions for an indoor robot.

Style: Natural and brief, like a human R2R annotator. Match how real humans describe indoor navigation.

RULES:
- When a turn has [object] in brackets: use that object as a turn reference naturally: \
"turn left at the grey pillar", "turn right past the dining table"
- When a turn has → [object ahead] after the arrow: add a BRIEF walking phrase (3-5 words) like \
"walk to the bookshelf" or "walk past the grey sofa" — keep it very short
- When route has [thru: description]: the robot passes through an architectural opening — write \
"through the X" (2-4 words). Prefer distinctive: "through the arched entry" over "through the doorway". \
Reserve "through the doorway" only when the description specifically mentions a plain doorway.
- When route has [pass: description]: the robot walks by an object — write "walk past the X" (3-5 words).
- When a turn has "→ enter [room]" notation (e.g., "turn left → enter [kitchen]"): write \
"turn left and enter the kitchen" or "turn left and walk into the kitchen". Use ONLY when this notation is in the route.
- When a turn has (→ room) after the arrow without "enter": ONLY mention the room if it is the final destination. \
Otherwise write "walk forward".
- Turns with no brackets: write ONLY "turn left" or "turn right" — no objects
- CRITICAL: Do NOT add turns unless the route shows them. Write "walk forward" not "walk toward the room".
- Do NOT repeat any direction or landmark. Each segment is described exactly once.
- ROOM NAMES: Do NOT write "walk into/through/down the hallway", "walk toward the hallway" as mid-route steps. \
Write "walk forward" instead. Only name a room when the route has "→ enter [room]" notation or it is the FINAL destination.
- DISTANCES: the route notation gives distances ("straight 4m") for your own reasoning ONLY. \
NEVER write a distance in the instruction — no "3 meters", "4m", "five feet". Measured on the \
10,819 R2R training instructions, a digit followed by a distance unit occurs in 10 of them (0.09%); \
humans write "walk down the hall", not "walk 4 meters".
- STYLE: Avoid starting with "Continue straight". Avoid overusing "wooden" for generic surfaces.
- Endings vary: "Stop near X." / "Wait near X." / no explicit ending
- Start verb is given — use it exactly\
"""
