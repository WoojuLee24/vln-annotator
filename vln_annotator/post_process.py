"""
VLN Auto-Annotator — Post-Processing (Surgical Fixes)

Four targeted fixes applied after LLM instruction generation:
  1. doorway_fix    — remove hallucinated "through the doorway" when no real passage exists
  2. walk_past_fix  — remove hallucinated "walk past the X" when no real objects to pass
  3. hallway_fix    — remove mid-route hallway phrases when goal is not a hallway
  4. floor_stop_fix — replace floor/tile/carpet-based stop phrases with navigable alternatives
"""
import re
from typing import Dict, Optional, Tuple


# ── Regex patterns ────────────────────────────────────────────────────────────

_THRU_DOORWAY_RE = re.compile(
    r'\bthrough\s+the\s+(?:doorway|open\s+doorway|wooden\s+doorframe|doorframe)\b',
    re.IGNORECASE,
)

_WALK_PAST_THE_RE = re.compile(r'\bwalk\s+past\s+the\b', re.IGNORECASE)

_MIDROUTE_HALLWAY_RE = re.compile(
    r'\b(?:walk(?:ing)?|go(?:ing)?|head(?:ing)?|continue(?:s)?|proceed(?:ing)?)\s+'
    r'(?:through|down|along|into|to|toward)\s+the\s+hallway\b',
    re.IGNORECASE,
)

# Floor/surface words that make bad stop landmarks (v65)
_FLOOR_STOP_RE = re.compile(
    r'\b(Stop|Wait)\b(?:.{0,80}?)'
    r'\b(floor(?:ing)?|tile(?:s)?|carpet(?:ing)?|linoleum|mat|'
    r'hardwood\s+floor|tiled\s+floor|marble\s+floor|wood\s+floor|'
    r'polished\s+floor|light[\-\s]colored\s+floor|white\s+floor|'
    r'grey\s+floor|gray\s+floor)\b.*$',
    re.IGNORECASE | re.DOTALL,
)
_RUG_STOP_RE = re.compile(
    r'\b(Stop|Wait)\s+(?:near|at|by|in\s+front\s+of|beside|next\s+to)\s+the\s+'
    r'(?:\w+\s+)*(?:rug|mat|carpet|runner|doormat)\b.*$',
    re.IGNORECASE | re.DOTALL,
)


def _has_thru_midpoint(midpoints: Dict) -> bool:
    """Check if this episode has any pass-through architectural opening."""
    from .route_builder import classify_pass_action
    return any(classify_pass_action(desc) == "thru"
               for desc in midpoints.values() if desc)


def _has_pass_midpoint(midpoints: Dict) -> bool:
    """Check if this episode has any furniture-passing midpoint."""
    from .route_builder import classify_pass_action
    return any(classify_pass_action(desc) == "pass"
               for desc in midpoints.values() if desc)


def apply_doorway_fix(text: str, episode_midpoints: Dict) -> str:
    """
    Remove 'through the doorway' when the episode has no architectural passage midpoint.
    Prevents LLM from hallucinating doorway passages on plain walking segments.
    """
    if not _THRU_DOORWAY_RE.search(text):
        return text
    if _has_thru_midpoint(episode_midpoints):
        return text
    return _THRU_DOORWAY_RE.sub("walk forward", text)


def apply_walk_past_fix(text: str, episode_midpoints: Dict) -> str:
    """
    Remove 'walk past the X' when the episode has no furniture-passing midpoint.
    Only ~6 episodes benefit from this — most 'walk past' are from real visual context.
    """
    if not _WALK_PAST_THE_RE.search(text):
        return text
    if _has_pass_midpoint(episode_midpoints):
        return text
    fixed = _WALK_PAST_THE_RE.sub("walk to the", text)
    fixed = re.sub(r'\bwalk to the\s+walk to the\b', 'walk to the', fixed, flags=re.I)
    return fixed


def apply_floor_stop_fix(text: str) -> str:
    """
    Replace floor/tile/carpet-based stop phrases with navigable alternatives (v65).
    'Stop in front of the white hallway floor' → 'Stop in the hallway.'
    'Stop at the light-colored floor' → 'Stop.'

    Root cause: VLM Phase 1 goal descriptions sometimes describe only the floor surface
    when the stop location is a featureless area. These non-landmarks confuse navigation.
    """
    for pattern in (_FLOOR_STOP_RE, _RUG_STOP_RE):
        m = pattern.search(text)
        if not m:
            continue
        action = m.group(1)  # "Stop" or "Wait"
        prefix = text[:m.start()].rstrip()
        full_lower = text.lower()
        if re.search(r'\bhallway\b', full_lower):
            replacement = f"{action} in the hallway."
        elif re.search(r'\bkitchen\b', full_lower):
            replacement = f"{action} in the kitchen."
        elif re.search(r'\bbedroom\b', full_lower):
            replacement = f"{action} in the bedroom."
        elif re.search(r'\bliving room\b', full_lower):
            replacement = f"{action} in the living room."
        elif re.search(r'\bdining room\b', full_lower):
            replacement = f"{action} in the dining room."
        elif re.search(r'\bbathroom\b', full_lower):
            replacement = f"{action} in the bathroom."
        elif re.search(r'\boffice\b', full_lower):
            replacement = f"{action} in the office."
        elif re.search(r'\bstair(?:case|s|way)\b', full_lower):
            replacement = f"{action} at the stairs."
        elif re.search(r'\bdoorway|doorframe\b', full_lower):
            replacement = f"{action} in the doorway."
        else:
            replacement = f"{action}."
        return prefix + " " + replacement
    return text


def apply_hallway_fix(text: str, stop_phrase: str) -> str:
    """
    Remove mid-route hallway phrases when the goal is not a hallway.
    Keeps hallway mentions only when the stop phrase references a hallway.
    Pattern: 'walk/go/head/continue through/down/into the hallway' → 'walk forward'
    """
    if "hallway" not in text.lower():
        return text
    if "hallway" in stop_phrase.lower():
        return text  # goal IS a hallway — keep all mentions
    if not _MIDROUTE_HALLWAY_RE.search(text):
        return text
    fixed = _MIDROUTE_HALLWAY_RE.sub("walk forward", text)
    fixed = re.sub(r'\bwalk forward\s+walk forward\b', 'walk forward', fixed, flags=re.I)
    return fixed


def apply_all_fixes(
    text: str,
    *,
    stop_phrase: str = "",
    episode_midpoints: Optional[Dict] = None,
) -> Tuple[str, int]:
    """
    Apply all surgical post-processing fixes in order.
    Returns (fixed_text, n_fixes_applied).
    """
    original = text
    n = 0
    mids = episode_midpoints or {}

    text = apply_doorway_fix(text, mids)
    if text != original:
        n += 1; original = text

    text = apply_walk_past_fix(text, mids)
    if text != original:
        n += 1; original = text

    text = apply_hallway_fix(text, stop_phrase)
    if text != original:
        n += 1; original = text

    text = apply_floor_stop_fix(text)
    if text != original:
        n += 1

    return text, n
