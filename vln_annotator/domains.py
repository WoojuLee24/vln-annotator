"""
VLN Auto-Annotator — scene-domain vocabulary profiles.

The annotator was calibrated on R2R / Matterport3D, which is residential
interiors. Every example landmark and every allowed room word is a house
word. Point it at a subway station and there is literally no room word it can
emit: `library` came out of the room whitelist, not out of the scene.

So the domain vocabulary is a profile, not a constant.

`house` reproduces the original behaviour byte-for-byte and stays the default,
because the published calibration (GT-match 0.951, avg_turns/ep 0.59, …)
describes that profile and nothing else. Selecting another profile changes
what those numbers mean, so run both and compare rather than assuming the new
one is better.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass(frozen=True)
class Domain:
    name: str
    # Shown to the VLM as examples of a good landmark noun phrase.
    landmark_examples: Tuple[str, ...]
    # Shown as examples of an object seen at a midpoint.
    midpoint_examples: Tuple[str, ...]
    # Room words allowed to survive post-processing (a whitelist).
    room_words: frozenset
    # Noun phrase naming the setting: "this {setting} scene".
    setting: str
    # Adverbial phrase: "A robot navigating {navigating} has just stepped ...".
    # Kept separate from `setting` because one grammatical slot cannot serve
    # both ("a indoor space" was the result of trying).
    navigating: str
    # Example sentence for the frame-description prompt.
    frame_example: str


HOUSE = Domain(
    name="house",
    landmark_examples=("grey stone pillar", "brown wooden cabinet", "white kitchen counter"),
    midpoint_examples=("dark wooden dining table", "grey upholstered sofa",
                       "white marble fireplace", "tall wooden bookshelf"),
    room_words=frozenset({
        "kitchen", "living room", "dining room", "bedroom", "bathroom",
        "office", "study", "library", "lounge", "family room", "game room",
        "home theater", "laundry", "pantry", "den", "nursery", "gym",
    }),
    setting="indoor",
    navigating="indoors",
    frame_example=("Starting in a bright hallway with brown wooden double doors on "
                   "the right. The polished stone floor leads forward."),
)

TRANSIT = Domain(
    name="transit",
    landmark_examples=("grey concrete pillar", "steel ticket gate", "blue vending machine"),
    midpoint_examples=("silver ticket gate", "blue vending machine",
                       "yellow tactile paving strip", "glass platform screen door"),
    room_words=frozenset({
        "concourse", "platform", "station hall", "ticket hall", "passageway",
        "corridor", "hallway", "underpass", "stairwell", "escalator landing",
        "waiting area", "transfer passage",
    }),
    setting="underground transit station",
    navigating="through a transit station",
    frame_example=("Starting in a bright station concourse with a yellow tactile paving "
                   "strip on the floor. Glass platform screen doors line the left side."),
)

DOMAINS: Dict[str, Domain] = {d.name: d for d in (HOUSE, TRANSIT)}
DEFAULT_DOMAIN = "house"


def get_domain(name: str) -> Domain:
    key = (name or DEFAULT_DOMAIN).lower()
    if key not in DOMAINS:
        raise ValueError(f"unknown domain {name!r}; choose from {sorted(DOMAINS)}")
    return DOMAINS[key]


def quoted(items) -> str:
    """Render examples the way the original prompts did: 'a', 'b', 'c'."""
    return ", ".join(f"'{x}'" for x in items)
