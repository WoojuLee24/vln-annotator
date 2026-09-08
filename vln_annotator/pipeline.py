"""
VLN Auto-Annotator — End-to-End Pipeline

Orchestrates all four phases to produce a GT-compatible .json.gz dataset
from raw episode paths + rendered scene images.

Pipeline phases:
  Phase 1a — describe_frames       : scene context at start + turn viewpoints
  Phase 1b — classify_midpoints    : object at each waypoint midpoint
  Phase 1c — describe_turn_sides   : landmark visible in turn direction
  Phase 2  — generate_instructions : LLM navigation instruction generation

Each phase writes a checkpoint so runs can be resumed without re-doing
completed API calls.
"""
import asyncio
import gzip
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import AnnotatorConfig
from .dataset_assembler import VLNTokenizer, assemble_dataset, load_source, save_dataset
from .instruction_gen import (
    build_instruction_prompt,
    choose_stop,
    compute_stats,
    generate_instructions,
    gt_start_verb,
    postprocess_instruction,
    print_stats,
    strip_material_adj,
)
from .domains import get_domain
from .path_analysis import analyze_path
from .route_builder import set_domain, build_route, classify_pass_action
from .vision import classify_midpoints, describe_frames, describe_turn_sides


# ── Scene similarity index (for same-building few-shot examples) ──────────────

class ScenePathIndex:
    """Index episodes by scene; retrieve top-K structurally-similar examples."""

    def __init__(self, all_episodes: List[Dict]) -> None:
        self.by_scene: Dict[str, List[Dict]] = defaultdict(list)
        for ep in all_episodes:
            feats = self._extract_features(ep)
            self.by_scene[ep["scene_id"]].append(
                {"episode_id": ep["episode_id"], "feats": feats,
                 "instruction": (ep.get("instruction") or {}).get("instruction_text", "")}
            )

    @staticmethod
    def _extract_features(ep: Dict) -> Dict:
        pa = analyze_path(ep["reference_path"], ep.get("start_rotation"))
        prims = pa.get("primitives", [])
        turns = []
        total_dist = 0.0
        for p in prims:
            if p["type"] == "left_turn":
                turns.append("L")
            elif p["type"] == "right_turn":
                turns.append("R")
            elif p["type"] == "straight":
                total_dist += p.get("distance_m", 0)
        summary = pa.get("summary", {})
        return {
            "turns": turns,
            "n_turns": len(turns),
            "total_dist": summary.get("total_distance_m", total_dist),
            "n_waypoints": summary.get("n_waypoints", len(ep["reference_path"])),
        }

    @staticmethod
    def _similarity(f1: Dict, f2: Dict) -> float:
        t1, t2 = f1["turns"], f2["turns"]
        m, n = len(t1), len(t2)
        if m == 0 and n == 0:
            turn_sim = 1.0
        else:
            dp = list(range(n + 1))
            for i in range(1, m + 1):
                prev = dp[:]
                dp[0] = i
                for j in range(1, n + 1):
                    dp[j] = prev[j - 1] if t1[i - 1] == t2[j - 1] else 1 + min(prev[j], dp[j - 1], prev[j - 1])
            turn_sim = 1.0 - dp[n] / max(m, n, 1)
        d1, d2 = f1["total_dist"], f2["total_dist"]
        dist_sim = 1.0 - abs(d1 - d2) / max(d1 + d2, 0.01)
        n1, n2 = f1["n_waypoints"], f2["n_waypoints"]
        wpt_sim = 1.0 - abs(n1 - n2) / max(n1, n2, 1)
        return 0.6 * turn_sim + 0.25 * dist_sim + 0.15 * wpt_sim

    def top_k(self, episode: Dict, k: int = 8) -> List[str]:
        scene = episode["scene_id"]
        eid = episode["episode_id"]
        feats = self._extract_features(episode)
        candidates = [e for e in self.by_scene.get(scene, []) if e["episode_id"] != eid]
        if not candidates:
            return []
        scored = [(self._similarity(feats, e["feats"]), e["instruction"]) for e in candidates]
        scored.sort(key=lambda x: -x[0])
        return [inst for _, inst in scored[:k] if inst]


# ── Landmark info extractor ───────────────────────────────────────────────────

_IS_GENERIC_TS_RE = re.compile(
    r'^(nothing|none|unclear|empty|n/a|wall|ceiling|floor|open\s+space|dark|light)$',
    re.IGNORECASE,
)


def _is_generic_turn_side(desc: str) -> bool:
    return not desc or bool(_IS_GENERIC_TS_RE.match(desc.strip()))


def _build_landmark_info(
    ep: Dict,
    primitives: List[Dict],
    vision_map: Dict,
    turn_sides_map: Dict,
) -> Dict:
    """
    Build per-turn landmark metadata for an episode.
    Uses turn-side vision descriptions (Phase 1c) as turn anchors.
    Falls back to start-scene context for the first turn.

    Returns dict with:
        "turns": [{direction, label, landmark, room, room_trans, ahead}]
        "goal_landmark":  str  (from vision or placeholder)
        "goal_full":      str
        "goal_room":      str
        "start_room":     str
        "start_context":  str
    """
    eid_str = str(ep["episode_id"])
    vis = vision_map.get(eid_str, {})
    ts = turn_sides_map.get(eid_str, {})

    turns_meta = []
    turn_idx = 0
    for p in primitives:
        if p["type"] not in ("left_turn", "right_turn"):
            continue
        direction = "left" if p["type"] == "left_turn" else "right"
        label = f"turn_{turn_idx + 1}"
        ts_desc = ts.get(label, "")
        landmark = "" if _is_generic_turn_side(ts_desc) else ts_desc
        turns_meta.append({
            "direction": direction,
            "label": label,
            "landmark": landmark,
            "room": "",
            "room_trans": "",
            "ahead": "",
        })
        turn_idx += 1

    goal_desc = vis.get("goal", "")
    start_desc = vis.get("start", "")
    start_first = re.split(r'(?<=[.!?])\s+', start_desc)[0] if start_desc else ""

    return {
        "turns": turns_meta,
        "goal_landmark": goal_desc or "the destination",
        "goal_full": goal_desc or "the destination",
        "goal_room": "",
        "start_room": "",
        "start_context": start_first,
    }



def _turn_directions(episodes: List[Dict]) -> Dict[str, Dict[str, str]]:
    """
    {episode_id: {"turn_1": "left", "turn_2": "right", ...}} for the first 3 turns.

    Phase 1c needs to know which way the robot turns, and analyze_path already
    knows -- it is pure geometry, no model call. It simply used to run after
    phase 1c. Labels match the exporter's turn_1..turn_3 image naming, i.e. the
    Nth turn along the path.
    """
    out: Dict[str, Dict[str, str]] = {}
    for ep in episodes:
        prims = analyze_path(ep["reference_path"], ep.get("start_rotation")).get("primitives", [])
        turns = [p["type"] for p in prims if p["type"] in ("left_turn", "right_turn")]
        out[str(ep["episode_id"])] = {
            f"turn_{i+1}": t.split("_")[0] for i, t in enumerate(turns[:3])
        }
    return out

def _select_turns(turns_meta: List[Dict], episode_id: int, cfg: AnnotatorConfig) -> List[Dict]:
    """
    Apply probabilistic anchor selection to match GT anchor_rate=16.5%.
    Episodes where NO turn has a landmark get a chance to be anchored (P_ANCHOR_EPISODE).
    """
    has_lm = any(t.get("landmark") for t in turns_meta)
    if has_lm:
        return turns_meta

    # Seed per-episode to ensure deterministic re-runs
    rng = random.Random(episode_id ^ 0xF1E2)
    if rng.random() < cfg.p_anchor_episode:
        # Randomly pick one turn to become the anchor using its landmark description
        eligible = [i for i, t in enumerate(turns_meta) if t.get("landmark")]
        if eligible:
            return turns_meta

    return turns_meta


def _get_midpoints(ep: Dict, midpoints_checkpoint: Dict) -> Dict[str, str]:
    """Extract midpoint descriptions for an episode from checkpoint."""
    eid_str = str(ep["episode_id"])
    mid = midpoints_checkpoint.get(eid_str, {})
    return {k: v for k, v in mid.items() if v}


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _load_ckpt(path: Path) -> Dict:
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            print(f"  Loaded checkpoint: {len(data)} entries from {path.name}")
            return data
        except Exception as e:
            print(f"  WARNING: Could not load checkpoint {path}: {e}")
    return {}


def _save_ckpt(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def run(
    gt_path: str,
    rendered_frames_dir: Path,
    midpoints_dir: Path,
    cfg: AnnotatorConfig,
    n_episodes: Optional[int] = None,
    skip_phase1: bool = False,
    skip_phase2: bool = False,
) -> None:
    """
    Full annotation pipeline. Reads GT episodes, runs all phases, writes output.

    Args:
        gt_path:              Path to GT .json.gz (e.g. val_unseen_patched.json.gz).
                              Used for episode metadata and vocabulary.
        rendered_frames_dir:  Dir containing episode_XXXXXX/ subdirs with frame images.
        midpoints_dir:        Dir containing episode_XXXXXX/midpoints.json files.
        cfg:                  AnnotatorConfig instance.
        n_episodes:           Limit to first N episodes (None = all).
        skip_phase1:          Reuse existing Phase 1 checkpoint, skip new API calls.
        skip_phase2:          Reuse existing Phase 2 checkpoint, only reassemble.
    """
    t_total = time.time()
    cfg.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load source data ──────────────────────────────────────────────────────
    print(f"Loading source episodes from {gt_path} …")
    all_episodes, instruction_vocab = load_source(gt_path)
    episodes = all_episodes[:n_episodes] if n_episodes else all_episodes
    print(f"Episodes: {len(episodes)}")

    tokenizer = VLNTokenizer(gt_path)
    scene_idx = ScenePathIndex(all_episodes)

    gt_map: Dict[int, str] = {}
    for ep in all_episodes:
        instr = ep.get("instruction")
        text = instr.get("instruction_text", "") if isinstance(instr, dict) else ""
        gt_map[ep["episode_id"]] = text

    # ── Phase 1 checkpoints ───────────────────────────────────────────────────
    p1a_ckpt_path = cfg.checkpoints_dir / "phase1a_frames.json"
    p1b_ckpt_path = cfg.checkpoints_dir / "phase1b_midpoints.json"
    p1c_ckpt_path = cfg.checkpoints_dir / "phase1c_turn_sides.json"
    p2_ckpt_path = cfg.checkpoints_dir / "phase2_instructions.json"

    # Room whitelist and prompt examples follow the scene domain (see domains.py).
    set_domain(cfg.domain)

    # Turn directions are pure geometry, so compute them up front: phase 1c
    # cannot name the object on the turn side without knowing which side it is.
    turn_dirs = _turn_directions(episodes)
    n_turns = sum(len(v) for v in turn_dirs.values())
    print(f"Domain: {cfg.domain} | turn directions resolved for "
          f"{len(turn_dirs)} episodes ({n_turns} turns)")

    # ── Phase 1a: Per-frame scene descriptions ────────────────────────────────
    p1a_ckpt = _load_ckpt(p1a_ckpt_path)
    if not skip_phase1:
        print("\n=== Phase 1a: Frame descriptions ===")
        p1a_ckpt = await describe_frames(
            episodes, rendered_frames_dir,
            checkpoint=p1a_ckpt,
            base_url=cfg.vllm_base_url, model=cfg.vllm_model, api_key=cfg.vllm_api_key,
            concurrency=cfg.concurrency_vision,
            backend=cfg.backend, max_calls=cfg.max_calls, dry_run_dir=cfg.dry_run_dir,
            domain=cfg.domain, turn_directions=turn_dirs,
        )
        _save_ckpt(p1a_ckpt, p1a_ckpt_path)
    else:
        print(f"Phase 1a skipped. {len(p1a_ckpt)} episodes in checkpoint.")

    # ── Phase 1b: Midpoint classification ────────────────────────────────────
    p1b_ckpt = _load_ckpt(p1b_ckpt_path)
    if not skip_phase1:
        print("\n=== Phase 1b: Midpoint classification ===")
        p1b_ckpt = await classify_midpoints(
            episodes, midpoints_dir,
            checkpoint=p1b_ckpt,
            base_url=cfg.vllm_base_url, model=cfg.vllm_model, api_key=cfg.vllm_api_key,
            concurrency=cfg.concurrency_vision,
            backend=cfg.backend, max_calls=cfg.max_calls, dry_run_dir=cfg.dry_run_dir,
            domain=cfg.domain, turn_directions=turn_dirs,
        )
        _save_ckpt(p1b_ckpt, p1b_ckpt_path)
    else:
        print(f"Phase 1b skipped. {len(p1b_ckpt)} episodes in checkpoint.")

    # ── Phase 1c: Turn-side descriptions ─────────────────────────────────────
    p1c_ckpt = _load_ckpt(p1c_ckpt_path)
    if not skip_phase1:
        print("\n=== Phase 1c: Turn-side descriptions ===")
        p1c_ckpt = await describe_turn_sides(
            episodes, rendered_frames_dir,
            checkpoint=p1c_ckpt,
            base_url=cfg.vllm_base_url, model=cfg.vllm_model, api_key=cfg.vllm_api_key,
            concurrency=cfg.concurrency_vision,
            backend=cfg.backend, max_calls=cfg.max_calls, dry_run_dir=cfg.dry_run_dir,
            domain=cfg.domain, turn_directions=turn_dirs,
        )
        _save_ckpt(p1c_ckpt, p1c_ckpt_path)
    else:
        print(f"Phase 1c skipped. {len(p1c_ckpt)} episodes in checkpoint.")

    # ── Phase 2: Instruction generation ──────────────────────────────────────
    p2_ckpt = _load_ckpt(p2_ckpt_path)
    task_meta: Dict[int, Dict] = {}
    tasks: List[Dict] = []
    skipped_p2 = 0

    for ep in episodes:
        eid = ep["episode_id"]
        gt_instr = gt_map.get(eid, "")
        start_verb = gt_start_verb(gt_instr)
        primitives = analyze_path(ep["reference_path"], ep.get("start_rotation")).get("primitives", [])
        n_wpts = len(ep["reference_path"])

        lm_info = _build_landmark_info(ep, primitives, p1a_ckpt, p1c_ckpt)
        sel_turns = _select_turns(lm_info["turns"], eid, cfg)
        midpoints = _get_midpoints(ep, p1b_ckpt)

        goal_lm_clean = strip_material_adj(lm_info["goal_landmark"])
        stop_phrase, stop_type = choose_stop(goal_lm_clean, lm_info["goal_room"], eid, cfg)

        task_meta[eid] = {
            "sv": start_verb,
            "stop_phrase": stop_phrase,
            "stop_type": stop_type,
            "midpoints": midpoints,
        }

        if str(eid) in p2_ckpt:
            skipped_p2 += 1
            continue

        route = build_route(
            primitives, sel_turns,
            midpoints=midpoints,
            p_pass_marker=cfg.p_pass_marker,
            p_thru_marker=cfg.p_thru_marker,
            p_turn_unanchored=cfg.p_turn_unanchored,
        )
        examples = scene_idx.top_k(ep)

        n_bracketed = sum(1 for t in sel_turns if t.get("landmark"))
        n_ahead = sum(1 for t in sel_turns if t.get("landmark") and t.get("ahead"))
        n_mid = sum(1 for desc in midpoints.values() if desc)

        prompt = build_instruction_prompt(
            route=route,
            start_verb=start_verb,
            stop_phrase=stop_phrase,
            stop_type=stop_type,
            goal_description=lm_info["goal_full"],
            goal_room=lm_info["goal_room"],
            start_room=lm_info["start_room"],
            start_context=lm_info["start_context"],
            n_waypoints=n_wpts,
            n_bracketed_turns=n_bracketed,
            n_ahead_markers=n_ahead,
            n_midpoint_markers=n_mid,
            similar_examples=examples,
            extra_rules=cfg.extra_rules,
        )
        tasks.append({"id": str(eid), "prompt": prompt})

    print(f"\n=== Phase 2: Instruction Generation ===")
    print(f"  Tasks: {len(tasks)}   Skipped (cached): {skipped_p2}")

    if tasks and not skip_phase2:
        print(f"  Sample prompt (ep {tasks[0]['id']}):\n{tasks[0]['prompt'][:800]}\n  …")
        results = await generate_instructions(tasks, cfg)
        p2_ckpt.update(results)
        _save_ckpt(p2_ckpt, p2_ckpt_path)
    elif skip_phase2:
        print("  Phase 2 skipped. Using checkpoint.")

    # ── Assemble ──────────────────────────────────────────────────────────────
    print("\n=== Assembling dataset ===")
    generated_texts: Dict[str, str] = {}
    n_pass = n_fix = n_fail = 0

    for ep in episodes:
        eid = ep["episode_id"]
        raw = p2_ckpt.get(str(eid), "")
        if not raw or raw.startswith("ERROR"):
            n_fail += 1
            continue

        meta = task_meta.get(eid, {})
        sv = meta.get("sv", "Walk")
        stop_phrase = meta.get("stop_phrase", "")
        stop_type = meta.get("stop_type", "none")
        midpoints = meta.get("midpoints", {})

        text = postprocess_instruction(raw, sv, stop_phrase, stop_type, midpoints)
        generated_texts[str(eid)] = text
        n_pass += 1
        if text != raw.strip():
            n_fix += 1

    print(f"  Pass: {n_pass}  Post-fixed: {n_fix}  Failed: {n_fail}")

    dataset = assemble_dataset(
        episodes, generated_texts, tokenizer, instruction_vocab
    )

    output_path = cfg.output_dir / cfg.output_name
    save_dataset(dataset, output_path)

    # ── Stats ─────────────────────────────────────────────────────────────────
    texts = [ep["instruction"]["instruction_text"] for ep in dataset["episodes"]]
    stats = compute_stats(texts)
    print_stats(stats)

    elapsed = time.time() - t_total
    print(f"\nDone in {elapsed/60:.1f} min → {output_path}")
