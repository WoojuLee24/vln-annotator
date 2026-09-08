#!/usr/bin/env python3
"""
VLN Auto-Annotator CLI
======================
Generate GT-compatible navigation instructions for a VLN dataset.

Usage:
  python annotate_dataset.py \\
    --gt-path /path/to/val_unseen_patched.json.gz \\
    --frames-dir /path/to/rendered_frames \\
    --midpoints-dir /path/to/midpoints \\
    --output-name val_unseen_annotated.json.gz

Run with --help for full option list.
"""
import argparse
import asyncio
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Auto-annotate a VLN dataset with LLM-generated navigation instructions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required paths
    p.add_argument("--gt-path", required=True,
                   help="Path to GT .json.gz (provides episode metadata + vocabulary)")
    p.add_argument("--frames-dir", required=True,
                   help="Directory with rendered frame images (episode_XXXXXX/ subdirs)")
    p.add_argument("--midpoints-dir", required=True,
                   help="Directory with midpoint metadata (episode_XXXXXX/midpoints.json)")

    # LLM backend
    p.add_argument("--backend", default="vllm",
                   choices=["vllm", "openai", "gemini", "anthropic", "dry"],
                   help="provider adapter. 'dry' makes no network call and dumps "
                        "prompts to disk (default: vllm)")
    p.add_argument("--dry-run", action="store_true",
                   help="shorthand for --backend dry")
    p.add_argument("--max-calls", type=int, default=None,
                   help="abort before sending if the run needs more than N calls. "
                        "Recommended for metered providers, e.g. --max-calls 20")
    p.add_argument("--vllm-url", default="http://10.77.32.231:8000/v1",
                   help="base URL for OpenAI-compatible backends "
                        "(default: http://10.77.32.231:8000/v1). Ignored by "
                        "--backend anthropic, which uses its own endpoint")
    p.add_argument("--vllm-model", default="cyankiwi/gemma-4-31B-it-AWQ-4bit",
                   help="Model name served by vLLM")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max-new-tokens", type=int, default=256)

    # Concurrency
    p.add_argument("--concurrency-vision", type=int, default=8,
                   help="Concurrent Phase 1 vision API calls")
    p.add_argument("--concurrency-instruct", type=int, default=12,
                   help="Concurrent Phase 2 instruction API calls")

    # Calibration parameters (pre-calibrated defaults, no need to change normally)
    p.add_argument("--p-stop", type=float, default=0.494,
                   help="P(stop ending) — calibrated to GT 50.8%%")
    p.add_argument("--p-wait", type=float, default=0.305,
                   help="P(wait ending) — calibrated to GT 30.5%%")
    p.add_argument("--p-anchor", type=float, default=0.231,
                   help="P(episode gets anchored turn) — calibrated to GT 16.5%%")
    p.add_argument("--p-turn-unanchored", type=float, default=0.42,
                   help="Keep-rate for unanchored turns — calibrated to GT 0.59 turns/ep")
    p.add_argument("--p-pass-marker", type=float, default=0.50,
                   help="P(midpoint becomes [pass:] marker)")
    p.add_argument("--p-thru-marker", type=float, default=0.85,
                   help="P(midpoint becomes [thru:] marker)")

    # Output / control
    p.add_argument("--output-dir", default="outputs/datasets",
                   help="Directory for output .json.gz (default: outputs/datasets)")
    p.add_argument("--checkpoints-dir", default="outputs/checkpoints",
                   help="Directory for phase checkpoints (default: outputs/checkpoints)")
    p.add_argument("--output-name", default="val_unseen_annotated.json.gz",
                   help="Output filename")
    p.add_argument("--n-episodes", type=int, default=None,
                   help="Limit to first N episodes (default: all)")
    p.add_argument("--skip-phase1", action="store_true",
                   help="Reuse existing Phase 1 checkpoints, skip vision API calls")
    p.add_argument("--skip-phase2", action="store_true",
                   help="Reuse existing Phase 2 checkpoint, only reassemble output")

    return p.parse_args()


async def main() -> None:
    # Parse first: --help must work in an environment with no provider SDKs
    # installed, so nothing may be imported above this line.
    args = parse_args()

    from vln_annotator.config import AnnotatorConfig
    from vln_annotator.pipeline import run

    backend = "dry" if args.dry_run else args.backend

    cfg = AnnotatorConfig(
        backend=backend,
        max_calls=args.max_calls,
        vllm_base_url=args.vllm_url,
        vllm_model=args.vllm_model,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        concurrency_vision=args.concurrency_vision,
        concurrency_instruct=args.concurrency_instruct,
        p_stop=args.p_stop,
        p_wait=args.p_wait,
        p_anchor_episode=args.p_anchor,
        p_turn_unanchored=args.p_turn_unanchored,
        p_pass_marker=args.p_pass_marker,
        p_thru_marker=args.p_thru_marker,
        output_dir=Path(args.output_dir),
        checkpoints_dir=Path(args.checkpoints_dir),
        output_name=args.output_name,
    )

    print("=" * 60)
    print("VLN Auto-Annotator")
    print("=" * 60)
    print(f"  GT path:       {args.gt_path}")
    print(f"  Frames dir:    {args.frames_dir}")
    print(f"  Midpoints dir: {args.midpoints_dir}")
    print(f"  Backend:       {cfg.backend}")
    print(f"  LLM:           {cfg.vllm_model if cfg.backend != 'dry' else '(none)'}")
    if cfg.backend in ("openai", "gemini", "anthropic") and cfg.max_calls is None:
        print("  WARNING:       metered backend with no --max-calls cap")
    print(f"  Output:        {cfg.output_dir / cfg.output_name}")
    print(f"  Calibration:   p_stop={cfg.p_stop}, p_wait={cfg.p_wait}, "
          f"p_anchor={cfg.p_anchor_episode}, p_turn_unanchored={cfg.p_turn_unanchored}")
    print()

    await run(
        gt_path=args.gt_path,
        rendered_frames_dir=Path(args.frames_dir),
        midpoints_dir=Path(args.midpoints_dir),
        cfg=cfg,
        n_episodes=args.n_episodes,
        skip_phase1=args.skip_phase1,
        skip_phase2=args.skip_phase2,
    )


if __name__ == "__main__":
    asyncio.run(main())
