"""
VLN Auto-Annotator — Configuration
All tunable hyperparameters in one place.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AnnotatorConfig:
    # ── LLM backend ──────────────────────────────────────────────────────────
    # backend selects the provider adapter (see vln_annotator/backends/):
    #   vllm | openai | gemini | anthropic | dry
    # "dry" makes no network call and dumps prompts to disk instead.
    backend: str = "vllm"
    vllm_base_url: str = "http://10.77.32.231:8000/v1"
    vllm_model: str = "cyankiwi/gemma-4-31B-it-AWQ-4bit"
    vllm_api_key: str = "EMPTY"
    temperature: float = 0.3
    max_new_tokens: int = 256

    # ── Scene domain ─────────────────────────────────────────────────────────
    # Vocabulary profile for prompts and the room whitelist (see domains.py).
    # "house" reproduces the original behaviour and is what the published
    # calibration numbers describe. "transit" retargets it to a station.
    domain: str = "house"

    # ── Cost guard (metered providers) ───────────────────────────────────────
    # Checked before the first request, so an over-budget run costs nothing.
    max_calls: Optional[int] = None

    # ── Concurrency ───────────────────────────────────────────────────────────
    concurrency_vision: int = 8   # parallel vision-description API calls
    concurrency_instruct: int = 12  # parallel instruction-generation API calls

    # ── Instruction structure sampling ───────────────────────────────────────
    # Controls what fraction of episodes get each structural element.
    # Calibrated to match R2R GT val_unseen distribution.
    p_stop: float = 0.494         # fraction with explicit "Stop near X" phrase (GT=50.8%)
    p_wait: float = 0.305         # fraction with "Wait near X" phrase (GT=30.5%)
    p_anchor_episode: float = 0.231  # fraction with a landmark-referenced turn (GT=16.5%)
    p_turn_unanchored: float = 0.42  # keep-rate for unanchored turns (GT avg_turns=0.59)
    p_pass_marker: float = 0.50   # sampling rate for [pass:] midpoint markers
    p_thru_marker: float = 0.85   # sampling rate for [thru:] midpoint markers

    # ── Path geometry ─────────────────────────────────────────────────────────
    turn_threshold_deg: float = 30.0  # minimum angle to count as a turn
    min_segment_dist_m: float = 0.3   # ignore sub-segments shorter than this

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: Path = Path("outputs/datasets")
    checkpoints_dir: Path = Path("outputs/checkpoints")
    dry_run_dir: Path = Path("outputs/dry_run")
    output_name: str = "val_unseen_annotated.json.gz"

    def output_path(self) -> Path:
        return self.output_dir / self.output_name


# Default configuration (best calibration as of v62/v63)
DEFAULT_CONFIG = AnnotatorConfig()
