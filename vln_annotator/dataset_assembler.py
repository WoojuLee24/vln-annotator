"""
VLN Auto-Annotator — Dataset Assembler

Combines path metadata + generated instructions + vocab tokens into a
GT-compatible .json.gz file that drops into Habitat / Isaac Lab eval configs.
"""
import gzip
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional


# ── VLN Tokenizer ─────────────────────────────────────────────────────────────

_PAD_LENGTH = 200
_PAD_IDX = 0
_UNK_IDX = 1
_EOS_IDX = 3


class VLNTokenizer:
    """
    Encodes instruction text to GT-compatible instruction_tokens arrays.
    Uses the word2idx_dict from the original dataset's instruction_vocab.
    """

    def __init__(self, vocab_source: str):
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.num_vocab = 0
        self._load_vocab(vocab_source)

    def _load_vocab(self, path: str) -> None:
        with gzip.open(path, "rt") as f:
            data = json.load(f)
        vocab = data.get("instruction_vocab", {})
        self.word2idx = vocab.get("word2idx_dict", {})
        word_list = vocab.get("word_list", [])
        self.idx2word = {i: w for i, w in enumerate(word_list)}
        self.num_vocab = vocab.get("num_vocab", len(word_list))

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+|[^\w\s]", text.lower().strip())

    def encode(self, text: str) -> List[int]:
        ids = [self.word2idx.get(tok, _UNK_IDX) for tok in self._tokenize(text)]
        ids.append(_EOS_IDX)
        ids = ids[:_PAD_LENGTH]
        ids += [_PAD_IDX] * (_PAD_LENGTH - len(ids))
        return ids

    def decode(self, ids: List[int]) -> str:
        words = []
        for idx in ids:
            if idx in (_PAD_IDX, _EOS_IDX):
                break
            words.append(self.idx2word.get(idx, "<unk>"))
        return " ".join(words)

    def coverage(self, text: str) -> float:
        tokens = self._tokenize(text)
        if not tokens:
            return 0.0
        return sum(1 for t in tokens if t in self.word2idx) / len(tokens)


# ── Episode assembler ─────────────────────────────────────────────────────────

def _geodesic(path: List[List[float]]) -> float:
    total = 0.0
    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i + 1]
        total += math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
    return round(total, 6)


def assemble_episode(source: Dict, text: str, tokenizer: VLNTokenizer) -> Dict:
    """
    Build one GT-compatible episode dict from source metadata + generated text.
    All structural fields (position, path, goals, scene) are copied verbatim;
    only instruction_text and instruction_tokens are replaced.
    """
    return {
        "episode_id": source["episode_id"],
        "trajectory_id": source.get("trajectory_id", 0),
        "scene_id": source["scene_id"],
        "start_position": source["start_position"],
        "start_rotation": source["start_rotation"],
        "info": {
            "geodesic_distance": source.get("info", {}).get(
                "geodesic_distance", _geodesic(source["reference_path"])
            )
        },
        "goals": source["goals"],
        "instruction": {
            "instruction_text": text,
            "instruction_tokens": tokenizer.encode(text),
        },
        "reference_path": source["reference_path"],
    }


def assemble_dataset(
    source_episodes: List[Dict],
    generated_texts: Dict,
    tokenizer: VLNTokenizer,
    instruction_vocab: Dict,
) -> Dict:
    """
    Build a complete dataset dict.

    Args:
        source_episodes:   Original episode list from GT json.gz.
        generated_texts:   {episode_id (int or str): instruction_text}
        tokenizer:         VLNTokenizer loaded from vocab source.
        instruction_vocab: Original instruction_vocab dict to carry forward.

    Returns:
        Dataset dict with "episodes" and "instruction_vocab" keys.
    """
    texts = {str(k): v for k, v in generated_texts.items()}
    assembled = []
    skipped = []

    for ep in source_episodes:
        eid = str(ep["episode_id"])
        text = texts.get(eid, "")
        if not text or text.startswith("ERROR"):
            skipped.append(eid)
            continue
        assembled.append(assemble_episode(ep, text, tokenizer))

    if skipped:
        n_shown = min(10, len(skipped))
        print(f"WARNING: Skipped {len(skipped)} episodes (no generated text). "
              f"First {n_shown}: {skipped[:n_shown]}")

    return {
        "episodes": assembled,
        "instruction_vocab": instruction_vocab,
    }


def save_dataset(dataset: Dict, output_path: Path) -> None:
    """Write dataset as gzip-compressed JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False)
    kb = output_path.stat().st_size / 1024
    print(f"Saved {len(dataset['episodes'])} episodes → {output_path} ({kb:.0f} KB)")


def load_source(gt_path: str) -> tuple:
    """Load GT json.gz and return (episodes, instruction_vocab)."""
    with gzip.open(gt_path, "rt") as f:
        data = json.load(f)
    return data["episodes"], data.get("instruction_vocab", {})
