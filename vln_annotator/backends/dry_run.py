"""
Dry-run backend — makes no network call.

Writes every prompt (and the image paths it would have sent) to disk and
returns a marker string. This is the cheapest way to prove that the upstream
adapter produced a complete, well-formed set of tasks: if a frame is missing
or a prompt is empty, it shows up here for free instead of after paying for
200 vision calls.

Returns a marker rather than plausible text on purpose — silently producing
something that looks like an instruction would let a broken run masquerade as
a good one.
"""
import json
from pathlib import Path
from typing import List, Optional

MARKER = "DRY_RUN__NO_MODEL_CALLED"

# A run makes one batch_call_async call per phase (1a, 1b, 1c, 2), so a new
# backend instance is built per phase. Truncate on the first instance only,
# then append -- otherwise each phase erases the previous one's prompts and
# the manifest ends up holding just the last phase.
_TRUNCATED: set = set()


class DryRunBackend:
    name = "dry"

    def __init__(self, *, model: Optional[str] = None, out_dir: Optional[Path] = None):
        self.model = model or "dry"
        self.out_dir = Path(out_dir) if out_dir else Path("outputs/dry_run")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._n = 0
        path = self.out_dir / "prompts.jsonl"
        key = str(path.resolve())
        mode = "a" if key in _TRUNCATED else "w"
        _TRUNCATED.add(key)
        self._manifest = path.open(mode, encoding="utf-8")

    async def chat(
        self,
        prompt: str,
        images: Optional[List[Path]] = None,
        *,
        max_tokens: int = 256,
        temperature: float = 0.3,
    ) -> str:
        imgs = [str(p) for p in (images or [])[:3]]
        missing = [p for p in imgs if not Path(p).exists()]
        self._manifest.write(json.dumps({
            "i": self._n,
            "n_images": len(imgs),
            "images": imgs,
            "missing_images": missing,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "prompt_chars": len(prompt),
            "prompt": prompt,
        }, ensure_ascii=False) + "\n")
        self._manifest.flush()
        self._n += 1
        return MARKER

    async def aclose(self) -> None:
        self._manifest.close()
        total = sum(1 for _ in (self.out_dir / "prompts.jsonl").open(encoding="utf-8"))
        print(f"dry-run: +{self._n} prompts ({total} total) "
              f"-> {self.out_dir / 'prompts.jsonl'}")
