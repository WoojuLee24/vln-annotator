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


class DryRunBackend:
    name = "dry"

    def __init__(self, *, model: Optional[str] = None, out_dir: Optional[Path] = None):
        self.model = model or "dry"
        self.out_dir = Path(out_dir) if out_dir else Path("outputs/dry_run")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._n = 0
        self._manifest = (self.out_dir / "prompts.jsonl").open("w", encoding="utf-8")

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
        print(f"dry-run: wrote {self._n} prompts -> {self.out_dir / 'prompts.jsonl'}")
