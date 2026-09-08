"""
VLN Auto-Annotator — LLM Backend

Owns concurrency, progress reporting and the cost guard. The provider-specific
part (how a prompt + images become one HTTP request) lives in ./backends/.

`batch_call_async` keeps its original keyword signature, so vision.py and
instruction_gen.py call it unchanged; `backend`, `max_calls` and `dry_run_dir`
are additive and default to the previous behaviour.
"""
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional

from .backends import get_backend, resolve_api_key

# NOTE: backends are imported lazily (inside get_backend and the shims below)
# so that `--backend dry` works with no provider SDK installed at all.


class CallBudgetExceeded(RuntimeError):
    """Raised before any request is sent when a run would exceed --max-calls."""


def image_to_base64(image_path: Path) -> str:
    """Kept for backwards compatibility; prefer backends.openai_compat."""
    import base64
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


def build_vision_message(prompt: str, image_paths: List[Path]) -> List[dict]:
    """Kept for backwards compatibility. Up to 3 images, then the text."""
    from .backends.openai_compat import build_messages
    return build_messages(prompt, image_paths)


def build_text_message(prompt: str) -> List[dict]:
    return [{"role": "user", "content": prompt}]


async def _one(
    backend,
    task: Dict,
    *,
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> Dict:
    async with semaphore:
        try:
            text = await backend.chat(
                task["prompt"],
                task.get("images"),
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return {"id": task["id"], "text": text, "ok": True, "error": None}
        except Exception as exc:
            return {"id": task["id"], "text": "", "ok": False, "error": str(exc)}


async def batch_call_async(
    tasks: List[Dict],
    *,
    base_url: str = None,
    model: str = None,
    api_key: str = "EMPTY",
    max_tokens: int = 256,
    temperature: float = 0.3,
    concurrency: int = 12,
    log_every: int = 200,
    backend: str = "vllm",
    max_calls: Optional[int] = None,
    dry_run_dir: Optional[Path] = None,
) -> Dict:
    """
    Run many LLM calls concurrently. Returns {id: text}; failures map to "".

    Each task: {"id": hashable, "prompt": str, "images": [Path] | None}

    max_calls is checked *before* the first request, so an over-budget run
    costs nothing. That matters for metered providers, where the mistake you
    want to prevent is discovering the size of a run from the invoice.
    """
    if not tasks:
        return {}

    if max_calls is not None and len(tasks) > max_calls:
        raise CallBudgetExceeded(
            f"{len(tasks)} calls requested but --max-calls={max_calls}. "
            f"Nothing was sent. Narrow the run (--n-episodes) or raise the cap."
        )

    be = get_backend(
        backend,
        base_url=base_url,
        model=model,
        api_key=resolve_api_key(backend, api_key if api_key != "EMPTY" else None),
        dry_run_dir=dry_run_dir,
    )

    # Count what will actually be sent: each task is truncated to 3 images.
    # This line is the cost estimate for metered providers, so it must not
    # over- or under-count.
    n_images = sum(min(3, len(t.get("images") or [])) for t in tasks)
    print(f"  backend={be.name} model={getattr(be, 'model', model)} "
          f"calls={len(tasks)} images={n_images} concurrency={concurrency}")

    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()
    try:
        coros = [
            _one(be, t, max_tokens=max_tokens, temperature=temperature, semaphore=sem)
            for t in tasks
        ]
        results: Dict = {}
        n_ok = n_err = 0
        first_error = None
        for i, result in enumerate(await asyncio.gather(*coros)):
            results[result["id"]] = result["text"]
            if result["ok"]:
                n_ok += 1
            else:
                n_err += 1
                if first_error is None:
                    first_error = result["error"]
            if (i + 1) % log_every == 0 or (i + 1) == len(tasks):
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-3)
                print(f"  [{i+1}/{len(tasks)}] {rate:.1f}/s  "
                      f"ETA={(len(tasks) - i - 1) / rate / 60:.1f}m  "
                      f"ok={n_ok} err={n_err}", flush=True)
    finally:
        await be.aclose()

    print(f"Batch done: {n_ok} ok, {n_err} errors in {time.time() - t0:.1f}s")
    if first_error:
        # One line, once — a wall of identical stack traces hides the cause.
        print(f"  first error: {first_error}")
    return results
