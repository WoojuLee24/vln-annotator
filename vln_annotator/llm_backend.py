"""
VLN Auto-Annotator — LLM Backend
OpenAI-compatible async client for vLLM inference servers.
Supports both text-only and vision (image + text) modes.
"""
import asyncio
import base64
import time
from pathlib import Path
from typing import Dict, List, Optional

from openai import AsyncOpenAI, OpenAI


def image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_vision_message(prompt: str, image_paths: List[Path]) -> List[dict]:
    """Single user message with up to 3 embedded images followed by text prompt."""
    content = []
    for img in image_paths[:3]:
        b64 = image_to_base64(img)
        suffix = img.suffix.lower().lstrip(".")
        mime = "jpeg" if suffix in ("jpg", "jpeg") else "png"
        content.append({"type": "image_url",
                         "image_url": {"url": f"data:image/{mime};base64,{b64}"}})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def build_text_message(prompt: str) -> List[dict]:
    return [{"role": "user", "content": prompt}]


async def call_llm_async(
    client: AsyncOpenAI,
    prompt: str,
    *,
    image_paths: Optional[List[Path]] = None,
    max_tokens: int = 256,
    temperature: float = 0.3,
    semaphore: asyncio.Semaphore,
    task_id: object = None,
) -> Dict:
    """Single async LLM call with semaphore throttling. Returns dict with text/ok/error."""
    async with semaphore:
        messages = (build_vision_message(prompt, image_paths)
                    if image_paths else build_text_message(prompt))
        try:
            resp = await client.chat.completions.create(
                model=client.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return {"id": task_id, "text": resp.choices[0].message.content.strip(),
                    "ok": True, "error": None}
        except Exception as exc:
            return {"id": task_id, "text": "", "ok": False, "error": str(exc)}


async def batch_call_async(
    tasks: List[Dict],
    *,
    base_url: str,
    model: str,
    api_key: str = "EMPTY",
    max_tokens: int = 256,
    temperature: float = 0.3,
    concurrency: int = 12,
    log_every: int = 200,
) -> Dict:
    """
    Run many LLM calls concurrently.

    Each task must have:
      - "id": unique identifier returned in results
      - "prompt": str
      - "images": optional list of Path (for vision calls)

    Returns {id: text} for successful calls, empty string for failures.
    """
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    client.model = model  # attach for use in call_llm_async
    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()

    coros = [
        call_llm_async(
            client,
            t["prompt"],
            image_paths=t.get("images"),
            max_tokens=max_tokens,
            temperature=temperature,
            semaphore=sem,
            task_id=t["id"],
        )
        for t in tasks
    ]

    results: Dict = {}
    n_ok = n_err = 0
    for i, result in enumerate(await asyncio.gather(*coros)):
        results[result["id"]] = result["text"]
        if result["ok"]:
            n_ok += 1
        else:
            n_err += 1
        if (i + 1) % log_every == 0 or (i + 1) == len(tasks):
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 0.001)
            eta = (len(tasks) - i - 1) / rate
            print(f"  [{i+1}/{len(tasks)}] {rate:.1f}/s  ETA={eta/60:.1f}m  "
                  f"ok={n_ok} err={n_err}", flush=True)

    print(f"Batch done: {n_ok} ok, {n_err} errors in {time.time()-t0:.1f}s")
    return results
