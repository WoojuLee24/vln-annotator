"""
OpenAI-compatible backend — vLLM (remote Gemma or local Cosmos), OpenAI, Gemini.

Images travel as `image_url` entries holding a data: URI. This is the path the
annotator has always used; it is kept byte-identical in behaviour so that the
calibration numbers in README.md still describe it.
"""
import base64
from pathlib import Path
from typing import List, Optional

from openai import AsyncOpenAI

_MIME = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}


def image_to_data_uri(path: Path) -> str:
    b64 = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
    mime = _MIME.get(Path(path).suffix.lower().lstrip("."), "jpeg")
    return f"data:image/{mime};base64,{b64}"


def build_messages(prompt: str, image_paths: Optional[List[Path]]) -> List[dict]:
    """Single user message: up to 3 images, then the text."""
    if not image_paths:
        return [{"role": "user", "content": prompt}]
    content = [
        {"type": "image_url", "image_url": {"url": image_to_data_uri(p)}}
        for p in image_paths[:3]
    ]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


class OpenAICompatBackend:
    def __init__(self, *, name: str, base_url: Optional[str], model: str, api_key: str):
        self.name = name
        self.model = model
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def chat(
        self,
        prompt: str,
        images: Optional[List[Path]] = None,
        *,
        max_tokens: int = 256,
        temperature: float = 0.3,
    ) -> str:
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=build_messages(prompt, images),
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()

    async def aclose(self) -> None:
        await self._client.close()
