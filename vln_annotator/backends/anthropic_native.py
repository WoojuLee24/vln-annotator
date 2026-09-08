"""
Anthropic backend.

Anthropic has no OpenAI-compatible endpoint, and its image envelope differs:

    OpenAI     {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    Anthropic  {"type": "image", "source": {"type": "base64",
                                            "media_type": "image/jpeg", "data": "..."}}

That difference is the only reason this file exists.

Requires the `anthropic` package (pyproject extra `api`).
Example model id: claude-sonnet-5
"""
import base64
from pathlib import Path
from typing import List, Optional

_MEDIA = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}


def _image_block(path: Path) -> dict:
    p = Path(path)
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": _MEDIA.get(p.suffix.lower().lstrip("."), "image/jpeg"),
            "data": base64.b64encode(p.read_bytes()).decode("utf-8"),
        },
    }


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, *, model: Optional[str], api_key: str):
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:                                  # pragma: no cover
            raise RuntimeError(
                "backend 'anthropic' needs the anthropic package: "
                "pip install 'vln-annotator[api]'"
            ) from exc
        self.model = model or "claude-sonnet-5"
        self._client = AsyncAnthropic(api_key=api_key)

    async def chat(
        self,
        prompt: str,
        images: Optional[List[Path]] = None,
        *,
        max_tokens: int = 256,
        temperature: float = 0.3,
    ) -> str:
        content = [_image_block(p) for p in (images or [])[:3]]
        content.append({"type": "text", "text": prompt})
        resp = await self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": content}],
        )
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "".join(parts).strip()

    async def aclose(self) -> None:
        await self._client.close()
