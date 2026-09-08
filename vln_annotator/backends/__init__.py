"""
VLN Auto-Annotator — LLM backends.

One interface, several providers. `batch_call_async` in ../llm_backend.py owns
concurrency, retries-by-omission and progress reporting; a backend owns only
"turn one prompt (+ up to 3 images) into one string".

    vllm | openai | gemini   -> openai_compat  (OpenAI-compatible HTTP)
    anthropic                -> anthropic_native (different image envelope)
    dry                      -> dry_run (no network; dumps prompts to disk)
"""
from typing import List, Optional, Protocol, runtime_checkable
from pathlib import Path

OPENAI_COMPATIBLE = ("vllm", "openai", "gemini", "local")
ALL_BACKENDS = OPENAI_COMPATIBLE + ("anthropic", "dry")

# Gemini exposes an OpenAI-compatible surface; the others are addressed directly.
DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
}

ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


@runtime_checkable
class Backend(Protocol):
    name: str

    async def chat(
        self,
        prompt: str,
        images: Optional[List[Path]] = None,
        *,
        max_tokens: int = 256,
        temperature: float = 0.3,
    ) -> str:
        """Return the assistant's text. Raise on failure; the caller records it."""
        ...

    async def aclose(self) -> None:
        ...


def get_backend(
    name: str,
    *,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    dry_run_dir: Optional[Path] = None,
) -> Backend:
    name = (name or "vllm").lower()
    if name not in ALL_BACKENDS:
        raise ValueError(f"unknown backend {name!r}; choose from {ALL_BACKENDS}")

    if name == "dry":
        from .dry_run import DryRunBackend
        return DryRunBackend(model=model, out_dir=dry_run_dir)

    if name == "anthropic":
        from .anthropic_native import AnthropicBackend
        return AnthropicBackend(model=model, api_key=api_key)

    from .openai_compat import OpenAICompatBackend
    return OpenAICompatBackend(
        name=name,
        base_url=base_url or DEFAULT_BASE_URLS.get(name),
        model=model,
        api_key=api_key,
    )


def resolve_api_key(name: str, explicit: Optional[str]) -> str:
    """Explicit value wins; else the provider's env var; else vLLM's 'EMPTY'."""
    import os
    if explicit:
        return explicit
    env = ENV_KEYS.get((name or "").lower())
    if env:
        key = os.environ.get(env, "")
        if not key:
            raise RuntimeError(
                f"backend {name!r} needs {env}. Put it in keys.env "
                f"(see keys.env.example) — docker/run.sh passes it with --env-file."
            )
        return key
    return "EMPTY"          # local vLLM ignores the key but the client requires one
