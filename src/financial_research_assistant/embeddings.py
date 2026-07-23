"""Text embeddings for semantic memory recall (``MEMORY_BACKEND=semantic``).

Uses an OpenAI-compatible embeddings endpoint via ``langchain-openai`` (already a
core dependency), so it honors the same ``OPENAI_API_BASE`` / ``OPENAI_API_KEY``
config as the chat model — a local server (llama.cpp / Ollama / LM Studio that
serves ``/v1/embeddings``) works with a dummy key, and cloud OpenAI works with a
real one. ``MEMORY_EMBED_MODEL`` selects the model (default
``text-embedding-3-small``).

Every function returns ``None`` on any failure (no key, no endpoint, network/model
error). Semantic memory treats ``None`` as "embeddings unavailable" and falls back
to keyword ranking — so enabling the semantic backend never hard-fails a turn, and
tests stay hermetic by monkeypatching ``embed_query``.
"""

from __future__ import annotations

import os
import math


def _client():
    """An OpenAIEmbeddings client from env, or None when no key/endpoint is set."""
    base = os.environ.get("OPENAI_API_BASE") or None
    key = os.environ.get("OPENAI_API_KEY") or ("dummy" if base else None)
    if not key:
        return None
    try:
        from langchain_openai import OpenAIEmbeddings
        from pydantic import SecretStr

        model = os.environ.get("MEMORY_EMBED_MODEL") or "text-embedding-3-small"
        return OpenAIEmbeddings(model=model, base_url=base, api_key=SecretStr(key))
    except Exception:
        return None


def embed_query(text: str) -> list[float] | None:
    """Embed a single string, or None if embeddings are unavailable/failed."""
    client = _client()
    if client is None:
        return None
    try:
        return client.embed_query(text)
    except Exception:
        return None


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 if either is degenerate)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
