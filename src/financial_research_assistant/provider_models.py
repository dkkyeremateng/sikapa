"""What a credential can actually use, asked of the provider right after login.

A login used to offer a hard-coded model list, which goes stale on its own: the
ChatGPT backend stopped serving the ``-codex`` names to ChatGPT accounts, and a
new model never showed up until someone edited the list. Each function here
asks one vendor which models this key or token can call, and returns them best
first, so the login can offer the newest as the default.

Each returns ``[{"name": ..., "context_window"?: ...}]``. The window is included
when the vendor states it, since the vendor's figure for this account beats the
built-in table's figure for the public model. Any failure raises; the login
reports it and falls back to the provider's built-in defaults.
"""

from __future__ import annotations

from typing import Any
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime

Listed = dict[str, Any]

#: Ids that are not chat models, on lists that mix in everything a key can call.
_NOT_CHAT = re.compile(
    r"embed|whisper|tts|dall-e|moderation|transcri|realtime|audio|image|rerank"
    r"|guard|search|computer-use|aqa|imagen|veo",
    re.IGNORECASE,
)

#: The ChatGPT backend hides any model whose ``minimal_client_version`` is above
#: the version the caller names, so an old number hides the newest models (0.130
#: offered gpt-5.5 alone; a current one offered eight, gpt-6.1 first). This asks
#: for everything; the proxy speaks Chat Completions to it either way.
CODEX_CLIENT_VERSION = "99.0.0"


def _get(url: str, headers: dict[str, str], timeout: float = 15.0) -> Any:
    req = urllib.request.Request(url, headers={"Accept": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (https)
        return json.loads(resp.read().decode("utf-8"))


def _created(row: dict[str, Any]) -> float:
    """When the vendor says the model was released, as a sortable number (0 when
    it doesn't say, which sorts last)."""
    value = row.get("created", row.get("created_at"))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0
    return 0.0


def _window(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) and value > 0 else None


def _listed(name: str, window: Any = None) -> Listed:
    out: Listed = {"name": name}
    if _window(window):
        out["context_window"] = _window(window)
    return out


def openai_compatible(base_url: str, key: str) -> list[Listed]:
    """``GET {base}/models``: OpenAI, Groq, OpenRouter and gateways such as
    gateframe. Newest first by ``created``; non-chat models dropped."""
    data = _get(base_url.rstrip("/") + "/models", {"Authorization": f"Bearer {key}"})
    rows = [r for r in (data.get("data") or []) if isinstance(r, dict) and r.get("id")]
    rows = [r for r in rows if not _NOT_CHAT.search(str(r["id"]))]
    rows.sort(key=_created, reverse=True)
    return [_listed(str(r["id"]), r.get("context_length")) for r in rows]


def anthropic(*, key: str = "", oauth_token: str = "") -> list[Listed]:
    """``GET /v1/models`` with an API key or a Claude subscription token."""
    headers = {"anthropic-version": "2023-06-01"}
    if oauth_token:
        headers.update({"Authorization": f"Bearer {oauth_token}",
                        "anthropic-beta": "oauth-2025-04-20"})
    else:
        headers["x-api-key"] = key
    data = _get("https://api.anthropic.com/v1/models?limit=1000", headers)
    rows = [r for r in (data.get("data") or []) if isinstance(r, dict) and r.get("id")]
    rows.sort(key=_created, reverse=True)
    return [_listed(str(r["id"]), r.get("max_input_tokens")) for r in rows]


def _gemini_rank(name: str) -> tuple[float, int, int]:
    """Newest version first; within one, stable before preview, then pro, flash,
    flash-lite. Gemini lists carry no release date, only the name."""
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    version = float(m.group(1)) if m else 0.0
    preview = 1 if re.search(r"preview|exp", name) else 0
    tier = 0 if "-pro" in name else 2 if "lite" in name else 1
    return (-version, preview, tier)


def google(key: str) -> list[Listed]:
    """The Gemini API's model list: those that can ``generateContent``."""
    data = _get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000",
                {"x-goog-api-key": key})
    out = []
    for r in data.get("models") or []:
        name = str(r.get("name", "")).removeprefix("models/")
        if (name.startswith("gemini") and not _NOT_CHAT.search(name)
                and "generateContent" in (r.get("supportedGenerationMethods") or [])):
            out.append(_listed(name, r.get("inputTokenLimit")))
    out.sort(key=lambda m: _gemini_rank(m["name"]))
    return out


def codex(api_base: str, token: str, account_id: str) -> list[Listed]:
    """The ChatGPT backend's Codex models for this account, in its own order
    (``priority``, lowest first), hidden ones dropped."""
    query = urllib.parse.urlencode({"client_version": CODEX_CLIENT_VERSION})
    from .codex_proxy import ORIGINATOR

    data = _get(f"{api_base.rstrip('/')}/models?{query}", {
        "Authorization": f"Bearer {token}",
        "chatgpt-account-id": account_id,
        "originator": ORIGINATOR,
    })
    rows = [r for r in (data.get("models") or [])
            if isinstance(r, dict) and r.get("slug") and r.get("visibility") == "list"]
    rows.sort(key=lambda r: r.get("priority", float("inf")))
    return [_listed(str(r["slug"]), r.get("context_window")) for r in rows]
