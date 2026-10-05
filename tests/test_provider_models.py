"""Asking each provider which models a credential can use.

The payloads are trimmed copies of what each vendor returned live on
2026-10-05; only the network call is faked.
"""

import pytest

from financial_research_assistant import provider_models


@pytest.fixture
def served(monkeypatch):
    """Serve ``payload`` for any request, recording the URL and headers asked."""
    calls: list[tuple[str, dict]] = []

    def serve(payload):
        def fake_get(url, headers, timeout=15.0):
            calls.append((url, headers))
            return payload
        monkeypatch.setattr(provider_models, "_get", fake_get)
        return calls

    return serve


def test_codex_lists_what_the_account_is_served_in_the_backends_order(served):
    calls = served({"models": [
        {"slug": "gpt-5.5", "visibility": "list", "priority": 13, "context_window": 272000},
        {"slug": "codex-auto-review", "visibility": "hide", "priority": 43},
        {"slug": "gpt-6.1-sol", "visibility": "list", "priority": 1, "context_window": 272000},
        {"slug": "gpt-6-sol", "visibility": "list", "priority": 3},
    ]})
    listed = provider_models.codex("https://chatgpt.com/backend-api/codex", "tok", "acct")
    assert [m["name"] for m in listed] == ["gpt-6.1-sol", "gpt-6-sol", "gpt-5.5"]
    assert listed[0]["context_window"] == 272000 and "context_window" not in listed[1]
    url, headers = calls[0]
    # An old client_version hides every newer model: 0.130 offered gpt-5.5 alone.
    assert url.startswith("https://chatgpt.com/backend-api/codex/models?")
    assert f"client_version={provider_models.CODEX_CLIENT_VERSION}" in url
    assert headers["Authorization"] == "Bearer tok" and headers["chatgpt-account-id"] == "acct"


def test_an_openai_compatible_list_is_newest_chat_model_first(served):
    calls = served({"data": [
        {"id": "auto", "created": 0},
        {"id": "text-embedding-3-large", "created": 1_800_000_000},
        {"id": "gpt-5.6", "created": 1_790_000_000, "context_length": 400000},
        {"id": "whisper-1", "created": 1_795_000_000},
        {"id": "gpt-5.5", "created": 1_780_000_000},
    ]})
    listed = provider_models.openai_compatible("https://router.gateframe.ai/v1/", "gf_key")
    assert [m["name"] for m in listed] == ["gpt-5.6", "gpt-5.5", "auto"]
    assert listed[0]["context_window"] == 400000
    assert calls[0][0] == "https://router.gateframe.ai/v1/models"
    assert calls[0][1]["Authorization"] == "Bearer gf_key"


def test_anthropic_takes_a_key_or_a_subscription_token(served):
    calls = served({"data": [
        {"id": "claude-haiku-4-5-20251001", "created_at": "2025-10-01T00:00:00Z"},
        {"id": "claude-opus-5-5", "created_at": "2026-08-20T00:00:00Z", "max_input_tokens": 1000000},
    ]})
    assert [m["name"] for m in provider_models.anthropic(key="sk-ant")] == [
        "claude-opus-5-5", "claude-haiku-4-5-20251001"]
    assert provider_models.anthropic(key="sk-ant")[0]["context_window"] == 1000000
    provider_models.anthropic(oauth_token="oat")
    key_headers, token_headers = calls[0][1], calls[-1][1]
    assert key_headers["x-api-key"] == "sk-ant" and "Authorization" not in key_headers
    assert token_headers["Authorization"] == "Bearer oat"
    assert token_headers["anthropic-beta"] == "oauth-2025-04-20"


def test_gemini_ranks_by_version_then_stable_then_tier(served):
    served({"models": [
        {"name": "models/gemini-2.5-pro", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-3.1-flash", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-3.1-pro-preview", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-3.1-pro", "supportedGenerationMethods": ["generateContent"],
         "inputTokenLimit": 2000000},
        {"name": "models/gemini-embedding-001", "supportedGenerationMethods": ["embedContent"]},
        {"name": "models/imagen-4", "supportedGenerationMethods": ["predict"]},
    ]})
    listed = provider_models.google("AIza")
    assert [m["name"] for m in listed] == [
        "gemini-3.1-pro", "gemini-3.1-flash", "gemini-3.1-pro-preview", "gemini-2.5-pro"]
    assert listed[0]["context_window"] == 2000000
