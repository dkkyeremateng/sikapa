"""Codex shim: chat.completions <-> Responses translation.

The translation half is pure and fully covered here. The transport half (the
upstream call to chatgpt.com) cannot be exercised without a live credential, so
it is deliberately thin and only its wiring is asserted.
"""

import json

import pytest

from financial_research_assistant import auth, codex_proxy, llm
from financial_research_assistant.codex_proxy import (
    ResponseStreamTranslator,
    to_completion,
    to_responses_request,
)


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("FINANCIAL_RESEARCH_AUTH_FILE", str(tmp_path / "auth.json"))


# --- request translation ------------------------------------------------------


def test_system_messages_become_instructions():
    out = to_responses_request({
        "model": "m",
        "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "system", "content": "cite sources"},
            {"role": "user", "content": "hi"},
        ],
    })
    assert out["instructions"] == "be terse\n\ncite sources"
    assert out["input"] == [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
    ]


def test_assistant_tool_calls_and_results_become_items():
    """The structural difference between the APIs: tool calls are top-level items,
    not fields on a message."""
    out = to_responses_request({
        "model": "m",
        "messages": [
            {"role": "user", "content": "quote AAPL"},
            {
                "role": "assistant",
                "content": "checking",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_quote", "arguments": '{"symbol":"AAPL"}'},
                }],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "231.40"},
        ],
    })
    kinds = [i["type"] for i in out["input"]]
    assert kinds == ["message", "message", "function_call", "function_call_output"]
    assert out["input"][2] == {
        "type": "function_call",
        "name": "get_quote",
        "arguments": '{"symbol":"AAPL"}',
        "call_id": "call_1",
    }
    assert out["input"][3] == {
        "type": "function_call_output", "call_id": "call_1", "output": "231.40",
    }


def test_tool_schemas_are_flattened():
    out = to_responses_request({
        "model": "m",
        "messages": [],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_quote",
                "description": "quote a symbol",
                "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}},
            },
        }],
    })
    assert out["tools"][0] == {
        "type": "function",
        "name": "get_quote",
        "description": "quote a symbol",
        "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}},
        "strict": False,
    }
    assert out["tool_choice"] == "auto"


def test_no_tools_key_when_none_given():
    out = to_responses_request({"model": "m", "messages": []})
    assert "tools" not in out and "tool_choice" not in out


def test_list_content_parts_are_flattened():
    out = to_responses_request({
        "model": "m",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "a"}, {"type": "text", "text": "b"},
        ]}],
    })
    assert out["input"][0]["content"][0]["text"] == "ab"


def test_upstream_always_streams():
    """We stream upstream regardless, and re-batch for a non-streaming caller —
    one code path instead of two."""
    assert to_responses_request({"model": "m", "messages": [], "stream": False})["stream"] is True


# --- response translation -----------------------------------------------------


def _events(*evs):
    t = ResponseStreamTranslator("m", completion_id="cmpl-x")
    return t, [c for ev in evs for c in t.event(ev)]


def test_text_deltas_become_content_chunks():
    _t, chunks = _events(
        {"type": "response.output_text.delta", "delta": "Hel"},
        {"type": "response.output_text.delta", "delta": "lo"},
    )
    assert [c["choices"][0]["delta"]["content"] for c in chunks] == ["Hel", "lo"]
    assert chunks[0]["object"] == "chat.completion.chunk"


def test_reasoning_deltas_surface_as_reasoning_content():
    """Keeps the TUI's 💭 panel working through the proxy — the adapter reads
    reasoning_content."""
    _t, chunks = _events({"type": "response.reasoning_text.delta", "delta": "hmm"})
    assert chunks[0]["choices"][0]["delta"]["reasoning_content"] == "hmm"


def test_tool_call_announce_then_argument_deltas():
    _t, chunks = _events(
        {"type": "response.output_item.added",
         "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_quote"}},
        {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": '{"sym'},
        {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": 'bol":"AAPL"}'},
    )
    first = chunks[0]["choices"][0]["delta"]["tool_calls"][0]
    assert first["index"] == 0 and first["id"] == "call_1"
    assert first["function"]["name"] == "get_quote"
    args = "".join(
        c["choices"][0]["delta"]["tool_calls"][0]["function"].get("arguments", "")
        for c in chunks
    )
    assert args == '{"symbol":"AAPL"}'


def test_parallel_tool_calls_get_distinct_indices():
    _t, chunks = _events(
        {"type": "response.output_item.added",
         "item": {"type": "function_call", "id": "a", "call_id": "c1", "name": "one"}},
        {"type": "response.output_item.added",
         "item": {"type": "function_call", "id": "b", "call_id": "c2", "name": "two"}},
        {"type": "response.function_call_arguments.delta", "item_id": "b", "delta": "{}"},
    )
    assert chunks[1]["choices"][0]["delta"]["tool_calls"][0]["index"] == 1
    # the argument delta must land on the SECOND call, not the first
    assert chunks[2]["choices"][0]["delta"]["tool_calls"][0]["index"] == 1


def test_completion_carries_usage_and_finish_reason():
    _t, chunks = _events({
        "type": "response.completed",
        "response": {"usage": {
            "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
            "input_tokens_details": {"cached_tokens": 40},
        }},
    })
    final = chunks[-1]
    assert final["choices"][0]["finish_reason"] == "stop"
    # renamed to the chat.completions spelling the adapter's usage events read
    assert final["usage"]["prompt_tokens"] == 100
    assert final["usage"]["completion_tokens"] == 20
    assert final["usage"]["prompt_tokens_details"]["cached_tokens"] == 40


def test_finish_reason_is_tool_calls_when_a_tool_was_called():
    _t, chunks = _events(
        {"type": "response.output_item.added",
         "item": {"type": "function_call", "id": "a", "call_id": "c", "name": "n"}},
        {"type": "response.completed", "response": {}},
    )
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_upstream_error_raises():
    t = ResponseStreamTranslator("m")
    with pytest.raises(codex_proxy.UpstreamError, match="rate limited"):
        t.event({"type": "response.failed",
                 "response": {"error": {"message": "rate limited"}}})


def test_unknown_events_are_ignored():
    t = ResponseStreamTranslator("m")
    assert t.event({"type": "response.output_item.done"}) == []
    assert t.event({"type": "something.new"}) == []


# --- re-batching for non-streaming callers ------------------------------------


def test_to_completion_folds_chunks():
    t = ResponseStreamTranslator("m", completion_id="cmpl-x")
    chunks = [c for ev in (
        {"type": "response.output_text.delta", "delta": "Hel"},
        {"type": "response.output_text.delta", "delta": "lo"},
        {"type": "response.completed", "response": {"usage": {"input_tokens": 5, "output_tokens": 1}}},
    ) for c in t.event(ev)]
    out = to_completion(chunks, "m", "cmpl-x")
    assert out["object"] == "chat.completion"
    assert out["choices"][0]["message"]["content"] == "Hello"
    assert out["usage"]["prompt_tokens"] == 5


def test_to_completion_reassembles_tool_calls():
    t = ResponseStreamTranslator("m", completion_id="cmpl-x")
    chunks = [c for ev in (
        {"type": "response.output_item.added",
         "item": {"type": "function_call", "id": "fc", "call_id": "call_1", "name": "get_quote"}},
        {"type": "response.function_call_arguments.delta", "item_id": "fc", "delta": '{"a":'},
        {"type": "response.function_call_arguments.delta", "item_id": "fc", "delta": "1}"},
        {"type": "response.completed", "response": {}},
    ) for c in t.event(ev)]
    call = to_completion(chunks, "m", "cmpl-x")["choices"][0]["message"]["tool_calls"][0]
    assert call["id"] == "call_1"
    assert call["function"] == {"name": "get_quote", "arguments": '{"a":1}'}


# --- server wiring ------------------------------------------------------------


def test_ensure_running_is_a_loopback_singleton():
    try:
        first = codex_proxy.ensure_running()
        assert first.startswith("http://127.0.0.1:") and first.endswith("/v1")
        assert codex_proxy.ensure_running() == first  # one server per process
    finally:
        codex_proxy.shutdown()


def test_proxy_reports_a_missing_credential_rather_than_hanging():
    import urllib.error
    import urllib.request

    try:
        base = codex_proxy.ensure_running()
        req = urllib.request.Request(
            base + "/chat/completions",
            data=json.dumps({"model": "m", "messages": []}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=10)
        assert exc.value.code == 401
        assert "login codex" in json.loads(exc.value.read())["error"]["message"]
    finally:
        codex_proxy.shutdown()


def test_codex_credential_found_in_any_tier():
    auth.set_credential("subagent", {"provider": "codex", "type": "oauth", "access": "t"})
    assert codex_proxy._codex_credential()["access"] == "t"
    assert codex_proxy._codex_scope() == "subagent"
    auth.delete("subagent")
    assert codex_proxy._codex_credential() is None
    assert codex_proxy._codex_scope() is None


def test_proxy_refreshes_the_tier_that_holds_the_credential(monkeypatch):
    """The credential is accepted from any tier, so refreshing only "default"
    would let one stored against quick/subagent expire with nothing renewing it."""
    import json
    import urllib.error
    import urllib.request

    from financial_research_assistant import oauth

    refreshed = []
    monkeypatch.setattr(oauth, "ensure_fresh", refreshed.append)
    auth.set_credential("subagent", {"provider": "codex", "type": "oauth", "access": "t"})
    monkeypatch.setattr(
        codex_proxy, "_upstream_events",
        lambda payload, cred, timeout=300.0: iter(()),
    )
    try:
        base = codex_proxy.ensure_running()
        req = urllib.request.Request(
            base + "/chat/completions",
            data=json.dumps({"model": "m", "messages": []}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10).read()
    finally:
        codex_proxy.shutdown()
    assert refreshed == ["subagent"]


def test_make_llm_routes_codex_through_the_proxy(monkeypatch):
    """The integration point: a stored Codex credential must not send the token
    straight at api.openai.com, which cannot accept it."""
    from financial_research_assistant import graph

    auth.set_credential(
        "default",
        {"provider": "codex", "type": "oauth", "access": "tok",
         "expires": 4_000_000_000_000, "base_url": "https://chatgpt.com/backend-api/codex"},
    )
    seen = {}
    monkeypatch.setattr("langchain_openai.ChatOpenAI", lambda **kw: seen.update(kw))
    try:
        llm._make_llm("gpt-5.5-codex")
        assert seen["base_url"].startswith("http://127.0.0.1:")
        assert seen["api_key"].get_secret_value() == "tok"
    finally:
        codex_proxy.shutdown()
