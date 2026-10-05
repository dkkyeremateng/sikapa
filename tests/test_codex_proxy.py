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


@pytest.fixture
def proxy():
    """The running proxy as ``(base_url, token)``, stopped afterwards so the next
    test gets a fresh server and a fresh token."""
    base = codex_proxy.ensure_running()
    try:
        yield base, codex_proxy.local_token()
    finally:
        codex_proxy.shutdown()


def _request(
    base,
    *,
    token=None,
    host=None,
    content_type="application/json",
    body=None,
    path="/chat/completions",
):
    """One request, spoken over a raw socket, returning the whole raw response.

    Raw rather than urllib because what is under test lives in the parts a
    convenience client hides: the ``Host`` header it fills in for you, and how many
    HTTP responses actually came back on the connection.
    """
    import socket
    import urllib.parse as up

    url = up.urlparse(base + path)
    payload = json.dumps(
        body if body is not None else {"model": "m", "messages": []}
    ).encode()
    head = [
        f"POST {url.path} HTTP/1.1",
        f"Host: {host or f'127.0.0.1:{url.port}'}",
        f"Content-Length: {len(payload)}",
        "Connection: close",
    ]
    if content_type:
        head.append(f"Content-Type: {content_type}")
    if token:
        head.append(f"Authorization: Bearer {token}")
    sock = socket.create_connection(("127.0.0.1", url.port), timeout=10)
    try:
        sock.sendall(("\r\n".join(head) + "\r\n\r\n").encode() + payload)
        out = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            out += chunk
    finally:
        sock.close()
    return out


def _status(raw: bytes) -> int:
    return int(raw.split(b" ", 2)[1])


def _payload(raw: bytes) -> dict:
    return json.loads(raw.partition(b"\r\n\r\n")[2])


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
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {codex_proxy.local_token()}",
            },
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
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {codex_proxy.local_token()}",
            },
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
        # the proxy's own token, not the spendable ChatGPT one — the proxy reads
        # that from the store per request and the client has no use for it
        assert seen["api_key"].get_secret_value() == codex_proxy.local_token()
        assert seen["api_key"].get_secret_value() != "tok"
    finally:
        codex_proxy.shutdown()


@pytest.mark.parametrize("model", ["gpt-5.5-codex", "gpt-5.5"])
def test_the_real_client_reaches_the_proxy_for_a_codex_model(model, monkeypatch):
    """The test above swaps ChatOpenAI for a recorder, so it can't see which path
    the real client calls. LangChain sends any model with "codex" in its name to
    /v1/responses by itself, and the proxy answered that with a 404 HTML page as
    the user's first message after `/login codex`. Here the real client talks to
    the real proxy; only the ChatGPT backend is faked."""
    auth.set_credential(
        "default",
        {"provider": "codex", "type": "oauth", "access": "tok",
         "expires": 4_000_000_000_000, "base_url": "https://chatgpt.com/backend-api/codex"},
    )
    sent = []

    def upstream(payload, cred, timeout=300.0):
        sent.append(payload)
        return iter([
            {"type": "response.output_text.delta", "delta": "O"},
            {"type": "response.output_text.delta", "delta": "K"},
            {"type": "response.completed", "response": {}},
        ])

    monkeypatch.setattr(codex_proxy, "_upstream_events", upstream)
    try:
        reply = llm._make_llm(model).invoke("say OK")
    finally:
        codex_proxy.shutdown()
    assert reply.content == "OK"
    assert sent and sent[0]["model"] == model


def test_only_the_proxy_gets_chat_completions_pinned(monkeypatch):
    """On OpenAI itself codex models are Responses-only, so the pin must not
    leak to an ordinary endpoint."""
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    plain = llm._make_llm("gpt-5.5-codex", api_key="sk-test")
    assert plain._use_responses_api({}) is True
    assert not codex_proxy.serves("https://api.openai.com/v1")


# --- who may talk to the proxy ------------------------------------------------


def test_a_request_without_the_token_is_rejected(proxy):
    """Loopback-only is not an access control: every process on the machine shares
    127.0.0.1, and the credential behind this endpoint spends money."""
    base, _token = proxy
    raw = _request(base)
    assert _status(raw) == 401
    assert "token" in _payload(raw)["error"]["message"]


def test_a_request_with_the_wrong_token_is_rejected(proxy):
    base, _token = proxy
    assert _status(_request(base, token="not-the-token")) == 401


def test_the_token_is_required_before_any_credential_is_touched(proxy, monkeypatch):
    """An unauthenticated caller must not be able to make the proxy read, refresh
    or spend the stored credential — the 401 comes first."""
    from financial_research_assistant import oauth

    auth.set_credential("default", {"provider": "codex", "type": "oauth", "access": "t"})
    touched = []
    monkeypatch.setattr(oauth, "ensure_fresh", touched.append)
    monkeypatch.setattr(
        codex_proxy, "_upstream_events",
        lambda payload, cred, timeout=300.0: touched.append("upstream") or iter(()),
    )
    base, _token = proxy
    assert _status(_request(base)) == 401
    assert touched == []


def test_an_unexpected_host_header_is_rejected(proxy):
    """A page whose domain resolves to 127.0.0.1 reaches this socket anyway; the
    name it asks for is the only thing that gives it away."""
    base, token = proxy
    raw = _request(base, token=token, host="rebound.example")
    assert _status(raw) == 403
    assert "Host" in _payload(raw)["error"]["message"]


def test_a_non_json_content_type_is_rejected(proxy):
    """text/plain is the one a cross-origin fetch can send with no preflight, so
    requiring JSON is what forces the browser to ask first — and be refused."""
    base, token = proxy
    assert _status(_request(base, token=token, content_type="text/plain")) == 415


def test_each_process_mints_its_own_token():
    """In memory, never stored: it identifies this process's client, and a value
    that outlived the process would be a secret to leak for no benefit."""
    assert codex_proxy.local_token() == ""
    try:
        codex_proxy.ensure_running()
        first = codex_proxy.local_token()
        assert len(first) >= 32
        assert codex_proxy.ensure_running() and codex_proxy.local_token() == first
    finally:
        codex_proxy.shutdown()
    assert codex_proxy.local_token() == ""
    try:
        codex_proxy.ensure_running()
        assert codex_proxy.local_token() != first
    finally:
        codex_proxy.shutdown()


# --- failing mid-stream --------------------------------------------------------


def test_mid_stream_failure_is_reported_as_an_event_not_a_second_response(
    proxy, monkeypatch
):
    """Once the 200 and the SSE headers are out, the response is committed: a
    second status line lands inside the body the client is parsing as events, and
    no [DONE] leaves the turn looking truncated rather than failed."""
    base, token = proxy
    auth.set_credential("default", {"provider": "codex", "type": "oauth", "access": "t"})

    def fails_after_a_chunk(payload, cred, timeout=300.0):
        yield {"type": "response.output_text.delta", "delta": "Hel"}
        raise codex_proxy.UpstreamError("rate limited")

    monkeypatch.setattr(codex_proxy, "_upstream_events", fails_after_a_chunk)
    raw = _request(base, token=token, body={"model": "m", "messages": [], "stream": True})
    head, _, body = raw.partition(b"\r\n\r\n")

    assert _status(raw) == 200
    assert b"text/event-stream" in head
    assert raw.count(b"HTTP/1.") == 1  # one response on the connection, not two

    lines = [line for line in body.split(b"\n\n") if line.startswith(b"data: ")]
    assert lines[-1] == b"data: [DONE]"
    events = [json.loads(line[6:]) for line in lines[:-1]]
    assert events[0]["choices"][0]["delta"]["content"] == "Hel"
    assert events[-1]["error"]["message"] == "rate limited"


def test_an_unexpected_mid_stream_failure_is_reported_the_same_way(proxy, monkeypatch):
    base, token = proxy
    auth.set_credential("default", {"provider": "codex", "type": "oauth", "access": "t"})

    def explodes(payload, cred, timeout=300.0):
        raise OSError("connection reset by peer")
        yield  # pragma: no cover - generator marker

    monkeypatch.setattr(codex_proxy, "_upstream_events", explodes)
    raw = _request(base, token=token, body={"model": "m", "messages": [], "stream": True})
    body = raw.partition(b"\r\n\r\n")[2]
    lines = [line for line in body.split(b"\n\n") if line.startswith(b"data: ")]
    assert _status(raw) == 200 and raw.count(b"HTTP/1.") == 1
    assert lines[-1] == b"data: [DONE]"
    assert "connection reset" in json.loads(lines[-2][6:])["error"]["message"]


def test_a_client_that_hangs_up_mid_stream_is_not_an_error():
    """Nobody is left to receive the error, so there is nothing to report — and a
    traceback per abandoned stream would bury the failures that do matter."""

    class HungUp:
        def write(self, _data):
            raise BrokenPipeError(32, "broken pipe")

        def flush(self):
            pass

    handler = codex_proxy._Handler.__new__(codex_proxy._Handler)
    handler.wfile = HungUp()
    handler.send_response = lambda *_a, **_k: None
    handler.send_header = lambda *_a, **_k: None
    handler.end_headers = lambda: None
    handler._stream(
        iter([{"type": "response.output_text.delta", "delta": "hi"}]),
        ResponseStreamTranslator("m"),
    )
