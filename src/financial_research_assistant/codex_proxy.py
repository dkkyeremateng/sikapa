"""Local OpenAI-compatible shim over the ChatGPT Codex backend.

A Codex OAuth token does not address the OpenAI API. It addresses
``chatgpt.com/backend-api/codex/responses``, which speaks the Responses API and
checks that the caller looks like the Codex CLI. ``ChatOpenAI`` speaks
``/v1/chat/completions``. This module bridges the two: a loopback HTTP server the
agent points ``OPENAI_API_BASE`` at, translating each request out and each
streamed event back.

Two halves, deliberately separated:

* **Translation** — ``to_responses_request`` / ``ResponseStreamTranslator`` are
  pure functions over dicts, so the mapping is tested without a socket or a
  credential.
* **Transport** — the server and the upstream call, which cannot be exercised
  offline and are kept as thin as possible for that reason.

Caveats worth knowing before relying on this. Since 4 April 2026 third-party
traffic bills as overage rather than drawing from a ChatGPT plan, so this buys no
flat-rate inference. The originator header is what makes the request pass as
Codex, and newer models have been reported resolving to a missing internal engine
for non-Codex originators. Neither is something this module can fix, and the
upstream shape is undocumented, so treat a breakage here as expected rather than
surprising.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing_extensions import override
from typing import Any
import json
import secrets
import threading
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = "https://chatgpt.com/backend-api/codex/responses"

# What marks the request as coming from the Codex CLI. The backend rejects
# callers it does not recognise, so these are load-bearing, not cosmetic.
ORIGINATOR = "codex_cli_rs"


# --- request translation ------------------------------------------------------


def _text_content(content: Any) -> str:
    """OpenAI content is a string or a list of typed parts; flatten to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") in ("text", "input_text", "output_text")
        )
    return "" if content is None else str(content)


def to_responses_request(body: dict[str, Any]) -> dict[str, Any]:
    """``/v1/chat/completions`` body → the Responses payload the backend takes.

    System messages become ``instructions`` (the Responses API keeps them out of
    the turn list); everything else becomes an ``input`` item. Assistant tool
    calls and their results are separate item types rather than message fields,
    which is the main structural difference between the two APIs.
    """
    instructions: list[str] = []
    items: list[dict[str, Any]] = []
    for msg in body.get("messages") or []:
        role = msg.get("role")
        if role == "system":
            instructions.append(_text_content(msg.get("content")))
            continue
        if role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id", ""),
                "output": _text_content(msg.get("content")),
            })
            continue
        if role == "assistant":
            text = _text_content(msg.get("content"))
            if text:
                items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                })
            for call in msg.get("tool_calls") or []:
                fn = call.get("function") or {}
                items.append({
                    "type": "function_call",
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments", "") or "{}",
                    "call_id": call.get("id", ""),
                })
            continue
        items.append({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": _text_content(msg.get("content"))}],
        })

    payload = {
        "model": body.get("model", "gpt-5.5"),
        "instructions": "\n\n".join(t for t in instructions if t),
        "input": items,
        "stream": True,  # always stream upstream; we re-batch if the caller didn't ask
        "store": False,
    }
    tools = []
    for tool in body.get("tools") or []:
        fn = tool.get("function") or {}
        tools.append({
            "type": "function",
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
            "strict": False,
        })
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = body.get("tool_choice", "auto")
        payload["parallel_tool_calls"] = bool(body.get("parallel_tool_calls", True))
    if body.get("reasoning_effort"):
        payload["reasoning"] = {"effort": body["reasoning_effort"]}
    return payload


# --- response translation -----------------------------------------------------


class ResponseStreamTranslator:
    """Responses SSE events → OpenAI ``chat.completion.chunk`` dicts.

    Stateful because the two protocols disagree about tool calls: Responses
    announces an item then streams its arguments, while chat.completions expects
    an index-keyed delta. The map from item id to index lives here.
    """

    def __init__(self, model: str, completion_id: str | None = None):
        self.model = model
        self.id = completion_id or f"chatcmpl-{uuid.uuid4().hex}"
        self._tool_index: dict[str, int] = {}
        self.usage: dict[str, Any] | None = None
        self.finish_reason = "stop"

    def _chunk(self, delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    def event(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        """Translate one upstream event into zero or more chunks."""
        kind = event.get("type", "")

        if kind == "response.output_text.delta":
            return [self._chunk({"content": event.get("delta", "")})]

        if kind in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            # Surfaced the way the OpenAI-compatible path already carries thinking,
            # so the TUI's 💭 panel keeps working through the proxy.
            return [self._chunk({"reasoning_content": event.get("delta", "")})]

        if kind == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") != "function_call":
                return []
            index = len(self._tool_index)
            self._tool_index[item.get("id", "")] = index
            self.finish_reason = "tool_calls"
            return [self._chunk({
                "tool_calls": [{
                    "index": index,
                    "id": item.get("call_id") or item.get("id", ""),
                    "type": "function",
                    "function": {"name": item.get("name", ""), "arguments": ""},
                }]
            })]

        if kind == "response.function_call_arguments.delta":
            index = self._tool_index.get(event.get("item_id", ""), 0)
            return [self._chunk({
                "tool_calls": [{
                    "index": index,
                    "function": {"arguments": event.get("delta", "")},
                }]
            })]

        if kind in ("response.completed", "response.incomplete"):
            resp = event.get("response") or {}
            raw = resp.get("usage") or {}
            if raw:
                # Responses names these differently from chat.completions; the
                # adapter's usage events read the chat.completions names.
                self.usage = {
                    "prompt_tokens": raw.get("input_tokens", 0),
                    "completion_tokens": raw.get("output_tokens", 0),
                    "total_tokens": raw.get("total_tokens", 0),
                    "prompt_tokens_details": {
                        "cached_tokens": (raw.get("input_tokens_details") or {}).get(
                            "cached_tokens", 0
                        )
                    },
                }
            chunk = self._chunk({}, finish_reason=self.finish_reason)
            if self.usage:
                chunk["usage"] = self.usage
            return [chunk]

        if kind in ("response.failed", "error"):
            detail = (event.get("response") or {}).get("error") or event.get("error") or {}
            raise UpstreamError(detail.get("message") or "upstream error")

        return []


class UpstreamError(Exception):
    """The Codex backend reported an error mid-stream."""


def to_completion(chunks: list[dict[str, Any]], model: str, completion_id: str) -> dict[str, Any]:
    """Fold streamed chunks into one ``chat.completion`` for a non-streaming caller."""
    content: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    finish_reason = "stop"
    usage = None
    for chunk in chunks:
        usage = chunk.get("usage") or usage
        choice = (chunk.get("choices") or [{}])[0]
        finish_reason = choice.get("finish_reason") or finish_reason
        delta = choice.get("delta") or {}
        if delta.get("content"):
            content.append(delta["content"])
        for call in delta.get("tool_calls") or []:
            slot = calls.setdefault(
                call.get("index", 0),
                {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
            )
            slot["id"] = call.get("id") or slot["id"]
            fn = call.get("function") or {}
            slot["function"]["name"] = fn.get("name") or slot["function"]["name"]
            slot["function"]["arguments"] += fn.get("arguments", "")
    message: dict[str, Any] = {"role": "assistant", "content": "".join(content) or None}
    if calls:
        message["tool_calls"] = [calls[i] for i in sorted(calls)]
    out: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }
    if usage:
        out["usage"] = usage
    return out


# --- transport ----------------------------------------------------------------


def _upstream_events(payload: dict[str, Any], cred: dict[str, Any], timeout: float = 300.0):
    """POST upstream and yield parsed SSE events."""
    req = urllib.request.Request(
        UPSTREAM,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {cred.get('access', '')}",
            "chatgpt-account-id": cred.get("account_id", ""),
            "OpenAI-Beta": "responses=experimental",
            "originator": ORIGINATOR,
            "session_id": str(uuid.uuid4()),
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (https)
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                yield json.loads(data)
            except ValueError:
                continue


#: Host header values that can only have come from something addressing the
#: loopback interface by its own name. A browser resolving an attacker's domain to
#: 127.0.0.1 (DNS rebinding) reaches the same socket but still sends that domain
#: here, so this is the one field that distinguishes the two.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _hostname(header: str) -> str:
    """The host part of a ``Host`` header, without the port and IPv6 brackets."""
    host = (header or "").strip().lower()
    if host.startswith("["):
        return host[1 : host.find("]")] if "]" in host else host[1:]
    return host.split(":", 1)[0]


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @override
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return  # the proxy is not an access log

    def _rejection(self) -> tuple[int, str] | None:
        """Why this request may not be served, or None if it may be.

        The proxy holds a credential that can spend money, so "listening only on
        127.0.0.1" is not by itself an access control: every process on the machine
        shares that interface, and so does every page in a browser running on it.
        Three checks, each closing a different way in:

        * **The bearer token.** A per-process secret, given only to the client this
          process builds. Nothing that merely found the port can present it.
        * **The Host header.** Loopback-only binding does not stop a page whose
          domain resolves to 127.0.0.1 from reaching this socket; the name it asks
          for is what gives it away.
        * **The content type.** ``application/json`` is not a value a form or a
          plain ``fetch`` can send without a CORS preflight, and this server answers
          no preflight. That turns a silent cross-origin POST into a blocked one.
        """
        if _hostname(self.headers.get("Host", "")) not in _LOOPBACK_HOSTS:
            return 403, "unexpected Host header"
        scheme, _, presented = (self.headers.get("Authorization") or "").partition(" ")
        if (
            not _token
            or scheme.lower() != "bearer"
            or not secrets.compare_digest(presented.strip(), _token)
        ):
            return 401, "missing or invalid proxy token"
        media_type = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip()
        if media_type.lower() != "application/json":
            return 415, "expected Content-Type: application/json"
        return None

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler's naming
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self.send_error(404)
            return
        rejection = self._rejection()
        if rejection is not None:
            # The body is deliberately left unread, so the connection cannot be
            # reused — whatever is still in the socket would otherwise be parsed as
            # the next request on it.
            self.close_connection = True
            self._json_error(*rejection)
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self.send_error(400, "invalid JSON")
            return

        from . import oauth

        # Read the credential per request rather than capturing it at startup, so
        # a refresh between turns is picked up without restarting the proxy.
        # Refresh the tier that actually HOLDS it: the credential is accepted from
        # any tier, so refreshing only "default" would let one stored against
        # `quick` or `subagent` expire with nothing renewing it.
        scope = _codex_scope()
        if scope:
            oauth.ensure_fresh(scope)
        cred = _codex_credential()
        if cred is None:
            self._json_error(401, "no codex credential stored — run /login codex")
            return

        wants_stream = bool(body.get("stream"))
        translator = ResponseStreamTranslator(body.get("model", "gpt-5.5"))
        try:
            events = _upstream_events(to_responses_request(body), cred)
            if wants_stream:
                # Nothing has reached the network yet — the generator only calls
                # upstream once it is iterated, which happens inside `_stream`,
                # after the status line is committed. So every failure of a
                # streaming request is a mid-stream one, and reporting it is
                # `_stream`'s job rather than this handler's.
                self._stream(events, translator)
                return
            chunks = [c for ev in events for c in translator.event(ev)]
            self._send_json(200, to_completion(chunks, translator.model, translator.id))
        except UpstreamError as exc:
            self._json_error(502, str(exc))
        except Exception as exc:
            self._json_error(502, f"codex proxy: {exc}")

    def _stream(self, events: Iterable[dict[str, Any]],
                translator: ResponseStreamTranslator) -> None:
        """Relay the translated chunks as SSE, failures included.

        Once the 200 and the event-stream headers are on the wire the response is
        committed: writing a second status line into the same connection puts an
        HTTP response in the middle of a body the client is already parsing as
        events, and it reads as corruption rather than as the error it is. Ending
        without ``[DONE]`` is no better — clients wait for that terminator, so the
        turn looks truncated or hangs. A failure here is therefore reported the one
        way the protocol still allows: a final error event, then the terminator.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        failure = ""
        try:
            for event in events:
                for chunk in translator.event(event):
                    self._sse(chunk)
        except (BrokenPipeError, ConnectionResetError):
            return  # the client hung up mid-answer; there is nobody left to tell
        except UpstreamError as exc:
            failure = str(exc)
        except Exception as exc:
            failure = f"codex proxy: {exc}"
        try:
            if failure:
                self._sse({"error": {"message": failure, "type": "codex_proxy"}})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

    def _sse(self, payload: dict[str, Any]) -> None:
        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
        self.wfile.flush()

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _json_error(self, code: int, message: str) -> None:
        self._send_json(code, {"error": {"message": message, "type": "codex_proxy"}})


def _codex_scope() -> str | None:
    """The tier whose credential is the Codex one, if any."""
    from . import auth

    for scope in auth.SCOPES:
        cred = auth.get(scope)
        if cred and cred.get("provider") == "codex":
            return scope
    return None


def _codex_credential() -> dict[str, Any] | None:
    """The stored Codex credential from whichever scope holds one."""
    from . import auth

    scope = _codex_scope()
    return auth.get(scope) if scope else None


_server = None
_lock = threading.Lock()
_token = ""


def local_token() -> str:
    """The bearer the proxy requires, or "" before it has started.

    Not a credential and never persisted: it is minted per process, lives only in
    memory, and dies with the process — its whole job is to tell the client this
    process built apart from everything else that can reach a loopback port.
    """
    return _token


def ensure_running() -> str:
    """Start the proxy once per process; return the base URL to point a client at.

    Bound to loopback only, and gated on ``local_token`` — it forwards a bearer
    token, so anything that can reach it can spend the account, and on a shared
    machine "reachable on 127.0.0.1" includes every other process and every page
    in a running browser.
    """
    global _server, _token
    with _lock:
        if _server is None:
            _token = secrets.token_urlsafe(32)
            _server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
            _server.daemon_threads = True
            threading.Thread(target=_server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{_server.server_port}/v1"


def serves(base_url: str | None) -> bool:
    """Whether ``base_url`` is this process's running proxy."""
    with _lock:
        return (_server is not None
                and base_url == f"http://127.0.0.1:{_server.server_port}/v1")


def shutdown() -> None:
    """Stop the proxy (tests; process exit does not need this)."""
    global _server, _token
    with _lock:
        if _server is not None:
            _server.shutdown()
            _server.server_close()
            _server = None
        _token = ""
