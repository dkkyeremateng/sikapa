"""OAuth login flows, kept UI-neutral so one implementation serves every surface.

A login has to talk to whoever is driving it — open a browser, show a device code,
ask for a pasted value. Wiring that to a specific interface is what forces a
second copy of the flow for every new one. Instead a flow receives
``LoginCallbacks`` and never learns whether it is talking to the Textual TUI, the
headless CLI, or a test, which is the same bargain ``events.AgentEvent`` makes for
output: one contract, many renderers.

Providers implement ``login`` (obtain a credential) and ``refresh`` (renew an
expiring one) and are looked up by name in a registry, so adding a provider is a
registration rather than an edit to the TUI. The credential they return is a plain
dict handed straight to ``auth.set_credential``; this module never writes it
itself, and ``auth`` never reaches the network.
"""

from __future__ import annotations

from typing_extensions import override
from typing import Any, Callable, Protocol, runtime_checkable
import base64
import hashlib
import json
import os
import secrets
import threading
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer

# How long a browser round-trip may take before we give up and offer the manual
# paste path. Generous: it includes the user finding the window and logging in.
LOGIN_TIMEOUT = 300.0


def _noop(*_a: Any, **_k: Any) -> None:
    return None


def _no_input(_prompt: str) -> str:
    return ""


@dataclass
class LoginCallbacks:
    """What a flow may ask of its driver. Every one has a default, so a caller
    only implements what the providers it offers actually use."""

    #: Send the user to this URL (open a browser, and show it as a fallback).
    on_auth: Callable[[str], None] = _noop
    #: Show a device code and the URL to enter it at (device-flow providers).
    on_device_code: Callable[[str, str], None] = _noop
    #: Ask for a value and return it — the paste-the-code path when no browser
    #: on this machine can reach our loopback (SSH, containers).
    on_prompt: Callable[[str], str] = _no_input
    #: Ask for a value that must not be echoed (an API key). Separate from
    #: ``on_prompt`` because the driver has to render it differently — a masked
    #: input, ``getpass`` — and a flow should not have to know which.
    on_secret: Callable[[str], str] = _no_input
    #: Progress narration; safe to ignore.
    on_status: Callable[[str], None] = _noop


class LoginError(Exception):
    """A login that failed for a reason worth showing the user verbatim."""


@runtime_checkable
class OAuthProvider(Protocol):
    name: str
    label: str

    def login(self, cb: LoginCallbacks) -> dict[str, Any]:
        """Run the flow and return a credential dict for ``auth.set_credential``."""
        ...

    def refresh(self, cred: dict[str, Any]) -> dict[str, Any]:
        """Return a renewed credential, or ``cred`` unchanged if it can't expire."""
        ...


# --- loopback callback server -------------------------------------------------


def _state_matches(expected: str, returned: str | None) -> bool:
    """Constant-time comparison of the returned ``state`` against the sent one."""
    return bool(returned) and secrets.compare_digest(returned, expected)


def _readable(text: str, limit: int = 200) -> str:
    """A vendor-supplied string made safe to print in a terminal.

    ``error_description`` arrives over a URL that anything on this machine can
    construct, so it is untrusted input on its way to a TTY: control characters
    there can rewrite the line, hide text, or repaint the screen. Keep the
    printable part and cap the length.
    """
    return "".join(c for c in text if c.isprintable())[:limit].strip()


def _denial(error: str, description: str = "") -> str:
    """Why the provider refused, phrased for the person who refused it.

    A declined consent screen is a decision, not a malfunction. Left as a bare
    "no code received" it becomes the paste-the-code prompt, which asks the user
    to produce something that was never issued.
    """
    known = {
        "access_denied": "the sign-in was declined in the browser",
        "consent_required": "the provider needs consent that was not granted",
        "invalid_scope": "the provider rejected the permissions this app asked for",
        "server_error": "the provider hit an error of its own during sign-in",
    }
    lead = known.get(error, f"the provider refused the sign-in ({_readable(error, 60)})")
    detail = _readable(description)
    return f"{lead}: {detail}" if detail else lead


@contextmanager
def loopback(
    path: str = "/callback",
    timeout: float | None = None,
    port: int | None = None,
    state: str | None = None,
):
    """Serve one OAuth redirect on an ephemeral localhost port.

    Yields ``(callback_url, wait)``; ``wait()`` blocks until the browser hits the
    redirect and returns the ``code`` query parameter, or None on timeout.
    ``timeout`` resolves from ``LOGIN_TIMEOUT`` at call time, not as a default
    argument — bound at definition time it would ignore the module attribute.

    ``port=None`` lets the OS pick, so concurrent logins never collide. Pass an
    explicit port only for a vendor that validates ``redirect_uri`` against an
    exact registered value (OpenAI requires 1455) — then a second simultaneous
    login does collide, and gets a clear error rather than a confusing timeout.

    ``state`` is the value the authorize URL carried; a callback that does not
    echo it back is not this sign-in and is ignored. Every process and every page
    on this machine can GET a loopback port, and the fixed one above is guessable
    by construction, so without the comparison any local caller could hand the
    flow an authorization code of its own choosing and have it redeemed and
    stored. Ignored rather than fatal: a stray request must not be able to cancel
    a login either, so the server keeps waiting for the real redirect.

    ``wait()`` raises ``LoginError`` when the provider itself refused — a declined
    consent screen has a reason worth repeating, and no code will ever arrive.
    """
    timeout = LOGIN_TIMEOUT if timeout is None else timeout
    received: dict[str, str | None] = {}
    done = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's naming
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != path:
                self.send_response(404)
                self.end_headers()
                return
            query = urllib.parse.parse_qs(parsed.query)
            if state is not None and not _state_matches(state, (query.get("state") or [None])[0]):
                self._page(
                    400,
                    "Not this sign-in.",
                    "This callback did not carry the value the sign-in sent, so it "
                    "was ignored. Nothing was stored.",
                )
                return
            error = (query.get("error") or [None])[0]
            if error:
                received["error"] = _denial(
                    error, (query.get("error_description") or [""])[0]
                )
                self._page(400, "Sign-in refused.", received["error"])
            else:
                received["code"] = (query.get("code") or [None])[0]
                self._page(
                    200,
                    "Signed in.",
                    "You can close this tab and return to the terminal.",
                )
            done.set()

        def _page(self, code: int, heading: str, message: str) -> None:
            from html import escape

            body = (
                "<!doctype html><meta charset=utf-8>"
                "<body style='font:16px system-ui;padding:3rem'>"
                f"<h2>{escape(heading)}</h2><p>{escape(message)}</p></body>"
            ).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        @override
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return  # keep the redirect out of stderr

    try:
        server = HTTPServer(("127.0.0.1", port or 0), Handler)
    except OSError as exc:
        raise LoginError(
            f"could not listen on port {port}: {exc}. This provider requires that "
            "exact port — close whatever is using it (another sign-in?) and retry."
        ) from exc
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def wait() -> str | None:
        if not done.wait(timeout):
            return None
        refusal = received.get("error")
        if refusal:
            raise LoginError(refusal)
        return received.get("code")

    try:
        yield f"http://localhost:{server.server_port}{path}", wait
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# --- PKCE ---------------------------------------------------------------------


def pkce_pair() -> tuple[str, str]:
    """A (verifier, S256 challenge) pair, base64url without padding per RFC 7636."""
    verifier = secrets.token_urlsafe(64)[:128]
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _send(req: urllib.request.Request, timeout: float) -> dict[str, Any]:
    """Perform the request, turning an HTTP error into one that says *why*.

    OAuth token endpoints put the useful part (``invalid_grant``,
    ``redirect_uri_mismatch``, …) in the response body, which urllib discards —
    leaving a bare "HTTP Error 400: Bad Request" that could mean any of a dozen
    things. Read the body and put it in the message.
    """
    from urllib.error import HTTPError  # a bare `import urllib.error` here would
    # rebind `urllib` as a local, hiding urllib.request from the line below.

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (https)
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", "replace").strip()[:400]
        except Exception:
            detail = ""
        raise LoginError(f"HTTP {exc.code} from {req.full_url}: {detail or exc.reason}") from exc


def _post_form(url: str, payload: dict[str, Any], headers: dict[str, Any] | None = None, timeout: float = 30.0) -> dict[str, Any]:
    """POST ``application/x-www-form-urlencoded`` — what RFC 6749 token endpoints
    take, and what all three subscription providers below use. Module-level so
    tests replace it instead of opening a socket."""
    data = urllib.parse.urlencode(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "User-Agent": "financial-research-assistant",
            **(headers or {}),
        },
        method="POST",
    )
    return _send(req, timeout)


def _expires_ms(token_response: dict[str, Any], default_s: float = 3600.0) -> float:
    """``expires_in`` (seconds, per RFC 6749) → the absolute ms timestamp the store
    keeps, so ``needs_refresh`` can compare it against the clock."""
    import time

    try:
        seconds = float(token_response.get("expires_in") or default_s)
    except (TypeError, ValueError):
        seconds = default_s
    return time.time() * 1000 + seconds * 1000


def _post_json(url: str, payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    """POST JSON, return the parsed response. Module-level so tests replace it
    rather than opening a socket — the same shape the other network modules use."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "financial-research-assistant",
        },
        method="POST",
    )
    return _send(req, timeout)


# --- providers ----------------------------------------------------------------


class OpenRouterProvider:
    """OpenRouter PKCE.

    The useful property for a long-lived agent: the exchange mints a *real,
    user-controlled API key* billed from your OpenRouter credits, not a token that
    expires. So there is nothing to refresh, nothing to re-login when a laptop
    sleeps through an expiry, and revocation is a button on their dashboard rather
    than a state this app has to model.
    """

    name = "openrouter"
    label = "OpenRouter — mints an API key billed from your credits"

    AUTH_URL = "https://openrouter.ai/auth"
    KEYS_URL = "https://openrouter.ai/api/v1/auth/keys"
    BASE_URL = "https://openrouter.ai/api/v1"

    MODELS: list[str] = []
    # Nothing to suggest: OpenRouter routes to whatever model you name, so the
    # window depends on that choice rather than on the credential.
    CONTEXT_WINDOW: int | None = None

    def login(self, cb: LoginCallbacks) -> dict[str, Any]:
        models = ask_models(cb, self.MODELS)
        window = ask_context_window(
            cb, suggested_window([m["name"] for m in models], self.CONTEXT_WINDOW)
        )
        models = apply_window(models, window)
        verifier, challenge = pkce_pair()
        # No state to compare: this authorize endpoint takes `callback_url` and the
        # challenge, and nothing else — a parameter it does not know is a parameter
        # it will not echo back. The port is ephemeral rather than fixed, so the
        # window is narrower than the vendors that pin one, and the code is still
        # worthless without the verifier.
        with loopback() as (callback_url, wait):
            params = {
                "callback_url": callback_url,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            }
            cb.on_auth(f"{self.AUTH_URL}?{urllib.parse.urlencode(params)}")
            cb.on_status("waiting for the browser to come back…")
            code = wait()
        if not code:
            # No browser on this machine could reach our loopback (SSH, container).
            # The redirect still carries ?code= in its URL, so a human can bring it
            # across by hand rather than being stuck.
            cb.on_status("no callback received — paste the code from the URL instead")
            code = (cb.on_prompt("authorization code: ") or "").strip()
        if not code:
            raise LoginError("no authorization code received")
        try:
            resp = _post_json(
                self.KEYS_URL,
                {
                    "code": code,
                    "code_verifier": verifier,
                    "code_challenge_method": "S256",
                },
            )
        except Exception as exc:  # network, HTTP error, bad JSON
            raise LoginError(f"could not exchange the code: {exc}") from exc
        key = (resp or {}).get("key")
        if not key:
            raise LoginError("OpenRouter returned no key")
        cred = {
            "provider": self.name,
            "type": "api_key",
            "key": key,
            # Pinned so the key is never paired with a stale OPENAI_API_BASE —
            # see graph._credentials.
            "base_url": self.BASE_URL,
            "models": models,
        }
        stash_window(cred, models, window)
        return cred

    def refresh(self, cred: dict[str, Any]) -> dict[str, Any]:
        return cred  # a minted key does not expire


class _SubscriptionProvider:
    """Shared machinery for the three consumer-subscription PKCE flows.

    READ THIS BEFORE USING ANY SUBCLASS. All three vendors restrict these tokens
    to their own first-party clients, and two of them enforce it server-side:

    * **Anthropic** — enforcement since January 2026. The flow completes and the
      token stores fine; every inference call then returns "This credential is
      only authorized for use with Claude Code and cannot be used for other API
      requests." OpenCode and Goose both shipped this and reverted to API keys.
    * **Google** — banned February 2026 with account suspensions (including paid
      Ultra), detection from 25 March 2026. Separately, Code Assist stopped
      serving the individual / AI Pro / AI Ultra tiers on 18 June 2026, so there
      is no endpoint left for those subscriptions at all.
    * **OpenAI** — still functional, but only through the Codex request shape
      (see ``codex_proxy``), and since 4 April 2026 third-party traffic bills as
      overage rather than drawing from the subscription.

    So none of these buys flat-rate inference. They are here because they were
    asked for; ``login_warning`` is surfaced by every caller so the failure mode
    is stated up front rather than discovered as an opaque 401.
    """

    name = ""
    label = ""
    login_warning = ""
    #: Environment variables naming the vendor's OAuth client. The values are
    #: public (each vendor ships them in its own client) but are kept out of the
    #: source; no secret variable means a public PKCE client with no secret.
    CLIENT_ID_ENV = ""
    CLIENT_SECRET_ENV = ""
    #: Where those values come from, named in the error when one is missing.
    CLIENT_SOURCE = ""
    AUTH_URL = ""
    TOKEN_URL = ""
    SCOPES = ""
    API_BASE = ""
    #: Fixed redirect port when the vendor validates an exact redirect_uri; None
    #: means any ephemeral loopback port is accepted.
    REDIRECT_PORT: int | None = None
    REDIRECT_PATH = "/callback"
    #: A vendor-hosted callback page instead of a loopback server. The browser
    #: lands on the vendor's own page showing the code and the user copies it —
    #: there is nothing for us to listen on, so the flow skips the loopback.
    REDIRECT_URI = ""
    #: Token endpoint body encoding. RFC 6749 says form; Anthropic wants JSON, and
    #: sending the wrong one is a bare 400 with no detail.
    EXCHANGE_AS_JSON = False
    #: Extra authorize-URL parameters this vendor requires.
    AUTH_EXTRA: dict[str, Any] = {}
    #: Whether the token exchange must echo the state value back.
    SEND_STATE = False

    def client_credentials(self) -> tuple[str, str]:
        """The OAuth client id and secret (the secret is empty for a public PKCE
        client), read from the environment at call time so it works whether or
        not .env was loaded before import. Raises LoginError when one is unset."""
        names = [n for n in (self.CLIENT_ID_ENV, self.CLIENT_SECRET_ENV) if n]
        values = {n: os.environ.get(n, "").strip() for n in names}
        if not self.CLIENT_ID_ENV or not all(values.values()):
            raise LoginError(
                f"{self.name} sign-in needs {' and '.join(names)} set: the public "
                f"client {'pair' if len(names) > 1 else 'id'} from {self.CLIENT_SOURCE}."
            )
        secret = values[self.CLIENT_SECRET_ENV] if self.CLIENT_SECRET_ENV else ""
        return values[self.CLIENT_ID_ENV], secret

    def _authorize_url(self, challenge: str, redirect_uri: str, state: str) -> str:
        params = {
            "client_id": self.client_credentials()[0],
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": self.SCOPES,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
            **self.AUTH_EXTRA,
        }
        return f"{self.AUTH_URL}?{urllib.parse.urlencode(params)}"

    def _exchange_payload(
        self, code: str, verifier: str, redirect_uri: str, state: str = ""
    ) -> dict[str, Any]:
        client_id, client_secret = self.client_credentials()
        payload = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "code_verifier": verifier,
        }
        if self.SEND_STATE:
            # This vendor's token endpoint checks state, and in its flow state
            # doubles as the verifier — see `login` for why that pairing is theirs
            # alone and not a default worth copying.
            payload["state"] = state or verifier
        if client_secret:
            payload["client_secret"] = client_secret
        return payload

    def _exchange(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.EXCHANGE_AS_JSON:
            return _post_json(self.TOKEN_URL, payload)
        return _post_form(self.TOKEN_URL, payload)

    #: LangChain integration + model this credential is valid for. Stored with the
    #: credential so ``/login`` actually switches the client, rather than leaving a
    #: token pointed at the previous provider's endpoint.
    MODEL_PROVIDER = "openai"
    #: Models this credential offers; the first is its default. A list because one
    #: key serves several, and picking between them should not need a re-login.
    MODELS: list[str] = []
    #: Context window in tokens, when the vendor publishes one for these models.
    #: None means "no better answer than the built-in table".
    CONTEXT_WINDOW: int | None = None

    def _credential(self, token: dict[str, Any], models: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        # No credential-wide context_window: it lives on each model config, which
        # is the only granularity that can be right for a key serving models with
        # different windows. Older stores that have one are still read.
        return {
            "provider": self.name,
            "type": "oauth",
            "access": token.get("access_token", ""),
            "refresh": token.get("refresh_token", ""),
            "expires": _expires_ms(token),
            "base_url": self.API_BASE,
            "model_provider": self.MODEL_PROVIDER,
            "models": [dict(m) for m in (models if models is not None else
                                         [_model_config_for(n) for n in self.MODELS])],
        }

    def login(self, cb: LoginCallbacks) -> dict[str, Any]:
        if self.login_warning:
            cb.on_status(self.login_warning)
        self.client_credentials()  # an unconfigured client fails before any prompt
        models = ask_models(cb, self.MODELS)
        window = ask_context_window(
            cb, suggested_window([m["name"] for m in models], self.CONTEXT_WINDOW)
        )
        models = apply_window(models, window)
        verifier, challenge = pkce_pair()
        # `state` is public and the verifier is not. state rides in the authorize
        # URL and comes back through the redirect, so it lands in browser history,
        # the vendor's request logs, and anything watching the loopback — while the
        # verifier is the single secret PKCE has, the proof that whoever redeems
        # the code is whoever started the flow. Sending one as the other publishes
        # it and leaves a plain code flow wearing PKCE's clothes. The exception is
        # a vendor whose token endpoint checks state AGAINST the verifier
        # (SEND_STATE), where the protocol defines them to be the same value.
        state = verifier if self.SEND_STATE else secrets.token_urlsafe(32)
        if self.REDIRECT_URI:
            # Vendor-hosted callback: no loopback to run, the user copies the code
            # off the vendor's page. The redirect_uri sent here must be byte-equal
            # to the one authorized, or the exchange comes back a bare 400.
            redirect_uri = self.REDIRECT_URI
            cb.on_auth(self._authorize_url(challenge, redirect_uri, state))
            cb.on_status("approve in the browser, then copy the code it shows")
            code = (cb.on_prompt("authorization code: ") or "").strip()
        else:
            with loopback(
                path=self.REDIRECT_PATH, port=self.REDIRECT_PORT, state=state
            ) as (
                redirect_uri,
                wait,
            ):
                cb.on_auth(self._authorize_url(challenge, redirect_uri, state))
                cb.on_status("waiting for the browser to come back…")
                code = wait()
            if not code:
                cb.on_status("no callback received — paste the code from the URL instead")
                code = (cb.on_prompt("authorization code: ") or "").strip()
        if not code:
            raise LoginError("no authorization code received")
        # Some vendors hand back "code#state" when the code is copied by hand.
        code = code.split("#", 1)[0].split("&", 1)[0].strip()
        try:
            token = self._exchange(
                self._exchange_payload(code, verifier, redirect_uri, state)
            )
        except Exception as exc:
            raise LoginError(f"could not exchange the code: {exc}") from exc
        if not token.get("access_token"):
            raise LoginError(f"{self.name} returned no access token")
        cred = self._credential(token, models)
        stash_window(cred, models, window)
        return cred

    def refresh(self, cred: dict[str, Any]) -> dict[str, Any]:
        client_id, client_secret = self.client_credentials()
        token_req = {
            "grant_type": "refresh_token",
            "refresh_token": cred.get("refresh", ""),
            "client_id": client_id,
        }
        if client_secret:
            token_req["client_secret"] = client_secret
        token = self._exchange(token_req)
        if not token.get("access_token"):
            return cred
        return {
            **cred,
            "access": token["access_token"],
            # A refresh response may omit refresh_token, meaning "keep the old one".
            "refresh": token.get("refresh_token") or cred.get("refresh", ""),
            "expires": _expires_ms(token),
        }


class AnthropicSubscriptionProvider(_SubscriptionProvider):
    name = "anthropic"
    label = "Claude Pro/Max — Haiku only; Sonnet/Opus blocked for third parties"
    login_warning = (
        "warning: since 28 Apr 2026 Anthropic blocks Sonnet and Opus for third-party "
        "OAuth clients — they return a bare 429 that extra-usage credit does NOT "
        "unlock. Only Haiku serves. For Sonnet/Opus use `/login anthropic-key` with "
        "a key from console.anthropic.com."
    )
    CLIENT_ID_ENV = "ANTHROPIC_OAUTH_CLIENT_ID"
    CLIENT_SOURCE = "Claude Code"
    AUTH_URL = "https://claude.ai/oauth/authorize"
    TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
    SCOPES = "org:create_api_key user:profile user:inference"
    API_BASE = "https://api.anthropic.com"
    # This client is registered against Anthropic's own console callback, not a
    # loopback, so the browser shows the code and the user copies it across. The
    # value must be sent back byte-equal at exchange time.
    REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
    # The token endpoint takes JSON. Form-encoding it — the RFC 6749 default the
    # other two providers use — returns a bare 400 with no body to explain why.
    EXCHANGE_AS_JSON = True
    AUTH_EXTRA = {"code": "true"}
    SEND_STATE = True
    MODEL_PROVIDER = "anthropic"
    # Haiku deliberately: it is the ONLY model this path still serves. Sonnet and
    # Opus return a 429 with no anthropic-ratelimit headers — a hard block, not a
    # quota — and loading an extra-usage balance does not lift it, because the gate
    # is on a promotional-credit flag rather than the balance. Defaulting to a
    # model that cannot answer would make every login look broken.
    MODELS = ["claude-haiku-4-5-20251001"]


class GoogleSubscriptionProvider(_SubscriptionProvider):
    name = "google"
    label = "Gemini (Code Assist) — BLOCKED; tiers discontinued 18 Jun 2026"
    login_warning = (
        "warning: Google banned third-party use of this token in Feb 2026 and has "
        "suspended accounts for it, and Code Assist stopped serving the individual "
        "/ AI Pro / AI Ultra tiers on 18 Jun 2026. Use a Gemini API key instead."
    )
    # GitHub's push protection rejects any commit carrying a Google OAuth client
    # secret, which is the other reason this pair can't live in the source.
    CLIENT_ID_ENV = "GEMINI_OAUTH_CLIENT_ID"
    CLIENT_SECRET_ENV = "GEMINI_OAUTH_CLIENT_SECRET"
    CLIENT_SOURCE = "Google's open-source Gemini CLI"
    AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    TOKEN_URL = "https://oauth2.googleapis.com/token"
    SCOPES = (
        "https://www.googleapis.com/auth/cloud-platform "
        "https://www.googleapis.com/auth/userinfo.email "
        "https://www.googleapis.com/auth/userinfo.profile"
    )
    # Not the public Gemini API: an OAuth token only addresses Code Assist.
    API_BASE = "https://cloudcode-pa.googleapis.com/v1internal"
    MODEL_PROVIDER = "google_genai"
    MODELS = ["gemini-2.5-flash", "gemini-2.5-pro"]

    @override
    def _authorize_url(self, challenge: str, redirect_uri: str, state: str) -> str:
        # access_type=offline + prompt=consent are what make Google return a
        # refresh_token; without them the credential dies in an hour with no way
        # back except a full re-login.
        base = super()._authorize_url(challenge, redirect_uri, state)
        return base + "&" + urllib.parse.urlencode(
            {"access_type": "offline", "prompt": "consent"}
        )


class CodexSubscriptionProvider(_SubscriptionProvider):
    name = "codex"
    label = "ChatGPT (Codex) — works via the Codex proxy; bills as overage"
    login_warning = (
        "note: since 4 Apr 2026 third-party traffic bills as overage rather than "
        "drawing from your ChatGPT plan, and calls are routed through a local "
        "proxy that speaks the Codex request shape."
    )
    CLIENT_ID_ENV = "CODEX_OAUTH_CLIENT_ID"
    CLIENT_SOURCE = "OpenAI's open-source Codex CLI"
    AUTH_URL = "https://auth.openai.com/oauth/authorize"
    TOKEN_URL = "https://auth.openai.com/oauth/token"
    SCOPES = "openid profile email offline_access"
    API_BASE = "https://chatgpt.com/backend-api/codex"
    # OpenAI validates redirect_uri against the registered value exactly, so this
    # port is not negotiable — unlike the others, a second concurrent login fails.
    REDIRECT_PORT = 1455
    REDIRECT_PATH = "/auth/callback"
    # Reached through codex_proxy, which presents an OpenAI-compatible surface.
    MODEL_PROVIDER = "openai"
    # What the backend serves a ChatGPT account (its /codex/models list, Oct
    # 2026). The API's "-codex" names are refused there: "not supported when
    # using Codex with a ChatGPT account".
    MODELS = ["gpt-5.5"]

    @override
    def _credential(self, token: dict[str, Any], models: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        cred = super()._credential(token, models)
        # The account id rides in the id_token and is required as a request header
        # by the Codex backend; pull it out now so the proxy doesn't re-parse a JWT
        # on every call.
        cred["account_id"] = _account_id_from_id_token(token.get("id_token", ""))
        return cred

    @override
    def refresh(self, cred: dict[str, Any]) -> dict[str, Any]:
        fresh = super().refresh(cred)
        fresh.setdefault("account_id", cred.get("account_id", ""))
        return fresh


def ask_models(cb: LoginCallbacks, suggested: list[str]) -> list[dict[str, Any]]:
    """Ask which models this credential should offer; the first is its default.

    Asked at login because the answer belongs with the credential: a model name is
    only meaningful alongside the key that can serve it. Blank keeps the suggested
    list, so the common case is one Enter.

    Each name is returned as a config dict with per-1M costs filled in from the
    built-in table where it knows them — which is the "set by default for
    non-OpenAI providers" case, since those are the models the table covers. A
    gateway model it has never heard of gets a bare entry to fill in by hand.
    """
    hint = ", ".join(suggested) if suggested else "none configured"
    answer = (cb.on_prompt(f"models, comma-separated [{hint}]: ") or "").strip()
    names: list[str] = []
    if answer:
        for part in answer.split(","):
            name = part.strip()
            if name and name not in names:
                names.append(name)
    return [_model_config_for(n) for n in (names or suggested)]


def _model_config_for(name: str) -> dict[str, Any]:
    """A model config seeded from the built-in pricing table, when it knows the
    model. Costs are per 1M tokens, matching the table's own units.

    Every cost field is always written. A model the table doesn't cover gets
    zeros rather than missing keys, so the shape to fill in is visible in the file
    instead of something you have to know to add. Zero is read back as "not set"
    when billing — see ``pricing.rates`` for why that beats reporting $0.00.
    """
    from .pricetables import known_context, table_cache_rates, table_rates

    rates = table_rates(name)
    if not rates:
        entry = {"name": name, **{field: 0 for field in COST_FIELDS}}
    else:
        write, read = table_cache_rates(name, rates[0])
        entry = {
            "name": name,
            "input_cost": rates[0],
            "output_cost": rates[1],
            "cache_write_cost": write,
            "cache_read_cost": read,
        }
    # Seed the window per model, not from one answer applied to all: an Anthropic
    # key serves a 1M Sonnet and a 200k Haiku, and one number cannot be right for
    # both. The login answer then only fills what the table doesn't know.
    window = known_context(name)
    if window:
        entry["context_window"] = window
    return entry


def stash_window(cred: dict[str, Any], models: list[dict[str, Any]], window: int | None) -> None:
    """Put the answered window where it can be found later.

    Normally that is each model config. With no models configured there is no
    entry to hold it, so it falls back to the credential — otherwise answering the
    question would silently discard the answer.
    """
    if window and not models:
        cred["context_window"] = window


def apply_window(models: list[dict[str, Any]], window: int | None) -> list[dict[str, Any]]:
    """Record an explicitly chosen context window on every model config.

    Only called with a value the user actually typed — a blank answer arrives as
    None and leaves each entry with the per-model window the table seeded, which
    is more accurate than one number across models whose windows differ.
    """
    if not window:
        return models
    return [{**m, "context_window": window} for m in models]


def suggested_window(models: list[str], declared: int | None) -> int | None:
    """What to offer as the context window at login.

    The built-in table first: it is keyed per model and already correct for public
    names, whereas anything declared here is one value for a credential that may
    serve models with different windows. A provider constant is the fallback for a
    vendor the table has never heard of.
    """
    from .pricetables import known_context

    for model in models:
        known = known_context(model)
        if known:
            return known
    return declared


def ask_context_window(cb: LoginCallbacks, suggested: int | None) -> int | None:
    """Ask how big this credential's context window is, in tokens.

    Worth asking because the window drives more than a display: the ctx% gauge,
    the tool-result clearing threshold, and auto-compaction all divide by it. A
    wrong value silently compacts too early or blows the window. The built-in
    table only knows public model names, so a gateway serving a model under its own
    name — the case where this matters most — has no entry to find.

    Returns None for a blank (or unparseable) answer, meaning "leave each model
    with the window the table gave it" — which is more accurate than one number
    for a credential serving a 1M Sonnet and a 200k Haiku. A TYPED value is a
    deliberate choice and overrides all of them. Accepts "200000", "200k" or "1m".
    """
    hint = f"{suggested:,}" if suggested else "auto-detect"
    answer = (cb.on_prompt(f"context window in tokens [{hint}]: ") or "").strip().lower()
    if not answer:
        return None
    multiplier = 1
    if answer.endswith("k"):
        multiplier, answer = 1_000, answer[:-1]
    elif answer.endswith("m"):
        multiplier, answer = 1_000_000, answer[:-1]
    try:
        value = int(float(answer) * multiplier)
    except ValueError:
        return None
    return value if value > 0 else None


def context_window_of(cred: dict[str, Any]) -> int | None:
    value = cred.get("context_window")
    return value if isinstance(value, int) and value > 0 else None


#: Per-model cost fields, in USD per 1M tokens.
COST_FIELDS = ("input_cost", "output_cost", "cache_write_cost", "cache_read_cost")


def _as_float(value: Any) -> float | None:
    """Costs may arrive as numbers or as strings (hand-edited, or as the example
    in the docs writes them); both mean the same rate."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out >= 0 else None


def model_entries(cred: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalized model configs: always ``{"name": ...}``, plus any costs set.

    Tolerates all three shapes this field has had — a bare string list, and the
    original singular ``model`` — because a stored credential outlives the code
    that wrote it and re-authenticating to pick up a format change is not a
    migration path anyone should have to follow.
    """
    raw = cred.get("models")
    if not isinstance(raw, list):
        single = cred.get("model")
        raw = [single] if isinstance(single, str) and single else []
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str) and item:
            out.append({"name": item})
        elif isinstance(item, dict) and isinstance(item.get("name"), str) and item["name"]:
            entry = {"name": item["name"]}
            for field_name in COST_FIELDS:
                value = _as_float(item.get(field_name))
                if value is not None:
                    entry[field_name] = value
            window = item.get("context_window")
            if isinstance(window, int) and window > 0:
                entry["context_window"] = window
            out.append(entry)
    if not out:
        # A credential written before this field existed lists nothing. Fall back
        # to what its provider declares, so the models `models_of` reports are the
        # same ones `model_config` can find — otherwise /models offers a model that
        # then resolves to no config at all. Bare names only: the tables already
        # cover these, and building full configs here would call back into pricing
        # from a path pricing itself uses.
        provider = _REGISTRY.get(cred.get("provider", ""))
        out = [{"name": n} for n in getattr(provider, "MODELS", []) or []]
    return out


def model_config(cred: dict[str, Any], model: str) -> dict[str, Any] | None:
    """The config for one model of this credential, or None if it serves none."""
    for entry in model_entries(cred):
        if entry["name"] == model:
            return entry
    return None


def models_of(cred: dict[str, Any]) -> list[str]:
    """Just the model names, in configured order (the first is the default)."""
    return [entry["name"] for entry in model_entries(cred)]


def _account_id_from_id_token(id_token: str) -> str:
    """Best-effort read of the ChatGPT account id from an OIDC id_token.

    Signature is NOT verified: this token came straight from the token endpoint
    over TLS and is only being read for a routing header, never trusted for
    authorization.
    """
    try:
        payload = id_token.split(".")[1]
        payload += "=" * (-len(payload) % 4)  # restore base64url padding
        claims = json.loads(base64.urlsafe_b64decode(payload).decode("utf-8"))
    except Exception:
        return ""
    auth_claims = claims.get("https://api.openai.com/auth") or {}
    return (
        auth_claims.get("chatgpt_account_id")
        or claims.get("chatgpt_account_id")
        or claims.get("sub")
        or ""
    )


class ApiKeyProvider:
    """Store a vendor API key in the credential store instead of ``.env``.

    Not an OAuth flow — it just prompts — but it belongs in the same registry so
    a key gets everything the store already provides: ``0600`` permissions, per-tier
    scoping, ``$VAR``/``!command`` indirection, redaction, and routing that
    switches the client instead of leaving it on the previous provider. Which is
    the whole reason to prefer it over an environment variable that every
    subprocess inherits.
    """

    def __init__(
        self,
        name: str,
        vendor: str,
        label: str,
        model_provider: str,
        models: list[str] | None = None,
        context_window: int | None = None,
        ask_base_url: bool = False,
        console: str = "",
    ):
        self.name = name
        self.vendor = vendor  # short name for the prompt; `label` is for the picker
        self.label = label
        self.MODEL_PROVIDER = model_provider
        # Mirrors graph._PROVIDER_DEFAULT_MODEL. Populated for every provider EXCEPT
        # the OpenAI lane, where OPENAI_MODEL is already the right answer and
        # pinning one here would override a deliberate choice.
        self.MODELS = list(models or [])
        # Published window for these models, when the vendor states one. None for
        # the OpenAI lane, where a gateway can serve anything.
        self.CONTEXT_WINDOW = context_window
        self.ask_base_url = ask_base_url
        self.console = console
        self.login_warning = ""

    def login(self, cb: LoginCallbacks) -> dict[str, Any]:
        if self.console:
            cb.on_status(f"create a key at {self.console}")
        key = (cb.on_secret(f"{self.vendor} API key: ") or "").strip()
        if not key:
            raise LoginError("no API key entered")
        cred: dict[str, object] = {
            "provider": self.name,
            "type": "api_key",
            "key": key,
            "model_provider": self.MODEL_PROVIDER,
        }
        # Endpoint BEFORE models: which gateway you point at determines which
        # models exist, so answering in the other order means naming models for an
        # endpoint you haven't chosen yet.
        if self.ask_base_url:
            # Only asked on the OpenAI-compatible lane, where pointing at a
            # gateway is the common case. Blank keeps the ambient default.
            base = (cb.on_prompt("base URL (blank for the default): ") or "").strip()
            if base:
                cred["base_url"] = base
                cb.on_status(f"models served by {base}")
        models = ask_models(cb, self.MODELS)
        window = ask_context_window(
            cb, suggested_window([m["name"] for m in models], self.CONTEXT_WINDOW)
        )
        cred["models"] = apply_window(models, window)
        stash_window(cred, models, window)
        return cred

    def refresh(self, cred: dict[str, Any]) -> dict[str, Any]:
        return cred  # an API key does not expire


# --- registry -----------------------------------------------------------------

_REGISTRY: dict[str, OAuthProvider] = {}


def register(provider: OAuthProvider) -> None:
    _REGISTRY[provider.name] = provider


def get(name: str) -> OAuthProvider | None:
    return _REGISTRY.get(name)


def routing_for(cred: dict[str, Any]) -> tuple[str | None, list[str]]:
    """``(model_provider, models)`` for a stored credential, falling back to what
    its provider declares.

    Credentials written before those fields existed have neither, and a stored
    credential is long-lived — expecting the user to notice and re-login to pick
    up a new field is not a real migration path. Reading the defaults back off the
    registry makes any credential self-healing, whenever it was written.
    """
    provider = _REGISTRY.get(cred.get("provider", ""))
    return (
        cred.get("model_provider") or getattr(provider, "MODEL_PROVIDER", None) or None,
        # model_entries already falls back to the provider's declared models when
        # the credential lists none, so no second fallback is needed here.
        models_of(cred),
    )


def names() -> list[str]:
    return sorted(_REGISTRY)


def choices() -> list[tuple[str, str]]:
    """``(name, label)`` pairs for a picker."""
    return [(n, _REGISTRY[n].label) for n in names()]


register(OpenRouterProvider())
register(AnthropicSubscriptionProvider())
register(GoogleSubscriptionProvider())
register(CodexSubscriptionProvider())

# Paste-a-key providers. These are the SUPPORTED path for every vendor whose
# subscription OAuth is blocked, so they are listed alongside it rather than
# hidden — the point of the store is that a key need not live in .env.
register(ApiKeyProvider(
    "anthropic-key", "Anthropic", "Anthropic — API key (the supported path)",
    "anthropic", ["claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5-20251001"],
    console="https://console.anthropic.com/settings/keys",
))
register(ApiKeyProvider(
    "google-key", "Google Gemini", "Google Gemini — API key (the supported path)",
    "google_genai", ["gemini-2.5-flash", "gemini-2.5-pro"],
    console="https://aistudio.google.com/apikey",
))
register(ApiKeyProvider(
    "groq-key", "Groq", "Groq — API key", "groq", ["llama-3.3-70b-versatile"],
    console="https://console.groq.com/keys",
))
register(ApiKeyProvider(
    # No MODEL: OPENAI_MODEL already names the model for this lane, and pinning
    # one here would silently override it.
    "openai-key", "OpenAI", "OpenAI or any compatible gateway — API key",
    "openai", [],
    ask_base_url=True, console="https://platform.openai.com/api-keys",
))


# Refresh this far ahead of the stated expiry. A turn can run for minutes across
# many tool calls, so renewing only once a token is already dead would kill it
# mid-flight; five minutes covers any realistic turn.
REFRESH_SKEW_MS = 5 * 60 * 1000


def needs_refresh(cred: dict[str, Any], now_ms: float | None = None) -> bool:
    """True when an OAuth credential is expired or about to be. A minted API key
    (``type: api_key``) never expires, and one with no ``expires`` can't be
    reasoned about, so both are left alone."""
    if not cred or cred.get("type") != "oauth":
        return False
    expires = cred.get("expires")
    if not isinstance(expires, (int, float)):
        return False
    if now_ms is None:
        import time

        now_ms = time.time() * 1000
    return expires - now_ms < REFRESH_SKEW_MS


def _store_refreshed(target: str, cred: dict[str, Any]) -> None:
    """Persist a renewed credential while ``auth._locked()`` is already held.

    ``auth.set_credential`` takes that same lock through a second file descriptor,
    and ``flock`` is per-descriptor rather than per-process — asking for it again
    from inside the lock blocks the caller against itself forever instead of
    serializing anything. So the store update is written out here, mirroring what
    ``set_credential`` does under its own lock.
    """
    from . import auth

    data = auth._read()
    key = cred.get("provider") or auth._anon_key(target)
    data["providers"][key] = cred
    data["active"][target] = key
    auth.save(data)


def ensure_fresh(scope: str = "default") -> None:
    """Renew ``scope``'s credential if it's near expiry. Cheap and safe to call
    before every model build — it's a timestamp comparison unless a refresh is
    actually due.

    Read, renew and write happen together under the credential-store lock, and the
    expiry check is made twice: once cheaply outside it, then again after acquiring
    it. A refresh token is single-use and rotates — the vendor invalidates it as it
    issues the next one — so two callers that both read "expired" and both POST
    leave the loser holding a token that has already been spent, and the login dies
    at the refresh after this one. Re-checking inside the lock means the second
    caller finds the first's fresh credential and simply uses it. Serializing only
    the write, as locking around it alone would, prevents nothing: the damage is
    done at the token endpoint, before either write.

    A failed refresh deliberately leaves the old credential in place rather than
    clearing it: a transient network blip would otherwise log the user out and
    throw away a refresh token that was still perfectly good. The stale token then
    fails at the API with a real error, which is recoverable; a deleted one is not.
    """
    from . import auth

    target = auth.effective_scope(scope)
    if target is None:
        return
    cred = auth.get(target)
    if cred is None or not needs_refresh(cred):
        return
    with auth._locked():
        cred = auth.get(target)
        if cred is None or not needs_refresh(cred):
            return  # somebody else renewed it while we were waiting for the lock
        provider = get(cred.get("provider", ""))
        if provider is None:
            return
        try:
            fresh = provider.refresh(cred)
        except Exception:
            return
        if fresh and fresh != cred:
            _store_refreshed(target, fresh)


def login(name: str, cb: LoginCallbacks, scope: str = "default") -> dict[str, Any]:
    """Run ``name``'s flow and persist the result for ``scope``. Returns the
    credential. Raises LoginError with something worth printing."""
    from . import auth

    provider = get(name)
    if provider is None:
        raise LoginError(f"unknown provider: {name} (try {', '.join(names())})")
    cred = provider.login(cb)
    auth.set_credential(scope, cred)
    return cred
