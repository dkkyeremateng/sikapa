"""OAuth flows: PKCE, the loopback callback server, the registry, OpenRouter.

Nothing here opens a socket to the internet — the loopback server is real (it is
the fiddly part, so it gets exercised) but the token exchange is stubbed at the
module-level _post_json, the same seam the other network modules use.
"""

import base64
import hashlib
import json
import threading
import urllib.request

import pytest

from financial_research_assistant import auth, llm, oauth


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("FINANCIAL_RESEARCH_AUTH_FILE", str(tmp_path / "auth.json"))
    monkeypatch.setenv("GEMINI_OAUTH_CLIENT_ID", "test-client.apps.example")
    monkeypatch.setenv("GEMINI_OAUTH_CLIENT_SECRET", "test-secret")


def _cb(**kw):
    """LoginCallbacks with recording defaults; override individual hooks."""
    seen = {"auth": [], "status": [], "device": []}
    base = dict(
        on_auth=seen["auth"].append,
        on_status=seen["status"].append,
        on_device_code=lambda c, u: seen["device"].append((c, u)),
    )
    base.update(kw)
    return oauth.LoginCallbacks(**base), seen


# --- PKCE ---------------------------------------------------------------------


def test_pkce_pair_is_rfc7636_shaped():
    verifier, challenge = oauth.pkce_pair()
    assert 43 <= len(verifier) <= 128
    expected = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("ascii")).digest()
    ).rstrip(b"=").decode()
    assert challenge == expected
    assert "=" not in challenge  # unpadded base64url


def test_pkce_pair_is_not_reused():
    assert oauth.pkce_pair()[0] != oauth.pkce_pair()[0]


# --- loopback server ----------------------------------------------------------


def test_loopback_captures_the_code():
    with oauth.loopback(timeout=10) as (url, wait):
        assert url.startswith("http://localhost:")
        threading.Thread(
            target=lambda: urllib.request.urlopen(url + "?code=abc123", timeout=10).read(),
            daemon=True,
        ).start()
        assert wait() == "abc123"


def test_loopback_times_out_without_a_callback():
    with oauth.loopback(timeout=0.2) as (_url, wait):
        assert wait() is None


def test_loopback_ports_do_not_collide():
    """Ephemeral ports, so a second instance logging in concurrently still works —
    a hardcoded port would fail here."""
    with oauth.loopback(timeout=1) as (a, _), oauth.loopback(timeout=1) as (b, _):
        assert a != b


def _hit(url: str) -> int:
    """GET the callback the way a browser would, returning the status code."""
    import urllib.error

    try:
        return urllib.request.urlopen(url, timeout=10).status
    except urllib.error.HTTPError as exc:
        return exc.code


def test_loopback_ignores_a_callback_that_does_not_echo_the_state():
    """Any process on this machine can GET a loopback port, and a vendor that pins
    one (Codex: 1455) makes it guessable — so a callback that cannot echo the value
    we sent is somebody else's, and must not be able to feed us a code."""
    with oauth.loopback(timeout=0.5, state="the-expected-state") as (url, wait):
        assert _hit(url + "?code=injected&state=guessed") == 400
        # and it did not get to cancel the real sign-in either
        assert wait() is None


def test_loopback_accepts_the_callback_that_echoes_the_state():
    with oauth.loopback(timeout=10, state="s-123") as (url, wait):
        threading.Thread(
            target=lambda: _hit(url + "?code=abc123&state=s-123"), daemon=True
        ).start()
        assert wait() == "abc123"


def test_loopback_reports_a_refusal_rather_than_swallowing_it():
    """A declined consent screen is a decision, not a missing code — reported as
    "no callback" it becomes a prompt to paste something never issued."""
    with oauth.loopback(timeout=10, state="s-1") as (url, wait):
        threading.Thread(
            target=lambda: _hit(url + "?error=access_denied&state=s-1"), daemon=True
        ).start()
        with pytest.raises(oauth.LoginError, match="declined in the browser"):
            wait()


def test_a_refusal_reason_cannot_repaint_the_terminal():
    """error_description is attacker-reachable text on its way to a TTY."""
    message = oauth._denial("access_denied", "user\x1b[2Jsaid no\nreally")
    assert "\x1b" not in message and "\n" not in message
    assert "said no" in message


# --- registry -----------------------------------------------------------------


def test_openrouter_is_registered():
    assert "openrouter" in oauth.names()
    assert isinstance(oauth.get("openrouter"), oauth.OAuthProvider)
    assert dict(oauth.choices())["openrouter"]


def test_unknown_provider_names_the_alternatives():
    with pytest.raises(oauth.LoginError) as exc:
        oauth.login("nope", oauth.LoginCallbacks())
    assert "openrouter" in str(exc.value)


# --- OpenRouter ---------------------------------------------------------------


def test_openrouter_login_end_to_end(monkeypatch):
    """Real loopback server, stubbed exchange: the browser is simulated by hitting
    the callback URL the flow hands to on_auth."""
    posted = {}

    def fake_post(url, payload, timeout=30.0):
        posted["url"] = url
        posted["payload"] = payload
        return {"key": "sk-or-minted"}

    monkeypatch.setattr(oauth, "_post_json", fake_post)

    auth_urls = []

    def open_browser(url):
        auth_urls.append(url)
        # the "browser": follow the redirect back to our loopback
        threading.Thread(
            target=lambda: urllib.request.urlopen(
                _callback_of(url) + "?code=the-code", timeout=10
            ).read(),
            daemon=True,
        ).start()

    cb, _seen = _cb(on_auth=open_browser)
    cred = oauth.login("openrouter", cb, scope="quick")

    assert cred["key"] == "sk-or-minted"
    assert cred["base_url"] == "https://openrouter.ai/api/v1"
    assert cred["type"] == "api_key"  # a minted key, so nothing to refresh
    assert posted["url"] == oauth.OpenRouterProvider.KEYS_URL
    assert posted["payload"]["code"] == "the-code"
    assert posted["payload"]["code_challenge_method"] == "S256"

    # the verifier sent must match the challenge that was in the auth URL
    import urllib.parse as up

    q = up.parse_qs(up.urlparse(auth_urls[0]).query)
    digest = hashlib.sha256(posted["payload"]["code_verifier"].encode()).digest()
    assert q["code_challenge"][0] == base64.urlsafe_b64encode(digest).rstrip(b"=").decode()

    # and it was persisted to the scope that was asked for, not "default"
    assert auth.resolve_key("quick") == "sk-or-minted"
    assert auth.get("default") is None


def _callback_of(auth_url: str) -> str:
    import urllib.parse as up

    return up.parse_qs(up.urlparse(auth_url).query)["callback_url"][0]


def test_openrouter_falls_back_to_pasted_code(monkeypatch):
    """No browser can reach our loopback (SSH); the user brings the code across."""
    monkeypatch.setattr(oauth, "_post_json", lambda u, p, timeout=30.0: {"key": "k"})
    monkeypatch.setattr(oauth, "LOGIN_TIMEOUT", 0.2)
    cb, seen = _cb(on_prompt=lambda _p: "  pasted-code  ")
    cred = oauth.OpenRouterProvider().login(cb)
    assert cred["key"] == "k"
    assert any("paste" in s for s in seen["status"])


def test_openrouter_no_code_raises(monkeypatch):
    monkeypatch.setattr(oauth, "LOGIN_TIMEOUT", 0.2)
    cb, _ = _cb(on_prompt=lambda _p: "")
    with pytest.raises(oauth.LoginError, match="no authorization code"):
        oauth.OpenRouterProvider().login(cb)


def test_openrouter_exchange_failure_is_reported(monkeypatch):
    def boom(*_a, **_k):
        raise OSError("connection refused")

    monkeypatch.setattr(oauth, "_post_json", boom)
    monkeypatch.setattr(oauth, "LOGIN_TIMEOUT", 0.2)
    cb, _ = _cb(on_prompt=lambda _p: "code")
    with pytest.raises(oauth.LoginError, match="connection refused"):
        oauth.OpenRouterProvider().login(cb)


def test_openrouter_missing_key_in_response(monkeypatch):
    monkeypatch.setattr(oauth, "_post_json", lambda u, p, timeout=30.0: {})
    monkeypatch.setattr(oauth, "LOGIN_TIMEOUT", 0.2)
    cb, _ = _cb(on_prompt=lambda _p: "code")
    with pytest.raises(oauth.LoginError, match="no key"):
        oauth.OpenRouterProvider().login(cb)


def test_failed_login_stores_nothing(monkeypatch):
    monkeypatch.setattr(oauth, "_post_json", lambda u, p, timeout=30.0: {})
    monkeypatch.setattr(oauth, "LOGIN_TIMEOUT", 0.2)
    cb, _ = _cb(on_prompt=lambda _p: "code")
    with pytest.raises(oauth.LoginError):
        oauth.login("openrouter", cb)
    assert auth.get("default") is None


def test_minted_key_refresh_is_identity():
    cred = {"provider": "openrouter", "type": "api_key", "key": "k"}
    assert oauth.OpenRouterProvider().refresh(cred) == cred


# --- subscription providers ---------------------------------------------------


def _patch_token_endpoint(monkeypatch, token):
    """Stub BOTH encodings. Patching only one lets a provider that uses the other
    reach the real token endpoint — which is how these tests briefly made live
    requests to console.anthropic.com."""
    posted = {}

    def form(url, payload, headers=None, timeout=30.0):
        posted.update(url=url, payload=payload, encoding="form")
        return token

    def as_json(url, payload, timeout=30.0):
        posted.update(url=url, payload=payload, encoding="json")
        return token

    monkeypatch.setattr(oauth, "_post_form", form)
    monkeypatch.setattr(oauth, "_post_json", as_json)
    return posted


def _login_via_paste(provider, monkeypatch, token, code="the-code"):
    """Drive a flow through the paste fallback (no browser in a test) and return
    (credential, the form posted to the token endpoint, the authorize URL)."""
    posted = _patch_token_endpoint(monkeypatch, token)
    urls = []
    monkeypatch.setattr(oauth, "LOGIN_TIMEOUT", 0.2)

    def prompt(question: str) -> str:
        # Login asks several things through on_prompt (models, context window, and
        # the pasted code). Answer only the code; blank accepts every default, so
        # adding a question can't break this helper.
        return code if "code" in question.lower() else ""

    cb = oauth.LoginCallbacks(on_auth=urls.append, on_prompt=prompt)
    return provider.login(cb), posted, urls[0]


ALL_SUBSCRIPTION = ("anthropic", "google", "codex")


def test_all_three_are_registered():
    for name in ALL_SUBSCRIPTION:
        assert oauth.get(name) is not None, name
    assert set(ALL_SUBSCRIPTION) <= set(oauth.names())


@pytest.mark.parametrize("name", ALL_SUBSCRIPTION)
def test_login_warning_is_surfaced_before_the_browser(name, monkeypatch):
    """Every one of these has a caveat that should be stated up front, not
    discovered later as an opaque 401."""
    provider = oauth.get(name)
    said = []
    _patch_token_endpoint(monkeypatch, {"access_token": "t"})
    monkeypatch.setattr(oauth, "LOGIN_TIMEOUT", 0.2)
    cb = oauth.LoginCallbacks(on_status=said.append, on_prompt=lambda _q: "c")
    provider.login(cb)
    assert said and provider.login_warning
    assert said[0] == provider.login_warning


@pytest.mark.parametrize("name", ALL_SUBSCRIPTION)
def test_authorize_url_is_pkce_shaped(name, monkeypatch):
    import urllib.parse as up

    provider = oauth.get(name)
    _cred, posted, url = _login_via_paste(
        provider, monkeypatch, {"access_token": "a", "refresh_token": "r", "expires_in": 3600}
    )
    q = up.parse_qs(up.urlparse(url).query)
    assert url.startswith(provider.AUTH_URL)
    assert q["client_id"] == [provider.client_credentials()[0]]
    assert q["response_type"] == ["code"]
    assert q["code_challenge_method"] == ["S256"]
    # the challenge must match the verifier that was later sent to the token endpoint
    digest = hashlib.sha256(posted["payload"]["code_verifier"].encode()).digest()
    assert q["code_challenge"][0] == base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


@pytest.mark.parametrize("name", ALL_SUBSCRIPTION)
def test_credential_shape_and_expiry(name, monkeypatch):
    import time

    provider = oauth.get(name)
    cred, posted, _url = _login_via_paste(
        provider, monkeypatch,
        {"access_token": "acc", "refresh_token": "ref", "expires_in": 3600},
    )
    assert posted["url"] == provider.TOKEN_URL
    assert posted["payload"]["grant_type"] == "authorization_code"
    assert cred["provider"] == name
    assert cred["type"] == "oauth"
    assert cred["access"] == "acc" and cred["refresh"] == "ref"
    assert cred["base_url"] == provider.API_BASE
    # expires_in (seconds) is stored as an absolute ms timestamp
    assert 3_500_000 < cred["expires"] - time.time() * 1000 <= 3_600_000


def test_anthropic_exchange_contract(monkeypatch):
    """Regression for a live "HTTP 400: Bad Request" on the exchange. Three things
    have to be right together, and getting any one wrong yields the same bare 400:
    JSON (not form) encoding, the console redirect_uri echoed back byte-equal, and
    the state value present."""
    import urllib.parse as up

    provider = oauth.get("anthropic")
    _cred, posted, url = _login_via_paste(provider, monkeypatch, {"access_token": "a"})

    assert posted["encoding"] == "json"
    assert posted["payload"]["redirect_uri"] == "https://console.anthropic.com/oauth/code/callback"
    assert posted["payload"]["state"] == posted["payload"]["code_verifier"]
    q = up.parse_qs(up.urlparse(url).query)
    assert q["code"] == ["true"]  # required by the authorize endpoint
    assert q["redirect_uri"] == ["https://console.anthropic.com/oauth/code/callback"]


@pytest.mark.parametrize("name", ("google", "codex"))
def test_state_is_not_the_pkce_verifier(name, monkeypatch):
    """The verifier is the one secret PKCE has. state is public — it rides in the
    authorize URL, the redirect and the vendor's logs — so sending the verifier as
    state publishes it and leaves a bare code flow wearing PKCE's clothes."""
    import urllib.parse as up

    _cred, posted, url = _login_via_paste(
        oauth.get(name), monkeypatch, {"access_token": "a"}
    )
    verifier = posted["payload"]["code_verifier"]
    state = up.parse_qs(up.urlparse(url).query)["state"][0]
    assert state != verifier
    assert verifier not in url
    assert len(state) >= 32  # still unguessable enough to be worth comparing


def test_anthropic_state_is_the_verifier_because_its_protocol_says_so(monkeypatch):
    """The one vendor whose token endpoint checks state against the verifier. Its
    flow breaks if the two differ, so the general rule above does not apply."""
    import urllib.parse as up

    _cred, posted, url = _login_via_paste(
        oauth.get("anthropic"), monkeypatch, {"access_token": "a"}
    )
    verifier = posted["payload"]["code_verifier"]
    assert up.parse_qs(up.urlparse(url).query)["state"] == [verifier]
    assert posted["payload"]["state"] == verifier


def test_a_denied_login_says_so_instead_of_asking_for_a_paste(monkeypatch):
    """The whole point of reading the callback's error: no code is ever coming, so
    prompting for one asks the user to produce something that does not exist."""
    import urllib.parse as up

    asked = []

    def open_browser(url):
        query = up.parse_qs(up.urlparse(url).query)
        callback = query["redirect_uri"][0]
        threading.Thread(
            target=lambda: _hit(
                f"{callback}?error=access_denied&state={query['state'][0]}"
            ),
            daemon=True,
        ).start()

    cb = oauth.LoginCallbacks(
        on_auth=open_browser, on_prompt=lambda q: asked.append(q) or ""
    )
    with pytest.raises(oauth.LoginError, match="declined in the browser"):
        oauth.get("google").login(cb)
    assert not any("authorization code" in q for q in asked)


def test_anthropic_skips_the_loopback_entirely(monkeypatch):
    """Its callback is vendor-hosted, so binding a local port would be pointless —
    and the code arrives by paste, not by redirect."""
    provider = oauth.get("anthropic")
    called = []
    monkeypatch.setattr(oauth, "loopback", lambda *a, **k: called.append(1))
    _login_via_paste(provider, monkeypatch, {"access_token": "a"})
    assert not called


def test_other_providers_still_use_form_encoding(monkeypatch):
    for name in ("google", "codex"):
        _cred, posted, _url = _login_via_paste(
            oauth.get(name), monkeypatch, {"access_token": "a"}
        )
        assert posted["encoding"] == "form", name


def test_http_error_body_is_surfaced(monkeypatch):
    """A bare "HTTP Error 400: Bad Request" hides which of several causes it was;
    the body names it."""
    import io
    import urllib.error

    def raise_400(req, timeout):
        raise urllib.error.HTTPError(
            req.full_url, 400, "Bad Request", {},
            io.BytesIO(b'{"error":"invalid_grant"}'),
        )

    monkeypatch.setattr(oauth.urllib.request, "urlopen",
                        lambda req, timeout=None: raise_400(req, timeout))
    with pytest.raises(oauth.LoginError, match="invalid_grant"):
        oauth._post_json("https://example.test/token", {})


def test_pasted_code_with_fragment_is_cleaned(monkeypatch):
    """Anthropic hands back "code#state" when the code is copied by hand."""
    cred, posted, _ = _login_via_paste(
        oauth.get("anthropic"), monkeypatch,
        {"access_token": "a"}, code="realcode#somestate",
    )
    assert posted["payload"]["code"] == "realcode"
    assert cred["access"] == "a"


def test_google_asks_for_a_refresh_token(monkeypatch):
    """Without access_type=offline + prompt=consent Google returns no refresh
    token and the credential dies in an hour with no way back."""
    _cred, _posted, url = _login_via_paste(
        oauth.get("google"), monkeypatch, {"access_token": "a", "refresh_token": "r"}
    )
    assert "access_type=offline" in url and "prompt=consent" in url


def test_google_sends_its_client_secret(monkeypatch):
    _cred, posted, url = _login_via_paste(
        oauth.get("google"), monkeypatch, {"access_token": "a"}
    )
    assert posted["payload"]["client_secret"] == "test-secret"
    assert posted["payload"]["client_id"] == "test-client.apps.example"
    assert "client_id=test-client.apps.example" in url


@pytest.mark.parametrize("missing", ("GEMINI_OAUTH_CLIENT_ID", "GEMINI_OAUTH_CLIENT_SECRET"))
def test_google_without_its_client_pair_fails_before_any_prompt(missing, monkeypatch):
    """The pair lives in the environment, not the source; without it the login
    says which variables to set instead of opening a browser that can't work."""
    monkeypatch.delenv(missing)
    asked = []
    cb = oauth.LoginCallbacks(on_prompt=lambda q: asked.append(q) or "",
                              on_auth=lambda url: asked.append(url))
    with pytest.raises(oauth.LoginError, match="GEMINI_OAUTH_CLIENT_ID and GEMINI_OAUTH_CLIENT_SECRET"):
        oauth.get("google").login(cb)
    assert asked == []
    with pytest.raises(oauth.LoginError):
        oauth.get("google").refresh({"refresh": "r"})


def test_anthropic_sends_no_client_secret(monkeypatch):
    """A public PKCE client has none; sending an empty one is rejected."""
    _cred, posted, _url = _login_via_paste(
        oauth.get("anthropic"), monkeypatch, {"access_token": "a"}
    )
    assert "client_secret" not in posted["payload"]


def test_codex_uses_the_registered_redirect_port():
    """OpenAI validates redirect_uri exactly, so this port is not negotiable."""
    assert oauth.CodexSubscriptionProvider.REDIRECT_PORT == 1455
    assert oauth.CodexSubscriptionProvider.REDIRECT_PATH == "/auth/callback"


def test_codex_extracts_the_account_id(monkeypatch):
    """The account id rides in the id_token and is required as a request header."""
    claims = {"https://api.openai.com/auth": {"chatgpt_account_id": "acct-123"}}
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    cred, _posted, _url = _login_via_paste(
        oauth.get("codex"), monkeypatch,
        {"access_token": "a", "id_token": f"hdr.{payload}.sig"},
    )
    assert cred["account_id"] == "acct-123"


def test_codex_survives_an_unreadable_id_token(monkeypatch):
    cred, _posted, _url = _login_via_paste(
        oauth.get("codex"), monkeypatch, {"access_token": "a", "id_token": "garbage"}
    )
    assert cred["account_id"] == ""


@pytest.mark.parametrize("name", ALL_SUBSCRIPTION)
def test_missing_access_token_raises(name, monkeypatch):
    provider = oauth.get(name)
    _patch_token_endpoint(monkeypatch, {})
    monkeypatch.setattr(oauth, "LOGIN_TIMEOUT", 0.2)
    cb = oauth.LoginCallbacks(on_prompt=lambda _q: "code")
    with pytest.raises(oauth.LoginError, match="no access token"):
        provider.login(cb)


@pytest.mark.parametrize("name", ALL_SUBSCRIPTION)
def test_refresh_rotates_the_token(name, monkeypatch):
    provider = oauth.get(name)
    _patch_token_endpoint(
        monkeypatch,
        {"access_token": "new", "refresh_token": "newref", "expires_in": 60},
    )
    out = provider.refresh({"provider": name, "type": "oauth", "access": "old", "refresh": "old-r"})
    assert out["access"] == "new" and out["refresh"] == "newref"


@pytest.mark.parametrize("name", ALL_SUBSCRIPTION)
def test_refresh_keeps_the_old_refresh_token_when_omitted(name, monkeypatch):
    """A refresh response may omit refresh_token, meaning "keep the one you have";
    overwriting it with "" would make the next refresh impossible."""
    provider = oauth.get(name)
    _patch_token_endpoint(monkeypatch, {"access_token": "new"})
    out = provider.refresh({"provider": name, "type": "oauth", "access": "o", "refresh": "keepme"})
    assert out["refresh"] == "keepme"


def test_codex_refresh_preserves_the_account_id(monkeypatch):
    _patch_token_endpoint(monkeypatch, {"access_token": "new"})
    out = oauth.get("codex").refresh(
        {"provider": "codex", "type": "oauth", "access": "o", "refresh": "r", "account_id": "acct-9"}
    )
    assert out["account_id"] == "acct-9"


def test_fixed_port_collision_is_reported_clearly():
    """Two concurrent Codex logins can't both bind 1455 — say so rather than
    letting the second one look like a timeout."""
    with oauth.loopback(port=1455, timeout=0.2):
        with pytest.raises(oauth.LoginError, match="could not listen on port 1455"):
            with oauth.loopback(port=1455, timeout=0.2):
                pass


# --- paste-a-key providers ----------------------------------------------------

KEY_PROVIDERS = ("anthropic-key", "google-key", "groq-key", "openai-key")


@pytest.mark.parametrize("name", KEY_PROVIDERS)
def test_key_providers_are_registered(name):
    assert oauth.get(name) is not None
    assert dict(oauth.choices())[name]


def test_key_is_asked_for_as_a_secret_not_a_plain_prompt():
    """The key must go through on_secret so the driver can mask it — a key echoed
    into the log would then be exported with the transcript. The models question is
    a plain prompt, since a model name is not a secret."""
    asked = {"secret": [], "plain": []}
    cb = oauth.LoginCallbacks(
        on_secret=lambda q: asked["secret"].append(q) or "sk-ant-key",
        on_prompt=lambda q: asked["plain"].append(q) or "",
    )
    cred = oauth.get("anthropic-key").login(cb)
    assert any("API key" in q for q in asked["secret"])
    assert any("models" in q for q in asked["plain"])
    assert cred["key"] == "sk-ant-key"


def test_login_asks_for_models_and_blank_keeps_the_suggestion():
    asked = []
    cb = oauth.LoginCallbacks(
        on_secret=lambda _q: "sk-ant", on_prompt=lambda q: asked.append(q) or ""
    )
    cred = oauth.get("anthropic-key").login(cb)
    # the suggestion is shown, so a blank answer is an informed one
    assert "claude-sonnet-5" in asked[0]
    assert [m["name"] for m in cred["models"]] == [
        "claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5-20251001",
    ]


def test_login_accepts_a_comma_separated_model_list():
    cb = oauth.LoginCallbacks(
        on_secret=lambda _q: "sk-ant",
        on_prompt=lambda _q: " claude-opus-5 , claude-sonnet-5 ,, claude-opus-5 ",
    )
    cred = oauth.get("anthropic-key").login(cb)
    # trimmed, blanks dropped, duplicates removed, order preserved
    assert [m["name"] for m in cred["models"]] == ["claude-opus-5", "claude-sonnet-5"]


def test_first_configured_model_is_the_default():
    from financial_research_assistant import graph

    oauth.login("anthropic-key", oauth.LoginCallbacks(
        on_secret=lambda _q: "sk-ant", on_prompt=lambda _q: "claude-opus-5, claude-sonnet-5",
    ))
    assert auth.models() == ["claude-opus-5", "claude-sonnet-5"]
    assert auth.routing()[1] == "claude-opus-5"
    assert llm.resolved_model() == "claude-opus-5"
    # and every configured model is offered, all bound to that provider
    offered = {m: p for m, _label, p in llm.configured_models()}
    assert offered["claude-opus-5"] == "anthropic-key"
    assert offered["claude-sonnet-5"] == "anthropic-key"


def test_legacy_singular_model_field_is_still_read():
    """Credentials written before `models` existed hold a single `model` string."""
    auth.set_credential("default", {
        "provider": "anthropic-key", "type": "api_key", "key": "k",
        "model_provider": "anthropic", "model": "claude-opus-5",
    })
    assert auth.models() == ["claude-opus-5"]
    assert auth.routing()[1] == "claude-opus-5"


def test_anthropic_key_routes_to_the_anthropic_integration():
    cb = oauth.LoginCallbacks(on_secret=lambda _q: "sk-ant-api-xyz")
    cred = oauth.get("anthropic-key").login(cb)
    assert cred["type"] == "api_key"          # not oauth: no bearer path, no refresh
    assert cred["model_provider"] == "anthropic"
    assert cred["models"][0]["name"] == "claude-sonnet-5"
    assert "base_url" not in cred             # the SDK's own default


def test_openai_key_leaves_the_model_to_the_environment():
    """OPENAI_MODEL already names the model for this lane; pinning one on the
    credential would silently override a deliberate choice."""
    cb = oauth.LoginCallbacks(on_secret=lambda _q: "sk-openai", on_prompt=lambda _q: "")
    cred = oauth.get("openai-key").login(cb)
    assert cred["models"] == []
    assert "base_url" not in cred


def _answer(**by_keyword):
    """on_prompt that replies based on which question was asked; blank otherwise."""

    def prompt(question: str) -> str:
        for keyword, reply in by_keyword.items():
            if keyword in question.lower():
                return reply
        return ""

    return prompt


def test_openai_key_can_pin_a_gateway():
    cb = oauth.LoginCallbacks(
        on_secret=lambda _q: "sk-gw",
        on_prompt=_answer(
            **{"base url": "https://gateway.example/v1",
               "models": "gw-model-a, gw-model-b",
               "context window": "32k"},
        ),
    )
    cred = oauth.get("openai-key").login(cb)
    assert cred["base_url"] == "https://gateway.example/v1"
    assert [m["name"] for m in cred["models"]] == ["gw-model-a", "gw-model-b"]
    # the window lands on each model config, not on the credential
    assert "context_window" not in cred
    assert all(m["context_window"] == 32_000 for m in cred["models"])


def test_gateway_endpoint_is_asked_before_its_models():
    """The endpoint decides which models exist, so naming models first would mean
    answering for a gateway not yet chosen."""
    asked = []
    inner = _answer(**{"base url": "https://gateway.example/v1", "models": "gw-model"})
    cb = oauth.LoginCallbacks(
        on_secret=lambda _q: "sk-gw",
        on_prompt=lambda q: (asked.append(q), inner(q))[1],
    )
    oauth.get("openai-key").login(cb)
    assert "base URL" in asked[0]
    assert "models" in asked[1]


def test_empty_key_is_rejected():
    cb = oauth.LoginCallbacks(on_secret=lambda _q: "   ")
    with pytest.raises(oauth.LoginError, match="no API key"):
        oauth.get("anthropic-key").login(cb)


def test_key_credential_never_refreshes():
    cred = {"provider": "anthropic-key", "type": "api_key", "key": "k"}
    assert oauth.get("anthropic-key").refresh(cred) == cred


def test_stored_key_does_not_take_the_oauth_bearer_path(monkeypatch):
    """An API-key credential for anthropic must use x-api-key, not the OAuth
    bearer path built for subscription tokens."""
    from financial_research_assistant import graph

    oauth.login("anthropic-key", oauth.LoginCallbacks(on_secret=lambda _q: "sk-ant-api-xyz"))
    seen = {}
    monkeypatch.setattr(
        "langchain.chat_models.init_chat_model",
        lambda model, **kw: seen.update(kw, model=model) or object(),
    )
    llm._make_llm()
    assert seen["model"] == "claude-sonnet-5"
    assert seen["model_provider"] == "anthropic"
    assert seen["api_key"] == "sk-ant-api-xyz"   # goes out as x-api-key, correctly
    assert "default_headers" not in seen


def test_storing_a_plain_openai_key_keeps_a_configured_gateway(monkeypatch):
    """Regression: a credential that pins no endpoint must not drop
    OPENAI_API_BASE, or storing a key silently breaks a gateway setup."""
    from financial_research_assistant import graph

    monkeypatch.setenv("OPENAI_API_BASE", "https://gateway.example/v1")
    oauth.login("openai-key", oauth.LoginCallbacks(
        on_secret=lambda _q: "sk-openai", on_prompt=lambda _q: ""
    ))
    assert llm._credentials("default", "openai", None, None) == (
        "https://gateway.example/v1", "sk-openai",
    )


def test_costs_are_seeded_for_non_openai_providers():
    """"Set it by default when authenticating with a non-OpenAI provider": those
    are exactly the models the built-in table covers, so each entry lands with its
    per-1M rates already filled in."""
    cred = oauth.get("anthropic-key").login(
        oauth.LoginCallbacks(on_secret=lambda _q: "k")
    )
    haiku = next(m for m in cred["models"] if m["name"].startswith("claude-haiku"))
    assert haiku == {
        "name": "claude-haiku-4-5-20251001",
        "input_cost": 1.0,
        "output_cost": 5.0,
        "cache_write_cost": 1.25,   # Anthropic writes bill at 1.25x input
        "cache_read_cost": 0.1,     # and reads at 0.1x
        "context_window": 200_000,
    }
    opus = next(m for m in cred["models"] if m["name"] == "claude-opus-5")
    assert (opus["input_cost"], opus["output_cost"]) == (5.0, 25.0)


def test_unknown_gateway_model_gets_a_bare_entry_to_fill_in():
    cb = oauth.LoginCallbacks(
        on_secret=lambda _q: "k",
        on_prompt=_answer(**{"models": "gateframe_ionix/dspark"}),
    )
    cred = oauth.get("openai-key").login(cb)
    # every cost field present, zeroed: the shape to fill in is visible in the
    # file rather than something you have to know to add
    assert cred["models"] == [{
        "name": "gateframe_ionix/dspark",
        "input_cost": 0, "output_cost": 0,
        "cache_write_cost": 0, "cache_read_cost": 0,
    }]


def test_zero_costs_read_as_unset_not_as_free():
    """A confident $0.00 on a gateway that bills real money is worse than a blank,
    and 0 cannot be told apart from "nobody filled this in"."""
    from financial_research_assistant import pricing

    auth.set_credential("default", {
        "provider": "gw", "type": "api_key", "key": "k",
        "models": [{"name": "gw/model", "input_cost": 0, "output_cost": 0,
                    "cache_write_cost": 0, "cache_read_cost": 0}],
    })
    assert pricing.rates("gw/model") is None
    assert pricing.cost_usd("gw/model", 1000, 100) is None


def test_zero_cache_read_falls_back_to_the_discount():
    from financial_research_assistant import pricing

    auth.set_credential("default", {
        "provider": "gw", "type": "api_key", "key": "k",
        "models": [{"name": "claude-sonnet-5", "input_cost": 3.0, "output_cost": 15.0,
                    "cache_read_cost": 0}],
    })
    # 1M cached input at Anthropic's 0.1x discount = 0.30, not free
    assert pricing.cost_usd("claude-sonnet-5", 1_000_000, 0, 1_000_000) == pytest.approx(0.30)


def test_stored_costs_drive_cost_usd():
    from financial_research_assistant import pricing

    auth.set_credential("default", {
        "provider": "gw", "type": "api_key", "key": "k", "model_provider": "openai",
        "models": [{"name": "gw/model", "input_cost": "2.0", "output_cost": "8.0",
                    "cache_read_cost": "0.5", "cache_write_cost": "2.5"}],
    })
    assert pricing.rates("gw/model") == (2.0, 8.0)
    # 500k fresh in @2 + 500k cached @0.5 + 100k out @8 = 1.0 + 0.25 + 0.8
    assert pricing.cost_usd("gw/model", 1_000_000, 100_000, 500_000) == pytest.approx(2.05)


def test_costs_are_read_from_strings_as_the_example_writes_them():
    from financial_research_assistant import pricing

    auth.set_credential("default", {
        "provider": "gw", "type": "api_key", "key": "k",
        "models": [{"name": "m", "input_cost": "5.0", "output_cost": "0.1",
                    "cache_write_cost": "0.1", "cache_read_cost": "0.1"}],
    })
    assert pricing.rates("m") == (5.0, 0.1)


def test_stored_costs_do_not_apply_to_other_models():
    from financial_research_assistant import pricing

    auth.set_credential("default", {
        "provider": "gw", "type": "api_key", "key": "k",
        "models": [{"name": "gw/model", "input_cost": "99", "output_cost": "99"}],
    })
    assert pricing.rates("gpt-4o") == (2.50, 10.00)  # the table, untouched


def test_partial_costs_fall_through_to_the_table():
    """Both rates or neither — half a pair would bill output at a guess."""
    from financial_research_assistant import pricing

    auth.set_credential("default", {
        "provider": "gw", "type": "api_key", "key": "k",
        "models": [{"name": "gpt-4o", "input_cost": "1.0"}],
    })
    assert pricing.rates("gpt-4o") == (2.50, 10.00)


def test_per_model_context_window_beats_the_credential_wide_one():
    """The model config is the finer-grained answer for a credential serving
    models whose windows differ."""
    auth.set_credential("default", {
        "provider": "gw", "type": "api_key", "key": "k",
        "context_window": 32_000,
        "models": [{"name": "gw/big", "context_window": 250_000}, {"name": "gw/small"}],
    })
    assert auth.context_window("gw/big") == 250_000
    assert auth.context_window("gw/small") == 32_000


# --- refresh ------------------------------------------------------------------


NOW = 1_800_000_000_000  # fixed instant for the pure needs_refresh checks


def _expiring(delta_ms, provider="fake-oauth"):
    """Expiry relative to the REAL clock — ensure_fresh reads it, so a fixed
    timestamp would drift into "not yet expired" depending on the run date."""
    import time

    return {"provider": provider, "type": "oauth", "access": "old",
            "refresh": "rt", "expires": time.time() * 1000 + delta_ms}


class _RefreshableProvider:
    name = "fake-oauth"
    label = "test provider"

    def __init__(self, result=None, boom=False):
        self.calls = 0
        self._result = result
        self._boom = boom

    def login(self, cb):
        return _expiring(3_600_000, self.name)

    def refresh(self, cred):
        self.calls += 1
        if self._boom:
            raise OSError("refresh endpoint down")
        return self._result or {**cred, "access": "new", "expires": NOW + 3_600_000}


@pytest.fixture
def refreshable(monkeypatch):
    p = _RefreshableProvider()
    monkeypatch.setitem(oauth._REGISTRY, p.name, p)
    return p


def test_needs_refresh_boundaries():
    def at(delta):  # expiry relative to the fixed instant
        return {"type": "oauth", "access": "a", "expires": NOW + delta}

    assert oauth.needs_refresh(at(-1), NOW) is True          # expired
    assert oauth.needs_refresh(at(60_000), NOW) is True      # inside the skew
    assert oauth.needs_refresh(at(3_600_000), NOW) is False  # plenty left
    # a minted API key has no expiry to reason about
    assert oauth.needs_refresh({"type": "api_key", "key": "k"}, NOW) is False
    assert oauth.needs_refresh({"type": "oauth", "access": "a"}, NOW) is False
    assert oauth.needs_refresh({}, NOW) is False


def test_ensure_fresh_renews_and_persists(refreshable):
    auth.set_credential("default", _expiring(-1))
    oauth.ensure_fresh()
    assert refreshable.calls == 1
    assert auth.get("default")["access"] == "new"


def test_ensure_fresh_is_a_noop_when_not_due(refreshable):
    auth.set_credential("default", _expiring(3_600_000))
    oauth.ensure_fresh()
    assert refreshable.calls == 0


def test_ensure_fresh_writes_back_to_the_inherited_scope(refreshable):
    """A tier inheriting the primary credential must refresh IT, not sprout its own
    copy — otherwise it silently stops inheriting future logins."""
    auth.set_credential("default", _expiring(-1))
    oauth.ensure_fresh("subagent")
    assert refreshable.calls == 1
    assert auth.get("subagent") is None
    assert auth.get("default")["access"] == "new"


def test_failed_refresh_keeps_the_old_credential(monkeypatch):
    """A network blip must not throw away a good refresh token: a stale access
    token fails recoverably at the API, a deleted one needs a re-login."""
    p = _RefreshableProvider(boom=True)
    monkeypatch.setitem(oauth._REGISTRY, p.name, p)
    auth.set_credential("default", _expiring(-1))
    oauth.ensure_fresh()
    assert auth.get("default")["access"] == "old"
    assert auth.get("default")["refresh"] == "rt"


def test_concurrent_ensure_fresh_refreshes_exactly_once(monkeypatch):
    """A refresh token is single-use: the vendor invalidates it as it issues the
    next one. Two callers that both POST leave the loser holding a spent token, so
    the second must wait, re-read, and find the first's fresh credential."""
    import time

    class _SlowRefresh(_RefreshableProvider):
        def refresh(self, cred):
            time.sleep(0.2)  # long enough that the other caller is at the lock
            return super().refresh(cred)

    provider = _SlowRefresh()
    monkeypatch.setitem(oauth._REGISTRY, provider.name, provider)
    auth.set_credential("default", _expiring(-1))

    ready = threading.Barrier(2)

    def racer():
        ready.wait()
        oauth.ensure_fresh()

    threads = [threading.Thread(target=racer) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert provider.calls == 1
    assert auth.get("default")["access"] == "new"


def test_ensure_fresh_tolerates_unknown_provider():
    auth.set_credential("default", _expiring(-1, provider="since-uninstalled"))
    oauth.ensure_fresh()  # must not raise
    assert auth.get("default")["access"] == "old"


def test_ensure_fresh_with_empty_store():
    oauth.ensure_fresh()  # must not raise


def test_make_llm_refreshes_before_building(monkeypatch, refreshable):
    """The integration point: building the client renews a near-dead token, so the
    key that reaches ChatOpenAI is the fresh one."""
    from financial_research_assistant import graph

    auth.set_credential("default", _expiring(-1))
    seen = {}
    monkeypatch.setattr("langchain_openai.ChatOpenAI", lambda **kw: seen.update(kw))
    llm._make_llm("m")
    assert refreshable.calls == 1
    assert seen["api_key"].get_secret_value() == "new"


# --- headless CLI -------------------------------------------------------------


def test_cli_bare_login_lists_providers(capsys):
    from financial_research_assistant.main import _handle_login

    assert _handle_login("", "default") == 0
    assert "openrouter" in capsys.readouterr().out


def test_cli_login_rejects_unknown_provider_and_tier(capsys):
    from financial_research_assistant.main import _handle_login, _handle_logout

    assert _handle_login("nope", "default") == 2
    assert "unknown provider" in capsys.readouterr().err
    assert _handle_login("openrouter", "nope") == 2
    assert "unknown tier" in capsys.readouterr().err
    assert _handle_logout("nope") == 2


def test_cli_login_stores_and_logout_clears(monkeypatch, capsys):
    from financial_research_assistant.main import _handle_login, _handle_logout

    monkeypatch.setattr(oauth, "_post_json", lambda u, p, timeout=30.0: {"key": "sk-cli"})
    monkeypatch.setattr(oauth, "LOGIN_TIMEOUT", 0.2)
    monkeypatch.setattr("webbrowser.open", lambda _u: True)
    monkeypatch.setattr("builtins.input", lambda _q: "code")

    assert _handle_login("openrouter", "quick") == 0
    assert auth.resolve_key("quick") == "sk-cli"
    assert "signed in to openrouter (quick tier)" in capsys.readouterr().out

    assert _handle_logout("quick") == 0
    assert auth.get("quick") is None
    assert _handle_logout("quick") == 0  # idempotent
    assert "no credential in use" in capsys.readouterr().out

    # the credential itself survives a tier logout and can be removed by name
    assert auth.providers() == ["openrouter"]
    assert _handle_logout("default", "openrouter") == 0
    assert auth.providers() == []
    assert _handle_logout("default", "openrouter") == 2


def test_cli_login_failure_exits_nonzero(monkeypatch, capsys):
    from financial_research_assistant.main import _handle_login

    monkeypatch.setattr(oauth, "LOGIN_TIMEOUT", 0.2)
    monkeypatch.setattr("webbrowser.open", lambda _u: True)
    monkeypatch.setattr("builtins.input", lambda _q: "")  # user gives up

    assert _handle_login("openrouter", "default") == 1
    assert "login failed" in capsys.readouterr().err
    assert auth.get() is None


def test_cli_login_cancelled_exits_nonzero(monkeypatch, capsys):
    from financial_research_assistant.main import _handle_login

    monkeypatch.setattr(oauth, "LOGIN_TIMEOUT", 0.2)
    monkeypatch.setattr("webbrowser.open", lambda _u: True)

    def interrupt(_q):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt)
    assert _handle_login("openrouter", "default") == 1
    assert "cancelled" in capsys.readouterr().err


def test_login_reaches_make_llm(monkeypatch):
    """The point of the whole phase: after /login the agent builds its client with
    the minted key AND OpenRouter's endpoint — not a stale OPENAI_API_BASE."""
    from financial_research_assistant import graph

    monkeypatch.setenv("OPENAI_API_KEY", "sk-stale-dotenv")
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    monkeypatch.setattr(oauth, "_post_json", lambda u, p, timeout=30.0: {"key": "sk-or"})
    monkeypatch.setattr(oauth, "LOGIN_TIMEOUT", 0.2)
    cb, _ = _cb(on_prompt=lambda _p: "code")
    oauth.login("openrouter", cb)

    seen = {}
    monkeypatch.setattr("langchain_openai.ChatOpenAI", lambda **kw: seen.update(kw))
    llm._make_llm("some-model")
    assert seen["base_url"] == "https://openrouter.ai/api/v1"
    assert seen["api_key"].get_secret_value() == "sk-or"


# --- a provider whose package is not installed ---------------------------------
#
# Every non-OpenAI provider is an OPTIONAL extra, so this is the first thing a
# working configuration hits on a fresh checkout. It surfaced as a bare
# "ModuleNotFoundError: No module named 'langchain_anthropic'" on an agent that
# had otherwise started fine — the module named, the remedy nowhere.


def _hide(monkeypatch, module):
    """Make `import <module>` fail exactly as an uninstalled package does."""
    import builtins

    real = builtins.__import__

    def fake(name, *args, **kwargs):
        if name == module or name.startswith(module + "."):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)


def _init_chat_model_raises(monkeypatch, exc):
    """Make `init_chat_model` fail with `exc`.

    Deliberately NOT done by hiding the real module. `init_chat_model` imports
    through `importlib.import_module`, which does not go through
    `builtins.__import__` — so a fake that hooks `__import__` models the wrong
    thing and silently stops failing. It did: these two tests passed against
    langchain-anthropic 1.4.8 and went green-but-meaningless on 1.5.6, where the
    real client was built instead. Pin the ERROR the library raises, not the
    machinery it happens to raise it from.
    """
    def boom(*_a, **_k):
        raise exc

    monkeypatch.setattr("langchain.chat_models.init_chat_model", boom)


#: What `init_chat_model` raises for an absent integration. Note it carries no
#: `name` attribute, which is why the message fallback exists.
_LANGCHAIN_IMPORT_ERROR = ImportError(
    "Initializing ChatAnthropic requires the langchain-anthropic package. "
    "Please install it with `pip install langchain-anthropic`"
)


def test_a_missing_provider_package_names_the_install_command(monkeypatch):
    _init_chat_model_raises(monkeypatch, _LANGCHAIN_IMPORT_ERROR)
    with pytest.raises(llm.ProviderSupportError) as got:
        llm._make_llm("claude-sonnet-4-5", provider="anthropic", api_key="k")
    message = str(got.value)
    assert "langchain-anthropic" in message
    assert "uv pip install langchain-anthropic" in message
    assert "financial-research-assistant[anthropic]" in message


def test_the_hint_warns_that_uv_sync_removes_other_extras(monkeypatch):
    """`uv sync --extra anthropic` syncs to exactly the extras named and removes
    the rest — run without `--extra dev` it uninstalls pytest, which is how this
    warning was earned."""
    _init_chat_model_raises(monkeypatch, _LANGCHAIN_IMPORT_ERROR)
    with pytest.raises(llm.ProviderSupportError) as got:
        llm._make_llm("claude-sonnet-4-5", provider="anthropic", api_key="k")
    assert "removes the rest" in str(got.value)


def test_a_missing_module_inside_an_installed_provider_is_not_masked(monkeypatch):
    """Answering a real bug with install advice sends the reader after the wrong
    thing entirely, so only the integration package itself converts."""
    _init_chat_model_raises(monkeypatch, ModuleNotFoundError(
        "No module named 'some_internal_dep'", name="some_internal_dep"
    ))
    with pytest.raises(ModuleNotFoundError) as got:
        llm._make_llm("claude-sonnet-4-5", provider="anthropic", api_key="k")
    assert not isinstance(got.value, llm.ProviderSupportError)
    assert "some_internal_dep" in str(got.value)


def test_a_provider_with_no_declared_extra_still_gets_a_command(monkeypatch):
    """The map covers what this project declares; anything else still beats a
    bare import error."""
    _init_chat_model_raises(monkeypatch, ModuleNotFoundError(
        "No module named 'langchain_cohere'", name="langchain_cohere"
    ))
    with pytest.raises(llm.ProviderSupportError) as got:
        llm._make_llm("some-model", provider="cohere", api_key="k")
    assert "uv pip install langchain-cohere" in str(got.value)


def test_the_openai_path_is_untouched(monkeypatch):
    """The guard must cost nothing to the default provider, which needs no extra."""
    seen = {}
    monkeypatch.setattr("langchain_openai.ChatOpenAI", lambda **kw: seen.update(kw))
    llm._make_llm("gpt-4o", provider="openai", api_key="k")
    assert seen["model"] == "gpt-4o"


def test_the_oauth_path_gets_the_hint_too(monkeypatch):
    """THE case that was actually reported. `init_chat_model` catches the import
    itself and re-raises a message of its own, but the OAuth subclass imports
    `langchain_anthropic` directly — so an OAuth user, and only an OAuth user, saw
    the bare "No module named 'langchain_anthropic'" with no remedy in it."""
    auth.set_credential("default", {
        "provider": "anthropic", "type": "oauth", "access": "tok",
        "refresh": "rt", "expires": 0,
    })
    monkeypatch.setattr(llm, "_is_oauth", lambda _scope: True)
    _hide(monkeypatch, "langchain_anthropic")
    with pytest.raises(llm.ProviderSupportError) as got:
        llm._make_llm("claude-sonnet-4-5", provider="anthropic", api_key="tok")
    assert "uv pip install langchain-anthropic" in str(got.value)


def test_the_oauth_path_still_builds_when_the_package_is_there():
    """The guard must cost nothing to the path it wraps: one auth header, the
    bearer one, even with an API key in the environment."""
    pytest.importorskip("langchain_anthropic")
    client = llm._anthropic_oauth_llm("claude-sonnet-4-5", "tok-abc", None)
    params = client._client_params
    assert params["auth_token"] == "tok-abc"
    assert params["api_key"] is None
