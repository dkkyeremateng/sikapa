"""Credential store: path resolution, 0600 writes, value indirection, scopes."""

import importlib.util
import json
import os
import stat

import pytest

from financial_research_assistant import auth, llm


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    """Never touch the real ~/.financial-research-assistant/auth.json."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_AUTH_FILE", str(tmp_path / "auth.json"))
    monkeypatch.delenv("FINANCIAL_RESEARCH_AUTH_ALLOW_EXEC", raising=False)


def test_path_env_override_and_default(tmp_path, monkeypatch):
    monkeypatch.setenv("FINANCIAL_RESEARCH_AUTH_FILE", str(tmp_path / "x.json"))
    assert auth.auth_file() == tmp_path / "x.json"
    monkeypatch.delenv("FINANCIAL_RESEARCH_AUTH_FILE")
    assert auth.auth_file().name == "auth.json"
    assert auth.auth_file().parent.name == ".financial-research-assistant"


def test_missing_file_reads_as_empty():
    assert auth.load()["credentials"] == {}
    assert auth.get() is None
    assert auth.resolve_key() is None


def test_round_trip_and_scopes():
    auth.set_credential("default", {"provider": "openai", "type": "api_key", "key": "k1"})
    auth.set_credential("quick", {"provider": "local", "type": "api_key", "key": "k2"})
    assert auth.resolve_key("default") == "k1"
    assert auth.resolve_key("quick") == "k2"
    # a scope with no credential of its own inherits the primary agent's
    assert auth.resolve_key("subagent") == "k1"


def test_written_file_is_0600():
    auth.set_credential("default", {"type": "api_key", "key": "secret"})
    mode = stat.S_IMODE(os.stat(auth.auth_file()).st_mode)
    assert mode == 0o600, f"secret written world-readable: {oct(mode)}"


def test_corrupt_file_does_not_raise():
    auth.auth_file().parent.mkdir(parents=True, exist_ok=True)
    auth.auth_file().write_text("{not json", encoding="utf-8")
    assert auth.load()["credentials"] == {}
    assert auth.resolve_key() is None


def test_wrong_shape_file_does_not_raise():
    auth.auth_file().parent.mkdir(parents=True, exist_ok=True)
    auth.auth_file().write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert auth.load()["credentials"] == {}


def test_delete():
    auth.set_credential("quick", {"type": "api_key", "key": "k"})
    assert auth.delete("quick") is True
    assert auth.delete("quick") is False
    assert auth.get("quick") is None


def test_env_var_indirection(monkeypatch):
    auth.set_credential("default", {"type": "api_key", "key": "$MY_SECRET"})
    monkeypatch.setenv("MY_SECRET", "from-env")
    assert auth.resolve_key() == "from-env"
    # unset indirection resolves to None so the caller falls through, rather
    # than sending an empty Authorization header
    monkeypatch.delenv("MY_SECRET")
    assert auth.resolve_key() is None


def test_command_indirection_is_off_by_default(tmp_path):
    marker = tmp_path / "ran"
    auth.set_credential(
        "default", {"type": "api_key", "key": f"!touch {marker} && echo pwned"}
    )
    assert auth.resolve_key() is None
    assert not marker.exists(), "config-file shell ran without the opt-in"


def test_command_indirection_when_enabled(monkeypatch):
    monkeypatch.setenv("FINANCIAL_RESEARCH_AUTH_ALLOW_EXEC", "1")
    auth.set_credential("default", {"type": "api_key", "key": "!echo from-cmd"})
    assert auth.resolve_key() == "from-cmd"


def test_failing_command_resolves_to_none(monkeypatch):
    monkeypatch.setenv("FINANCIAL_RESEARCH_AUTH_ALLOW_EXEC", "1")
    auth.set_credential("default", {"type": "api_key", "key": "!exit 3"})
    assert auth.resolve_key() is None


def test_oauth_credential_returns_access_token():
    auth.set_credential(
        "default",
        {"provider": "openrouter", "type": "oauth",
         "access": "at", "refresh": "rt", "expires": 1786000000000},
    )
    assert auth.resolve_key() == "at"


def test_describe_never_leaks_the_secret():
    auth.set_credential("default", {"provider": "openai", "type": "api_key", "key": "sk-super-secret"})
    line = auth.describe()
    assert "sk-super-secret" not in line
    assert "openai" in line

    auth.set_credential(
        "quick",
        {"provider": "openrouter", "type": "oauth", "access": "tok", "expires": 1786000000000},
    )
    line = auth.describe("quick")
    assert "tok" not in line and "openrouter" in line and "expires" in line
    assert auth.describe("subagent") == "none"


def test_base_url_travels_with_the_credential():
    auth.set_credential(
        "default",
        {"provider": "openrouter", "type": "api_key", "key": "k",
         "base_url": "https://openrouter.ai/api/v1"},
    )
    assert auth.base_url() == "https://openrouter.ai/api/v1"
    assert auth.base_url("quick") == "https://openrouter.ai/api/v1"  # inherits
    auth.set_credential("quick", {"type": "api_key", "key": "q"})
    assert auth.base_url("quick") is None  # its own credential pins no endpoint


# --- llm._credentials: precedence ------------------------------------------


def _creds(scope="default", provider="openai", base_url=None, api_key=None):
    from financial_research_assistant import llm

    return llm._credentials(scope, provider, base_url, api_key)


def test_explicit_argument_wins_over_everything(monkeypatch):
    auth.set_credential("default", {"type": "api_key", "key": "stored"})
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    assert _creds(api_key="explicit") == (None, "explicit")


def test_store_outranks_env(monkeypatch):
    """A stale .env key must not shadow a fresh /login — load_dotenv() makes it a
    real env var, so ranked the other way /login would silently do nothing."""
    monkeypatch.setenv("OPENAI_API_KEY", "stale-dotenv-key")
    auth.set_credential("default", {"type": "api_key", "key": "fresh-login"})
    assert _creds() == (None, "fresh-login")


def test_env_used_when_store_is_empty(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_API_BASE", "http://env/v1")
    assert _creds() == ("http://env/v1", "env-key")


def test_endpoint_and_key_come_from_the_same_source(monkeypatch):
    """The pairing invariant: a /login credential that pins an endpoint must not be
    combined with a leftover OPENAI_API_BASE pointing at a different vendor."""
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    auth.set_credential(
        "default",
        {"type": "api_key", "key": "gw-token", "base_url": "https://gateway/v1"},
    )
    assert _creds() == ("https://gateway/v1", "gw-token")


def test_non_openai_provider_never_gets_openai_env_key(monkeypatch):
    """With nothing stored, anthropic/google must fall through to init_chat_model
    reading their OWN env var — handing them OPENAI_API_KEY would break them."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("OPENAI_API_BASE", "http://openai/v1")
    assert _creds(provider="anthropic") == (None, None)


def test_stored_credential_applies_to_any_provider():
    """The store is tier-scoped, not provider-scoped: whatever that tier logged in
    with is what it uses."""
    auth.set_credential("default", {"type": "api_key", "key": "tok"})
    assert _creds(provider="anthropic") == (None, "tok")


# --- llm.endpoint: what the header reports -----------------------------------


def _endpoint(scope="default"):
    from financial_research_assistant import llm

    return llm.endpoint(scope)


def test_endpoint_reports_the_credentials_own_endpoint(monkeypatch):
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    auth.set_credential(
        "default",
        {"provider": "openrouter", "type": "api_key", "key": "k",
         "base_url": "https://openrouter.ai/api/v1"},
    )
    assert _endpoint() == "https://openrouter.ai/api/v1"


def test_endpoint_names_the_providers_own_default_when_nothing_is_pinned(monkeypatch):
    """An Anthropic key pins no endpoint — the SDK's default is where the request
    goes, so reporting OPENAI_API_BASE would name the one endpoint it does not."""
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    auth.set_credential(
        "default",
        {"provider": "anthropic-key", "type": "api_key", "key": "sk-ant-x",
         "model_provider": "anthropic", "models": [{"name": "claude-haiku-4-5-20251001"}]},
    )
    assert _endpoint() == "https://api.anthropic.com"


def test_endpoint_falls_back_to_the_env_with_nothing_stored(monkeypatch):
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost:1234/v1")
    assert _endpoint() == "http://localhost:1234/v1"
    monkeypatch.delenv("OPENAI_API_BASE")
    assert _endpoint() == "https://api.openai.com/v1"


def test_endpoint_agrees_with_the_client_when_a_key_cannot_be_resolved(monkeypatch):
    """A credential whose `$VAR` is unset supplies no key, so `_credentials` uses
    the env for BOTH halves of the pair — the endpoint shown must follow."""
    monkeypatch.delenv("MISSING_KEY_VAR", raising=False)
    monkeypatch.setenv("OPENAI_API_BASE", "http://env/v1")
    auth.set_credential(
        "default",
        {"provider": "openrouter", "type": "api_key", "key": "$MISSING_KEY_VAR",
         "base_url": "https://openrouter.ai/api/v1"},
    )
    assert _creds()[0] == "http://env/v1"
    assert _endpoint() == "http://env/v1"


def test_endpoint_is_per_scope(monkeypatch):
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    auth.set_credential(
        "default",
        {"provider": "openrouter", "type": "api_key", "key": "k",
         "base_url": "https://openrouter.ai/api/v1"},
    )
    auth.set_credential(
        "quick",
        {"provider": "local", "type": "api_key", "key": "q",
         "base_url": "http://localhost:8080/v1"},
    )
    assert _endpoint() == "https://openrouter.ai/api/v1"
    assert _endpoint("quick") == "http://localhost:8080/v1"
    assert _endpoint("subagent") == "https://openrouter.ai/api/v1"  # inherits


def test_scope_isolation_and_inheritance(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    auth.set_credential("default", {"type": "api_key", "key": "primary"})
    auth.set_credential("quick", {"type": "api_key", "key": "cheap"})
    assert _creds(scope="quick")[1] == "cheap"
    assert _creds(scope="subagent")[1] == "primary"  # inherits default


# --- credential-driven routing ------------------------------------------------


ANTHROPIC_CRED = {
    "provider": "anthropic", "type": "oauth", "access": "sk-ant-oat01-xxxxxxxxxxxxxxxx",
    "expires": 4_000_000_000_000, "base_url": "https://api.anthropic.com",
    "model_provider": "anthropic", "model": "claude-sonnet-4-5",
}


def test_login_switches_provider_and_model(monkeypatch):
    """Regression: /login stored a token but left MODEL_PROVIDER/OPENAI_MODEL in
    charge, so a Claude token was built into a ChatOpenAI aimed at
    api.anthropic.com — which has no /v1/chat/completions and failed opaquely."""
    from financial_research_assistant import llm

    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "gateframe_ionix/dspark")
    monkeypatch.setenv("OPENAI_API_BASE", "https://gateway.example/v1")
    auth.set_credential("default", ANTHROPIC_CRED)

    assert llm.credential_provider() == "anthropic"
    assert llm.credential_model() == "claude-sonnet-4-5"
    assert llm.resolved_model() == "claude-sonnet-4-5"


LEGACY_ANTHROPIC_CRED = {
    # Exactly what /login wrote before model_provider/model were added — a stored
    # credential outlives the code that wrote it.
    "provider": "anthropic", "type": "oauth", "access": "sk-ant-oat01-xxxxxxxxxxxxxxxx",
    "refresh": "r", "expires": 4_000_000_000_000, "base_url": "https://api.anthropic.com",
}


def test_credential_without_routing_fields_still_routes(monkeypatch):
    """Regression: a credential stored before the routing fields existed was
    invisible — absent from /models, and ignored when building the client. The
    provider registry supplies the defaults, so no re-login is needed."""
    from financial_research_assistant import llm

    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "gateframe_ionix/dspark")
    auth.set_credential("default", LEGACY_ANTHROPIC_CRED)

    assert auth.routing() == ("anthropic", "claude-haiku-4-5-20251001")
    assert llm.credential_provider() == "anthropic"
    assert llm.resolved_model() == "claude-haiku-4-5-20251001"

    models = {m: (label, provider) for m, label, provider in llm.configured_models()}
    assert models["claude-haiku-4-5-20251001"] == ("active · anthropic", "anthropic")
    # and the provider default row follows the credential, not the env
    assert "claude-sonnet-5" in models


def _offered():
    from financial_research_assistant import llm

    return {model: (label, provider) for model, label, provider in llm.configured_models()}


def test_every_configured_provider_is_offered(monkeypatch):
    """Several providers can be configured at once, and each contributes a
    selectable entry carrying the provider it needs."""
    from financial_research_assistant import oauth

    oauth.login("anthropic-key", oauth.LoginCallbacks(on_secret=lambda _q: "sk-ant"))
    oauth.login("groq-key", oauth.LoginCallbacks(on_secret=lambda _q: "gsk"))

    offered = _offered()
    assert offered["claude-sonnet-5"] == ("configured · anthropic-key", "anthropic-key")
    assert offered["llama-3.3-70b-versatile"] == ("active · groq-key", "groq-key")
    # both credentials survive: the second login did not replace the first
    assert set(auth.providers()) == {"anthropic-key", "groq-key"}


def test_model_entry_carries_the_provider_that_serves_it(monkeypatch):
    """Regression: picking an OpenAI-gateway model while signed in to Anthropic
    sent that name to Anthropic and got `404 model: <name>`. Each entry now names
    the credential it must be used with, so selection can switch both together."""
    monkeypatch.setenv("OPENAI_MODEL", "gateframe_ionix/dspark")
    auth.set_credential("default", ANTHROPIC_CRED)

    offered = _offered()
    # the env model has no stored key to pair with while a credential is active
    assert "gateframe_ionix/dspark" not in offered
    assert offered["claude-sonnet-4-5"][1] == "anthropic"


def test_env_models_are_offered_when_nothing_is_signed_in(monkeypatch):
    from financial_research_assistant import llm

    monkeypatch.setenv("OPENAI_MODEL", "gateframe_ionix/dspark")
    monkeypatch.setenv("QUICK_MODEL", "cheap-mini")
    offered = _offered()
    assert offered["gateframe_ionix/dspark"] == ("OPENAI_MODEL", "")
    assert "cheap-mini" in offered
    assert llm.active_lane_note() == ""


def test_quick_tier_on_another_lane_is_excluded(monkeypatch):
    """A quick tier pointed at its own provider is a different lane, even though
    it lives in the same env family."""
    monkeypatch.setenv("MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("QUICK_MODEL", "local-mini")
    monkeypatch.setenv("QUICK_MODEL_PROVIDER", "openai")
    assert "local-mini" not in _offered()


def test_activate_switches_without_re_login():
    from financial_research_assistant import oauth

    oauth.login("anthropic-key", oauth.LoginCallbacks(on_secret=lambda _q: "sk-ant"))
    oauth.login("groq-key", oauth.LoginCallbacks(on_secret=lambda _q: "gsk"))
    assert auth.active() == "groq-key"

    assert auth.activate("anthropic-key") is True
    assert auth.active() == "anthropic-key"
    assert auth.resolve_key() == "sk-ant"          # the earlier key was kept
    assert auth.routing()[0] == "anthropic"

    assert auth.activate("never-stored") is False  # nothing to point at


def test_forget_removes_a_provider_and_any_tier_using_it():
    from financial_research_assistant import oauth

    oauth.login("anthropic-key", oauth.LoginCallbacks(on_secret=lambda _q: "sk-ant"))
    oauth.login("groq-key", oauth.LoginCallbacks(on_secret=lambda _q: "gsk"), scope="quick")

    assert auth.forget("groq-key") is True
    assert auth.providers() == ["anthropic-key"]
    assert auth.get("quick") is None               # the tier stopped pointing at it
    assert auth.forget("groq-key") is False


def test_logout_keeps_the_credential_for_later():
    """/logout stops using a provider; it must not destroy the key, or switching
    back means re-authenticating."""
    from financial_research_assistant import oauth

    oauth.login("anthropic-key", oauth.LoginCallbacks(on_secret=lambda _q: "sk-ant"))
    assert auth.delete("default") is True
    assert auth.get("default") is None
    assert auth.providers() == ["anthropic-key"]   # still stored
    assert auth.activate("anthropic-key") is True  # and reusable
    assert auth.resolve_key() == "sk-ant"


def test_v1_store_migrates_to_multi_provider(tmp_path, monkeypatch):
    """An existing single-provider file must keep working and gain the new shape."""
    store = tmp_path / "v1.json"
    store.write_text(json.dumps({"version": 1, "credentials": {
        "default": {"provider": "anthropic", "type": "oauth", "access": "tok"},
        "quick": {"provider": "groq-key", "type": "api_key", "key": "gsk"},
    }}), encoding="utf-8")
    monkeypatch.setenv("FINANCIAL_RESEARCH_AUTH_FILE", str(store))

    assert auth.active() == "anthropic"
    assert auth.resolve_key() == "tok"
    assert auth.resolve_key("quick") == "gsk"
    assert set(auth.providers()) == {"anthropic", "groq-key"}


def test_explicit_routing_beats_the_provider_default():
    """A credential that names its own model keeps it — the registry is a
    fallback, not an override."""
    auth.set_credential("default", {**LEGACY_ANTHROPIC_CRED, "model": "claude-opus-5"})
    assert auth.routing()[1] == "claude-opus-5"


def test_routing_of_an_unknown_provider_is_empty():
    auth.set_credential("default", {"provider": "since-removed", "type": "api_key", "key": "k"})
    assert auth.routing() == (None, None)


def test_env_still_wins_when_nothing_is_stored(monkeypatch):
    from financial_research_assistant import llm

    monkeypatch.setenv("MODEL_PROVIDER", "groq")
    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    assert llm.credential_provider() == "groq"
    assert llm.credential_model() == "env-model"


def test_explicit_model_override_beats_the_credential():
    """/model NAME is the user choosing live, so it outranks everything."""
    from financial_research_assistant import llm

    auth.set_credential("default", ANTHROPIC_CRED)
    assert llm.resolved_model("some-other-model") == "some-other-model"


@pytest.mark.skipif(
    importlib.util.find_spec("langchain_anthropic") is None,
    reason="the [anthropic] extra is not installed",
)
def test_anthropic_oauth_sends_bearer_and_no_x_api_key(monkeypatch):
    """Regression for a live 401 'invalid x-api-key'.

    Asserts the actual outgoing headers rather than the constructor arguments —
    the bug was that the SDK emits X-Api-Key whenever api_key is not None, so a
    kwargs-level assertion passed while the wire format was still wrong. The env
    key is set to prove it doesn't leak back in via the SDK's own fallback.
    """
    from anthropic._base_client import FinalRequestOptions

    from financial_research_assistant import llm

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-a-real-key-from-env")
    auth.set_credential("default", ANTHROPIC_CRED)

    chat = llm._make_llm()
    request = chat._client._build_request(
        FinalRequestOptions(method="post", url="/v1/messages")
    )
    assert request.headers.get("authorization") == "Bearer " + ANTHROPIC_CRED["access"]
    assert request.headers.get("x-api-key") is None
    assert request.headers.get("anthropic-beta") == "oauth-2025-04-20"
    assert chat.model == "claude-sonnet-4-5"


def test_api_key_credential_gets_no_oauth_headers(monkeypatch):
    from financial_research_assistant import llm

    auth.set_credential(
        "default",
        {"provider": "x", "type": "api_key", "key": "k", "model_provider": "anthropic"},
    )
    seen = {}
    monkeypatch.setattr(
        "langchain.chat_models.init_chat_model",
        lambda model, **kw: seen.update(kw, model=model) or object(),
    )
    llm._make_llm()
    assert "default_headers" not in seen
    assert seen["api_key"] == "k"


# --- robustness ---------------------------------------------------------------


def test_concurrent_writes_do_not_corrupt_the_store():
    """What the lock is for: read-modify-write from several threads must not lose
    entries or leave half a file behind."""
    import threading

    def write(i):
        auth.set_credential("default", {"provider": f"p{i}", "type": "api_key", "key": f"k{i}"})

    threads = [threading.Thread(target=write, args=(i,)) for i in range(25)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(auth.providers()) == 25
    json.loads(auth.auth_file().read_text(encoding="utf-8"))  # still valid JSON


def test_hand_edited_junk_in_models_is_survived():
    """auth.json is meant to be edited by hand, so a typo must degrade rather than
    crash a turn."""
    from financial_research_assistant import pricing

    auth.auth_file().parent.mkdir(parents=True, exist_ok=True)
    auth.auth_file().write_text(json.dumps({"version": 2, "providers": {"x": {
        "provider": "x", "type": "api_key", "key": "k",
        "models": ["ok-name", 42, None, {"noname": 1}, {"name": ""},
                   {"name": "n2", "input_cost": "abc", "context_window": "big"}]}},
        "active": {"default": "x"}}), encoding="utf-8")

    assert auth.models() == ["ok-name", "n2"]      # junk dropped, valid kept
    assert auth.model_config("n2") == {"name": "n2"}  # unparseable fields ignored
    assert pricing.rates("n2") is None                # not billed at a guess
    assert pricing.context_cap("n2") == 128_000


def test_active_pointing_at_a_missing_provider_is_harmless():
    """Deleting a provider by hand while a tier still names it."""
    auth.auth_file().parent.mkdir(parents=True, exist_ok=True)
    auth.auth_file().write_text(
        json.dumps({"version": 2, "providers": {}, "active": {"default": "ghost"}}),
        encoding="utf-8",
    )
    assert auth.get() is None
    assert auth.resolve_key() is None
    assert auth.models() == []
    assert auth.routing() == (None, None)


def test_models_reported_are_models_that_resolve():
    """Regression: a credential written before `models` existed reported its
    provider's models but had no config for any of them, so /models offered
    entries that resolved to nothing."""
    auth.set_credential("default", {"provider": "anthropic-key", "type": "api_key", "key": "k"})
    listed = auth.models()
    assert listed  # the provider's declared models
    for name in listed:
        assert auth.model_config(name) is not None, name


# --- cache-write billing --------------------------------------------------------


def test_cache_write_bills_at_its_own_rate():
    """Writing to the cache costs MORE than fresh input on Anthropic (1.25x) while
    reading costs far less (0.1x), so folding either into the input rate misreports
    a cache-heavy turn in opposite directions."""
    from financial_research_assistant import pricing

    # claude-sonnet: $3 in / $15 out; write 3.75, read 0.30
    # 1M input = 400k fresh + 400k read + 200k write
    expected = (400_000 * 3.0 + 400_000 * 0.30 + 200_000 * 3.75) / 1_000_000
    got = pricing.cost_usd("claude-sonnet-5", 1_000_000, 0, 400_000, 200_000)
    assert got == pytest.approx(expected)
    # and it is strictly more than treating the written portion as fresh input
    assert got > pricing.cost_usd("claude-sonnet-5", 1_000_000, 0, 400_000, 0)


def test_stored_cache_write_rate_wins():
    from financial_research_assistant import pricing

    auth.set_credential("default", {
        "provider": "gw", "type": "api_key", "key": "k",
        "models": [{"name": "gw/m", "input_cost": 2.0, "output_cost": 8.0,
                    "cache_write_cost": 10.0, "cache_read_cost": 1.0}],
    })
    # 100k written at the stored 10.0/1M = 1.0
    assert pricing.cost_usd("gw/m", 100_000, 0, 0, 100_000) == pytest.approx(1.0)


def test_openai_lane_cache_write_is_not_a_premium():
    """OpenAI-compatible endpoints don't charge extra to write, so a write bills
    like fresh input rather than at Anthropic's 1.25x."""
    from financial_research_assistant import pricing

    plain = pricing.cost_usd("gpt-4o", 100_000, 0, 0, 0)
    written = pricing.cost_usd("gpt-4o", 100_000, 0, 0, 100_000)
    assert written == pytest.approx(plain)


def test_cache_counts_are_clamped_to_the_input_total():
    """A provider reporting slightly inconsistent counts must not drive the fresh
    portion negative and refund the turn."""
    from financial_research_assistant import pricing

    cost = pricing.cost_usd("claude-sonnet-5", 1_000, 0, 900, 900)
    assert cost > 0
    # never cheaper than pricing every input token at the cheapest cache rate
    assert cost >= 1_000 * 0.30 / 1_000_000


def test_cache_write_is_free_when_the_model_is_unpriced():
    from financial_research_assistant import pricing

    assert pricing.cost_usd("totally-unknown-model", 100, 10, 5, 5) is None


# --- context window -----------------------------------------------------------


def test_non_openai_providers_default_a_known_window():
    """"Set it by default when available." The value comes from the per-model
    table, NOT a per-provider constant: one Anthropic key serves a 1M Sonnet and a
    200k Haiku, so a single hardcoded number would be wrong for one of them."""
    from financial_research_assistant import oauth

    for provider, expected in (("anthropic-key", 1_000_000),   # claude-sonnet-5
                               ("google-key", 1_048_576),      # gemini-2.5
                               ("groq-key", 128_000)):         # llama-3
        oauth.login(provider, oauth.LoginCallbacks(on_secret=lambda _q: "k"))
        assert auth.context_window() == expected, provider


def test_openai_lane_has_no_default_window():
    """A gateway can serve anything, so there is nothing honest to default to."""
    from financial_research_assistant import oauth

    oauth.login("openai-key", oauth.LoginCallbacks(on_secret=lambda _q: "k"))
    assert auth.context_window() is None


def test_window_is_configurable_at_login():
    from financial_research_assistant import oauth

    def answer(question):
        return "48k" if "context window" in question else ""

    oauth.login("openai-key", oauth.LoginCallbacks(
        on_secret=lambda _q: "k", on_prompt=answer,
    ))
    assert auth.context_window() == 48_000


def test_window_accepts_k_m_and_plain_forms():
    from financial_research_assistant import oauth

    for typed, expected in (("200000", 200_000), ("128k", 128_000), ("1m", 1_000_000)):
        cb = oauth.LoginCallbacks(on_prompt=lambda _q, t=typed: t)
        assert oauth.ask_context_window(cb, None) == expected


def test_blank_or_garbage_window_leaves_per_model_defaults():
    """None means "keep what the table gave each model" — more accurate than one
    number across models whose windows differ."""
    from financial_research_assistant import oauth

    for typed in ("", "lots", "-5", "0"):
        cb = oauth.LoginCallbacks(on_prompt=lambda _q, t=typed: t)
        assert oauth.ask_context_window(cb, 200_000) is None, typed


def test_stored_window_drives_context_cap():
    """The case this exists for: a gateway model the built-in table has never heard
    of would otherwise silently fall through to the 128k default, which the ctx%
    gauge and the compaction thresholds all divide by."""
    from financial_research_assistant import oauth, pricing

    assert pricing.context_cap("gateframe_ionix/dspark") == 128_000  # the fallback

    def answer(question):
        return {"base url": "https://gw/v1", "models": "gateframe_ionix/dspark",
                "context window": "48k"}.get(
            next((k for k in ("base url", "models", "context window") if k in question), ""), "")

    oauth.login("openai-key", oauth.LoginCallbacks(on_secret=lambda _q: "k", on_prompt=answer))
    assert pricing.context_cap("gateframe_ionix/dspark") == 48_000


def test_blank_window_keeps_each_models_own_size():
    """Accepting the default must not stamp one number over models the table sizes
    differently — an Anthropic key serves a 1M Sonnet and a 200k Haiku."""
    from financial_research_assistant import oauth, pricing

    oauth.login("anthropic-key", oauth.LoginCallbacks(on_secret=lambda _q: "k"))
    assert pricing.context_cap("claude-sonnet-5") == 1_000_000
    assert pricing.context_cap("claude-haiku-4-5-20251001") == 200_000
    assert pricing.context_cap("gpt-4o") == 128_000  # unrelated, untouched


def test_typed_window_overrides_every_model():
    """A value typed at the prompt is a deliberate choice, so it wins over the
    table for this credential's models — and only for those."""
    from financial_research_assistant import oauth, pricing

    def answer(question):
        return "777k" if "context window" in question else ""

    oauth.login("anthropic-key", oauth.LoginCallbacks(
        on_secret=lambda _q: "k", on_prompt=answer,
    ))
    assert auth.context_window() == 777_000
    assert auth.context_window("claude-opus-5") == 777_000
    assert pricing.context_cap("claude-sonnet-5") == 777_000
    assert pricing.context_cap("gpt-4o") == 128_000  # not this credential's model


def test_credential_wide_window_is_dead_config_once_models_set_their_own():
    """The field moved into the model object, so a leftover credential-wide value
    silently does nothing — the exact confusion of editing 1000000 there and still
    seeing 256k. /config calls this out; the resolution itself is per model."""
    from financial_research_assistant import pricing

    auth.set_credential("default", {
        "provider": "anthropic", "type": "oauth", "access": "t",
        "model_provider": "anthropic",
        "context_window": 1_000_000,
        "models": [{"name": "claude-haiku-4-5-20251001", "context_window": 256_000}],
    })
    assert pricing.context_cap("claude-haiku-4-5-20251001") == 256_000


def test_credential_wide_window_still_covers_models_without_one():
    """It is a fallback, not dead weight: entries that set nothing still use it,
    which is what keeps older stores working."""
    from financial_research_assistant import pricing

    auth.set_credential("default", {
        "provider": "gw", "type": "api_key", "key": "k",
        "context_window": 48_000,
        "models": [{"name": "gw/unknown-model"}],
    })
    assert pricing.context_cap("gw/unknown-model") == 48_000


def test_hand_edited_per_model_window_wins():
    """Editing one entry in auth.json is the way to correct a single model."""
    from financial_research_assistant import pricing

    auth.set_credential("default", {
        "provider": "anthropic-key", "type": "api_key", "key": "k",
        "model_provider": "anthropic",
        "models": [{"name": "claude-sonnet-5", "context_window": 250_000}],
    })
    assert pricing.context_cap("claude-sonnet-5") == 250_000


def test_context_cap_survives_a_broken_store(monkeypatch):
    from financial_research_assistant import pricing

    monkeypatch.setattr(
        "financial_research_assistant.auth.context_window",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert pricing.context_cap("claude-sonnet-5") == 1_000_000  # falls back, never raises


# --- redaction ----------------------------------------------------------------


def test_redact_masks_a_stored_literal_key():
    auth.set_credential("default", {"type": "api_key", "key": "sk-thisisthelivekey123456"})
    out = auth.redact("calling with sk-thisisthelivekey123456 now")
    assert "sk-thisisthelivekey123456" not in out
    assert "***" in out


def test_redact_masks_oauth_tokens():
    auth.set_credential(
        "default",
        {"type": "oauth", "access": "access-token-value-x", "refresh": "refresh-token-value-y"},
    )
    out = auth.redact("access-token-value-x / refresh-token-value-y")
    assert "access-token-value-x" not in out and "refresh-token-value-y" not in out


def test_redact_masks_keys_this_process_never_held():
    """A tool result or traceback can carry someone else's key."""
    for probe in (
        "sk-ant-oat01-AAAAAAAAAAAAAAAAAAAA",
        "sk-or-v1-abcdefghijklmnopqrst",
        "sk-abcdefghijklmnopqrstuvwx",
        "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345",
    ):
        assert probe not in auth.redact(f"key={probe} tail"), probe


def test_redact_masks_auth_headers():
    out = auth.redact("Authorization: Bearer abcdef123456789")
    assert "abcdef123456789" not in out
    assert "Authorization" in out  # the field name survives; the value doesn't


def test_redact_keeps_json_parseable():
    """The .jsonl export is redacted as serialized text, so masking must not eat
    the structural punctuation around a value."""
    auth.set_credential("default", {"type": "api_key", "key": "sk-or-v1-LIVEKEY0123456789"})
    payload = {
        "tool": "fetch",
        "args": {"headers": {"Authorization": "Bearer sk-or-v1-LIVEKEY0123456789"}},
        "answer": "AAPL closed at 231.40",
    }
    out = auth.redact(json.dumps(payload))
    parsed = json.loads(out)  # must still parse
    assert "sk-or-v1-LIVEKEY0123456789" not in out
    assert parsed["answer"] == "AAPL closed at 231.40"
    assert parsed["tool"] == "fetch"


def test_redact_masks_env_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "totally-secret-value")
    assert "totally-secret-value" not in auth.redact("x totally-secret-value y")


def test_redact_does_not_run_shell_indirection(tmp_path):
    """Redacting a log line must not execute a config file's command."""
    marker = tmp_path / "ran"
    auth.set_credential("default", {"type": "api_key", "key": f"!touch {marker}"})
    auth.redact("nothing to see")
    assert not marker.exists()


def test_redact_leaves_ordinary_text_alone():
    text = "AAPL closed at 231.40, up 1.2% — see the 10-K risk factors."
    assert auth.redact(text) == text
    assert auth.redact("") == ""


def test_redact_handles_overlapping_secrets():
    """Longest-first, so masking a short secret can't strand part of a longer one."""
    auth.set_credential("default", {"type": "api_key", "key": "abcdefgh"})
    auth.set_credential("quick", {"type": "api_key", "key": "abcdefghijklmnop"})
    assert "abcdefgh" not in auth.redact("token abcdefghijklmnop end")


def test_set_preserves_other_scopes():
    auth.set_credential("default", {"type": "api_key", "key": "a"})
    auth.set_credential("quick", {"type": "api_key", "key": "b"})
    auth.set_credential("default", {"type": "api_key", "key": "c"})
    assert auth.resolve_key("default") == "c"
    assert auth.resolve_key("quick") == "b"
