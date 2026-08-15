"""Model selection, credential resolution, and chat-model construction.

Split out of ``graph.py`` to break an import cycle. ``research``, ``subagents``
and ``reflection`` each need one thing from the graph module — a chat model — but
importing ``graph`` for it pulled in the whole agent, and ``graph`` imports the
tool catalog, which imports those same modules for their ``*_TOOLS`` lists.
Everything that answers "which model, on whose credential, built how" lives here
instead, so the LLM factory is a leaf the agent and its subagents both sit on.

Precedence for every knob is the same and is applied in ``_credentials``:
explicit argument, then the stored credential for the tier, then the environment.
Provider and key resolve as a PAIR, so a stored credential can never be paired
with an environment key belonging to a different provider.
"""

from __future__ import annotations

from typing import Any
import os
import re


#: Provider → (extra name, the module its integration imports as). Only the
#: providers this project declares an extra for; anything else falls back to the
#: generic half of the message, which is still better than a bare import error.
_PROVIDER_EXTRAS = {
    "anthropic": ("anthropic", "langchain_anthropic"),
    "google_genai": ("google", "langchain_google_genai"),
    "google-genai": ("google", "langchain_google_genai"),
    "groq": ("groq", "langchain_groq"),
}


class ProviderSupportError(RuntimeError):
    """A provider was selected whose LangChain integration is not installed.

    Distinct from a credential or config error so the message can carry an
    install command. Every non-OpenAI provider is an OPTIONAL extra, so this is
    the first thing a working configuration hits on a fresh checkout — and the
    bare ``ModuleNotFoundError: No module named 'langchain_anthropic'`` it
    replaced named the module but not the remedy, on an agent that had otherwise
    started fine.
    """


def _provider_support_error(
    provider: str, exc: ImportError
) -> ProviderSupportError | None:
    """An actionable error for a missing provider package, or None to re-raise.

    Catches ImportError rather than ModuleNotFoundError because the two
    construction paths fail differently, and only one of them was ever reported.
    ``init_chat_model`` catches the import itself and re-raises its own
    ImportError ("requires the langchain-anthropic package"); the OAuth subclass
    imports ``langchain_anthropic`` directly and lets the bare
    ModuleNotFoundError out. An OAuth user therefore got the message with no
    remedy in it, which is the one that was actually hit.

    Only converts when the missing module IS an integration package. An
    ImportError from somewhere INSIDE an installed provider is a real bug, and
    answering it with install advice would send the reader after the wrong thing.
    """
    extra, module = _PROVIDER_EXTRAS.get(provider, ("", ""))
    # `name` is set on ModuleNotFoundError but not on the ImportError
    # `init_chat_model` constructs, so the message is the fallback.
    missing = (getattr(exc, "name", "") or "").split(".")[0]
    if not missing:
        found = re.search(r"\blangchain[-_][a-z0-9_-]+", str(exc))
        missing = found.group(0).replace("-", "_") if found else ""
    if not missing or (missing != module and not missing.startswith("langchain")):
        return None
    dist = missing.replace("_", "-")
    how = (
        f"  uv pip install {dist}\n"
        f"  pip install 'financial-research-assistant[{extra}]'"
        if extra
        else f"  uv pip install {dist}\n  pip install {dist}"
    )
    return ProviderSupportError(
        f"The {provider!r} provider needs the optional {dist!r} package, which is "
        "not installed. Every non-OpenAI provider is an optional extra.\n"
        f"Install it with one of:\n{how}\n"
        # `uv sync --extra X` is deliberately NOT suggested: it syncs to exactly
        # the extras named, so it removes any others already installed — running
        # it without `--extra dev` here uninstalled pytest.
        "(`uv sync --extra …` syncs to exactly the extras you name and removes "
        "the rest, so pass every extra you already have, or use the commands above.)"
    )


def _anthropic_oauth_llm(model: str, token: str, base_url: str | None):
    """ChatAnthropic authenticating with a Bearer token instead of an x-api-key.

    An OAuth token is a bearer credential. Passing it as ``api_key`` sends it as
    ``x-api-key`` and the API answers ``401 invalid x-api-key`` — and setting the
    header by hand doesn't help either, because the SDK emits ``X-Api-Key``
    whenever ``api_key`` is anything other than ``None`` (an empty string still
    emits it), so both headers go out and the wrong one is validated.

    ``ChatAnthropic`` hardcodes ``api_key`` into its client params and types
    ``default_headers`` as ``dict[str, str]``, so the SDK's ``Omit`` sentinel
    can't get through pydantic either. Overriding the params is the one seam that
    reaches ``anthropic.Client(auth_token=…)``, which is the SDK's supported way
    to send a bearer credential. Verified against langchain-anthropic 1.5.3 /
    anthropic 0.120.2: exactly one auth header goes out, even with
    ``ANTHROPIC_API_KEY`` set in the environment.
    """
    from functools import cached_property

    from langchain_anthropic import ChatAnthropic

    class _OAuthChatAnthropic(ChatAnthropic):
        @cached_property
        def _client_params(self) -> dict[str, Any]:
            params = dict(ChatAnthropic._client_params.func(self))
            params["api_key"] = None  # None specifically: "" still emits the header
            params["auth_token"] = token
            return params

    kwargs: dict[str, Any] = {
        "model": model,
        # Never sent; the client requires the field to be populated.
        "api_key": "unused-oauth-placeholder",
        # Consumer OAuth tokens are only accepted with this beta opt-in.
        "default_headers": {"anthropic-beta": "oauth-2025-04-20"},
    }
    if base_url:
        kwargs["base_url"] = base_url
    return _OAuthChatAnthropic(**kwargs)


def _is_oauth(scope: str) -> bool:
    from . import auth

    cred = auth.get(auth.effective_scope(scope) or scope) or {}
    return cred.get("type") == "oauth"


def credential_provider(scope: str = "default") -> str | None:
    """The LangChain integration a stored credential requires, else ``MODEL_PROVIDER``.

    The credential wins for the same reason it wins for the key: routing and
    credential must come from ONE source. A Claude token built into a ChatOpenAI
    client — because ``MODEL_PROVIDER`` still said openai — POSTs
    ``/v1/chat/completions`` at an API with no such route, which surfaces as an
    opaque transport error rather than anything pointing at the mismatch.
    """
    from . import auth

    stored, _model = auth.routing(scope)
    return stored or os.environ.get("MODEL_PROVIDER") or None


def credential_model(scope: str = "default") -> str | None:
    """The model a stored credential is valid for, else ``OPENAI_MODEL``.

    Ordered this way because the model name has to match the provider: keeping a
    leftover ``OPENAI_MODEL`` after signing in to Anthropic sends a model name
    Anthropic has never heard of. An explicit ``/models`` override still wins over
    both — that is the user choosing, live.
    """
    from . import auth

    _provider, stored = auth.routing(scope)
    return stored or os.environ.get("OPENAI_MODEL") or None


def _credentials(
    scope: str, provider: str, base_url: str | None, api_key: str | None
) -> tuple[str | None, str | None]:
    """Resolve the endpoint and key as a PAIR: explicit arg → auth store → env.

    Two rules, both of them load-bearing:

    *Pair them.* Whichever source supplies the key also supplies the endpoint it
    was issued for. Mixing sources is the failure this exists to prevent — a key
    from ``/login`` combined with a leftover ``OPENAI_API_BASE`` sends a gateway
    token to a different vendor's API.

    *Store outranks the environment.* ``main.py`` calls ``load_dotenv()``, so a
    stale key in ``.env`` is a real environment variable by the time we get here.
    Ranked the other way it would silently shadow a fresh ``/login``, which fails
    as "the command did nothing" with no error to go on.

    The store is provider-neutral — it holds whatever credential that tier was
    logged in with. The env fallback is not: ``OPENAI_API_*`` only applies to the
    OpenAI-compatible path. For any other provider we return no key at all so
    ``init_chat_model`` goes on reading ``ANTHROPIC_API_KEY`` / ``GOOGLE_API_KEY``
    / … itself, exactly as before.
    """
    from . import auth, oauth

    if api_key:  # an explicit override is already a complete, deliberate pair
        return base_url, api_key
    # Renew a near-expired token before the client is built. adapter.py rebuilds
    # the real graph per turn, so this runs once a turn and costs a timestamp
    # comparison unless a refresh is genuinely due.
    oauth.ensure_fresh(scope)
    stored = auth.resolve_key(scope)
    if stored:
        cred_scope = auth.effective_scope(scope)
        cred = auth.get(cred_scope) if cred_scope else None
        if cred and cred.get("provider") == "codex":
            # A Codex token addresses the ChatGPT backend, which speaks the
            # Responses API and checks the caller looks like the Codex CLI — not
            # something ChatOpenAI can talk to. Route through the local shim, which
            # re-reads the credential per request so refreshes are picked up.
            from . import codex_proxy

            # The key handed to the client is the proxy's own per-process token,
            # never the Codex one: the proxy reads that from the store itself on
            # every request, so the real token would be a spendable secret sent to
            # a local socket that has no use for it. Evaluated in order — the token
            # only exists once the server has been started.
            return codex_proxy.ensure_running(), codex_proxy.local_token()
        pinned = auth.base_url(scope)
        if pinned:
            return base_url or pinned, stored
        # The credential pins no endpoint (a plain vendor key). On the
        # OpenAI-compatible lane the ambient base URL is still the right one, so
        # storing a key must not quietly drop a configured gateway. Other
        # providers get None and use their SDK's own default.
        if provider in ("", "openai"):
            return base_url or os.environ.get("OPENAI_API_BASE") or None, stored
        return base_url, stored
    if provider in ("", "openai"):
        return (
            base_url or os.environ.get("OPENAI_API_BASE") or None,
            os.environ.get("OPENAI_API_KEY") or None,
        )
    return base_url, None


# Where each LangChain integration sends requests when nothing pins an endpoint.
# DISPLAY ONLY — the SDKs hold their own copies and are what actually route; this
# exists so the header can name the endpoint a client would be built against
# instead of asserting OpenAI's for every provider.
_PROVIDER_ENDPOINT = {
    "": "https://api.openai.com/v1",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "google_genai": "https://generativelanguage.googleapis.com",
    "groq": "https://api.groq.com",
}


def endpoint(scope: str = "default") -> str:
    """The endpoint ``_make_llm`` would build ``scope``'s client against.

    Resolved by the same rules as ``_credentials`` — credential first, then the
    environment — because the header describing the client has to agree with the
    client. Reading ``OPENAI_API_BASE`` on its own instead reports ``api.openai.com``
    for an Anthropic key, i.e. the one endpoint the request definitely does not go
    to. Side-effect free: no refresh, no proxy start, since this runs on render.

    A provider that pins nothing falls back to its SDK's own default, named from
    ``_PROVIDER_ENDPOINT`` (or described generically for an integration not listed
    there — a guess would be worse than saying we don't know).
    """
    from . import auth

    provider = (credential_provider(scope) or "openai").strip().lower()
    if auth.resolve_key(scope):
        # Only when a key is actually resolvable: a credential whose `$VAR` is
        # unset contributes no key, and `_credentials` falls through to the env
        # for both halves of the pair — so the endpoint must fall through too.
        pinned = auth.base_url(scope)
        if pinned:
            return pinned
    if provider in ("", "openai"):
        return os.environ.get("OPENAI_API_BASE") or _PROVIDER_ENDPOINT["openai"]
    return _PROVIDER_ENDPOINT.get(provider) or f"{provider} default"


def _make_llm(
    model: str | None = None,
    *,
    provider: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    scope: str = "default",
):
    """Construct the chat model from env, shared by the ReAct agent, the standalone
    summarizer (compaction), and subagents.

    ``model`` / ``provider`` / ``base_url`` / ``api_key`` are optional overrides;
    each falls back to its env var (``OPENAI_MODEL`` / ``MODEL_PROVIDER`` /
    ``OPENAI_API_BASE`` / ``OPENAI_API_KEY``) when None. With nothing stored by
    ``/login``, passing none reproduces the original env-only behavior exactly.
    Subagents use these overrides to point at a SEPARATE OpenAI-compatible
    endpoint/key/model (``SUBAGENT_*``) while still going through this one builder
    — so they get the same OpenAI-compatible support the primary agent has.

    ``scope`` names the credential tier to read from the ``auth`` store
    (``"default"`` / ``"quick"`` / ``"subagent"``), so ``/login`` can give the
    cheap tier its own key; a scope with nothing stored inherits ``"default"``.
    See ``_credentials`` for the precedence rules.

    Default (``MODEL_PROVIDER`` unset or ``openai``): an OpenAI-compatible
    ``ChatOpenAI`` — cloud OpenAI, or any local server via ``OPENAI_API_BASE``
    (llama.cpp / Ollama / LM Studio). This path is unchanged, so existing setups
    behave identically.

    Other providers (``MODEL_PROVIDER=anthropic`` | ``google_genai`` | ``groq`` |
    …): built through LangChain's ``init_chat_model``, which reads that provider's
    own key env var (``ANTHROPIC_API_KEY``, ``GOOGLE_API_KEY``, …) unless a
    ``base_url``/``api_key`` override is given — those are forwarded here too, so
    the tier overrides work against an Anthropic-compatible gateway exactly as
    they do against an OpenAI-compatible one. The provider's integration package
    is an OPTIONAL extra and must be installed (e.g. ``pip install
    '.[anthropic]'``); a missing one raises `ProviderSupportError` naming the
    package and the command that installs it, which surfaces as an error event.
    That is the first thing a working configuration hits on a fresh checkout, and
    it used to surface as a bare ``No module named 'langchain_anthropic'``."""
    provider = (provider or credential_provider(scope) or "openai").strip().lower()
    model = model or credential_model(scope) or _default_model(provider)
    base_url, api_key = _credentials(scope, provider, base_url, api_key)
    if provider in ("", "openai"):
        from langchain_openai import ChatOpenAI
        from pydantic import SecretStr

        # A local server (llama.cpp / Ollama / LM Studio) needs no real key, but
        # the client insists on one being present.
        api_key = api_key or ("dummy" if base_url else None)
        return ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=SecretStr(api_key) if api_key else None,
        )
    # Every provider below is an optional extra, so a missing package is the
    # expected first failure rather than an exceptional one. One wrapper covers
    # both construction paths — the OAuth subclass imports `langchain_anthropic`
    # directly, and `init_chat_model` imports its provider package internally.
    try:
        return _build_provider_llm(provider, model, base_url, api_key, scope)
    except ImportError as e:
        raise (_provider_support_error(provider, e) or e) from e


def _build_provider_llm(
    provider: str, model: str, base_url: str | None, api_key: str | None, scope: str
):
    """Construct a non-OpenAI provider's chat model. See `_make_llm`."""
    from langchain.chat_models import init_chat_model

    # Forward the endpoint overrides on this path too. Without them `SUBAGENT_*`
    # and `QUICK_*` silently did nothing for every non-OpenAI provider — the
    # values were accepted as parameters and dropped — so a subagent pointed at
    # its own Anthropic-compatible gateway quietly used the primary agent's.
    # init_chat_model forwards **kwargs to the provider class, but not every
    # integration names them the same way, so a provider that rejects one falls
    # back to the env-only construction rather than raising mid-turn.
    kwargs = {}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    if provider == "anthropic" and _is_oauth(scope) and api_key:
        return _anthropic_oauth_llm(model, api_key, base_url)
    if not kwargs:
        return init_chat_model(model, model_provider=provider)
    try:
        return init_chat_model(model, model_provider=provider, **kwargs)
    except TypeError:
        return init_chat_model(model, model_provider=provider)


def _quick_overrides() -> dict[str, Any]:
    """Endpoint overrides for the 'quick' model tier (``QUICK_API_BASE`` /
    ``QUICK_API_KEY`` / ``QUICK_MODEL_PROVIDER``); each unset value is None so
    ``_make_llm`` falls back to the primary agent's config — exactly like the
    ``SUBAGENT_*`` overrides."""
    return {
        "provider": os.environ.get("QUICK_MODEL_PROVIDER") or None,
        "base_url": os.environ.get("QUICK_API_BASE") or None,
        "api_key": os.environ.get("QUICK_API_KEY") or None,
    }


def quick_llm(model: str | None = None):
    """Build the 'quick'/cheap model tier for summarization & extraction tasks
    (context compaction, and any other high-volume, low-reasoning call) — the
    deep-vs-quick split trading firms use to cut cost. ``QUICK_MODEL`` (+ optional
    ``QUICK_API_BASE`` / ``QUICK_API_KEY`` / ``QUICK_MODEL_PROVIDER``) selects it and
    takes precedence for these tasks; when ``QUICK_MODEL`` is unset it falls back to
    ``model`` and then the primary agent's config, so unset = identical to today
    (the primary model does the summarizing, no behavior change)."""
    return _make_llm(
        os.environ.get("QUICK_MODEL") or model or None,
        scope="quick",
        **_quick_overrides(),
    )


# Sensible default model per provider when neither the caller nor OPENAI_MODEL
# names one, so `MODEL_PROVIDER=anthropic` alone works without also setting a model.
#
# The Anthropic default stays on the balanced Sonnet tier (the tier this agent
# was already pointed at) rather than jumping to Opus: this is a research
# assistant doing multi-step tool calls, and Sonnet 5 reaches near-Opus quality
# on agentic work at a fifth of the Opus input price. Set OPENAI_MODEL to
# `claude-opus-5` for the hardest analysis.
_PROVIDER_DEFAULT_MODEL = {
    "anthropic": "claude-sonnet-5",
    "google_genai": "gemini-2.5-flash",
    "groq": "llama-3.3-70b-versatile",
}


def _default_model(provider: str) -> str:
    return _PROVIDER_DEFAULT_MODEL.get(provider, "gpt-4.1-mini")


def configured_models() -> list[tuple[str, str, str]]:
    """Every model this install can reach, as ``(model, label, provider)``.

    ``provider`` is the stored credential the model belongs to, or ``""`` for one
    configured through the environment. It is part of the entry because a model is
    only usable with the credential from its own lane: selecting one has to switch
    the active provider too, or the request goes out with this provider's key and
    that provider's model name — which is how picking an OpenAI-gateway model while
    signed in to Anthropic produced ``404 model: <name>``.

    Environment-configured models are listed only when they belong to the lane
    that would be used with no credential, since there is no stored key to pair
    them with otherwise.
    """
    from . import auth

    def lane(*env_vars: str) -> str:
        for var in env_vars:
            value = os.environ.get(var)
            if value:
                return value.strip().lower()
        return "openai"

    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()

    def add(model: str, label: str, provider: str = "") -> None:
        if model and model not in seen:
            seen.add(model)
            out.append((model, label, provider))

    # One entry per stored credential, most useful first: each is selectable and
    # switches the active provider.
    current = auth.active()
    stored = auth.credentials()  # read once: this runs per /models keystroke
    for name in sorted(n for n in stored if not n.startswith("_scope:")):
        cred = stored.get(name) or {}
        provider_id, models = _routing_of(cred)
        lane_default = _default_model(provider_id or "openai")
        mark = "active" if name == current else "configured"
        for model in models or [lane_default]:
            add(model, f"{mark} · {name}", name)
        # The lane's default too, so a model outside the configured list is still
        # one selection away rather than requiring the free-text path.
        add(lane_default, f"{provider_id} default · {name}", name)

    env_lane = (os.environ.get("MODEL_PROVIDER") or "openai").strip().lower()
    if not current:  # nothing stored for this tier: the env lane is what runs
        add(os.environ.get("OPENAI_MODEL") or "", "OPENAI_MODEL")
        if lane("QUICK_MODEL_PROVIDER", "MODEL_PROVIDER") == env_lane:
            add(os.environ.get("QUICK_MODEL") or "", "quick tier")
        if lane("SUBAGENT_MODEL_PROVIDER", "MODEL_PROVIDER") == env_lane:
            add(os.environ.get("SUBAGENT_MODEL") or "", "subagent tier")
        add(_default_model(env_lane), f"{env_lane} default")
    return out


def _routing_of(cred: dict[str, Any]) -> tuple[str | None, list[str]]:
    """``(model_provider, models)`` for a stored credential — a LIST since one
    credential serves several models."""
    from . import oauth

    return oauth.routing_for(cred)


def active_lane_note() -> str:
    """What the picker's title should say, or "" when nothing is signed in."""
    from . import auth

    current = auth.active()
    if not current:
        return ""
    return f"signed in: {current} — selecting another switches provider"


def resolved_model(model: str | None = None) -> str:
    """The model name ``_make_llm`` would actually build, for pricing, context-window,
    and token-budget math — so a non-OpenAI provider's default is reflected rather
    than a hardcoded gpt-4.1-mini."""
    if model:
        return model
    stored = credential_model()
    if stored:
        return stored
    provider = (credential_provider() or "openai").strip().lower()
    return _default_model(provider)
