"""Suite-wide fixtures."""

import pytest

from financial_research_assistant import alerts


@pytest.fixture(autouse=True)
def _never_read_the_real_credential_store(tmp_path_factory, monkeypatch):
    """Point every test at a throwaway auth.json.

    Without this a test that builds an LLM picks up whatever the developer is
    actually signed in with — which is how a live OAuth token ended up printed in
    a pytest assertion diff. Suite-wide and autouse because the leak happens in
    tests that have nothing to do with auth (any `_make_llm` call reaches the
    store now), so per-file fixtures cannot be relied on to cover it.
    """
    store = tmp_path_factory.mktemp("auth") / "auth.json"
    monkeypatch.setenv("FINANCIAL_RESEARCH_AUTH_FILE", str(store))


@pytest.fixture(autouse=True)
def _never_write_to_the_real_portfolio(tmp_path_factory, monkeypatch):
    """Point every test at a throwaway statements DB and Flex directory.

    Most tests that store a statement already redirect the DB themselves, which
    is exactly why this was easy to miss: the one that didn't was a Flex test
    written when `flex_sync` only saved a file and touched no store. Teaching it
    to import turned that test into a writer, and a fixture account landed in the
    developer's real portfolio — alongside genuine imports, indistinguishable
    from them at a glance and silently feeding every position and gain figure.

    Suite-wide and autouse because the hazard follows the *code* changing under a
    test, not the test's own subject. Redirecting the Flex directory too keeps a
    fetch test from writing into the real statement archive.
    """
    monkeypatch.setenv(
        "FINANCIAL_RESEARCH_STATEMENTS_DB",
        str(tmp_path_factory.mktemp("statements") / "statements.db"),
    )
    monkeypatch.setenv(
        "FINANCIAL_RESEARCH_FLEX_DIR",
        str(tmp_path_factory.mktemp("flex")),
    )
    # Reading the portfolio now pulls a fresh Flex statement when the one on
    # file is old. With a developer's real token in the environment, that would
    # fetch the real account during a test, so no test starts with one.
    monkeypatch.delenv("IBKR_FLEX_TOKEN", raising=False)
    monkeypatch.delenv("IBKR_FLEX_QUERY_ID", raising=False)
    monkeypatch.setenv(
        "FRA_FLEX_REFRESH_FILE",
        str(tmp_path_factory.mktemp("flex-refresh") / "flex-refresh.json"),
    )


@pytest.fixture(autouse=True)
def _never_actually_play_audio(monkeypatch):
    """Stop any test from making noise or raising a desktop notification.

    Alert delivery ends in real subprocesses — an audio player and an OS
    notifier — so a test that fires a rule would otherwise play a sound and pop
    a banner on whoever's running the suite, and on CI spawn processes that
    aren't going anywhere. Stubbing the one spawn point keeps the surrounding
    logic (enabled? which file? which command?) under test while the only thing
    lost is the subprocess itself. Tests that care assert on the recorded
    commands instead.
    """
    spawned: list[list[str]] = []
    monkeypatch.setattr(alerts, "_spawn", spawned.append)
    return spawned


@pytest.fixture(autouse=True)
def _never_ask_a_real_provider_for_its_models(monkeypatch):
    """Every login now asks the provider for its model list, so a login test
    would otherwise call the vendor with a fake key. Offline, the login falls back
    to its built-in defaults, as it does on a real network failure; a test that
    wants a list patches ``provider_models._get`` itself."""
    from financial_research_assistant import provider_models

    def offline(url, headers, timeout=15.0):
        raise OSError(f"no network in tests: {url}")

    monkeypatch.setattr(provider_models, "_get", offline)


@pytest.fixture(autouse=True)
def _never_touch_the_real_task_store(tmp_path_factory, monkeypatch):
    """Point every test at a throwaway tasks.json and journal.json, and unset the
    delivery channels.

    Three hazards, one fixture. A test that schedules something would otherwise
    queue work in the developer's real store — which a later `--run-due` would
    faithfully execute, spending real tokens on a test fixture's prompt. A test
    that runs the scheduler would deliver to whatever channel the developer has
    configured, i.e. send a Telegram message from the suite. And the journal is
    read by the capability probe on EVERY graph build, so without redirecting it
    the developer's real recorded calls would decide which tools the suite sees
    bound — making the toolset tests pass or fail on the contents of a file
    outside the repo.
    """
    monkeypatch.setenv(
        "FINANCIAL_RESEARCH_TASKS_FILE",
        str(tmp_path_factory.mktemp("tasks") / "tasks.json"),
    )
    # Alert rules too: a watcher test added a rule and it landed in the
    # developer's real alerts.json, where the live service would have fired it.
    monkeypatch.setenv(
        "FINANCIAL_RESEARCH_ALERTS_FILE",
        str(tmp_path_factory.mktemp("alerts") / "alerts.json"),
    )
    monkeypatch.setenv(
        "FINANCIAL_RESEARCH_JOURNAL_FILE",
        str(tmp_path_factory.mktemp("journal") / "journal.json"),
    )
    # Every store resolved through `storage.state_dir()` (report ledger, event and
    # run logs, the investor profile, the autonomy switch) lands here too.
    monkeypatch.setenv(
        "FINANCIAL_RESEARCH_HOME", str(tmp_path_factory.mktemp("state"))
    )
    # Transcripts too: a scheduled run logs one, and without this it appended to
    # the developer's real `task-s1.jsonl`.
    monkeypatch.setenv(
        "FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path_factory.mktemp("sessions"))
    )
    monkeypatch.setenv(
        "FINANCIAL_RESEARCH_TELEGRAM_STATE",
        str(tmp_path_factory.mktemp("tg") / "offset.json"),
    )
    for var in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "TELEGRAM_ALLOWED_CHAT_IDS"):
        monkeypatch.delenv(var, raising=False)
    # An unrecognised key selects no channel, so the default for the suite is
    # "deliver nowhere". Tests that exercise delivery set this themselves.
    monkeypatch.setenv("NOTIFY_CHANNELS", "none")


@pytest.fixture(autouse=True)
def _never_touch_the_real_home(tmp_path_factory, monkeypatch):
    """Redirect every remaining store that defaults under the real home.

    The fixtures above cover the stores a test was once caught writing to; the
    rest had simply never been caught. Rendered reports, ingested documents,
    exports, the pricing table, the eval dir and the prompt addendum all default
    under ``~/.financial-research-assistant`` — a full run left ~150 test PDFs in
    the developer's real reports folder, and a watcher test put an alert rule into
    their real alerts.json, where the live service would have fired it.

    Not by moving HOME itself: headless Chrome keeps its profile under HOME and
    stalls on a fresh one, turning every rendering test into a two-minute timeout.
    """
    root = tmp_path_factory.mktemp("home")
    for var, name in (
        ("FINANCIAL_RESEARCH_REPORTS_DIR", "reports"),
        ("FINANCIAL_RESEARCH_DOCS_DIR", "documents"),
        ("FINANCIAL_RESEARCH_EXPORT_DIR", "exports"),
        ("FINANCIAL_RESEARCH_PRICING_FILE", "pricing.json"),
        ("FINANCIAL_RESEARCH_EVAL_DIR", "eval"),
        ("FINANCIAL_RESEARCH_PROMPT_ADDENDUM_FILE", "prompt_addendum.txt"),
    ):
        monkeypatch.setenv(var, str(root / name))
