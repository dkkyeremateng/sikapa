"""Delivery-channel registry: selection, overrides, and best-effort sending.

Offline — every channel is replaced with a recorder, so nothing here sends a real
message.
"""

import pytest

from financial_research_assistant import channels


@pytest.fixture
def registry(monkeypatch):
    """A registry holding two recording channels plus one that always fails."""
    sent: dict[str, list[str]] = {"alpha": [], "beta": []}
    monkeypatch.setattr(channels, "CHANNEL_REGISTRY", {}, raising=False)
    channels.register_channel(channels.Channel(
        "alpha", lambda: True, lambda t: bool(sent["alpha"].append(t) or True), "Alpha",
    ))
    channels.register_channel(channels.Channel(
        "beta", lambda: True, lambda t: bool(sent["beta"].append(t) or True), "Beta",
    ))
    channels.register_channel(channels.Channel(
        "offline", lambda: False, lambda t: True, "Offline",
    ))
    monkeypatch.delenv("NOTIFY_CHANNELS", raising=False)
    return sent


def test_unconfigured_channels_are_never_selected(registry):
    """A channel is opt-in by the presence of its env: one with no token must not
    be attempted, and must not be reported as a delivery target."""
    keys = [c.key for c in channels.active_channels()]
    assert keys == ["alpha", "beta"]
    assert "Offline" not in channels.describe_targets()


def test_notify_channels_pins_a_subset(registry, monkeypatch):
    monkeypatch.setenv("NOTIFY_CHANNELS", "beta")
    delivered, failed = channels.deliver("hello")
    assert delivered == ["beta"] and not failed
    assert registry["alpha"] == [] and registry["beta"] == ["hello"]


def test_a_task_channel_overrides_the_global_selection(registry, monkeypatch):
    """Naming a channel on the task IS the selection — one job can go to your phone
    while everything else stays local."""
    monkeypatch.setenv("NOTIFY_CHANNELS", "alpha")
    delivered, _failed = channels.deliver("just this one", prefer="beta")
    assert delivered == ["beta"]
    assert registry["alpha"] == []


def test_an_unknown_or_unconfigured_preference_delivers_nowhere_and_says_so(registry):
    delivered, failed = channels.deliver("x", prefer="offline")
    assert not delivered and not failed
    assert "not configured" in channels.describe_targets("offline")


def test_a_throwing_channel_is_recorded_as_failed_not_raised(monkeypatch, registry):
    """Completed work must never be lost to a notification API being down — the
    caller decides what to do, and `deliver` never raises."""
    def boom(_text):
        raise RuntimeError("network down")

    channels.register_channel(channels.Channel("bad", lambda: True, boom, "Bad"))
    delivered, failed = channels.deliver("hello")
    assert "bad" in failed
    assert set(delivered) == {"alpha", "beta"}, "one bad channel must not stop the others"


def test_a_channel_reporting_failure_is_not_counted_as_delivered(registry):
    channels.register_channel(channels.Channel("mute", lambda: True, lambda t: False, "Mute"))
    delivered, failed = channels.deliver("hello")
    assert "mute" in failed and "mute" not in delivered


def test_empty_text_is_not_sent(registry):
    assert channels.deliver("   ") == ([], [])


# --- who actually carried the answer --------------------------------------------


def test_a_banner_channel_does_not_count_as_carrying_the_answer(registry):
    """A truncating channel is a courtesy, not delivery: it showed the user that
    something happened, not what it said. The caller has to be able to tell the
    difference without knowing any channel by name."""
    channels.register_channel(channels.Channel(
        "banner", lambda: True, lambda t: True, "Banner", full_content=False,
    ))
    assert channels.carried_full_text(["banner"]) is False
    assert channels.carried_full_text(["banner", "alpha"]) is True
    assert channels.carried_full_text([]) is False


def test_channels_carry_the_whole_message_unless_they_say_otherwise(registry):
    """The default has to be the safe one — a new channel that forgot to declare
    itself must not silently start parking every answer for redelivery."""
    assert channels.carried_full_text(["alpha"]) is True
    assert channels.carried_full_text(["gone-from-the-registry"]) is True


def test_the_desktop_channel_declares_itself_a_banner():
    """It is enabled by default and returns True the moment the notifier process is
    spawned — before, and regardless of, anything appearing on screen. So a failed
    Telegram send plus a "successful" toast used to count as a delivered analysis,
    and the answer was never parked for redelivery."""
    assert channels.CHANNEL_REGISTRY["desktop"].full_content is False
    assert channels.CHANNEL_REGISTRY["telegram"].full_content is True


def test_describe_targets_is_honest_when_nothing_is_configured(monkeypatch):
    """A user told "it will be sent to you" who configured no channel would go
    looking for a message that never arrives."""
    monkeypatch.setattr(channels, "CHANNEL_REGISTRY", {}, raising=False)
    text = channels.describe_targets()
    assert "TELEGRAM_BOT_TOKEN" in text and "task log" in text


# --- the shipped channels ------------------------------------------------------


def test_stdout_is_never_implicitly_active(monkeypatch):
    """Left in the default set it would 'deliver' every scheduled answer into a log
    nobody reads — and report success for it."""
    monkeypatch.delenv("NOTIFY_CHANNELS", raising=False)
    assert "stdout" not in [c.key for c in channels.active_channels()]
    monkeypatch.setenv("NOTIFY_CHANNELS", "stdout")
    assert [c.key for c in channels.active_channels()] == ["stdout"]


def test_telegram_channel_tracks_its_env(monkeypatch):
    monkeypatch.delenv("NOTIFY_CHANNELS", raising=False)
    assert "telegram" not in [c.key for c in channels.active_channels()]
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    assert "telegram" in [c.key for c in channels.active_channels()]


# --- what the model is told -----------------------------------------------------


def test_the_prompt_states_a_configured_channel_instead_of_offering_setup(monkeypatch):
    """The reported bug: a user with TELEGRAM_BOT_TOKEN set was told to go configure
    TELEGRAM_TOKEN — a variable this project does not have. The model had no view of
    delivery at all, so it answered from general knowledge."""
    from financial_research_assistant import graph

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    monkeypatch.setenv("NOTIFY_CHANNELS", "telegram")
    text = graph._delivery_guidance()
    assert "Telegram" in text and "ALREADY CONFIGURED" in text
    assert "TELEGRAM_BOT_TOKEN" not in text, "configured means no setup instructions"


def test_the_prompt_names_the_real_variables_when_nothing_is_configured(monkeypatch):
    from financial_research_assistant import graph

    monkeypatch.setenv("NOTIFY_CHANNELS", "none")
    text = graph._delivery_guidance()
    assert "TELEGRAM_BOT_TOKEN" in text and "TELEGRAM_CHAT_ID" in text
    assert "--notify-test" in text
