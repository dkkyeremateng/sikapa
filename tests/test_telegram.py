"""Telegram Bot API client: chunking, the inbound allowlist, and token hygiene.

Offline — ``_call`` is replaced with a recorder in every test that would otherwise
reach the network.
"""

import json
import urllib.error

import pytest

from financial_research_assistant import telegram


@pytest.fixture
def bot(monkeypatch):
    """A configured bot whose API calls are recorded instead of sent."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "12345:SECRET-TOKEN-VALUE")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "999")
    calls: list[tuple[str, dict]] = []

    def fake_call(method, payload, timeout=30.0):
        calls.append((method, payload))
        return {"ok": True, "result": []}

    monkeypatch.setattr(telegram, "_call", fake_call)
    return calls


def test_not_configured_sends_nothing(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    assert telegram.configured() is False
    assert telegram.send_message("hello") is False


def test_send_goes_to_the_default_chat(bot):
    assert telegram.send_message("scheduled answer") is True
    method, payload = bot[0]
    assert method == "sendMessage"
    assert payload["chat_id"] == "999" and payload["text"] == "scheduled answer"


def test_a_long_answer_is_split_not_truncated(bot):
    """Telegram caps a message at 4096 chars and an analysis routinely runs past
    it. Truncating would cut off the end — which is where the recommendation is."""
    body = "\n".join(f"line {i} " + "x" * 80 for i in range(200))
    assert len(body) > telegram.MAX_MESSAGE
    telegram.send_message(body)
    assert len(bot) > 1
    assert all(len(p["text"]) <= telegram.MAX_MESSAGE for _m, p in bot)
    rejoined = "".join(p["text"] for _m, p in bot)
    assert "line 199" in rejoined, "the tail of the answer must survive"


def test_splitting_prefers_line_boundaries(bot):
    body = ("A" * 4000) + "\n" + ("B" * 500)
    telegram.send_message(body)
    assert bot[0][1]["text"].endswith("A")
    assert bot[1][1]["text"].startswith("B")


def test_messages_are_sent_as_plain_text(bot):
    """Not Markdown: an analysis is full of *, _ and $ from tickers and figures,
    and Telegram rejects the whole message when they don't parse as entities."""
    telegram.send_message("NVDA *beat* on EPS_1 ($0.85 vs $0.80)")
    _method, payload = bot[0]
    assert "parse_mode" not in payload


# --- inbound: the allowlist ----------------------------------------------------


def test_inbound_is_off_without_an_allowlist(monkeypatch):
    """Anyone can message a bot whose username they guess, so polling without an
    allowlist would hand a stranger an agent that reads your portfolio."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "12345:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "999")
    monkeypatch.delenv("TELEGRAM_ALLOWED_CHAT_IDS", raising=False)
    assert telegram.inbound_enabled() is False
    assert telegram.get_updates() == []


def test_outbound_config_does_not_silently_enable_inbound(monkeypatch):
    """Defaulting the allowlist to TELEGRAM_CHAT_ID would turn a notification
    channel into a remote-control channel for everyone who set one up."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "12345:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "999")
    assert telegram.configured() is True
    assert telegram.inbound_enabled() is False


def _updates(*chat_ids):
    return {"ok": True, "result": [
        {"update_id": 100 + i,
         "message": {"chat": {"id": c}, "text": f"hi from {c}", "from": {"username": "u"}}}
        for i, c in enumerate(chat_ids)
    ]}


def test_messages_from_other_chats_are_dropped(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "12345:abc")
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", "999")
    monkeypatch.setattr(telegram, "_call", lambda *a, **k: _updates(999, 777))
    got = telegram.get_updates()
    assert [m["chat_id"] for m in got] == ["999"]


def test_the_cursor_advances_past_dropped_messages(monkeypatch):
    """Otherwise a stranger's message is re-fetched forever and the queue never
    moves past it, blocking every later message from the owner."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "12345:abc")
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", "999")
    monkeypatch.setattr(telegram, "_call", lambda *a, **k: _updates(777))
    assert telegram.get_updates() == []
    assert telegram._read_offset() == 101


def test_the_cursor_persists_so_a_restart_does_not_re_answer(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "12345:abc")
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", "999")
    seen: list[dict] = []

    def fake_call(method, payload, timeout=30.0):
        seen.append(payload)
        return _updates(999)

    monkeypatch.setattr(telegram, "_call", fake_call)
    telegram.get_updates()
    telegram.get_updates()
    assert seen[1]["offset"] == 101, "the second poll must resume after the first"


# --- token hygiene -------------------------------------------------------------


def test_errors_never_echo_the_bot_token(monkeypatch):
    """The token is a bearer credential sitting in the URL path, and urllib puts
    the full URL in its exception text — an unmasked error would print a working
    credential into a cron log and into the task's stored result."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "12345:SECRET-TOKEN-VALUE")

    def boom(*_a, **_k):
        raise urllib.error.URLError(
            "failed to reach https://api.telegram.org/bot12345:SECRET-TOKEN-VALUE/sendMessage"
        )

    monkeypatch.setattr(telegram.urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError) as err:
        telegram._call("sendMessage", {"chat_id": "1", "text": "x"})
    assert "SECRET-TOKEN-VALUE" not in str(err.value)
    assert "12345" not in str(err.value)


def test_an_api_rejection_surfaces_the_reason(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "12345:abc")

    class FakeResp:
        def read(self):
            return json.dumps({"ok": False, "description": "chat not found"}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(telegram.urllib.request, "urlopen", lambda *a, **k: FakeResp())
    with pytest.raises(RuntimeError, match="chat not found"):
        telegram._call("sendMessage", {"chat_id": "1", "text": "x"})


# --- files ----------------------------------------------------------------------


def test_an_image_goes_as_a_photo_and_other_files_as_documents(bot, tmp_path, monkeypatch):
    """A photo previews inline in the chat — the entire point of rendering a report
    as a picture is that it reads without opening anything."""
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        telegram, "_upload",
        lambda method, path, fields, field: bool(calls.append((method, field)) or True),
    )
    png = tmp_path / "r.png"; png.write_bytes(b"\x89PNG" + b"0" * 100)
    pdf = tmp_path / "r.pdf"; pdf.write_bytes(b"%PDF" + b"0" * 100)
    assert telegram.send_file(str(png)) is True
    assert telegram.send_file(str(pdf)) is True
    assert calls == [("sendPhoto", "photo"), ("sendDocument", "document")]


def test_an_oversized_image_falls_back_to_a_document(bot, tmp_path, monkeypatch):
    """Telegram recompresses a large photo to mush; as a document it arrives intact."""
    calls: list[str] = []
    monkeypatch.setattr(
        telegram, "_upload",
        lambda method, path, fields, field: bool(calls.append(method) or True),
    )
    monkeypatch.setattr(telegram, "_PHOTO_MAX_BYTES", 50)
    png = tmp_path / "big.png"; png.write_bytes(b"0" * 500)
    telegram.send_file(str(png))
    assert calls == ["sendDocument"]


def test_a_missing_or_unconfigured_file_send_is_false(bot, tmp_path):
    assert telegram.send_file(str(tmp_path / "nope.png")) is False


def test_a_long_caption_is_trimmed_rather_than_losing_the_upload(bot, tmp_path, monkeypatch):
    """Telegram rejects a caption over 1024 chars and takes the file down with it."""
    seen: dict = {}
    monkeypatch.setattr(
        telegram, "_upload",
        lambda method, path, fields, field: bool(seen.update(fields) or True),
    )
    png = tmp_path / "r.png"; png.write_bytes(b"\x89PNG" + b"0" * 10)
    telegram.send_file(str(png), caption="x" * 3000)
    assert len(seen["caption"]) <= 1024


def test_the_multipart_body_carries_the_file_and_fields(tmp_path):
    f = tmp_path / "r.png"; f.write_bytes(b"\x89PNGDATA")
    body, ctype = telegram._multipart({"chat_id": "42", "caption": "hi"}, "photo", f)
    assert ctype.startswith("multipart/form-data; boundary=")
    assert b'name="chat_id"' in body and b"42" in body
    assert b'filename="r.png"' in body and b"\x89PNGDATA" in body
