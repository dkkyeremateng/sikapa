"""The shared private/atomic store write.

Four files depend on this: the credential store, the task queue, the ingested-
document index, and long-term memory. Each is rewritten in full on every change
and each holds something the user would not publish, so both properties are
tested here once rather than four times over.
"""

import json
import os
import stat

import pytest

from financial_research_assistant.storage import write_private


def test_the_file_is_0600(tmp_path):
    path = tmp_path / "store.json"
    write_private(path, '{"secret": 1}')
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_it_is_never_briefly_world_readable(tmp_path):
    """`write_text` + `chmod` leaves a window where the contents are readable at
    the process umask. `mkstemp` starts the file at 0600 instead, so there is no
    such window — including for the temporary file, which lives in the destination
    directory (same filesystem, so the rename below is atomic rather than a copy)."""
    path = tmp_path / "store.json"
    write_private(path, "x" * 100_000, prefix=".probe-")
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".probe-")]
    assert leftovers == [], "the temporary file must not survive a successful write"


def test_the_parent_directory_is_created(tmp_path):
    path = tmp_path / "nested" / "deeper" / "store.json"
    write_private(path, "hello")
    assert path.read_text() == "hello"


def test_a_failed_write_leaves_the_previous_contents_intact(tmp_path, monkeypatch):
    """The reason this is atomic at all: every caller rewrites the WHOLE store, so
    a torn write doesn't corrupt one record — it truncates the file, which then
    parses as 'you have nothing saved'."""
    path = tmp_path / "store.json"
    write_private(path, json.dumps({"kept": True}))

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        write_private(path, json.dumps({"kept": False}), prefix=".probe-")

    assert json.loads(path.read_text()) == {"kept": True}
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".probe-")]
    assert leftovers == [], "a failed write must not leave a temporary file behind"


def test_the_bytes_are_fsynced_before_the_rename(tmp_path, monkeypatch):
    """`os.replace` orders the rename, not the data. Without an fsync first, a
    crash after the rename can leave the destination present and zero-length —
    which is exactly the "you have nothing saved" outcome the atomic rename was
    put there to prevent."""
    order: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        order.append("fsync")
        return real_fsync(fd)

    def replace(src, dst):
        order.append("replace")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    write_private(tmp_path / "store.json", json.dumps({"credential": "kept"}))

    assert order == ["fsync", "replace"]


# --- the stores that go through it ----------------------------------------------


def test_the_document_index_is_0600(tmp_path, monkeypatch):
    """It holds the extracted TEXT of whatever was ingested — routinely a brokerage
    statement or a private research PDF."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_DOCS_DIR", str(tmp_path / "docs"))
    from financial_research_assistant import documents

    src = tmp_path / "note.md"
    src.write_text("Revenue rose 12% on strong demand across every segment.")
    documents.ingest_document(str(src))

    mode = stat.S_IMODE(documents._index_path().stat().st_mode)
    assert mode == 0o600, f"document index is {oct(mode)}"


def test_the_memory_store_is_0600(tmp_path, monkeypatch):
    """Risk tolerance, tax situation, holdings — personal by definition."""
    monkeypatch.setenv("MEMORY_BACKEND", "local")
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("MEMORY_USER", "alice")
    from financial_research_assistant.memory import get_memory

    mem = get_memory()
    assert mem is not None
    mem.save("My risk tolerance is moderate")

    written = list(tmp_path.glob("*.jsonl"))
    assert len(written) == 1
    mode = stat.S_IMODE(written[0].stat().st_mode)
    assert mode == 0o600, f"memory store is {oct(mode)}"


def test_state_dir_follows_its_override(monkeypatch, tmp_path):
    from financial_research_assistant import storage

    monkeypatch.setenv("FINANCIAL_RESEARCH_HOME", str(tmp_path / "srv"))
    assert storage.state_dir() == tmp_path / "srv"
    assert storage.state_file("ledger.json") == tmp_path / "srv" / "ledger.json"
    monkeypatch.setenv("MY_OVERRIDE", str(tmp_path / "elsewhere.json"))
    assert storage.state_file("ledger.json", "MY_OVERRIDE") == tmp_path / "elsewhere.json"


def test_a_jsonl_log_is_private_and_survives_a_torn_line(tmp_path):
    import os
    import stat

    from financial_research_assistant import storage

    log = tmp_path / "events.jsonl"
    storage.append_jsonl(log, {"a": 1})
    storage.append_jsonl(log, {"a": 2})
    with open(log, "a", encoding="utf-8") as fh:
        fh.write('{"a": 3')  # a write cut off by a crash
    assert stat.S_IMODE(os.stat(log).st_mode) == 0o600
    assert storage.read_jsonl(log) == [{"a": 1}, {"a": 2}]
    assert storage.read_jsonl(log, limit=1) == []  # the last line is the torn one
    assert storage.read_jsonl(tmp_path / "missing.jsonl") == []


def test_read_json_treats_a_corrupt_store_as_its_default(tmp_path):
    from financial_research_assistant import storage

    bad = tmp_path / "x.json"
    bad.write_text("{nope", encoding="utf-8")
    assert storage.read_json(bad, {"k": []}) == {"k": []}
    assert storage.read_json(tmp_path / "missing.json", []) == []


def test_an_append_tightens_a_log_created_world_readable(tmp_path):
    """Transcripts were created 0644 before they went through this helper; the
    next append must close that, not leave old files readable for ever."""
    import os
    import stat

    from financial_research_assistant import storage

    log = tmp_path / "cli.jsonl"
    log.write_text('{"old": 1}\n', encoding="utf-8")
    os.chmod(log, 0o644)
    storage.append_jsonl(log, {"new": 2})
    assert stat.S_IMODE(os.stat(log).st_mode) == 0o600
    assert storage.read_jsonl(log) == [{"old": 1}, {"new": 2}]
