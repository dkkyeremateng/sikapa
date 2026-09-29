"""One way to write a local store: privately, and all-or-nothing.

Every file this project keeps under ``~/.financial-research-assistant`` (and the
memory store under ``MEMORY_DIR``) holds something the user would not want
world-readable — a provider credential, a task prompt quoting position sizes, the
text of an ingested brokerage statement, a remembered fact about their tax
situation. And every one of them is rewritten in full on each change, so a crash
mid-write truncates the whole store rather than one record.

``write_private`` is the small answer to both, factored out of ``auth.py`` and
``tasks.py`` where it was already written twice:

* ``mkstemp`` creates the temporary file ``0600`` in the destination directory,
  so the contents are never briefly world-readable the way ``write_text`` +
  ``chmod`` would leave them, and never on a different filesystem (which would
  make the rename below a copy).
* ``fsync`` forces the bytes to disk *before* the rename. A rename is atomic with
  respect to other readers, but it says nothing about durability: the filesystem
  is free to commit the new directory entry while the data blocks are still in
  the page cache, so a crash right after ``os.replace`` can leave ``path``
  present and zero-length. That is the exact shape the atomicity below exists to
  prevent.
* ``os.replace`` swaps it in atomically. A reader either sees the whole old file
  or the whole new one — never a half-written one, which for the JSON/JSONL
  stores here parses as "empty" and reads as "you have nothing saved".
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def state_dir() -> Path:
    """Where this assistant keeps its state: ``~/.financial-research-assistant``.

    ``FINANCIAL_RESEARCH_HOME`` moves it — a server mounts its data volume
    somewhere of its own choosing, and the test suite points it at a throwaway
    directory. The older stores each take their own ``FINANCIAL_RESEARCH_*_FILE``
    override and default under the same directory; newer ones resolve through here.
    """
    raw = os.environ.get("FINANCIAL_RESEARCH_HOME")
    if raw:
        return Path(os.path.expandvars(raw)).expanduser()
    return Path.home() / ".financial-research-assistant"


def state_file(name: str, env_var: str = "") -> Path:
    """A file under ``state_dir()``, or wherever ``env_var`` points if it is set."""
    raw = os.environ.get(env_var) if env_var else None
    if raw:
        return Path(os.path.expandvars(raw)).expanduser()
    return state_dir() / name


@contextmanager
def locked(path: Path) -> Iterator[None]:
    """Hold an exclusive cross-process lock for a read-modify-write of ``path``.

    The same ``fcntl`` pattern ``tasks.py`` and ``alerts.py`` each wrote for
    themselves, for the stores added since: a sidecar ``.lock`` file, blocking,
    and degrading to no locking where ``fcntl`` doesn't exist rather than failing.
    """
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path.with_suffix(path.suffix + ".lock"), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def read_json(path: Path, default: Any) -> Any:
    """Parse ``path``, or return ``default`` when it is missing or unreadable.

    A corrupt store reads as empty rather than raising, the same trade every
    store here makes: a bad hand-edit must not take down the tick that reads it.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON line to a ``0600`` log, creating it if needed.

    Append-only logs (events, runs) are never rewritten, so the atomic swap
    ``write_private`` does is unnecessary here; what matters is that the file is
    private from its first byte and that one record is one line.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def read_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    """Every record in a JSONL log, oldest first (the last ``limit`` if given).
    Unparseable lines — a torn final write — are skipped, not fatal."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in lines[-limit:] if limit else lines:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def write_private(path: Path, text: str, prefix: str = ".tmp-") -> None:
    """Write ``text`` to ``path`` as a ``0600`` file, atomically.

    Creates the parent directory if needed. ``prefix`` names the temporary file
    so a leftover from a hard kill is identifiable; it is otherwise cosmetic.
    Any failure removes the temporary file and re-raises, leaving the previous
    contents of ``path`` intact.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=prefix, suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
