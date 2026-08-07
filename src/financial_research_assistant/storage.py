"""One way to write a local store: privately, and all-or-nothing.

Every file this project keeps under ``~/.financial-research-assistant`` (and the
memory store under ``MEMORY_DIR``) holds something the user would not want
world-readable — a provider credential, a task prompt quoting position sizes, the
text of an ingested brokerage statement, a remembered fact about their tax
situation. And every one of them is rewritten in full on each change, so a crash
mid-write truncates the whole store rather than one record.

``write_private`` is the two-line answer to both, factored out of ``auth.py`` and
``tasks.py`` where it was already written twice:

* ``mkstemp`` creates the temporary file ``0600`` in the destination directory,
  so the contents are never briefly world-readable the way ``write_text`` +
  ``chmod`` would leave them, and never on a different filesystem (which would
  make the rename below a copy).
* ``os.replace`` swaps it in atomically. A reader either sees the whole old file
  or the whole new one — never a half-written one, which for the JSON/JSONL
  stores here parses as "empty" and reads as "you have nothing saved".
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


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
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
