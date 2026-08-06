"""Rendered reports — turn an answer into a PNG/PDF a phone can actually read.

Everything else this agent produces is text, which is right for a terminal and
poor on a phone: a scheduled task's analysis arrives as a wall of Telegram
message. This renders the same content as a typeset sheet — a headline figure,
stat tiles, then the body — and hands it to the delivery channels as a file.

Three deliberate choices:

**Markdown in, not a schema.** A model produces good markdown and poor deeply
nested JSON. The only structure beyond markdown is ``highlights``: one stat tile
per line as ``label | value | note``, the same pipe-delimited shape ``--schedule``
and ``dispatch_subagents`` already use.

**Chrome renders it.** HTML/CSS is the one layout engine that is already on the
machine, needs no new dependency, and produces both a raster preview and a
print-quality vector PDF from one source. The trade is a hard dependency on a
Chrome/Chromium binary — absent one, the HTML is still written and its path
returned, so the work is never lost.

**The palette is the validated one** from the project's data-viz reference
(surface ``#fcfcfb``, ink ``#0b0b0b``, blue ``#2a78d6`` / red ``#e34948`` as the
diverging pair) rather than colours chosen per report, so every sheet looks like
the same publication and the contrast is known-good.
"""

from __future__ import annotations

from typing import Any
import html as _html
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

#: Where Chrome might live. Checked in order; ``FINANCIAL_RESEARCH_CHROME`` wins.
_CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/snap/bin/chromium",
)

#: Render width in CSS pixels. 1080 is a phone-friendly portrait width that also
#: prints sensibly.
_WIDTH = 1080

#: Device pixel ratio. 3 puts a 1080-wide sheet out at 3240px, so body text stays
#: crisp when a phone lets you pinch into it — the difference between "an image of
#: a report" and something readable. Override for a smaller file.
_DEFAULT_SCALE = 3
_MAX_SCALE = 4

#: Fallback viewport height when measurement fails. Chrome screenshots the
#: VIEWPORT, not the page (verified), so this is not a floor to grow from — it is
#: exactly what gets captured, and a wrong value either clips the sheet or pads it
#: with dead space. Hence `_measure_height` below.
_FALLBACK_H = 2200
_MAX_H = 12000
_RENDER_TIMEOUT = 120


def render_scale() -> int:
    raw = (os.environ.get("FINANCIAL_RESEARCH_REPORT_SCALE") or "").strip()
    if raw.isdigit() and 1 <= int(raw) <= _MAX_SCALE:
        return int(raw)
    return _DEFAULT_SCALE


def chrome_path() -> str:
    """The Chrome/Chromium binary, or "" when there isn't one."""
    explicit = (os.environ.get("FINANCIAL_RESEARCH_CHROME") or "").strip()
    if explicit:
        return explicit if Path(explicit).exists() else ""
    for candidate in _CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    for name in ("google-chrome", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            return found
    return ""


def reports_dir() -> Path:
    from .research import reports_dir as _dir

    return _dir()


def _slug(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in (text or "report")]
    out = "".join(keep).strip("-")
    while "--" in out:
        out = out.replace("--", "-")
    return (out or "report")[:48]


def parse_highlights(raw: str) -> list[dict[str, str]]:
    """``label | value | note`` per line -> stat tiles. Blank lines ignored.

    Tolerant on purpose: a missing note is fine, extra pipes fold into the note,
    and a line with no pipe becomes a value with no label rather than an error —
    a malformed tile should degrade, not sink the report.
    """
    tiles: list[dict[str, str]] = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 1:
            tiles.append({"label": "", "value": parts[0], "note": ""})
        else:
            tiles.append({
                "label": parts[0],
                "value": parts[1],
                "note": " · ".join(p for p in parts[2:] if p),
            })
    return tiles[:6]  # past six they stop being scannable


def _markdown_html(text: str) -> str:
    """Markdown -> HTML via markdown-it (already present through rich)."""
    try:
        from markdown_it import MarkdownIt

        return MarkdownIt("commonmark", {"html": False}).enable("table").render(text or "")
    except Exception:  # pragma: no cover - fallback if the dep ever goes away
        return "".join(
            f"<p>{_html.escape(p)}</p>" for p in (text or "").split("\n\n") if p.strip()
        )


_CSS = """
:root{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--rule:#c3c2b7;--accent:#2a78d6;--warn:#e34948;--wash:#f4f4f1;}
*{box-sizing:border-box;margin:0;padding:0}
body{width:1080px;background:var(--page);color:var(--ink);
 font-family:system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}
.sheet{background:var(--surface);padding:44px 48px 40px}
.eyebrow{font-size:15px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-weight:600}
h1.doc{font-size:44px;line-height:1.1;font-weight:650;letter-spacing:-.02em;margin-top:10px}
.sub{font-size:18px;color:var(--ink2);margin-top:10px}
.rule{height:1px;background:var(--grid);margin:26px 0}
.tiles{display:grid;gap:2px;background:var(--grid);margin-bottom:4px}
.tile{background:var(--surface);padding:20px 18px}
.tile .lab{font-size:13.5px;color:var(--muted);font-weight:500}
.tile .val{font-size:31px;font-weight:650;margin-top:8px;letter-spacing:-.02em}
.tile .note{font-size:14px;color:var(--ink2);margin-top:6px}
.body{font-size:17px;line-height:1.62;color:var(--ink)}
.body h1{font-size:27px;font-weight:650;margin:30px 0 12px;letter-spacing:-.01em}
.body h2{font-size:15px;letter-spacing:.11em;text-transform:uppercase;color:var(--muted);
 font-weight:650;margin:30px 0 14px}
.body h3{font-size:19px;font-weight:650;margin:22px 0 8px}
.body p{margin:12px 0}
.body ul,.body ol{margin:12px 0 12px 24px}
.body li{margin:7px 0}
.body strong{font-weight:650}
.body code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:15px;
 background:var(--wash);padding:2px 6px;border-radius:4px}
.body blockquote{border-left:4px solid var(--warn);background:#fdf5f5;padding:14px 20px;margin:18px 0}
.body blockquote p{margin:0;color:var(--ink2)}
.body table{border-collapse:collapse;width:100%;margin:18px 0;font-size:16px}
.body th{text-align:left;font-size:13px;letter-spacing:.08em;text-transform:uppercase;
 color:var(--muted);font-weight:650;padding:8px 12px;border-bottom:1px solid var(--rule)}
.body td{padding:9px 12px;border-bottom:1px solid var(--grid);font-variant-numeric:tabular-nums}
.body tr:last-child td{border-bottom:none}
.body hr{border:none;height:1px;background:var(--grid);margin:26px 0}
.body a{color:var(--accent);text-decoration:none}
footer{padding:22px 48px 30px;background:var(--page);font-size:12.5px;color:var(--muted);line-height:1.7}
footer .disc{margin-top:12px;border-top:1px solid var(--grid);padding-top:12px}
"""

_DOC = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{title}</title><style>{css}</style></head><body>
<div class="sheet">
  <div class="eyebrow">{eyebrow}</div>
  <h1 class="doc">{title}</h1>
  {subtitle}
  <div class="rule"></div>
  {tiles}
  <div class="body">{body}</div>
</div>
<footer>{footer}<div class="disc">{disclaimer}</div></footer>
<script>document.title="__FRA_H:"+document.documentElement.scrollHeight;</script>
</body></html>"""

_DISCLAIMER = (
    "Generated by financial-research-assistant from delayed public data. Figures are "
    "as reported by the cited sources and have not been independently audited. "
    "For research only — not investment advice, not an offer to buy or sell."
)


def build_html(
    title: str,
    markdown: str,
    highlights: str = "",
    subtitle: str = "",
    eyebrow: str = "",
) -> str:
    """The full HTML document. Pure — no filesystem, no Chrome, so it is testable."""
    tiles = parse_highlights(highlights)
    tiles_html = ""
    if tiles:
        cols = min(len(tiles), 4)
        cells = "".join(
            f'<div class="tile"><div class="lab">{_html.escape(t["label"])}</div>'
            f'<div class="val">{_html.escape(t["value"])}</div>'
            + (f'<div class="note">{_html.escape(t["note"])}</div>' if t["note"] else "")
            + "</div>"
            for t in tiles
        )
        tiles_html = (
            f'<div class="tiles" style="grid-template-columns:repeat({cols},1fr)">{cells}</div>'
        )
    stamp = datetime.now().strftime("%d %B %Y, %H:%M")
    return _DOC.format(
        css=_CSS,
        title=_html.escape(title or "Report"),
        eyebrow=_html.escape(eyebrow or "Financial research assistant"),
        subtitle=f'<div class="sub">{_html.escape(subtitle)}</div>' if subtitle else "",
        tiles=tiles_html,
        body=_markdown_html(markdown),
        footer=f"Generated {stamp}",
        disclaimer=_DISCLAIMER,
    )


def _run_chrome(args: list[str]) -> subprocess.CompletedProcess[bytes] | None:
    try:
        return subprocess.run(
            args, capture_output=True, timeout=_RENDER_TIMEOUT, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _measure_height(chrome: str, url: str) -> int:
    """The document's true pixel height, via a measuring pass.

    Chrome's ``--screenshot`` captures the VIEWPORT, so the window height has to
    equal the content height or the sheet is either clipped or padded with a band
    of dead space (which is what the first delivered reports had). There is no CLI
    flag for "fit the page", but ``--dump-dom`` runs the page's scripts first — so
    the document stamps its own ``scrollHeight`` into the title and this reads it
    back. Costs one extra headless run of a local file.
    """
    proc = _run_chrome([
        chrome, "--headless", "--disable-gpu", "--no-sandbox",
        "--virtual-time-budget=3000", "--dump-dom", url,
    ])
    if proc is None:
        return _FALLBACK_H
    found = re.search(rb"__FRA_H:(\d+)", proc.stdout or b"")
    if not found:
        return _FALLBACK_H
    return max(200, min(int(found.group(1)), _MAX_H))


def render(html: str, name: str) -> dict[str, str]:
    """Write the HTML and render PNG + PDF beside it. Returns the paths that exist.

    Never raises: a missing Chrome or a failed render still leaves the HTML on
    disk, and the caller reports what it got. Losing a finished analysis to a
    rendering problem would be the worst outcome here.
    """
    out_dir = reports_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{_slug(name)}-{datetime.now():%Y%m%d-%H%M%S}"
    paths: dict[str, str] = {}

    html_path = out_dir / f"{stem}.html"
    html_path.write_text(html, encoding="utf-8")
    paths["html"] = str(html_path)

    chrome = chrome_path()
    if not chrome:
        return paths

    base = [chrome, "--headless", "--disable-gpu", "--no-sandbox", "--hide-scrollbars"]
    url = html_path.as_uri()
    height = _measure_height(chrome, url)
    png = out_dir / f"{stem}.png"
    if _run_chrome(base + [
        f"--force-device-scale-factor={render_scale()}",
        f"--screenshot={png}", f"--window-size={_WIDTH},{height}", url,
    ]) and png.exists():
        paths["png"] = str(png)

    pdf = out_dir / f"{stem}.pdf"
    if _run_chrome(base + ["--no-pdf-header-footer", f"--print-to-pdf={pdf}", url]) and pdf.exists():
        paths["pdf"] = str(pdf)
    return paths


# --- Model-facing tool ---------------------------------------------------------


def render_report(
    title: str,
    markdown: str,
    highlights: str = "",
    subtitle: str = "",
    deliver: bool = True,
) -> str:
    """Typeset a summary as a PNG + PDF sheet and send it to the user's channels.

    Use when the user asks for a report/infographic/PDF/one-pager, or wants
    something "sent"/"pushed" to them as a file rather than as chat text — and for
    a scheduled task's output, where a picture reads far better on a phone than a
    wall of message text.

    ``title`` is the headline. ``markdown`` is the body — normal markdown works:
    headings, bold, lists, tables, `>` blockquote for a warning callout.
    ``highlights`` is optional stat tiles, ONE PER LINE as ``label | value | note``
    (up to 6), e.g. "Adjusted EPS | $1.84 | vs $1.91 consensus". Put the numbers
    that matter here, not in the body. ``subtitle`` is one line under the title.
    ``deliver=False`` renders without sending.

    Returns where the files went and whether delivery succeeded. Requires Chrome
    or Chromium for image output; without one you still get the HTML.
    """
    if not (title or "").strip() and not (markdown or "").strip():
        return "Nothing to render — give at least a title or some body text."
    paths = render(build_html(title, markdown, highlights, subtitle), title)

    lines = []
    if "png" not in paths and "pdf" not in paths:
        lines.append(
            "Rendered the HTML but could not produce an image: no Chrome/Chromium "
            "found. Install one, or set FINANCIAL_RESEARCH_CHROME to its path."
        )
    lines.append("Saved: " + ", ".join(f"{k.upper()} {v}" for k, v in paths.items()))

    if deliver:
        from . import channels

        best = paths.get("png") or paths.get("pdf")
        sent: list[str] = []
        if best:
            delivered, failed = channels.deliver_file(best, caption=title, full_quality=True)
            sent += delivered
            if "pdf" in paths and best != paths["pdf"]:
                more, _ = channels.deliver_file(
                    paths["pdf"], caption=f"{title} (PDF)", full_quality=True
                )
                sent += [c for c in more if c not in sent]
            if failed:
                lines.append(f"Delivery failed on: {', '.join(failed)}")
        if sent:
            lines.append(f"Sent to: {', '.join(sorted(set(sent)))}.")
        else:
            lines.append(
                "No channel accepted a file — check TELEGRAM_BOT_TOKEN / "
                "TELEGRAM_CHAT_ID, or read the saved file directly."
            )
    return "\n".join(lines)


REPORT_TOOLS = [render_report]
