"""Rendered reports — turn an answer into a PDF + cover image a phone can read.

Everything else this agent produces is text, which is right for a terminal and
poor on a phone: a scheduled task's analysis arrives as a wall of Telegram
message. This renders the same content as a typeset sheet — headline, stat tiles,
body — and hands it to the delivery channels as files.

**The PDF is the document; the PNG is a purpose-built infographic of it** — the
headline figures plus whatever series the body contains, charted. It replaces the
old "screenshot the first page" cover at every length: page 1 of a long report is
the masthead and whatever happened to fit, and page 1 of a short one is a wall of
prose that a picture summarises better. A report with neither tiles nor series
falls back to page 1, since a sheet with nothing to visualise has no value over the
document itself. Everything is PDF-first, with ``pypdfium2`` rasterising one page.

**Two renderers, one content model.** Both consume the same markdown and the same
``highlights``:

* **Chrome** (preferred) renders the HTML template. Better typography, and the CSS
  is the design. ``@page`` is sized to the sheet, so a short report is one page
  identical to the on-screen design and a long one paginates at that size.
* **fpdf2** (fallback) draws the same structure when no browser is installed —
  the case in this project's own container, where the feature otherwise produced
  HTML and nothing else. Plainer, and it always works.

A fallback nobody looks at rots quietly, so the mitigation is in the tests: they
run *both* renderers over the same inputs and assert the same structural facts, so
a divergence fails CI rather than surfacing months later as an ugly PDF.

**Markdown in, not a schema.** A model produces good markdown and poor deeply
nested JSON. The only structure beyond markdown is ``highlights``: one stat tile
per line as ``label | value | note``, the pipe-delimited shape ``--schedule`` and
``dispatch_subagents`` already use.

**The palette is the validated one** from the project's data-viz reference
(surface ``#fcfcfb``, ink ``#0b0b0b``, red ``#e34948`` for the callout rule)
rather than colours chosen per report, so every sheet looks like one publication
and the contrast is known-good.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
import html as _html
import os
import re
import shutil
import subprocess
from contextlib import contextmanager
from contextvars import ContextVar
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

#: Sheet width in CSS pixels — phone-friendly portrait that also prints sensibly.
_WIDTH = 1080

#: A report shorter than this becomes ONE page sized exactly to its content, so the
#: common case carries no trailing dead band. Longer ones paginate at `_PAGE_H`.
_SINGLE_MAX_H = 3200
_PAGE_H = 1500
_FALLBACK_H = 2200
_MAX_H = 20000

#: Device pixel ratio for the cover image. 3 keeps body text crisp when a phone
#: lets you pinch into it.
_DEFAULT_SCALE = 3
_MAX_SCALE = 4
_RENDER_TIMEOUT = 120

#: CSS px -> PDF points.
_PT = 0.75

#: Light and dark are both SELECTED, not one flipped into the other: the dark row
#: takes its own steps from the same ramps, chosen for the dark surface. Both
#: diverging pairs pass the data-viz validator against their own surface (CVD dE
#: 21.6 light / 19.2 dark, contrast >= 3:1).
_THEMES = {
    "light": {
        "surface": "#fcfcfb", "page": "#f9f9f7", "ink": "#0b0b0b", "ink2": "#52514e",
        "muted": "#898781", "grid": "#e1e0d9", "rule": "#c3c2b7", "track": "#f0efec",
        "accent": "#2a78d6", "warn": "#e34948", "wash": "#f4f4f1",
        "pos": "#2a78d6", "neg": "#e34948", "callout": "#fdf5f5",
    },
    "dark": {
        "surface": "#1a1a19", "page": "#0d0d0d", "ink": "#ffffff", "ink2": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "rule": "#383835", "track": "#232322",
        "accent": "#3987e5", "warn": "#e66767", "wash": "#232322",
        "pos": "#3987e5", "neg": "#e66767", "callout": "#2a1e1e",
    },
}


#: The theme in force for the artifact currently being drawn. The cover and the
#: document are separate artifacts with different jobs — one is read on a phone,
#: the other may be printed — so they are themed independently, and a context is
#: cleaner than threading a parameter through every drawing call.
_active_theme: ContextVar[str | None] = ContextVar("fra_report_theme", default=None)


@contextmanager
def use_theme(name: str):
    """Draw everything inside this block in ``name``."""
    token = _active_theme.set(name if name in _THEMES else "light")
    try:
        yield
    finally:
        _active_theme.reset(token)


def _env_theme(var: str, default: str = "light") -> str:
    name = (os.environ.get(var) or "").strip().lower()
    return name if name in _THEMES else default


#: Which engine draws the sheet. Operator-level rather than a tool argument: the
#: model has no basis for choosing an engine, and every parameter costs schema on
#: every call. "auto" prefers Chrome and falls back; the explicit values are for
#: pinning a container or reproducing a bug.
_RENDERERS = ("auto", "chrome", "fpdf2")


def renderer_mode() -> str:
    name = (os.environ.get("FINANCIAL_RESEARCH_REPORT_RENDERER") or "").strip().lower()
    return name if name in _RENDERERS else "auto"


#: What the tool produces and delivers. "both" is the default; "image" suits a
#: phone-only workflow, "pdf" a filing one. The PDF is rendered either way — the
#: cover is rasterised FROM it — so "image" withholds the file rather than skipping
#: the work.
_OUTPUTS = ("both", "image", "pdf")


def output_mode(override: str = "") -> str:
    name = (override or os.environ.get("FINANCIAL_RESEARCH_REPORT_OUTPUT") or "").strip().lower()
    return name if name in _OUTPUTS else "both"


def cover_theme(override: str = "") -> str:
    """Theme for the image — the artifact that lands in a chat. ``override`` is one
    report's own choice, which wins over the configured default."""
    name = (override or "").strip().lower()
    return name if name in _THEMES else _env_theme("FINANCIAL_RESEARCH_REPORT_THEME")


def pdf_theme() -> str:
    """Theme for the document. Defaults to light INDEPENDENTLY of the cover: a dark
    PDF lays down a full page of ink when printed, so wanting a dark sheet on a
    phone should not quietly commit you to that."""
    return _env_theme("FINANCIAL_RESEARCH_REPORT_PDF_THEME")


def report_theme() -> str:
    """The theme in force right now (the cover's, outside a render)."""
    return _active_theme.get() or cover_theme()


def _palette() -> dict[str, str]:
    return _THEMES[report_theme()]


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

#: Added to the measured height before sizing the page. Print layout computes
#: fractionally taller boxes than `scrollHeight` reports, and being 1px short spills
#: the sheet onto a second, near-empty page — measured at 1013px needing 1021.
_HEIGHT_SLACK = 12


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


#: Where a figure starts: at a word gap, an optional currency mark and sign, then
#: a digit. The gap is what stops "Top-5 concentration 64.2%" splitting at the
#: hyphen inside "Top-5" and yielding the label "Top".
_TILE_FIGURE_RE = re.compile(r"(?:^|(?<=\s))[$€£¥]?\s*[+\-−–—]?\d")


def _split_at_the_figure(text: str) -> dict[str, str]:
    """A pipe-less tile split into label and figure at the first number.

    "Analyst target $62.37 (+19.4%)" arrived with no pipes and became one long
    value under an empty label — a tile with a heading and nothing beneath it on a
    delivered sheet. The split the writer meant is obvious and mechanical: words
    first, figure after. A line with no figure at all, or one that opens with it,
    is left exactly as it was, since there is nothing to divide.
    """
    text = text.strip()
    found = _TILE_FIGURE_RE.search(text)
    if not found:
        return {"label": "", "value": text, "note": ""}
    # No separate guard for "the figure is first": the slice before it is then
    # empty, which is the same empty label that branch would have produced.
    return {
        "label": text[:found.start()].strip(" .:—–-"),
        "value": text[found.start():].strip(),
        "note": "",
    }


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
        # Several tiles on ONE line: a model that forgets the newlines produces
        # `A | 1 | note | B | 2 | note | ...`, which folded into a single tile with
        # an unreadable run-on note (observed). Six or more fields is not one tile.
        if len(parts) >= 6:
            for start in range(0, len(parts), 3):
                group = parts[start:start + 3]
                if not group[0]:
                    continue
                tiles.append({
                    "label": group[0],
                    "value": group[1] if len(group) > 1 else "",
                    "note": group[2] if len(group) > 2 else "",
                })
            continue
        if len(parts) == 1:
            tiles.append(_split_at_the_figure(parts[0]))
        else:
            tiles.append({
                "label": parts[0],
                "value": parts[1],
                "note": " · ".join(p for p in parts[2:] if p),
            })
    return tiles[:6]  # past six they stop being scannable


#: A tile value should be a FIGURE. When the model writes a phrase instead it wraps
#: onto a second line, making that tile taller than its neighbours and breaking the
#: grid rhythm (observed: "Q1 2026: +24.0% surprise"). Rather than rewrite the
#: model's text — an early attempt turned "Trailing P/E: 11.33" into the meaningless
#: "Trailing P/E: 11" — the type steps down so every character survives on one line.
_TILE_STEPS = ((16, ""), (26, " sm"), (999, " xs"))


def _tile_size(value: str) -> str:
    """The size-class suffix for a tile value."""
    return next(cls for limit, cls in _TILE_STEPS if len(value or "") <= limit)


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
:root{{{vars}}}
@page{{size:{w}px {h}px;margin:0}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{w}px;background:var(--page);color:var(--ink);
 font-family:system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}}
.sheet{{background:var(--surface);padding:44px 48px 40px}}
.eyebrow{{font-size:15px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-weight:600}}
h1.doc{{font-size:44px;line-height:1.1;font-weight:650;letter-spacing:-.02em;margin-top:10px}}
.sub{{font-size:18px;color:var(--ink2);margin-top:10px}}
{stance_css}
.rule{{height:1px;background:var(--grid);margin:26px 0}}
.tiles{{display:grid;gap:2px;background:var(--grid);margin-bottom:4px}}
.tile{{background:var(--surface);padding:20px 18px}}
.tile .lab{{font-size:13.5px;color:var(--muted);font-weight:500}}
.tile .val{{font-size:31px;font-weight:650;margin-top:8px;letter-spacing:-.02em;white-space:nowrap}}
.tile .val.sm{{font-size:23px}}
.tile .val.xs{{font-size:17px;white-space:normal}}
.tile .note{{font-size:14px;color:var(--ink2);margin-top:6px}}
.body{{font-size:17px;line-height:1.62;color:var(--ink)}}
.body h1{{font-size:27px;font-weight:650;margin:30px 0 12px;letter-spacing:-.01em}}
.body h2{{font-size:15px;letter-spacing:.11em;text-transform:uppercase;color:var(--muted);
 font-weight:650;margin:30px 0 14px}}
.body h3{{font-size:19px;font-weight:650;margin:22px 0 8px}}
.body p{{margin:12px 0}}
.body ul,.body ol{{margin:12px 0 12px 24px}}
.body li{{margin:7px 0}}
.body strong{{font-weight:650}}
.body code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:15px;
 background:var(--wash);padding:2px 6px;border-radius:4px}}
.body blockquote{{border-left:4px solid var(--warn);background:var(--callout);padding:14px 20px;margin:18px 0}}
.body blockquote p{{margin:0;color:var(--ink2)}}
.body table{{border-collapse:collapse;width:100%;margin:18px 0;font-size:16px}}
.body th{{text-align:left;font-size:13px;letter-spacing:.08em;text-transform:uppercase;
 color:var(--muted);font-weight:650;padding:8px 12px;border-bottom:1px solid var(--rule)}}
.body td{{padding:9px 12px;border-bottom:1px solid var(--grid);font-variant-numeric:tabular-nums}}
.body tr:last-child td{{border-bottom:none}}
.body hr{{border:none;height:1px;background:var(--grid);margin:26px 0}}
.body a{{color:var(--accent);text-decoration:none}}
footer{{padding:22px 48px 30px;background:var(--page);font-size:12.5px;color:var(--muted);line-height:1.7}}
footer .disc{{margin-top:12px;border-top:1px solid var(--grid);padding-top:12px}}
/* Pagination. Without these a long report strands a heading at the foot of a page
   (observed in a real render) and splits tables mid-row. */
h1,h2,h3,.body h1,.body h2,.body h3{{break-after:avoid;page-break-after:avoid}}
.body table,.body blockquote,.tiles,.tile{{break-inside:avoid;page-break-inside:avoid}}
.body tr{{break-inside:avoid;page-break-inside:avoid}}
"""

#: The stance badge, shared by both sheets so the call cannot look like a
#: different thing on the image than in the document.
#:
#: 20px/700 is a contrast decision as much as a typographic one. `pos` and `neg`
#: are validated at >= 3:1 against their own surface, which WCAG accepts for LARGE
#: text (>= 18.66px bold) but not for normal text — light-theme `pos` measures
#: 4.30:1 and `neg` 3.85:1, both under the 4.5:1 body-text bar. Sizing the badge
#: to what it should be anyway is what makes those colours legitimate on it;
#: shrinking it would quietly put the sheet out of conformance.
#:
#: Outlined rather than filled: a filled pill puts text on `pos` instead of on the
#: surface, and that pairing is not what the palette validated.
#:
#: SINGLE braces, unlike the sheets this is spliced into. Those are `.format()`
#: templates and double their braces to survive it; this block is a substituted
#: VALUE, and a value is not re-processed — doubled braces reached the stylesheet
#: literally as `.stance{{...}}`, which every rule in the block then silently
#: dropped. The badge still emitted correct markup, so it rendered as unstyled
#: text run together with its reason.
_STANCE_CSS = """.stance{display:flex;align-items:center;gap:14px;margin-top:18px;flex-wrap:wrap}
.stance .pill{font-size:20px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;
 padding:6px 18px;border-radius:999px;border:2px solid;line-height:1.25}
.stance .pill.pos{color:var(--pos);border-color:var(--pos)}
.stance .pill.neg{color:var(--neg);border-color:var(--neg)}
.stance .pill.hold{color:var(--ink2);border-color:var(--rule)}
.stance .why{font-size:16px;color:var(--ink2)}"""

_DOC = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{title}</title><style>{css}</style></head><body>
<div class="sheet">
  <div class="eyebrow">{eyebrow}</div>
  <h1 class="doc">{title}</h1>
  {subtitle}
  {stance}
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

_EYEBROW = "Financial research assistant"


def _css_vars() -> str:
    """The theme as CSS custom properties, so the sheet is written against roles."""
    return "".join(f"--{k}:{v};" for k, v in _palette().items())


def _stamp() -> str:
    return f"Generated {datetime.now():%d %B %Y, %H:%M}"


def build_html(
    title: str,
    markdown: str,
    highlights: str = "",
    subtitle: str = "",
    eyebrow: str = "",
    page_height: int = _PAGE_H,
    stance: str = "",
) -> str:
    """The full HTML document. Pure — no filesystem, no Chrome, so it is testable.

    ``page_height`` sizes ``@page``: the measured content height for a single-page
    sheet, or the default to paginate.
    """
    tiles = cover_tiles(highlights, markdown)
    tiles_html = ""
    if tiles:
        cols = min(len(tiles), 4)
        cells = "".join(
            f'<div class="tile"><div class="lab">{_html.escape(t["label"])}</div>'
            f'<div class="val{_tile_size(t["value"])}">{_html.escape(t["value"])}</div>'
            + (f'<div class="note">{_html.escape(t["note"])}</div>' if t["note"] else "")
            + "</div>"
            for t in tiles
        )
        tiles_html = (
            f'<div class="tiles" style="grid-template-columns:repeat({cols},1fr)">{cells}</div>'
        )
    return _DOC.format(
        css=_CSS.format(
            w=_WIDTH, h=max(200, min(int(page_height), _MAX_H)), vars=_css_vars(),
            stance_css=_STANCE_CSS,
        ),
        title=_html.escape(title or "Report"),
        eyebrow=_html.escape(eyebrow or _EYEBROW),
        subtitle=f'<div class="sub">{_html.escape(subtitle)}</div>' if subtitle else "",
        stance=_stance_html(stance_of(stance, markdown)),
        tiles=tiles_html,
        body=_markdown_html(markdown),
        footer=_stamp(),
        disclaimer=_DISCLAIMER,
    )


# --- Chrome renderer -----------------------------------------------------------


def _sandbox_disabled() -> bool:
    """Whether to pass ``--no-sandbox``.

    The page being rendered is written by the MODEL, and a report routinely
    summarises `web_search` results and filing text — i.e. content this project
    treats as untrusted everywhere else. Nothing here is known to be exploitable
    (the markdown is rendered with ``html=False`` and every interpolated field goes
    through ``html.escape``), so this is defence in depth, not a fix. But it is the
    cheap kind: the sandbox costs nothing on a normal desktop render, and it is the
    layer that still holds if one of those escapes is ever missed.

    Two cases genuinely need it off, and only these:
      - running as root, where Chrome refuses to start sandboxed at all (the usual
        container case — detected rather than configured, so a Docker run still
        renders without anyone having to know this flag exists);
      - ``FINANCIAL_RESEARCH_CHROME_NO_SANDBOX=1``, for a confined environment where
        the sandbox can't acquire the namespaces it needs.
    """
    if (os.environ.get("FINANCIAL_RESEARCH_CHROME_NO_SANDBOX") or "").strip().lower() in (
        "1", "true", "yes", "on",
    ):
        return True
    geteuid = getattr(os, "geteuid", None)  # absent on Windows
    return geteuid is not None and geteuid() == 0


def _chrome_flags(chrome: str) -> list[str]:
    """The command prefix every headless run shares."""
    flags = [chrome, "--headless", "--disable-gpu"]
    if _sandbox_disabled():
        flags.append("--no-sandbox")
    return flags


def _run_chrome(args: list[str]) -> subprocess.CompletedProcess[bytes] | None:
    try:
        return subprocess.run(
            args, capture_output=True, timeout=_RENDER_TIMEOUT, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _measure_height(chrome: str, url: str) -> int:
    """The document's true pixel height, via a measuring pass.

    There is no CLI flag for "fit the page", but ``--dump-dom`` runs the page's
    scripts first — so the document stamps its own ``scrollHeight`` into the title
    and this reads it back. One extra headless run over a local file.
    """
    proc = _run_chrome(
        _chrome_flags(chrome) + ["--virtual-time-budget=3000", "--dump-dom", url]
    )
    if proc is None:
        return _FALLBACK_H
    found = re.search(rb"__FRA_H:(\d+)", proc.stdout or b"")
    return max(200, min(int(found.group(1)), _MAX_H)) if found else _FALLBACK_H


def _chrome_pdf(
    chrome: str, html_path: Path, pdf: Path, single_page: bool = False
) -> bool:
    """Print the sheet, sizing the page to the content when it fits on one.

    ``single_page`` forces one page however tall the content is. The cover
    infographic needs it: paginating a cover would make it a slice of a summary of
    a report — which is the problem it exists to solve.
    """
    url = html_path.as_uri()
    height = _measure_height(chrome, url) + _HEIGHT_SLACK
    if not single_page and height > _SINGLE_MAX_H:
        height = _PAGE_H
    html_path.write_text(
        re.sub(
            r"@page\{size:\d+px \d+px",
            f"@page{{size:{_WIDTH}px {height}px",
            html_path.read_text(encoding="utf-8"),
        ),
        encoding="utf-8",
    )
    _run_chrome(
        _chrome_flags(chrome)
        + ["--hide-scrollbars", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", url]
    )
    return pdf.exists()


# --- fpdf2 renderer (no browser) -----------------------------------------------

def _ink(role: str) -> tuple[int, int, int]:
    """A theme colour as RGB, for the browser-free renderer."""
    return _rgb(_palette()[role])


def _draw_stance(pdf: Any, stance: dict[str, str] | None, x: float, family: str,
                 conv: Callable[[str], str], inner: float) -> None:
    """Draw the stance badge at the cursor, in the same shape the CSS produces.

    Shared by both fpdf sheets: the fallback renderer is what runs with no browser
    installed, and a call that appeared only under Chrome would be a call the
    reader sometimes does not get.
    """
    if not stance:
        return
    label = conv(stance["label"])
    pdf.set_font(family, "B", 15)  # 20px CSS at 0.75pt/px
    text_w = pdf.get_string_width(label)
    pad, height = 13.0, 26.0
    top = pdf.get_y()
    colour = _ink("ink2" if stance["tone"] == "hold" else stance["tone"])
    pdf.set_draw_color(*(_ink("rule") if stance["tone"] == "hold" else colour))
    pdf.set_line_width(1.5)
    pdf.rect(x, top, text_w + 2 * pad, height, style="D",
             round_corners=True, corner_radius=height / 2)
    pdf.set_xy(x, top + 6)
    pdf.set_text_color(*colour)
    pdf.cell(text_w + 2 * pad, 14, label, align="C")
    if stance["note"]:
        pdf.set_xy(x + text_w + 2 * pad + 11, top + 6)
        pdf.set_font(family, "", 12)
        pdf.set_text_color(*_ink("ink2"))
        pdf.cell(inner - (text_w + 2 * pad + 11), 14, conv(stance["note"]), align="L")
    pdf.set_y(top + height)

#: System fonts carrying the punctuation a report actually uses (— · ▼ ✓ →). The
#: core PDF fonts are Latin-1 only and RAISE on an em dash, so without one of these
#: the text is transliterated instead (see `_ascii`). Nothing is vendored: a font
#: file in the repo is a licence to track, and the transliteration is a fine floor.
#: (regular, bold) pairs. fpdf2 does NOT synthesize bold for an embedded TTF, so
#: registering one file for both weights renders `**bold**` identically to body
#: text and the emphasis is simply lost. Pairs are preferred for that reason;
#: `Arial Unicode` has no bold companion and sits last as a single-weight
#: fallback that at least keeps the punctuation.
_FONT_CANDIDATES = (
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/usr/share/fonts/TTF/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"),
    ("/usr/share/fonts/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
    ("/System/Library/Fonts/Supplemental/Arial.ttf",
     "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"),
    ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", ""),
)

#: Always stripped, whatever the font. No text face carries emoji, and a missing
#: glyph in a TTF renders as an empty box rather than raising — silently ugly.
_EMOJI_MAP = {"🔔": "*", "🤖": "", "📈": "", "📉": "", "⚠️": "!"}

#: Only applied on the core-font path, which is Latin-1 and RAISES on an em dash.
#: With a real Unicode face these characters render properly and are left alone.
_LATIN1_MAP = {
    "—": "-", "–": "-", "‘": "'", "’": "'", "“": '"', "”": '"', "…": "...",
    "→": "->", "≈": "~", "×": "x", "✓": "[ok]", "▼": "v", "▲": "^", "⚠": "!",
    "€": "EUR ", "≥": ">=", "≤": "<=", "·": "-",
}


def _strip_emoji(text: str) -> str:
    for src, dst in _EMOJI_MAP.items():
        text = text.replace(src, dst)
    return text


def _ascii(text: str) -> str:
    """Transliterate everything the core PDF fonts cannot encode.

    Only used when no Unicode TTF is available at all. Losing an em dash beats
    ``FPDFUnicodeEncodingException`` taking the whole render down.
    """
    text = _strip_emoji(text)
    for src, dst in _LATIN1_MAP.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", "replace").decode("latin-1")


def _bundled_font() -> tuple[str, str]:
    """The vendored Inter faces, or ``("", "")`` if the package data is missing.

    Vendored so a sheet renders identically everywhere: relying on system fonts
    made the output depend on the machine — Arial on macOS, DejaVu on Linux, and
    transliterated ASCII in a slim container. Inter is SIL OFL 1.1 (licence beside
    the files) and is the closest free face to the ``system-ui`` stack the Chrome
    path renders, so the two renderers look like the same publication.
    """
    base = Path(__file__).parent / "assets" / "fonts"
    regular, bold = base / "Inter-Regular.ttf", base / "Inter-Bold.ttf"
    if regular.exists():
        return str(regular), (str(bold) if bold.exists() else "")
    return "", ""


def _tag_styles(family: str) -> dict[str, Any]:
    """Restyle fpdf2's HTML defaults to this project's palette.

    Its stock heading colour is dark red (``rgb(150,0,0)``) and blockquote is
    maroon. In this palette red means *miss / critical*, so a report whose every
    section heading is red reads as a page of alarms. Headings take ink and muted
    grey instead, matching the CSS template.
    """
    try:
        from fpdf.fonts import TextStyle
    except ImportError:  # pragma: no cover - older fpdf2
        return {}
    return {
        "h1": TextStyle(font_family=family, font_style="B", font_size_pt=17,
                        color=_ink("ink"), t_margin=10, b_margin=3),
        "h2": TextStyle(font_family=family, font_style="B", font_size_pt=11,
                        color=_ink("muted"), t_margin=11, b_margin=2),
        "h3": TextStyle(font_family=family, font_style="B", font_size_pt=13,
                        color=_ink("ink"), t_margin=8, b_margin=2),
        "blockquote": TextStyle(font_family=family, font_size_pt=10, color=_ink("ink2"),
                                l_margin=12, t_margin=5, b_margin=5),
    }


def _unicode_font() -> tuple[str, str]:
    """``(regular, bold)`` paths for an embeddable Unicode font, or ``("", "")``."""
    explicit = (os.environ.get("FINANCIAL_RESEARCH_REPORT_FONT") or "").strip()
    if explicit and Path(explicit).exists():
        bold = re.sub(r"(-Regular)?\.(ttf|otf)$", r"-Bold.\2", explicit)
        return explicit, (bold if Path(bold).exists() else "")
    bundled = _bundled_font()
    if bundled[0]:
        return bundled
    for regular, bold in _FONT_CANDIDATES:
        if Path(regular).exists():
            return regular, (bold if bold and Path(bold).exists() else "")
    return "", ""


def _fpdf_pdf(
    pdf_path: Path, title: str, markdown: str, highlights: str = "",
    subtitle: str = "", eyebrow: str = "", stance: str = "",
) -> bool:
    """Draw the sheet without a browser. Same content, plainer typography.

    Rendered twice when it fits on one page: fpdf2 costs ~10ms, so measuring by
    rendering is cheaper than estimating, and it buys the same "no trailing dead
    band" the Chrome path gets from its measuring pass.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        return False

    font_regular, font_bold = _unicode_font()
    width_pt = _WIDTH * _PT
    margin = 36.0
    call = stance_of(stance, markdown)

    class Sheet(FPDF):
        """Footer as an override, not a manual write at the end.

        Writing it by hand at ``set_y(-44)`` lands inside the auto-page-break
        margin, which triggers a break and adds a blank page — the sheet then
        reports two pages and never gets refitted. fpdf2 calls this hook with
        break handling suppressed, and repeats it on every page, which a
        multi-page report wants anyway.
        """

        conv: Callable[[str], str] = staticmethod(lambda s: s)
        family: str = "helvetica"

        def footer(self) -> None:
            self.set_y(-40)
            self.set_x(margin)
            self.set_font(self.family, "", 7)
            self.set_text_color(*_ink("muted"))
            self.multi_cell(
                width_pt - 2 * margin, 9,
                self.conv(f"{_stamp()}  ·  {_DISCLAIMER}"), align="L",
            )

    def draw(height_pt: float | None):
        page_h = height_pt or (_PAGE_H * _PT)
        pdf = Sheet(unit="pt", format=(width_pt, page_h))
        pdf.set_auto_page_break(True, margin=52)
        pdf.set_left_margin(margin)
        pdf.set_right_margin(margin)
        pdf.add_page()
        pdf.set_fill_color(*_ink("surface"))
        pdf.rect(0, 0, width_pt, page_h, style="F")

        family = "helvetica"
        if font_regular:
            try:
                pdf.add_font("sheet", "", font_regular)
                # Only claim a bold face when a real one exists — registering the
                # regular file under "B" silently renders **bold** as body text.
                pdf.add_font("sheet", "B", font_bold or font_regular)
                family = "sheet"
            except Exception:  # noqa: BLE001 - unreadable font: fall back to core
                family = "helvetica"
        conv = _strip_emoji if family != "helvetica" else _ascii
        pdf.family = family
        # Bound as a plain attribute holding a function: `staticmethod(...)` on an
        # instance is not callable through the instance on 3.10.
        pdf.conv = conv

        pdf.set_xy(margin, 33)
        pdf.set_font(family, "", 8)
        pdf.set_text_color(*_ink("muted"))
        pdf.cell(0, 10, conv((eyebrow or _EYEBROW).upper()), new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(margin)
        pdf.set_font(family, "B", 23)
        pdf.set_text_color(*_ink("ink"))
        pdf.multi_cell(width_pt - 2 * margin, 28, conv(title or "Report"), align="L")
        if subtitle:
            pdf.set_x(margin)
            pdf.set_font(family, "", 11)
            pdf.set_text_color(*_ink("ink2"))
            pdf.multi_cell(width_pt - 2 * margin, 15, conv(subtitle), align="L")
        if call:
            pdf.ln(12)
            _draw_stance(pdf, call, margin, family, conv, width_pt - 2 * margin)
        pdf.ln(10)
        pdf.set_draw_color(*_ink("grid"))
        pdf.set_line_width(0.6)
        pdf.line(margin, pdf.get_y(), width_pt - margin, pdf.get_y())
        pdf.ln(12)

        tiles = cover_tiles(highlights, markdown)
        if tiles:
            col = (width_pt - 2 * margin) / len(tiles)
            top = pdf.get_y()
            for i, t in enumerate(tiles):
                x = margin + i * col
                if i:
                    pdf.line(x - 4, top, x - 4, top + 52)
                pdf.set_xy(x, top)
                pdf.set_font(family, "", 8)
                pdf.set_text_color(*_ink("muted"))
                pdf.cell(col, 11, conv(t["label"]), align="L")
                pdf.set_xy(x, top + 13)
                pdf.set_font(family, "B", 19)
                pdf.set_text_color(*_ink("ink"))
                pdf.cell(col, 24, conv(t["value"]), align="L")
                if t["note"]:
                    pdf.set_xy(x, top + 38)
                    pdf.set_font(family, "", 8)
                    pdf.set_text_color(*_ink("ink2"))
                    pdf.multi_cell(col - 8, 10, conv(t["note"]), align="L")
            pdf.set_y(top + 60)
            pdf.set_draw_color(*_ink("grid"))
            pdf.line(margin, pdf.get_y(), width_pt - margin, pdf.get_y())
            pdf.ln(10)

        pdf.set_x(margin)
        pdf.set_font(family, "", 11)
        pdf.set_text_color(*_ink("ink"))
        body_html = _markdown_html(markdown).replace("<th>", '<th align="left">')
        pdf.write_html(
            conv(body_html),
            tag_styles=_tag_styles(family),
            table_line_separators=True,
        )
        return pdf, pdf.get_y()

    try:
        # Measure on a deliberately tall single page: whether the sheet FITS one
        # page has to be decided before the page height is chosen, and drawing at
        # the paginating height first answers the wrong question (a short sheet
        # spills, so it never gets refitted).
        doc, end_y = draw(_SINGLE_MAX_H * _PT)
        doc, _ = draw(end_y + 62) if doc.page_no() == 1 else draw(None)
        doc.output(str(pdf_path))
    except Exception:  # noqa: BLE001 - a render must never take the turn down
        return False
    return pdf_path.exists()


# --- rasterise page 1 ----------------------------------------------------------


def rasterize_first_page(pdf_path: str, png_path: str, scale: int | None = None) -> bool:
    """PDF page 1 -> PNG. The cover image for a report of any length."""
    try:
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(pdf_path)
        doc[0].render(scale=(scale or render_scale())).to_pil().save(png_path)
    except Exception:  # noqa: BLE001 - no rasteriser, unreadable PDF: skip the image
        return False
    return Path(png_path).exists()


def page_count(pdf_path: str) -> int:
    try:
        import pypdfium2 as pdfium

        return len(pdfium.PdfDocument(pdf_path))
    except Exception:  # noqa: BLE001
        return 0


# --- orchestration -------------------------------------------------------------


def _unique_stem(out_dir: Path, base: str) -> str:
    """``base``, or the first ``base-2``/``base-3``… nothing has claimed.

    The stem is the title plus a to-the-second timestamp, so two reports on one
    subject rendered in the same second (a scheduled batch, two tool calls in one
    turn) resolve to the same name — and every artifact of the first is silently
    overwritten by the second, including the PDF the user was told was saved.
    """
    stem, n = base, 1
    while any(out_dir.glob(f"{stem}.*")):
        n += 1
        stem = f"{base}-{n}"
    return stem


def render(
    html: str,
    name: str,
    *,
    content: dict[str, str] | None = None,
    theme: str = "",
    output: str = "",
) -> dict[str, str]:
    """Write the HTML, produce the PDF, then rasterise page 1. Never raises.

    ``content`` carries the raw fields so the browser-free renderer can draw the
    same sheet; without it only the Chrome path can run. A failure at any stage
    still leaves the earlier artifacts on disk — losing a finished analysis to a
    rendering problem is the one outcome worth engineering against.

    ``theme`` and ``output`` are this report's own choices, overriding the
    configured defaults. They are arguments rather than environment writes because
    the adapter runs tools in threads: a report that announced its theme by setting
    a process-wide variable coloured whatever another report was drawing at the
    same moment, and got that report's theme back.
    """
    out_dir = reports_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _unique_stem(out_dir, f"{_slug(name)}-{datetime.now():%Y%m%d-%H%M%S}")
    paths: dict[str, str] = {}

    html_path = out_dir / f"{stem}.html"
    pdf = out_dir / f"{stem}.pdf"
    made = False
    mode = renderer_mode()
    chrome = "" if mode == "fpdf2" else chrome_path()
    # The document is drawn under its own theme — see `pdf_theme`. The HTML is
    # REBUILT here when the raw fields are available: taking it pre-built made the
    # document's theme depend on whoever called this, which silently produced a
    # dark PDF for a caller that only asked for a dark cover.
    with use_theme(pdf_theme()):
        if content is not None:
            html = build_html(
                content.get("title", ""), content.get("markdown", ""),
                content.get("highlights", ""), content.get("subtitle", ""),
                content.get("eyebrow", ""), stance=content.get("stance", ""),
            )
        html_path.write_text(html, encoding="utf-8")
        paths["html"] = str(html_path)
        if chrome:
            made = _chrome_pdf(chrome, html_path, pdf)
            if made:
                paths["renderer"] = "chrome"
        if not made and content is not None and mode != "chrome":
            made = _fpdf_pdf(
                pdf,
                content.get("title", ""),
                content.get("markdown", ""),
                content.get("highlights", ""),
                content.get("subtitle", ""),
                content.get("eyebrow", ""),
                content.get("stance", ""),
            )
            if made:
                paths["renderer"] = "fpdf2"
    if not made:
        return paths
    paths["pdf"] = str(pdf)

    png = out_dir / f"{stem}.png"
    pages = page_count(str(pdf))
    # The infographic is the cover whenever there is anything to visualise — the
    # distilled figures read better than the document's first page at any length.
    # It falls back to page 1 if the sheet would be empty or the render fails,
    # since a cover is worth less than the report it introduces.
    wanted = output_mode(output)
    if wanted == "pdf":
        return paths
    if content is not None and _worth_charting(content):
        with use_theme(cover_theme(theme)):
            cover = _build_cover(
                out_dir, stem, content, pages, chrome, attached=wanted != "image"
            )
        if cover and rasterize_first_page(str(cover), str(png)):
            paths["png"] = str(png)
            paths["cover"] = "infographic"
            return paths
    if rasterize_first_page(str(pdf), str(png)):
        paths["png"] = str(png)
        paths["cover"] = "page-1"
    return paths


def _worth_charting(content: dict[str, str]) -> bool:
    """Whether a summary sheet would carry anything the reader cannot get faster.

    Tiles or charts are the whole value of a cover. Without either it is a title
    over some bullets, and page 1 — which for a one-page report IS the report —
    strictly beats it.
    """
    return bool(
        cover_tiles(content.get("highlights", ""), content.get("markdown", ""))
        or extract_series(content.get("markdown", ""))
    )


def _build_cover(
    out_dir: Path, stem: str, content: dict[str, str], pages: int, chrome: str,
    attached: bool = True,
) -> Path | None:
    """Render the summary infographic to its own single-page PDF, or None."""
    info_pdf = out_dir / f"{stem}-cover.pdf"
    title = content.get("title", "")
    markdown = content.get("markdown", "")
    highlights = content.get("highlights", "")
    subtitle = content.get("subtitle", "")
    stance = content.get("stance", "")
    if chrome:
        info_html = out_dir / f"{stem}-cover.html"
        info_html.write_text(
            build_infographic_html(
                title, markdown, highlights, subtitle, pages=pages, attached=attached,
                stance=stance,
            ),
            encoding="utf-8",
        )
        if _chrome_pdf(chrome, info_html, info_pdf, single_page=True):
            return info_pdf
    if _fpdf_infographic(
        info_pdf, title, markdown, highlights, subtitle, pages, attached=attached,
        stance=stance,
    ):
        return info_pdf
    return None


# --- the portfolio-review guard -------------------------------------------------
#
# Routing a performance review to `render_review` was asked for in the system
# prompt AND in this module's docstring, and three runs with both loaded ignored
# both. The last one did not merely retype a figure: it invented the entire
# monthly series (January "+5.2%" against a real +0.10%, April "+1.9%" against a
# real +17.87%), invented the annual track record, and put the trailing-twelve-
# month return back on the sheet as year-to-date. Every number a reader would act
# on was wrong, on a sheet that looked exactly as authoritative as a correct one.
#
# Prompts request; a refusal decides. This is the same shape as the chartless-body
# refusal below — it declines and says precisely what to call instead.

#: Set while `render_review` is driving, so the sheet it builds is not refused.
_rendering_review: ContextVar[bool] = ContextVar("fra_rendering_review", default=False)


@contextmanager
def reviewing():
    """Mark this render as coming FROM `render_review` — see `_looks_like_a_review`."""
    token = _rendering_review.set(True)
    try:
        yield
    finally:
        _rendering_review.reset(token)


#: Set while `render_stock_report` is driving, so the sheet it builds is not refused.
_rendering_stock: ContextVar[bool] = ContextVar("fra_rendering_stock", default=False)


@contextmanager
def analysing():
    """Mark this render as coming FROM `render_stock_report` — see
    `_looks_like_a_stock_report`."""
    token = _rendering_stock.set(True)
    try:
        yield
    finally:
        _rendering_stock.reset(token)


_PORTFOLIO_WORDS = ("portfolio", "my holdings", "account performance")
_REVIEW_WORDS = ("review", "performance", "year-to-date", "year to date", "ytd",
                 "quarter", "monthly", "annual", "recap")
#: Tile labels that mark a sheet as a PERFORMANCE review rather than, say, a
#: portfolio risk or allocation sheet — which stay allowed.
_REVIEW_TILES = ("return", "unrealised", "unrealized", "drawdown", "dividend",
                 "investment gain", "portfolio value", "deposit", "nav")


def _looks_like_a_review(title: str, subtitle: str, highlights: str) -> bool:
    """Whether this call is a portfolio performance review built by hand.

    Deliberately narrow, and requiring all three: the words for a portfolio, the
    words for a review of one, and at least two performance figures in the tiles.
    A single-stock sheet has no portfolio word; a portfolio RISK or ALLOCATION
    sheet has no review word and different tiles. Both keep working.
    """
    head = f"{title} {subtitle}".lower()
    if not any(w in head for w in _PORTFOLIO_WORDS):
        return False
    if not any(w in head for w in _REVIEW_WORDS):
        return False
    labels = " ".join(t["label"] for t in parse_highlights(highlights)).lower()
    return sum(1 for w in _REVIEW_TILES if w in labels) >= 2


_REVIEW_REFUSAL = (
    "NOT RENDERED — this is a portfolio performance review, and building one here "
    "means writing its figures by hand.\n"
    "Call `render_review(period=..., observations=..., stance=...)` instead. It "
    "computes the return, deposits, investment gain, drawdown, monthly path, "
    "holdings, income and concentration, renders the sheet and delivers it. You "
    "supply only `observations` — 3-6 bullets on what the numbers mean.\n"
    "This is refused rather than warned about because a hand-built review last "
    "shipped an entire monthly series that was invented: January '+5.2%' where the "
    "account returned +0.10%, April '+1.9%' where it returned +17.87%, and a "
    "trailing-twelve-month return labelled year-to-date.\n"
    "If you genuinely need a custom portfolio sheet that is NOT a performance "
    "review — an allocation breakdown, a risk profile, a tax-lot summary — title it "
    "for what it is and it will render."
)


#: Words that make a sheet an EARNINGS write-up rather than some other stock page.
_EARNINGS_WORDS = ("earnings", "results", "quarter", "10-q", "q1 ", "q2 ", "q3 ",
                   "q4 ", "fy20", "fiscal")
#: Tile labels holding the figures a quarter is reported in.
_STOCK_TILES = ("revenue", "eps", "earnings per share", "net income", "margin",
                "operating income", "drawdown", "free cash flow")
#: Heads that name more than one company. A comparison or screen legitimately
#: carries revenue and margin tiles for several names, and `render_stock_report`
#: is single-symbol, so refusing those would leave no way to build them at all.
_MULTI_NAME_WORDS = (" vs ", " vs. ", "versus", "comparison", "compare", "screen",
                     "peers", "peer group", "sector", "watchlist", "basket")


def _looks_like_a_stock_report(title: str, subtitle: str, highlights: str) -> bool:
    """Whether this call is a single-stock earnings sheet built by hand.

    Narrow, and requiring all four: an earnings word, no portfolio word (that is
    the review guard's territory), no word naming several companies, and at least
    two of a quarter's figures in the tiles. A valuation or risk sheet has
    different tiles; a peer comparison is exempted outright.
    """
    head = f"{title} {subtitle}".lower()
    if not any(w in head for w in _EARNINGS_WORDS):
        return False
    if any(w in head for w in _PORTFOLIO_WORDS + _MULTI_NAME_WORDS):
        return False
    labels = " ".join(t["label"] for t in parse_highlights(highlights)).lower()
    return sum(1 for w in _STOCK_TILES if w in labels) >= 2


_STOCK_REFUSAL = (
    "NOT RENDERED — this is a single-stock earnings sheet, and building one here "
    "means writing its figures by hand.\n"
    "Call `render_stock_report(symbol=..., observations=..., stance=...)` instead. "
    "It pulls the quarter from SEC 10-Q XBRL and the price side from daily history, "
    "renders the sheet and delivers it. You supply only `observations` — 3-6 "
    "bullets on what the quarter means.\n"
    "This is refused rather than warned about because a hand-built earnings sheet "
    "last shipped revenue of $4.96B where the as-reported figure was $5.29B, called "
    "the stock down 63% in one bullet and 70% in another over a correctly-computed "
    "-66.3%, and charted a net margin LEVEL of 11.8% among year-over-year changes "
    "as the one thing that rose in a bad quarter.\n"
    "If you genuinely need a stock sheet that is NOT an earnings write-up — a "
    "valuation, a risk profile, a peer comparison — title it for what it is and it "
    "will render."
)


# --- Model-facing tool ---------------------------------------------------------


def render_report(
    title: str,
    markdown: str,
    highlights: str = "",
    subtitle: str = "",
    deliver: bool = True,
    allow_prose: bool = False,
    theme: str = "",
    output: str = "",
    stance: str = "",
) -> str:
    """Typeset a summary as a PDF + cover image and send it to the user's channels.

    NOT FOR A PORTFOLIO PERFORMANCE REVIEW — call `render_review(period,
    observations, stance)` instead, which computes the figures and renders in one
    step. Building one here means writing the tiles by hand, and every delivered
    review that carried a wrong number carried one that had been typed rather than
    read.

    NOT FOR A SINGLE-STOCK EARNINGS WRITE-UP either — call
    `render_stock_report(symbol, observations, stance)`, which pulls the quarter
    from SEC 10-Q XBRL and the price side from daily history, each figure labelled
    with the window and basis it is on. A peer comparison, valuation or risk sheet
    is not an earnings write-up and still belongs here.

    Use when the user asks for a report/infographic/PDF/one-pager, or wants
    something "sent"/"pushed" to them as a file rather than as chat text — and for
    a scheduled task's output, where a typeset sheet reads far better on a phone
    than a wall of message text.

    ``title`` is the headline. ``markdown`` is the body — normal markdown works:
    headings, bold, lists, tables, `>` blockquote for a warning callout.

    WRITE BREAKDOWNS AS LISTS OR TABLES, NOT SENTENCES. The image sent alongside
    the PDF is an infographic charted from the SHAPE of this markdown: any run of
    3+ list items each carrying a percentage under one heading, OR a markdown table
    with a label column and one numeric column, becomes a bar chart (up to 3).
    Write ``- **Healthcare:** 28.4% — UNH, NVO, MOH``, or a table whose first
    column names the row and one column holds a single number per row. The same
    numbers inside a paragraph produce no chart. Keep gains and
    losses signed (``+28.6%`` / ``-20.7%``) so they render as up/down bars around
    zero rather than as magnitudes. Short bullets with no percentage become the
    "key observations" block.
    ``highlights`` is optional stat tiles, ONE PER LINE as ``label | value | note``
    (up to 6), e.g. "Adjusted EPS | $1.84 | vs $1.91 consensus". Put the numbers
    that matter there, not in the body — INCLUDING the report's verdict figure
    (fear price, fair value, price target) if it has one, since that is the number
    the reader looks for first. A verdict you instead write as a heading
    (``## Fear Price: $32.00 – $38.00``) is promoted onto the sheet automatically,
    taking the last tile's slot if all six are full — but a tile you write
    yourself keeps the label and note you chose. ``subtitle`` is one line under
    the title.

    ``stance`` is THIS REPORT'S CALL on the stock, badged under the title on both
    the image and the PDF — pass it whenever the report reaches one, since what to
    do about a stock is the reader's first question. Write the verdict, optionally
    a short reason after a pipe: ``"HOLD | trim 50% at $40–$42"``. Buy / accumulate
    / overweight / outperform badge positive, sell / reduce / trim / underweight
    negative, hold / neutral / watch neutral. It is YOUR conclusion, not the
    street's — when analysts disagree with you, put their rating in a
    ``highlights`` tile ("Analyst Consensus | BUY | 12 analysts, mean $47.28") so
    the sheet shows the disagreement instead of hiding it. A stance written as a
    heading (``## Rating: HOLD``) is picked up automatically.
    ``theme`` is "light" or "dark" for the IMAGE — use it when the user asks for a
    dark (or light) one-pager; blank follows the configured default, and the PDF
    stays print-friendly either way. ``output`` is "both" (default), "image" for a
    phone-only send, or "pdf" to skip the cover entirely.
    ``deliver=False`` renders without sending. ``allow_prose=True`` renders a
    report that genuinely has no numbers to chart — without it, a body containing
    no list, table or inline breakdown is REFUSED so you can restructure it.

    Returns where the files went and whether delivery succeeded.
    """
    if not (title or "").strip() and not (markdown or "").strip():
        return "Nothing to render — give at least a title or some body text."

    # Before anything is drawn: a performance review assembled by hand is refused,
    # because its figures were typed rather than read. See `_looks_like_a_review`.
    if not _rendering_review.get() and _looks_like_a_review(title, subtitle, highlights):
        return _REVIEW_REFUSAL
    # And the same for a single-stock earnings sheet, for the same reason: its
    # figures were typed. See `_looks_like_a_stock_report`.
    if (not _rendering_stock.get()
            and _looks_like_a_stock_report(title, subtitle, highlights)):
        return _STOCK_REFUSAL

    # Refuse a chartless body rather than shipping a cover of tiles and text.
    # A narrative report is a legitimate outcome, but it should be a decision:
    # `allow_prose=True` says "I looked, there is genuinely nothing to chart",
    # which is the difference between choosing prose and defaulting into it.
    if not allow_prose and not extract_series(markdown):
        return (
            "NOT RENDERED — nothing in this body can be charted, so the cover image "
            "would be tiles and text with no visual summary.\n"
            "Restructure at least one section into a shape that charts, then call "
            "render_report again:\n"
            "  • a list of 3+ items each with a percentage, under a heading — "
            "`- **Healthcare:** 28.4% — UNH, NVO`\n"
            "  • a markdown table whose first column names the row and one column "
            "holds a single number per row — `| Q1 2026 | $0.19 to $0.23 | +24.0% |`\n"
            "  • or an enumeration inside one line — "
            "`(North America ~65%, Europe ~35%)`\n"
            "Look for a composition, ranking, history, scenario ladder or peer "
            "comparison the report already discusses in prose. If this report "
            "genuinely has no such figures, call render_report again with "
            "allow_prose=True."
        )

    wanted = output_mode(output)
    content = {
        "title": title, "markdown": markdown,
        "highlights": highlights, "subtitle": subtitle, "stance": stance,
    }
    # A per-report theme overrides the configured one for the COVER only; the
    # document keeps its own default so a dark request never produces a PDF that
    # prints as a full page of ink. Passed down as arguments: these are one call's
    # choices, and announcing them in the environment leaked them into whatever
    # other report a concurrent tool call was drawing.
    with use_theme(pdf_theme()):
        document_html = build_html(title, markdown, highlights, subtitle)
    paths = render(document_html, title, content=content, theme=theme, output=wanted)

    lines = []
    if "pdf" not in paths:
        lines.append(
            "Could not produce a PDF (no browser, and the built-in renderer failed) "
            "— the HTML is saved and opens in any browser."
        )
    else:
        pages = page_count(paths["pdf"])
        cover = paths.get("cover")
        if cover == "infographic":
            summarising = f" summarising all {pages} pages" if pages > 1 else ""
            shape = f"; the image is a one-sheet infographic{summarising}."
        elif pages > 1:
            shape = "; the image is page 1."
        else:
            shape = "."
        # Counting pages needs pypdfium2, which is optional — and "Rendered 0
        # page(s)" reads as a failed render of a document that is on disk and about
        # to be delivered. Without the count, say what is known instead.
        extent = f"{pages} page(s)" if pages else "a PDF"
        lines.append(
            f"Rendered {extent} with {paths.get('renderer', '?')}{shape}"
        )
    lines.append("Saved: " + ", ".join(
        f"{k.upper()} {v}" for k, v in paths.items() if k != "renderer"
    ))
    # Say so rather than dropping it. A stance we cannot tone is left off the
    # sheet, and silence there reads as "rendered fine" while the one thing the
    # reader looks for first is missing.
    if (stance or "").strip() and not parse_stance(stance):
        lines.append(
            f"No stance badge: {stance.strip()!r} is not a verdict this can colour. "
            f"Use one of {', '.join(sorted(_STANCE_TONES))} — optionally with a "
            f"reason after a pipe, e.g. \"HOLD | trim 50% at $40\"."
        )

    if deliver:
        from . import channels

        sent: list[str] = []
        failed: list[str] = []
        wanted_keys = {"both": ("png", "pdf"), "image": ("png",), "pdf": ("pdf",)}[wanted]
        for key, caption in (("png", title), ("pdf", f"{title} (PDF)")):
            if key in paths and key in wanted_keys:
                ok, bad = channels.deliver_file(
                    paths[key], caption=caption, full_quality=True
                )
                sent += [c for c in ok if c not in sent]
                failed += [c for c in bad if c not in failed]
        if failed:
            lines.append(f"Delivery failed on: {', '.join(failed)}")
        if sent:
            lines.append(f"Sent to: {', '.join(sorted(set(sent)))}.")
        else:
            lines.append(
                "No channel accepted a file — check TELEGRAM_BOT_TOKEN / "
                "TELEGRAM_CHAT_ID, or read the saved file directly."
            )
    else:
        # Said outright, because silence here reads as success. With deliver=False
        # the result was "Rendered … Saved: …" and nothing else, which is
        # indistinguishable from a delivered report — so a report that was never
        # sent could be reported to the user as sent.
        lines.append(
            "NOT SENT — deliver=False, so this is on disk only and the user has "
            "NOT received it. Call again with deliver=True if they asked for it."
        )
    return "\n".join(lines)


REPORT_TOOLS = [render_report]


# --- infographic cover (multi-page reports) -------------------------------------
#
# Page 1 is the right cover for a one-page sheet and a poor one for a five-page
# report: it shows the masthead and whatever happened to fit, which is the least
# informative slice of the document. So a multi-page report gets a purpose-built
# summary sheet instead — the headline figures plus whatever series the body
# actually contains, charted.
#
# Extraction is deterministic. The model already decided what matters when it
# wrote the report; this reads the shapes it produced rather than asking a second
# model what to draw, so the cover costs nothing and cannot invent a number.

#: What counts as a MINUS in front of a figure. Writers reach for four glyphs —
#: hyphen, unicode minus, en dash, em dash — and a delivered "EPS BEAT/MISS
#: TRACK RECORD" showed four misses as beats because the model wrote "–8.7%"
#: with an en dash.
#:
#: The en dash is also a RANGE separator ("10–20%", "$68–$76"), so position
#: decides: a dash with a digit before it joins two numbers, a dash with
#: nothing numeric before it negates the one after. That reading is the same
#: one a person makes, and it needs no vocabulary of dashes.
_MINUS = r"(?:(?<![\d.,])[+\-\u2212\u2013\u2014])?"
#: The glyphs that mean "negative" once one has been matched.
_MINUS_CHARS = "-\u2212\u2013\u2014"

#: A percentage anywhere in a list item, sign preserved — INCLUDING the unicode
#: minus. `_NUM_RE` learned that for table cells and these four did not, so a
#: figure written "−29.5%" parsed as +29.5 and a maximum drawdown charted as a
#: GAIN. The en and em dashes stay out: they separate a range ("$68–$76"), and
#: reading one as a sign would break every range on a sheet.
_PCT_RE = re.compile(rf"({_MINUS}\d+(?:\.\d+)?)\s*%")
def _pct_float(text: str) -> float:
    """A captured percentage as a number, unicode minus included.

    `float()` does not accept U+2212, so widening the patterns to match it
    without widening this raised ValueError on the first negative it saw.
    """
    for glyph in _MINUS_CHARS[1:]:
        text = text.replace(glyph, "-")
    return float(text)


#: A parenthetical aside — context hung off a figure, never the figure itself.
_ASIDE_RE = re.compile(r"\s*\([^)]*\)")
#: Where a label stops: a dash/colon separator, or the figure itself.
_LABEL_SPLIT = re.compile(rf"\s*[–—:|]\s*|\s+(?={_MINUS}\d+(?:\.\d+)?\s*%)")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
_BOLD_HEADING_RE = re.compile(r"^\s{0,3}\*\*(.+?)\*\*:?\s*$")
_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.+)$")

_MAX_SERIES = 3
_MAX_ITEMS = 8
_MIN_ITEMS = 3
#: Longest a chart label can be before it stops being a category. Real ones are
#: short ("Healthcare", "VOO", "Communication Services" at 22); a prose observation
#: that happens to quote a percentage is far longer, and charting it produces a row
#: of ellipsised sentences — observed in a live run.
_MAX_LABEL = 24
#: A label left dangling on a connective is a sentence the splitter cut mid-phrase
#: — "High volatility of 29.3% means..." yields "High volatility of", which is
#: short enough to pass the length rule and means nothing on an axis.
_DANGLING_RE = re.compile(
    r"\b(?:of|at|by|to|in|on|is|are|was|were|from|with|and|or|the|an?|for|near|"
    r"about|around|over|under|than|that|which|means?|reached|hit)$",
    re.IGNORECASE,
)


#: Headings that introduce COMMENTARY, not a breakdown. Bullets under them each
#: quote whatever figure their sentence is about, so charting the run puts a
#: month's return, a position weight and a two-month total on one axis — four
#: bars that share nothing but the % sign. (Observed: "April carried the year
#: 17.9 / Concentration risk 30.4 / Crypto hedge modest 5.5".) It also collided
#: with the sheet's own "Key observations" block, printing that heading twice.
#:
#: Matched on the heading rather than inferred from the values, because the
#: numbers alone cannot say whether four percentages are comparable — but a
#: section called "Key observations" has already said it is not a breakdown.
_PROSE_HEADINGS = {
    "key observations", "observations", "key takeaways", "takeaways",
    "key points", "notes", "summary", "bottom line", "conclusion",
    "commentary", "analysis", "what stands out", "highlights",
    "recommendation", "recommendations",
}


def _is_prose_heading(title: str) -> bool:
    return re.sub(r"[^a-z ]", "", (title or "").lower()).strip() in _PROSE_HEADINGS


#: Words that state a direction, so a label carrying one has already said which
#: way its own number goes. Stems, matched as prefixes, so "compressed",
#: "compression" and "compressing" all count once.
_FELL = ("compress", "contract", "declin", "decreas", "fell", "fall", "drop",
         "shrank", "shrink", "narrow", "weaken", "slump", "sank", "lower",
         "loss", "lost", "miss")
_ROSE = ("expand", "grew", "grow", "rose", "rise", "rising", "gain", "climb",
         "improv", "widen", "strengthen", "surge", "jump", "beat", "higher")
_WORD_RE = re.compile(r"[a-z]+")


def _direction(label: str) -> int:
    """+1 / -1 when a label names a direction, 0 when it does not or says both.

    A label holding both ("declining growth") has not settled the question, so it
    is left alone — this only ever speaks when the label is unambiguous.
    """
    words = _WORD_RE.findall((label or "").lower())
    fell = any(w.startswith(_FELL) for w in words)
    rose = any(w.startswith(_ROSE) for w in words)
    return 0 if fell == rose else (-1 if fell else 1)


def _is_coherent(label: str, value: float) -> bool:
    """Whether a bar's sign agrees with the direction its own label states.

    A LEVEL written into a chart of CHANGES is the failure: "Net margin compressed
    +11.8%" is the margin itself, not its move, and charted beside a -25.5% EPS
    change it drew as the one thing that went up in a bad quarter. English carries
    the sign here and the digits do not — "fell 3%" negates, "fell to 11.8%" does
    not — so no rule over the number alone can separate them.

    Narrow on purpose. It reads the label's own claim and only ever drops a bar
    that contradicts it; a label naming no direction is never touched, so the
    failure mode is missing a bad bar, never removing a good one.
    """
    direction = _direction(label)
    return direction == 0 or value == 0 or (value > 0) == (direction > 0)


def _coherent_series(series: dict[str, Any]) -> dict[str, Any] | None:
    """`series` with self-contradicting bars removed, or None if too little is left.

    Only SIGNED series are checked. Where the sign is not being displayed as
    meaning anything, a directional word in a label is describing the category
    rather than claiming which way the bar points.
    """
    if not series.get("signed"):
        return series
    items = [(l, v) for l, v in series["items"] if _is_coherent(l, v)]
    if len(items) == len(series["items"]):
        return series
    if len(items) < _MIN_ITEMS:
        return None
    return {**series, "items": items}


def _distinct_enough(items: list[tuple[str, float]]) -> bool:
    """Whether these labels form a category axis at all.

    A run of bullets written ``- **Problem:** ...`` all reduce to the same label,
    because the split takes the text before the colon — and that is where the
    label lives in the shape this was built for (``- **Healthcare:** 28.4%``).
    The result charts six bars reading "Problem", which looks like six categories
    and names none of them: worse than no chart, since the reader cannot tell
    which figure belongs to what.

    Rejecting is the right repair rather than guessing a better label. The bullets
    then fall through to Key Observations, where the full sentence is kept and
    reads correctly.
    """
    return len({label for label, _v in items}) >= _MIN_ITEMS


def _is_label(text: str) -> bool:
    """Whether a cleaned label reads as a category rather than a cut sentence.

    The dangling check only applies to multi-word labels: a one-word label is a
    name even when the word happens to be "A", while "High volatility of" is a
    sentence the splitter cut mid-phrase.
    """
    if not text or len(text) > _MAX_LABEL:
        return False
    return " " not in text.strip() or not _DANGLING_RE.search(text)


def _clean_label(text: str) -> str:
    """The label a list item is charted under. NOT truncated — the caller needs the
    real length to tell a category from a sentence fragment."""
    text = re.sub(r"\*\*|__|`", "", text).strip()
    return _LABEL_SPLIT.split(text, maxsplit=1)[0].strip(" .:–—-")


#: A single number in a table cell: currency, percent or bare, sign preserved.
#
# The dash is ESCAPED. Written `[+-−]` it is a range operator, not a literal, so
# the class spans U+002B to U+2212 — every digit included. The sign group then
# matched the leading digit of any unsigned multi-digit number and the rest still
# parsed, so `74.1%` charted as `4.1%`: a plausible figure, silently wrong, on a
# sheet whose whole promise is that it cannot invent a number. Signed cells
# (`+11.4%`) and thousands-separated ones happened to survive, which is why this
# stood for so long.
_NUM_RE = re.compile(rf"({_MINUS})\s*\$?\s*(\d[\d,]*(?:\.\d+)?)\s*(%?)")
_TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


#: A magnitude suffix immediately after the number: 115.2B, €38.6bn, 980M, 1.2T.
#: Anchored to the number's end so a stray capital in prose ("$4.98 Beat") is not
#: read as billions.
_SCALE_RE = re.compile(r"\d\s*(k|m|bn?|t)\b", re.IGNORECASE)

#: ``(2.30)`` — accounting notation for a negative, and the default in anything
#: transcribed from a financial statement. Read as positive it does not merely
#: misstate the magnitude, it points the bar the wrong way: a cash burn charts as
#: cash generated. Deliberately narrow — the parentheses must wrap the number and
#: nothing else, so "11.65x (trailing)" stays positive and so does
#: "(4.2% of total)".
_ACCOUNTING_NEG_RE = re.compile(r"^\$?\s*\(\s*[-−]?\s*\$?\s*[\d,]+(?:\.\d+)?\s*[%x]?\s*\)$",
                                re.IGNORECASE)


def _cell_number(cell: str) -> tuple[float, str] | None:
    """``(value, unit)`` when a cell holds exactly one number.

    Exactly one on purpose: a cell like ``$0.19 → $0.23`` is a transition, not a
    measure, and charting either end of it would be arbitrary.
    """
    text = cell.strip()
    if not text:
        return None
    hits = _NUM_RE.findall(text)
    if len(hits) != 1:
        return None
    sign, digits, pct = hits[0]
    try:
        value = float(digits.replace(",", ""))
    except ValueError:
        return None
    # `sign and` first: an empty string is `in` every string, so without it a
    # cell with no sign at all negated itself.
    if (sign and sign in _MINUS_CHARS) or _ACCOUNTING_NEG_RE.match(text):
        value = -value
    unit = "%" if pct else ("$" if "$" in text else "")
    # The MAGNITUDE travels with the unit. Dropped, "$115.2B" charted as "$115.20"
    # — a figure a billion times smaller than the source, and one that reads as
    # perfectly ordinary. The value itself stays as written, so it remains a token
    # of the source; only the label gains the suffix back.
    scale = _SCALE_RE.search(text)
    if scale and not pct:
        # Normalised to ONE letter: "bn" left a two-character unit that the
        # formatter's single-character check could not see, so the suffix was
        # dropped again one layer further on.
        unit += scale.group(1).upper()[0]
    return value, unit


def _table_series(rows: list[list[str]], heading: str) -> dict[str, Any] | None:
    """Turn a markdown table into a series: first column labels, best numeric column.

    Reports carry their most chartable data in tables — a beat history, a
    scenario ladder, a metric comparison — and ignoring them was why an
    earnings preview full of numbers produced a cover with no charts at all.

    The chosen column is the one where the most rows hold exactly one number,
    preferring percentages: a mixed column of ``$13.58`` and ``+15.3%`` cannot
    share an axis.
    """
    if len(rows) < _MIN_ITEMS + 1:  # header + rows
        return None
    header, body = rows[0], rows[1:]
    best: tuple[int, int, str] | None = None  # (score, column, unit)
    for col in range(1, len(header)):
        parsed = [_cell_number(r[col]) for r in body if col < len(r)]
        good = [p for p in parsed if p]
        if len(good) < _MIN_ITEMS:
            continue
        units = {u for _v, u in good}
        if len(units) > 1:
            continue  # mixed units cannot share one axis
        unit = good[0][1]
        score = len(good) * 2 + (1 if unit == "%" else 0)
        if best is None or score > best[0]:
            best = (score, col, unit)
    if best is None:
        return None
    _score, col, unit = best
    items: list[tuple[str, float]] = []
    signed = False
    for row in body:
        if col >= len(row):
            continue
        parsed = _cell_number(row[col])
        if not parsed:
            continue
        # A table's first column IS the label, so no prose guard — but strip the
        # parenthetical asides reports like to hang off them.
        label = re.sub(r"\s*\(.*?\)", "", re.sub(r"\*\*|__|`", "", row[0])).strip()
        if not label:
            continue
        value, _unit = parsed
        # A negative value IS a signed one, however it was written. Keyed only off
        # a leading +/- glyph, an accounting negative left the series "unsigned",
        # which drew a loss in the gain colour and dropped the minus from its label.
        signed = signed or value < 0 or bool(re.match(r"\s*[+\-−]", row[col].strip()))
        items.append((label[:_MAX_LABEL], value))
    if len(items) < _MIN_ITEMS or not _distinct_enough(items):
        return None
    return {
        "title": heading or (header[0].strip() or "Breakdown"),
        "signed": signed,
        "unit": unit,
        "items": items[:_MAX_ITEMS],
    }


#: ``North America ~65%``: a capitalised label immediately followed by a percentage.
_PAIR_RE = re.compile(
    rf"([A-Z][A-Za-z][A-Za-z &/'\-]{{1,26}}?)\s*[:~≈]?\s*({_MINUS}\d+(?:\.\d+)?)\s*%"
)
#: ``65% North America``: the same split written the other way round. The label
#: must end on punctuation or a conjunction, so ``29.3%) means a wide outcome``
#: yields nothing rather than a label of "means a wide outcome".
_PAIR_REV_RE = re.compile(
    rf"({_MINUS}\d+(?:\.\d+)?)\s*%\s+([A-Z][A-Za-z][A-Za-z &/'\-]{{1,26}}?)"
    r"(?=\s*[,;)]|\s+(?:and|or)\b|$)"
)
#: An enumeration inside one sentence needs only two members to be worth a chart —
#: a 65/35 split is a comparison, not a lone statistic.
_MIN_INLINE = 2


def _inline_series(line: str, heading: str) -> dict[str, Any] | None:
    """A breakdown written inside a sentence: ``(North America ~65%, Europe ~35%)``.

    Narrative reports bury their only real data this way, and requiring a list or
    a table meant such a report charted nothing at all. Two or more pairs on one
    line is the guard: a single ``5-10% move higher`` is prose, and only an actual
    enumeration clears it.
    """
    text = re.sub(r"\*\*|__|`", "", line)
    pairs = _PAIR_RE.findall(text)
    if len(pairs) < _MIN_INLINE:
        pairs = [(label, digits) for digits, label in _PAIR_REV_RE.findall(text)]
    if len(pairs) < _MIN_INLINE:
        return None
    items: list[tuple[str, float]] = []
    for label, digits in pairs:
        clean = label.strip(" .:-–—,").strip()
        # Drop leading connectives the regex may have swallowed.
        clean = re.sub(r"^(?:and|or|with|plus|vs\.?)\s+", "", clean, flags=re.I).strip()
        # And the trailing one the pattern swallowed on its way to the figure:
        # "Healthcare is 27.9%" yields the label "Healthcare is", which charts as a
        # sentence cut mid-phrase. The list path already refuses those via
        # `_is_label`; this path had no equivalent. (Observed on a delivered sheet:
        # bars reading "Healthcare is" and "Consumer Cyclical at".)
        clean = _DANGLING_RE.sub("", clean).strip()
        if len(clean) < 2 or len(clean) > _MAX_LABEL:
            return None
        items.append((clean, _pct_float(digits)))
    title = re.split(r"\s*[(:]", text, maxsplit=1)[0].strip(" -–—•*")
    # A heading that is ALSO one of its own bars is not a heading. "Discount rate:
    # 9%, Terminal growth: 2.5%" charted two DCF assumptions against each other
    # under the title "Discount rate" — the first label doing double duty because
    # the sentence began with it, which is the tell that this is a sentence and not
    # a breakdown. A real one names the whole ("Split by region" over North America
    # and Europe; "Revenue" over QoQ and YTD) and never repeats a part.
    # Compared as they are: both come from the same substring of the same
    # sentence, so they cannot differ by case or punctuation, and normalising
    # would be a branch no input could reach.
    if title and any(title == label for label, _v in items):
        return None
    return {
        "title": (title[:48] if len(title) >= 4 else heading) or "Breakdown",
        "signed": False,
        "unit": "%",
        "items": items[:_MAX_ITEMS],
    }


def extract_series(markdown: str) -> list[dict[str, Any]]:
    """Charted series found in the body: ``[{title, signed, items:[(label, value)]}]``.

    A run of list items under one heading counts as a series when at least three
    of them carry a percentage — the shape of "top holdings", "sector exposure",
    "movers". Signed values (``+28.5%`` / ``-20.7%``) mark it diverging, so gains
    and losses read as opposites rather than as magnitudes.
    """
    series: list[dict[str, Any]] = []
    inline: list[dict[str, Any]] = []
    heading = ""
    items: list[tuple[str, float]] = []
    signed = False
    table: list[list[str]] = []

    def flush_table() -> None:
        nonlocal table
        if table:
            found = _table_series(table, heading)
            if found:
                series.append(found)
        table = []

    def flush() -> None:
        nonlocal items, signed
        # Prose, not a series. ANY over-long label disqualifies the run: if a label
        # has to be cut to fit an axis it was never a category, and a "majority"
        # rule let through a bear-case list that charted a drawdown, a geographic
        # share and a volatility figure together on one axis.
        if (len(items) >= _MIN_ITEMS and all(_is_label(l) for l, _v in items)
                and _distinct_enough(items)):
            series.append({
                "title": heading or "Breakdown",
                "signed": signed,
                "unit": "%",
                "items": items[:_MAX_ITEMS],
            })
        items, signed = [], False

    for raw in (markdown or "").splitlines():
        line = raw.rstrip()
        row = _TABLE_ROW_RE.match(line)
        if row and not _TABLE_SEP_RE.match(line):
            flush()
            table.append([c.strip() for c in row.group(1).split("|")])
            continue
        if row:  # the |---|---| separator
            continue
        flush_table()

        head = _HEADING_RE.match(line) or _BOLD_HEADING_RE.match(line)
        if head:
            flush()
            heading = re.sub(r"\*\*|__|`", "", head.group(1)).strip()
            continue
        item = _ITEM_RE.match(line)
        if not item:
            if not line.strip():
                continue
            flush()  # prose ends a run
            continue
        text = item.group(1)
        if not _PCT_RE.search(text):
            continue
        # An EMBEDDED breakdown is tried first, and on the whole line, because it
        # habitually lives inside the parentheses: "(North America ~65%, Europe
        # ~35%)". Stripping asides before this point erased those breakdowns
        # entirely — the fix below, applied one step too early.
        embedded = _inline_series(text, heading)
        if embedded:
            # Filtered on the SOURCE heading, not the series title: an inline
            # breakdown names itself after the sentence it came from, so a
            # commentary section slipped past the prose-heading check that already
            # governs list and table series. It shipped a chart headed
            # "Concentration risk is material and rising. VOO a" — a sentence, cut
            # at the title limit — with bars labelled "VOO alone" and "AMZN
            # together represent".
            if not _is_prose_heading(heading):
                inline.append(embedded)
            continue
        # For a SINGLE measure, though, an aside cannot be it. Searched over the
        # whole line, "Sharpe ratio: 1.06 (risk-free rate = 0%)" charts as 0.0% —
        # the first percent sign belongs to the aside, and a ratio is not a
        # percentage at all, so the bar reads as the metric while showing a number
        # from its own footnote. (Observed on a delivered risk profile.)
        found = _PCT_RE.search(_ASIDE_RE.sub("", text))
        if not found:
            continue
        label = _clean_label(text)
        if not label:
            continue
        value = _pct_float(found.group(1))
        signed = signed or found.group(1)[0] in "+" + _MINUS_CHARS
        items.append((label, value))
    flush()
    flush_table()
    # Inline breakdowns are the weakest signal, so they only fill space a list or
    # table did not claim.
    for extra in inline:
        if len(series) >= _MAX_SERIES:
            break
        if extra["title"] not in {s["title"] for s in series}:
            series.append(extra)
    # ONE chart per heading. A section holding both a table and a run of bullets
    # yields two series with the same title, and the sheet then prints that heading
    # twice over two different charts — observed as "KEY BUSINESS TRENDS" above a
    # revenue table and again above three unrelated percentages. The first is kept
    # because a table is the more structured of the two; the inline path already
    # deduped this way, and only list-against-table was left out.
    kept: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for s in series:
        if s["title"] in seen_titles:
            continue
        seen_titles.add(s["title"])
        kept.append(s)
    # Applied last so they catch every path into `series` — list runs, tables and
    # inline breakdowns alike.
    checked = [_coherent_series(s) for s in kept if not _is_prose_heading(s["title"])]
    return [s for s in checked if s][:_MAX_SERIES]


_INFO_CSS = """
:root{{{vars}}}
@page{{size:{w}px {h}px;margin:0}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{w}px;background:var(--surface);color:var(--ink);
 font-family:system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}}
.sheet{{padding:48px 52px 40px}}
.eyebrow{{font-size:15px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-weight:600}}
h1{{font-size:46px;line-height:1.06;font-weight:650;letter-spacing:-.025em;margin-top:12px}}
.sub{{font-size:18px;color:var(--ink2);margin-top:10px}}
{stance_css}
.rule{{height:1px;background:var(--grid);margin:28px 0}}
.tiles{{display:grid;gap:2px;background:var(--grid)}}
.tile{{background:var(--surface);padding:20px 18px}}
.tile .lab{{font-size:13px;color:var(--muted);font-weight:500}}
.tile .val{{font-size:34px;font-weight:650;margin-top:6px;letter-spacing:-.025em;line-height:1.05;white-space:nowrap}}
.tile .val.sm{{font-size:25px}}
.tile .val.xs{{font-size:18px;white-space:normal}}
.tile .note{{font-size:13.5px;color:var(--ink2);margin-top:7px;line-height:1.4}}
.charts{{display:grid;grid-template-columns:repeat({cols},1fr);gap:34px 40px;margin-top:32px}}
.card.wide{{grid-column:1 / -1}}
.card h2{{font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);
 font-weight:650;margin-bottom:18px}}
/* The value column grows past its floor rather than being a hard 72px: a money
   series keeps its cents now, and "+$12,480.50" has no break opportunity, so a
   fixed track let it overflow leftward into the bar it labels. */
.row{{display:grid;grid-template-columns:164px 1fr minmax(72px,max-content);align-items:center;gap:12px;margin-bottom:11px}}
.row .k{{font-size:13.5px;color:var(--ink);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.row .v{{font-size:14.5px;font-weight:600;text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}}
.track{{height:22px;background:var(--track);border-radius:4px;position:relative}}
.fill{{height:22px;background:var(--pos);border-radius:0 4px 4px 0}}
.dv{{position:relative;height:22px;background:var(--track);border-radius:4px}}
.dv .zero{{position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:var(--rule)}}
.dv .bar{{position:absolute;top:0;height:22px}}
.dv .bar.p{{left:50%;background:var(--pos);border-radius:0 4px 4px 0}}
.dv .bar.n{{right:50%;background:var(--neg);border-radius:4px 0 0 4px}}
.legend{{display:flex;gap:18px;font-size:13px;color:var(--ink2);margin-top:14px}}
.sw{{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px}}
.notes{{margin-top:34px;border-top:1px solid var(--grid);padding-top:22px}}
.notes h2{{font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);
 font-weight:650;margin-bottom:14px}}
.notes li{{font-size:15.5px;line-height:1.55;color:var(--ink);margin:0 0 9px 18px}}
footer{{padding:20px 52px 30px;background:var(--page);font-size:12.5px;color:var(--muted);line-height:1.7}}
footer .disc{{margin-top:10px;border-top:1px solid var(--grid);padding-top:10px}}
"""

_INFO_DOC = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{title}</title><style>{css}</style></head><body>
<div class="sheet">
  <div class="eyebrow">{eyebrow}</div>
  <h1>{title}</h1>
  {subtitle}
  {stance}
  <div class="rule"></div>
  {tiles}
  {charts}
  {notes}
</div>
<footer>{footer}<div class="disc">{disclaimer}</div></footer>
<script>document.title="__FRA_H:"+document.documentElement.scrollHeight;</script>
</body></html>"""


def _fmt_value(value: float, unit: str, signed: bool) -> str:
    """Label a bar in its own unit. A dollar P/L rendered as "180.0%" is a lie.

    Money keeps its cents, and drops them only when they are literally ".00".
    Rounding to whole dollars flattened an EPS table — the most ordinary
    dollar-denominated shape a report has — into uselessness: $1.84 became "$2",
    and $0.19 became "$0", which is not merely imprecise but says the opposite of
    what the source did. A magnitude threshold instead would have mixed "$180" and
    "$1.84" inside one chart, so the rule is the same at every size: never show a
    cent the source did not have, never drop one it did.
    """
    sign = "+" if signed and value > 0 else ("-" if value < 0 else "")
    # A trailing K/M/B/T is the magnitude the source wrote; it goes back on the
    # label, and `unit` reduces to the currency or percent it was before.
    scale = unit[-1] if unit[-1:] in ("K", "M", "B", "T") else ""
    unit = unit[:-1] if scale else unit
    if unit == "$":
        text = f"{abs(value):,.2f}".removesuffix(".00")
        return f"{sign}${text}{scale}"
    if unit == "%":
        return f"{sign}{abs(value):.1f}%"
    return f"{sign}{abs(value):,.2f}{scale}"


def _is_diverging(series: dict[str, Any]) -> bool:
    """Zero-centred only when values actually straddle zero.

    A run of five positive surprises centred on zero wastes half the width and
    squeezes +0.3% into an invisible sliver. Same sign throughout is a magnitude
    comparison that happens to carry a sign.
    """
    values = [v for _l, v in series["items"]]
    return bool(series.get("signed")) and any(v < 0 for v in values) and any(v > 0 for v in values)


def _bars_html(series: dict[str, Any]) -> str:
    pal = _palette()
    items = series["items"]
    rows = []
    if _is_diverging(series):
        span = max((abs(v) for _l, v in items), default=1) or 1
        for label, value in items:
            width = min(abs(value) / span * 50.0, 50.0)
            side = "n" if value < 0 else "p"
            style = f"{'right' if value < 0 else 'left'}:50%;width:{width:.1f}%"
            rows.append(
                f'<div class="row"><div class="k">{_html.escape(label)}</div>'
                f'<div class="dv"><div class="zero"></div>'
                f'<div class="bar {side}" style="{style}"></div></div>'
                f'<div class="v" style="color:{pal["neg"] if value < 0 else pal["pos"]}">'
                f'{_fmt_value(value, series.get("unit", "%"), True)}</div></div>'
            )
        legend = (f'<div class="legend"><span><span class="sw" style="background:'
                  f'{pal["pos"]}"></span>Up</span><span><span class="sw" '
                  f'style="background:{pal["neg"]}"></span>Down</span></div>')
    else:
        signed = bool(series.get("signed"))
        top = max((abs(v) for _l, v in items), default=1) or 1
        negative = all(v <= 0 for _l, v in items) and signed
        colour = pal["neg"] if negative else pal["pos"]
        for label, value in items:
            rows.append(
                f'<div class="row"><div class="k">{_html.escape(label)}</div>'
                f'<div class="track"><div class="fill" style="width:'
                f'{abs(value) / top * 100:.1f}%;background:{colour}"></div></div>'
                f'<div class="v"'
                + (f' style="color:{colour}"' if signed else "")
                + f'>{_fmt_value(value, series.get("unit", "%"), signed)}</div></div>'
            )
        legend = ""
    return (f'<div class="card"><h2>{_html.escape(series["title"])}</h2>'
            + "".join(rows) + legend + "</div>")


#: Bullets shorter than this are labels, not observations.
_MIN_NOTE = 24
#: Past this a "bullet" is a paragraph, and the cover is a summary sheet.
_MAX_NOTE = 420
#: How much of one reaches the cover; the rest is in the document. Sized from real
#: output rather than guessed: the agent's observations run 185-215 characters, so
#: 190 clipped almost every one of them a few words from the end — "Top-5
#: concentration is…" cut immediately before its number. All five fit at this.
_NOTE_DISPLAY = 240


def extract_notes(markdown: str, used_titles: set[str]) -> list[str]:
    """Bullets from sections the charts did not consume — the observations that
    are prose rather than numbers.

    Long ones are CLIPPED rather than dropped. A hard 150-character ceiling
    excluded every real observation the agent writes: "April carried the year: a
    +17.87% month against seven others near flat or negative…" runs to 218, and
    the analysis reached the PDF while the cover — the thing actually read on a
    phone — carried none of it.
    """
    picks: list[str] = []
    heading = ""
    for raw in (markdown or "").splitlines():
        head = _HEADING_RE.match(raw) or _BOLD_HEADING_RE.match(raw)
        if head:
            heading = re.sub(r"\*\*|__|`", "", head.group(1)).strip()
            continue
        item = _ITEM_RE.match(raw)
        if not item or heading in used_titles:
            continue
        text = re.sub(r"\*\*|__|`", "", item.group(1)).strip()
        if _MIN_NOTE <= len(text) <= _MAX_NOTE and not _PCT_RE.match(text):
            picks.append(_clip(text, _NOTE_DISPLAY))
    return picks[:5]


#: A figure a tile can carry: money, a plain number, a percentage, a multiple.
_FIGURE = r"[+\-−]?\s*\$?\s*\d[\d,]*(?:\.\d+)?\s*(?:%|[xX]\b)?"
#: A heading whose text is ``Label: figure`` — the figure optionally a range, and
#: optionally trailed by a parenthetical that becomes the tile's note. Anchored at
#: both ends on purpose: "Coverage: 12 analysts" is a sentence, not a tile value,
#: and only the anchor tells them apart.
_HEADING_FIGURE_RE = re.compile(
    rf"^(?P<label>[^:]{{2,32}}):\s*"
    rf"(?P<value>{_FIGURE}(?:\s*(?:[-–—/]|to)\s*{_FIGURE})*)"
    rf"(?:\s*\((?P<note>[^)]{{1,48}})\))?$"
)

#: How many heading figures may be promoted. A report that prices several
#: scenarios as headings should not be able to evict every tile the model chose.
_MAX_PROMOTED = 2

#: Tiles past six stop being scannable — the cap ``parse_highlights`` already
#: applies, restated here because promotion has to respect the same budget.
_MAX_TILES = 6


def heading_figures(markdown: str) -> list[dict[str, str]]:
    """Tiles promoted from headings shaped ``Label: figure``.

    A model puts a number in a *heading* only when that number is the section's
    answer — a fear price, a fair value, a price target. Those are the figures a
    reader wants first, and they were the one place the cover could not reach:
    tiles come from ``highlights``, charts from list and table shapes,
    observations from bullets. A verdict written as a heading matched none of
    them, so it landed on page 2 of the PDF and nowhere on the image at all
    (observed: an NVO report whose entire second half priced a
    ``Fear Price: $32.00 – $38.00`` the cover never mentioned).

    Deterministic like the rest of this section: it re-reads what the model
    already wrote rather than asking a second model what mattered, so it costs
    nothing and cannot invent a number.
    """
    out: list[dict[str, str]] = []
    for raw in (markdown or "").splitlines():
        head = _HEADING_RE.match(raw) or _BOLD_HEADING_RE.match(raw)
        if not head:
            continue
        text = re.sub(r"\*\*|__|`", "", head.group(1)).strip()
        found = _HEADING_FIGURE_RE.match(text)
        if not found:
            continue
        out.append({
            "label": found.group("label").strip(),
            "value": re.sub(r"\s+", " ", found.group("value")).strip(),
            "note": (found.group("note") or "").strip(),
        })
        if len(out) == _MAX_PROMOTED:
            break
    return out


def cover_tiles(highlights: str, markdown: str = "") -> list[dict[str, str]]:
    """The stat tiles a sheet shows: the model's ``highlights``, plus any verdict
    figure it wrote as a heading instead of as a tile.

    A promoted figure DISPLACES the last highlight when all six slots are taken.
    The model orders highlights most-important-first, so the last one is its own
    least-important choice, whereas a figure it promoted into a heading is the
    report's conclusion — dropping the conclusion to keep a sixth context figure
    is the failure this exists to fix.
    """
    tiles = parse_highlights(highlights)
    seen = {t["label"].strip().lower() for t in tiles if t["label"].strip()}
    promoted: list[dict[str, str]] = []
    for extra in heading_figures(markdown):
        key = extra["label"].lower()
        if key in seen:  # the model already gave this figure a tile of its own
            continue
        seen.add(key)
        promoted.append(extra)
    if not promoted:
        return tiles
    # Trim the MODEL'S tiles, never what was already promoted. Dropping the last
    # entry one promotion at a time instead made a second verdict evict the first,
    # so a report pricing two scenarios showed only the later one.
    if len(tiles) + len(promoted) > _MAX_TILES:
        tiles = tiles[: _MAX_TILES - len(promoted)]
    return tiles + promoted


# --- the call: buy, sell or hold ------------------------------------------------
#
# A stance is the one thing on the sheet that is not a figure, so it does not
# belong in a tile — tile values are numbers, which is why `_TILE_STEPS` exists at
# all. It gets its own badge: the reader's first question about a stock is what to
# do about it, and a sheet that answers it in prose on page 3 has buried the lede.
#
# Tone, not verdict. The badge colours the stance and states it; it does not
# decide it. Whatever the report concluded is what shows.

#: Stance vocabulary → tone. Ordered longest-phrase-first at use, so "strong buy"
#: is not read as "buy" with "strong" left dangling in the note.
_STANCE_TONES = {
    "strong buy": "pos", "buy": "pos", "accumulate": "pos", "add": "pos",
    "overweight": "pos", "outperform": "pos", "long": "pos",
    "strong sell": "neg", "sell": "neg", "reduce": "neg", "trim": "neg",
    "underweight": "neg", "underperform": "neg", "avoid": "neg", "exit": "neg",
    "hold": "hold", "neutral": "hold", "market perform": "hold",
    "equal weight": "hold", "market weight": "hold", "watch": "hold", "wait": "hold",
}

_STANCE_RE = re.compile(
    r"^(?P<label>"
    + "|".join(re.escape(p) for p in sorted(_STANCE_TONES, key=len, reverse=True))
    + r")\b[\s:,|(—–-]*(?P<note>.*)$",
    re.IGNORECASE,
)

#: Headings that introduce a stance. Same heading-only rule the figure promotion
#: uses, and for the same reason: the NVO report carried "Rating: BUY" as a BULLET
#: under Analyst Consensus — the street's view, not its own, which it argued was
#: stale before concluding HOLD. Reading bullets would have badged the sheet with
#: the opinion the report existed to disagree with.
_STANCE_HEADING_RE = re.compile(
    r"^(?:rating|recommendation|verdict|stance|call|position|action)\s*:\s*(?P<rest>.+)$",
    re.IGNORECASE,
)


def _clip(text: str, limit: int) -> str:
    """Cut at a word boundary, marked. A hard slice ended a stance reason on a
    bare letter ("…the cheapest leverage i"), which reads as a rendering fault
    rather than as text that was shortened."""
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:.—–-")
    return f"{cut or text[:limit]}…"


def parse_stance(raw: str) -> dict[str, str] | None:
    """``BUY`` / ``HOLD | trim 50% at $40`` -> ``{label, tone, note}``, else None.

    Unrecognised words return None rather than an uncoloured badge: a stance whose
    tone we cannot name is one we would have to guess at, and guessing wrong on
    this particular field is worse than leaving it off. ``render_report`` says so
    in its result, so a model that writes "maybe" learns it rather than shipping a
    sheet quietly missing the call.
    """
    text = re.sub(r"[*_`]", "", raw or "").strip()
    if not text:
        return None
    found = _STANCE_RE.match(text)
    if not found:
        return None
    label = re.sub(r"\s+", " ", found.group("label")).strip()
    note = found.group("note").strip().rstrip(")").strip()
    return {
        "label": label.upper(),
        "tone": _STANCE_TONES[label.lower()],
        "note": _clip(re.sub(r"\s+", " ", note), 88),
    }


def stance_of(stance: str = "", markdown: str = "") -> dict[str, str] | None:
    """The sheet's call: what the caller passed, else a stance written as a heading.

    The explicit argument wins — it is the model saying so deliberately, with the
    note it chose — and the heading is the fallback that makes an existing report
    work without being rewritten.
    """
    explicit = parse_stance(stance)
    if explicit:
        return explicit
    for raw in (markdown or "").splitlines():
        head = _HEADING_RE.match(raw) or _BOLD_HEADING_RE.match(raw)
        if not head:
            continue
        text = re.sub(r"\*\*|__|`", "", head.group(1)).strip()
        intro = _STANCE_HEADING_RE.match(text)
        if intro:
            found = parse_stance(intro.group("rest"))
            if found:
                return found
    return None


def _stance_html(stance: dict[str, str] | None) -> str:
    if not stance:
        return ""
    note = (f'<span class="why">{_html.escape(stance["note"])}</span>'
            if stance["note"] else "")
    return (f'<div class="stance"><span class="pill {stance["tone"]}">'
            f'{_html.escape(stance["label"])}</span>{note}</div>')


def _notes_html(markdown: str, used_titles: set[str]) -> str:
    picks = extract_notes(markdown, used_titles)
    if not picks:
        return ""
    lis = "".join(f"<li>{_html.escape(p)}</li>" for p in picks)
    return f'<div class="notes"><h2>Key observations</h2><ul>{lis}</ul></div>'


def _cover_tail(pages: int, attached: bool) -> str:
    """The cover's footer line — what this sheet is, and where the rest of it is.

    It may only promise a PDF when one is actually being delivered. Under
    ``output="image"`` the document stays on disk and never reaches the reader, so
    "full PDF attached" would be pointing at a file they don't have. The page count
    still earns its place there — it tells them how much was distilled — but for a
    single-page report with nothing attached the whole line says nothing, and the
    generation stamp is better use of the space.
    """
    if not attached:
        return f"Summary of a {pages}-page report" if pages > 1 else ""
    return (f"Summary of a {pages}-page report — full PDF attached" if pages > 1
            else "Summary — full report attached as PDF")


def build_infographic_html(
    title: str,
    markdown: str,
    highlights: str = "",
    subtitle: str = "",
    eyebrow: str = "",
    pages: int = 0,
    page_height: int = _PAGE_H,
    attached: bool = True,
    stance: str = "",
) -> str:
    """A one-page visual summary of a multi-page report. Pure and testable."""
    tiles = cover_tiles(highlights, markdown)
    tiles_html = ""
    if tiles:
        cols = min(len(tiles), 3 if len(tiles) in (3, 5, 6) else 4)
        cells = "".join(
            f'<div class="tile"><div class="lab">{_html.escape(t["label"])}</div>'
            f'<div class="val{_tile_size(t["value"])}">{_html.escape(t["value"])}</div>'
            + (f'<div class="note">{_html.escape(t["note"])}</div>' if t["note"] else "")
            + "</div>"
            for t in tiles
        )
        # Pad the last row: an unfilled grid cell shows the gap colour as a grey
        # block, which reads as a missing tile rather than as empty space.
        filler = (-len(tiles)) % cols
        cells += '<div class="tile"></div>' * filler
        tiles_html = (f'<div class="tiles" style="grid-template-columns:'
                      f'repeat({cols},1fr)">{cells}</div>')

    series = extract_series(markdown)
    charts_html = ""
    if series:
        cards = [_bars_html(s) for s in series]
        # An odd chart in a two-column grid would leave a hole; let it span instead.
        if len(cards) > 1 and len(cards) % 2 == 1:
            cards[-1] = cards[-1].replace('<div class="card">', '<div class="card wide">', 1)
        charts_html = '<div class="charts">' + "".join(cards) + "</div>"
    notes = _notes_html(markdown, {s["title"] for s in series})

    tail = _cover_tail(pages, attached)
    return _INFO_DOC.format(
        css=_INFO_CSS.format(
            w=_WIDTH, h=max(200, min(int(page_height), _MAX_H)),
            cols=2 if len(series) > 1 else 1, vars=_css_vars(),
            stance_css=_STANCE_CSS,
        ),
        title=_html.escape(title or "Report"),
        eyebrow=_html.escape(eyebrow or _EYEBROW),
        subtitle=(f'<div class="sub">{_html.escape(subtitle)}</div>' if subtitle else ""),
        stance=_stance_html(stance_of(stance, markdown)),
        tiles=tiles_html,
        charts=charts_html,
        notes=notes,
        footer=_html.escape(tail) if tail else _stamp(),
        disclaimer=_DISCLAIMER,
    )


def _fpdf_infographic(
    pdf_path: Path, title: str, markdown: str, highlights: str = "",
    subtitle: str = "", pages: int = 0, attached: bool = True, stance: str = "",
) -> bool:
    """The same summary sheet without a browser: tiles, then bars drawn as rects."""
    try:
        from fpdf import FPDF
    except ImportError:
        return False

    font_regular, font_bold = _unicode_font()
    width_pt = _WIDTH * _PT
    margin = 39.0
    series = extract_series(markdown)
    tiles = cover_tiles(highlights, markdown)
    call = stance_of(stance, markdown)

    def draw(height_pt: float | None):
        pdf = FPDF(unit="pt", format=(width_pt, height_pt or (_PAGE_H * _PT)))
        pdf.set_auto_page_break(False)
        pdf.set_left_margin(margin)
        pdf.set_right_margin(margin)
        pdf.add_page()
        pdf.set_fill_color(*_ink("surface"))
        pdf.rect(0, 0, width_pt, height_pt or (_PAGE_H * _PT), style="F")

        family = "helvetica"
        if font_regular:
            try:
                pdf.add_font("sheet", "", font_regular)
                pdf.add_font("sheet", "B", font_bold or font_regular)
                family = "sheet"
            except Exception:  # noqa: BLE001
                family = "helvetica"
        conv = _strip_emoji if family != "helvetica" else _ascii
        inner = width_pt - 2 * margin

        pdf.set_xy(margin, 36)
        pdf.set_font(family, "", 8)
        pdf.set_text_color(*_ink("muted"))
        pdf.cell(0, 10, conv(_EYEBROW.upper()), new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(margin)
        pdf.set_font(family, "B", 26)
        pdf.set_text_color(*_ink("ink"))
        pdf.multi_cell(inner, 31, conv(title or "Report"), align="L")
        if subtitle:
            pdf.set_x(margin)
            pdf.set_font(family, "", 11)
            pdf.set_text_color(*_ink("ink2"))
            pdf.multi_cell(inner, 15, conv(subtitle), align="L")
        if call:
            pdf.ln(12)
            _draw_stance(pdf, call, margin, family, conv, inner)
        pdf.ln(10)
        pdf.set_draw_color(*_ink("grid"))
        pdf.set_line_width(0.6)
        pdf.line(margin, pdf.get_y(), width_pt - margin, pdf.get_y())
        pdf.ln(14)

        if tiles:
            per_row = min(len(tiles), 3 if len(tiles) in (3, 5, 6) else 4)
            col = inner / per_row
            for index, tile in enumerate(tiles):
                if index and index % per_row == 0:
                    pdf.set_y(pdf.get_y() + 62)
                top = pdf.get_y()
                x = margin + (index % per_row) * col
                pdf.set_xy(x, top)
                pdf.set_font(family, "", 8)
                pdf.set_text_color(*_ink("muted"))
                pdf.cell(col, 11, conv(tile["label"]), align="L")
                pdf.set_xy(x, top + 13)
                size = {"": 21, " sm": 16, " xs": 12}[_tile_size(tile["value"])]
                pdf.set_font(family, "B", size)
                pdf.set_text_color(*_ink("ink"))
                pdf.cell(col, 25, conv(tile["value"]), align="L")
                if tile["note"]:
                    pdf.set_xy(x, top + 39)
                    pdf.set_font(family, "", 8)
                    pdf.set_text_color(*_ink("ink2"))
                    pdf.multi_cell(col - 8, 10, conv(tile["note"]), align="L")
                pdf.set_y(top)
            pdf.set_y(pdf.get_y() + 70)

        for chart in series:
            pdf.set_x(margin)
            pdf.set_font(family, "B", 9)
            pdf.set_text_color(*_ink("muted"))
            pdf.cell(0, 14, conv(chart["title"].upper()), new_x="LMARGIN", new_y="NEXT")
            # Same reason the CSS column grew: "+$12,480.50" does not fit 56pt,
            # and an overflowing fpdf cell writes over the bar to its left.
            label_w, value_w = 118.0, 72.0
            bar_w = inner - label_w - value_w - 16
            values = chart["items"]
            diverging = _is_diverging(chart)
            span = max((abs(v) for _l, v in values), default=1) or 1
            for label, value in values:
                y = pdf.get_y()
                pdf.set_xy(margin, y)
                pdf.set_font(family, "", 10)
                pdf.set_text_color(*_ink("ink"))
                pdf.cell(label_w, 17, conv(label), align="L")
                bar_x = margin + label_w + 8
                pdf.set_fill_color(240, 239, 236)
                pdf.rect(bar_x, y + 2, bar_w, 14, style="F", round_corners=True,
                         corner_radius=3)
                if diverging:
                    mid = bar_x + bar_w / 2
                    length = abs(value) / span * (bar_w / 2)
                    pdf.set_fill_color(*(_ink("neg") if value < 0 else _ink("pos")))
                    pdf.rect(mid - length if value < 0 else mid, y + 2, length, 14,
                             style="F", round_corners=True, corner_radius=3)
                    pdf.set_draw_color(*_ink("rule"))
                    pdf.line(mid, y, mid, y + 18)
                else:
                    negative = all(v <= 0 for _l, v in values) and chart["signed"]
                    pdf.set_fill_color(*(_ink("neg") if negative else _ink("pos")))
                    pdf.rect(bar_x, y + 2, max(bar_w * abs(value) / span, 1), 14,
                             style="F", round_corners=True, corner_radius=3)
                pdf.set_xy(bar_x + bar_w + 8, y)
                pdf.set_font(family, "B", 10)
                pdf.set_text_color(*(_ink("neg") if chart["signed"] and value < 0 else _ink("ink")))
                pdf.cell(
                    value_w, 17,
                    conv(_fmt_value(value, chart.get("unit", "%"), chart["signed"])),
                    align="R",
                )
                pdf.set_y(y + 19)
            pdf.ln(10)

        notes = extract_notes(markdown, {c["title"] for c in series})
        if notes:
            pdf.set_draw_color(*_ink("grid"))
            pdf.line(margin, pdf.get_y(), width_pt - margin, pdf.get_y())
            pdf.ln(12)
            pdf.set_x(margin)
            pdf.set_font(family, "B", 9)
            pdf.set_text_color(*_ink("muted"))
            pdf.cell(0, 14, conv("KEY OBSERVATIONS"), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font(family, "", 10)
            pdf.set_text_color(*_ink("ink"))
            for note in notes:
                pdf.set_x(margin)
                pdf.multi_cell(inner, 14, conv(f"•  {note}"), align="L")
            pdf.ln(4)

        end = pdf.get_y()
        pdf.set_font(family, "", 7)
        pdf.set_text_color(*_ink("muted"))
        tail = _cover_tail(pages, attached) or _stamp()
        pdf.set_xy(margin, end + 8)
        pdf.multi_cell(inner, 9, conv(f"{tail}  ·  {_DISCLAIMER}"), align="L")
        return pdf, pdf.get_y()

    try:
        _probe, end_y = draw(_MAX_H * _PT)
        doc, _ = draw(end_y + 30)
        doc.output(str(pdf_path))
    except Exception:  # noqa: BLE001 - a cover must never take the report down
        return False
    return pdf_path.exists()
