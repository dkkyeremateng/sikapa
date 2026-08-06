"""Rendered reports — turn an answer into a PDF + cover image a phone can read.

Everything else this agent produces is text, which is right for a terminal and
poor on a phone: a scheduled task's analysis arrives as a wall of Telegram
message. This renders the same content as a typeset sheet — headline, stat tiles,
body — and hands it to the delivery channels as files.

**The PDF is the document; the PNG is its first page.** A long report paginates,
and the cover image is page 1 — which a full-page screenshot cannot express, since
a screenshot has no pages. So everything is PDF-first and ``pypdfium2`` rasterises
page 1 afterwards, whichever renderer produced the PDF.

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
:root{{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--rule:#c3c2b7;--accent:#2a78d6;--warn:#e34948;--wash:#f4f4f1;}}
@page{{size:{w}px {h}px;margin:0}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{w}px;background:var(--page);color:var(--ink);
 font-family:system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}}
.sheet{{background:var(--surface);padding:44px 48px 40px}}
.eyebrow{{font-size:15px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-weight:600}}
h1.doc{{font-size:44px;line-height:1.1;font-weight:650;letter-spacing:-.02em;margin-top:10px}}
.sub{{font-size:18px;color:var(--ink2);margin-top:10px}}
.rule{{height:1px;background:var(--grid);margin:26px 0}}
.tiles{{display:grid;gap:2px;background:var(--grid);margin-bottom:4px}}
.tile{{background:var(--surface);padding:20px 18px}}
.tile .lab{{font-size:13.5px;color:var(--muted);font-weight:500}}
.tile .val{{font-size:31px;font-weight:650;margin-top:8px;letter-spacing:-.02em}}
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
.body blockquote{{border-left:4px solid var(--warn);background:#fdf5f5;padding:14px 20px;margin:18px 0}}
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

_EYEBROW = "Financial research assistant"


def _stamp() -> str:
    return f"Generated {datetime.now():%d %B %Y, %H:%M}"


def build_html(
    title: str,
    markdown: str,
    highlights: str = "",
    subtitle: str = "",
    eyebrow: str = "",
    page_height: int = _PAGE_H,
) -> str:
    """The full HTML document. Pure — no filesystem, no Chrome, so it is testable.

    ``page_height`` sizes ``@page``: the measured content height for a single-page
    sheet, or the default to paginate.
    """
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
    return _DOC.format(
        css=_CSS.format(w=_WIDTH, h=max(200, min(int(page_height), _MAX_H))),
        title=_html.escape(title or "Report"),
        eyebrow=_html.escape(eyebrow or _EYEBROW),
        subtitle=f'<div class="sub">{_html.escape(subtitle)}</div>' if subtitle else "",
        tiles=tiles_html,
        body=_markdown_html(markdown),
        footer=_stamp(),
        disclaimer=_DISCLAIMER,
    )


# --- Chrome renderer -----------------------------------------------------------


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
    proc = _run_chrome([
        chrome, "--headless", "--disable-gpu", "--no-sandbox",
        "--virtual-time-budget=3000", "--dump-dom", url,
    ])
    if proc is None:
        return _FALLBACK_H
    found = re.search(rb"__FRA_H:(\d+)", proc.stdout or b"")
    return max(200, min(int(found.group(1)), _MAX_H)) if found else _FALLBACK_H


def _chrome_pdf(chrome: str, html_path: Path, pdf: Path) -> bool:
    """Print the sheet, sizing the page to the content when it fits on one."""
    url = html_path.as_uri()
    height = _measure_height(chrome, url)
    height = _PAGE_H if height > _SINGLE_MAX_H else height + _HEIGHT_SLACK
    html_path.write_text(
        re.sub(
            r"@page\{size:\d+px \d+px",
            f"@page{{size:{_WIDTH}px {height}px",
            html_path.read_text(encoding="utf-8"),
        ),
        encoding="utf-8",
    )
    _run_chrome([
        chrome, "--headless", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
        "--no-pdf-header-footer", f"--print-to-pdf={pdf}", url,
    ])
    return pdf.exists()


# --- fpdf2 renderer (no browser) -----------------------------------------------

_SURFACE = (252, 252, 251)
_INK = (11, 11, 11)
_INK2 = (82, 81, 78)
_MUTED = (137, 135, 129)
_GRID = (225, 224, 217)

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

_ASCII_MAP = {
    "—": "-", "–": "-", "‘": "'", "’": "'", "“": '"', "”": '"', "…": "...",
    "→": "->", "≈": "~", "×": "x", "✓": "[ok]", "▼": "v", "▲": "^", "⚠": "!",
    "🔔": "*", "🤖": "", "€": "EUR ", "≥": ">=", "≤": "<=", "·": "-",
}


def _ascii(text: str) -> str:
    """Transliterate the typographic characters the core PDF fonts cannot encode.

    Only used when no Unicode TTF is on the machine. Losing an em dash beats
    ``FPDFUnicodeEncodingException`` taking the whole render down.
    """
    for src, dst in _ASCII_MAP.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", "replace").decode("latin-1")


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
                        color=_INK, t_margin=10, b_margin=3),
        "h2": TextStyle(font_family=family, font_style="B", font_size_pt=11,
                        color=_MUTED, t_margin=11, b_margin=2),
        "h3": TextStyle(font_family=family, font_style="B", font_size_pt=13,
                        color=_INK, t_margin=8, b_margin=2),
        "blockquote": TextStyle(font_family=family, font_size_pt=10, color=_INK2,
                                l_margin=12, t_margin=5, b_margin=5),
    }


def _unicode_font() -> tuple[str, str]:
    """``(regular, bold)`` paths for an embeddable Unicode font, or ``("", "")``."""
    explicit = (os.environ.get("FINANCIAL_RESEARCH_REPORT_FONT") or "").strip()
    if explicit and Path(explicit).exists():
        bold = re.sub(r"(-Regular)?\.(ttf|otf)$", r"-Bold.\2", explicit)
        return explicit, (bold if Path(bold).exists() else "")
    for regular, bold in _FONT_CANDIDATES:
        if Path(regular).exists():
            return regular, (bold if bold and Path(bold).exists() else "")
    return "", ""


def _fpdf_pdf(
    pdf_path: Path, title: str, markdown: str, highlights: str = "",
    subtitle: str = "", eyebrow: str = "",
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
            self.set_text_color(*_MUTED)
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
        pdf.set_fill_color(*_SURFACE)
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
        conv = (lambda s: s) if family != "helvetica" else _ascii
        pdf.family = family
        # Bound as a plain attribute holding a function: `staticmethod(...)` on an
        # instance is not callable through the instance on 3.10.
        pdf.conv = conv

        pdf.set_xy(margin, 33)
        pdf.set_font(family, "", 8)
        pdf.set_text_color(*_MUTED)
        pdf.cell(0, 10, conv((eyebrow or _EYEBROW).upper()), new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(margin)
        pdf.set_font(family, "B", 23)
        pdf.set_text_color(*_INK)
        pdf.multi_cell(width_pt - 2 * margin, 28, conv(title or "Report"), align="L")
        if subtitle:
            pdf.set_x(margin)
            pdf.set_font(family, "", 11)
            pdf.set_text_color(*_INK2)
            pdf.multi_cell(width_pt - 2 * margin, 15, conv(subtitle), align="L")
        pdf.ln(10)
        pdf.set_draw_color(*_GRID)
        pdf.set_line_width(0.6)
        pdf.line(margin, pdf.get_y(), width_pt - margin, pdf.get_y())
        pdf.ln(12)

        tiles = parse_highlights(highlights)
        if tiles:
            col = (width_pt - 2 * margin) / len(tiles)
            top = pdf.get_y()
            for i, t in enumerate(tiles):
                x = margin + i * col
                if i:
                    pdf.line(x - 4, top, x - 4, top + 52)
                pdf.set_xy(x, top)
                pdf.set_font(family, "", 8)
                pdf.set_text_color(*_MUTED)
                pdf.cell(col, 11, conv(t["label"]), align="L")
                pdf.set_xy(x, top + 13)
                pdf.set_font(family, "B", 19)
                pdf.set_text_color(*_INK)
                pdf.cell(col, 24, conv(t["value"]), align="L")
                if t["note"]:
                    pdf.set_xy(x, top + 38)
                    pdf.set_font(family, "", 8)
                    pdf.set_text_color(*_INK2)
                    pdf.multi_cell(col - 8, 10, conv(t["note"]), align="L")
            pdf.set_y(top + 60)
            pdf.set_draw_color(*_GRID)
            pdf.line(margin, pdf.get_y(), width_pt - margin, pdf.get_y())
            pdf.ln(10)

        pdf.set_x(margin)
        pdf.set_font(family, "", 11)
        pdf.set_text_color(*_INK)
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


def render(html: str, name: str, *, content: dict[str, str] | None = None) -> dict[str, str]:
    """Write the HTML, produce the PDF, then rasterise page 1. Never raises.

    ``content`` carries the raw fields so the browser-free renderer can draw the
    same sheet; without it only the Chrome path can run. A failure at any stage
    still leaves the earlier artifacts on disk — losing a finished analysis to a
    rendering problem is the one outcome worth engineering against.
    """
    out_dir = reports_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{_slug(name)}-{datetime.now():%Y%m%d-%H%M%S}"
    paths: dict[str, str] = {}

    html_path = out_dir / f"{stem}.html"
    html_path.write_text(html, encoding="utf-8")
    paths["html"] = str(html_path)

    pdf = out_dir / f"{stem}.pdf"
    made = False
    chrome = chrome_path()
    if chrome:
        made = _chrome_pdf(chrome, html_path, pdf)
        if made:
            paths["renderer"] = "chrome"
    if not made and content is not None:
        made = _fpdf_pdf(
            pdf,
            content.get("title", ""),
            content.get("markdown", ""),
            content.get("highlights", ""),
            content.get("subtitle", ""),
            content.get("eyebrow", ""),
        )
        if made:
            paths["renderer"] = "fpdf2"
    if not made:
        return paths
    paths["pdf"] = str(pdf)

    png = out_dir / f"{stem}.png"
    if rasterize_first_page(str(pdf), str(png)):
        paths["png"] = str(png)
    return paths


# --- Model-facing tool ---------------------------------------------------------


def render_report(
    title: str,
    markdown: str,
    highlights: str = "",
    subtitle: str = "",
    deliver: bool = True,
) -> str:
    """Typeset a summary as a PDF + cover image and send it to the user's channels.

    Use when the user asks for a report/infographic/PDF/one-pager, or wants
    something "sent"/"pushed" to them as a file rather than as chat text — and for
    a scheduled task's output, where a typeset sheet reads far better on a phone
    than a wall of message text.

    ``title`` is the headline. ``markdown`` is the body — normal markdown works:
    headings, bold, lists, tables, `>` blockquote for a warning callout. A long
    body paginates, and the image sent alongside is page 1.
    ``highlights`` is optional stat tiles, ONE PER LINE as ``label | value | note``
    (up to 6), e.g. "Adjusted EPS | $1.84 | vs $1.91 consensus". Put the numbers
    that matter there, not in the body. ``subtitle`` is one line under the title.
    ``deliver=False`` renders without sending.

    Returns where the files went and whether delivery succeeded.
    """
    if not (title or "").strip() and not (markdown or "").strip():
        return "Nothing to render — give at least a title or some body text."

    paths = render(
        build_html(title, markdown, highlights, subtitle),
        title,
        content={
            "title": title,
            "markdown": markdown,
            "highlights": highlights,
            "subtitle": subtitle,
        },
    )

    lines = []
    if "pdf" not in paths:
        lines.append(
            "Could not produce a PDF (no browser, and the built-in renderer failed) "
            "— the HTML is saved and opens in any browser."
        )
    else:
        pages = page_count(paths["pdf"])
        lines.append(
            f"Rendered {pages} page(s) with {paths.get('renderer', '?')}"
            + ("; the image is page 1." if pages > 1 else ".")
        )
    lines.append("Saved: " + ", ".join(
        f"{k.upper()} {v}" for k, v in paths.items() if k != "renderer"
    ))

    if deliver:
        from . import channels

        sent: list[str] = []
        failed: list[str] = []
        for key, caption in (("png", title), ("pdf", f"{title} (PDF)")):
            if key in paths:
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
    return "\n".join(lines)


REPORT_TOOLS = [render_report]
