"""Rendered reports: document building, stat tiles, and file delivery.

Offline — Chrome is never invoked (the one test that would is skipped without a
binary), and no channel sends anything real.
"""

from pathlib import Path

import pytest

from financial_research_assistant import channels, reports


# --- highlights ----------------------------------------------------------------


def test_highlights_parse_into_tiles():
    tiles = reports.parse_highlights(
        "Adjusted EPS | $1.84 | vs $1.91 consensus\nClose | $52.68 | down 2.64%"
    )
    assert [t["label"] for t in tiles] == ["Adjusted EPS", "Close"]
    assert tiles[0]["value"] == "$1.84" and tiles[0]["note"] == "vs $1.91 consensus"


def test_malformed_highlights_degrade_rather_than_raise():
    """A model will get the delimiter wrong eventually; a bad tile must not sink a
    report that took a dozen tool calls to produce."""
    tiles = reports.parse_highlights("just a value\n\n  \nA | B | C | D")
    assert tiles[0] == {"label": "", "value": "just a value", "note": ""}
    assert tiles[1]["note"] == "C · D", "extra pipes fold into the note"


def test_tiles_are_capped_so_they_stay_scannable():
    assert len(reports.parse_highlights("\n".join(f"L{i} | V{i}" for i in range(12)))) == 6


# --- the document ---------------------------------------------------------------


def test_the_document_carries_title_tiles_and_body():
    html = reports.build_html(
        "FISV Q2 2026", "## Results\n\nEPS **missed** by 3.7%.",
        highlights="Adjusted EPS | $1.84 | vs $1.91", subtitle="Reported 6 August",
    )
    assert "FISV Q2 2026" in html and "Reported 6 August" in html
    assert "$1.84" in html
    assert "<strong>missed</strong>" in html, "markdown must be rendered, not escaped"
    assert "grid-template-columns:repeat(1,1fr)" in html


def test_markdown_tables_and_callouts_render():
    html = reports.build_html(
        "t", "| A | B |\n|---|---|\n| 1 | 2 |\n\n> watch out\n",
    )
    assert "<table>" in html and "<blockquote>" in html


def test_html_in_the_body_is_not_executed():
    """The body is model-written text that can quote a filing or a web page, so it
    is untrusted: a <script> in it must render as characters, not run in Chrome."""
    html = reports.build_html("t", "<script>alert(1)</script> and <b>raw</b>")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_a_title_with_markup_is_escaped():
    assert "&lt;img" in reports.build_html('<img src=x onerror=y>', "body")


def test_the_disclaimer_is_always_present():
    """Every sheet leaves the machine as a standalone artifact; it must say what it
    is and is not."""
    assert "not investment advice" in reports.build_html("t", "b")


# --- delivery -------------------------------------------------------------------


@pytest.fixture
def file_channels(monkeypatch):
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(channels, "CHANNEL_REGISTRY", {}, raising=False)
    channels.register_channel(channels.Channel(
        "filey", lambda: True, lambda t: True, "Filey",
        send_file=lambda p, c, fq=False: bool(sent.append((p, c, fq)) or True),
    ))
    channels.register_channel(channels.Channel(
        "texty", lambda: True, lambda t: True, "Texty",  # no send_file
    ))
    monkeypatch.delenv("NOTIFY_CHANNELS", raising=False)
    return sent


def test_a_file_only_goes_to_channels_that_can_carry_one(file_channels):
    delivered, failed = channels.deliver_file("/tmp/x.png", "caption")
    assert delivered == ["filey"]
    assert failed == [], "a text-only channel is not a failure, it is not a target"
    assert file_channels == [("/tmp/x.png", "caption", False)]
    assert channels.file_capable() == ["filey"]


def test_a_throwing_file_channel_is_recorded_not_raised(file_channels):
    def boom(_p, _c, _fq=False):
        raise RuntimeError("upload died")

    channels.register_channel(channels.Channel(
        "bad", lambda: True, lambda t: True, "Bad", send_file=boom,
    ))
    delivered, failed = channels.deliver_file("/tmp/x.png")
    assert delivered == ["filey"] and failed == ["bad"]


# --- the tool -------------------------------------------------------------------


def test_the_tool_reports_where_the_file_went(monkeypatch, tmp_path):
    monkeypatch.setattr(reports, "render", lambda html, name, content=None: {
        "html": str(tmp_path / "r.html"), "png": str(tmp_path / "r.png"),
    })
    monkeypatch.setattr(channels, "deliver_file", lambda p, caption="", prefer="", full_quality=False: (["telegram"], []))
    out = reports.render_report("FISV Q2", "body text", deliver=True, allow_prose=True)
    assert "Sent to: telegram" in out and "PNG" in out


def test_the_tool_says_so_when_nothing_can_receive_a_file(monkeypatch, tmp_path):
    monkeypatch.setattr(reports, "render", lambda html, name, content=None: {"png": str(tmp_path / "r.png")})
    monkeypatch.setattr(channels, "deliver_file", lambda p, caption="", prefer="", full_quality=False: ([], []))
    assert "No channel accepted a file" in reports.render_report("t", "b", allow_prose=True)


def test_with_no_renderer_at_all_the_html_still_survives(monkeypatch, tmp_path):
    """A finished analysis must never be lost to a rendering problem."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORTS_DIR", str(tmp_path))
    monkeypatch.setattr(reports, "chrome_path", lambda: "")
    monkeypatch.setattr(reports, "_fpdf_pdf", lambda *a, **k: False)
    paths = reports.render(
        reports.build_html("t", "b"), "t", content={"title": "t", "markdown": "b"}
    )
    assert set(paths) == {"html"}

    monkeypatch.setattr(reports, "render", lambda html, name, content=None: paths)
    out = reports.render_report("t", "b", deliver=False, allow_prose=True)
    assert "Could not produce a PDF" in out


def test_empty_input_is_refused_before_rendering():
    assert "Nothing to render" in reports.render_report("", "   ")


def test_the_tool_is_registered():
    from financial_research_assistant import catalog

    assert "render_report" in {catalog.tool_name(t) for t in catalog.TOOLS}


@pytest.mark.skipif(not reports.chrome_path(), reason="no Chrome/Chromium on this machine")
def test_chrome_actually_produces_a_png_and_pdf(monkeypatch, tmp_path):
    """The one test that shells out — proves the flags and the file paths are right,
    which no amount of HTML assertion can."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORTS_DIR", str(tmp_path))
    paths = reports.render(
        reports.build_html("Render check", "# Heading\n\nSome body text."), "render check"
    )
    assert "png" in paths and "pdf" in paths
    from pathlib import Path

    assert Path(paths["png"]).stat().st_size > 5_000
    assert Path(paths["pdf"]).read_bytes()[:4] == b"%PDF"


# --- render quality -------------------------------------------------------------


def test_the_page_stamps_its_own_height_for_the_measuring_pass():
    """Chrome screenshots the viewport, not the page, and has no fit-to-content
    flag — so the document reports its own height and `_measure_height` reads it
    back. Without it every sheet is clipped or padded with dead space."""
    assert "__FRA_H:" in reports.build_html("t", "body")
    assert "scrollHeight" in reports.build_html("t", "body")


def test_the_render_scale_is_high_by_default_and_bounded(monkeypatch):
    """A phone lets you pinch into the sheet; body text has to survive it."""
    monkeypatch.delenv("FINANCIAL_RESEARCH_REPORT_SCALE", raising=False)
    assert reports.render_scale() == 3
    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORT_SCALE", "2")
    assert reports.render_scale() == 2
    for junk in ("0", "9", "huge", ""):
        monkeypatch.setenv("FINANCIAL_RESEARCH_REPORT_SCALE", junk)
        assert reports.render_scale() == 3, junk


def test_a_failed_measurement_falls_back_rather_than_rendering_nothing(monkeypatch):
    monkeypatch.setattr(reports, "_run_chrome", lambda args: None)
    assert reports._measure_height("chrome", "file:///x") == reports._FALLBACK_H


def test_the_measured_height_is_clamped(monkeypatch):
    class P:
        stdout = b"<title>__FRA_H:999999</title>"

    monkeypatch.setattr(reports, "_run_chrome", lambda args: P())
    assert reports._measure_height("c", "u") == reports._MAX_H


def test_the_sheet_is_sent_uncompressed(monkeypatch, tmp_path):
    """sendPhoto re-encodes to JPEG and downscales, which turns dense body text to
    mush regardless of the render resolution. The sheet must go as a document."""
    seen: list[bool] = []
    monkeypatch.setattr(reports, "render", lambda html, name, content=None: {"png": str(tmp_path / "r.png")})
    monkeypatch.setattr(
        channels, "deliver_file",
        lambda p, caption="", prefer="", full_quality=False: (
            seen.append(full_quality), (["telegram"], []))[1],
    )
    reports.render_report("t", "b", allow_prose=True)
    assert seen == [True]


# --- the two renderers must not diverge -----------------------------------------
#
# A fallback nobody looks at rots. These run BOTH renderers over identical input
# and assert the same structural facts, so a divergence fails here rather than
# surfacing months later as an ugly PDF nobody asked for.

import pytest  # noqa: E402

_SAMPLE = {
    "title": "FISV Q2 2026 — earnings miss",
    "subtitle": "Reported 6 August 2026",
    "highlights": "Adjusted EPS | $1.84 | vs $1.91\nClose | $52.68 | down 2.64%",
    "markdown": (
        "## Headline\n\nFiserv **missed** and cut guidance — the first quarter "
        "under a new CEO.\n\n> Detail sits in the 8-K.\n\n"
        "| Metric | Q1 | Q2 |\n|---|---|---|\n| EPS | $1.79 | $1.84 |\n\n"
        "- median $62.50\n- mean $66.62\n"
    ),
}

_LONG = dict(
    _SAMPLE,
    markdown="\n".join(
        f"## Section {i}\n\n" + ("Body text for this section. " * 45)
        for i in range(1, 26)
    ),
)


def _render_with(monkeypatch, tmp_path, payload, renderer):
    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORTS_DIR", str(tmp_path))
    if renderer == "fpdf2":
        monkeypatch.setenv("FINANCIAL_RESEARCH_CHROME", "/nonexistent/chrome")
    else:
        monkeypatch.delenv("FINANCIAL_RESEARCH_CHROME", raising=False)
        if not reports.chrome_path():
            pytest.skip("no Chrome/Chromium on this machine")
    return reports.render(
        reports.build_html(**payload), payload["title"], content=payload
    )


@pytest.mark.parametrize("renderer", ["chrome", "fpdf2"])
def test_both_renderers_produce_a_pdf_with_the_content_as_real_text(
    monkeypatch, tmp_path, renderer
):
    paths = _render_with(monkeypatch, tmp_path, _SAMPLE, renderer)
    assert paths.get("renderer") == renderer
    assert "pdf" in paths and "png" in paths

    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(paths["pdf"])
    text = " ".join(doc[0].get_textpage().get_text_range().split())
    # Case-insensitive: the CSS uppercases section headings via `text-transform`,
    # which Chrome bakes into the text layer ("HEADLINE") while fpdf2 has no such
    # transform ("Headline"). A cosmetic divergence, not a missing heading.
    lower = text.lower()
    assert "fisv q2 2026" in lower
    assert "$1.84" in text, "a stat-tile value went missing"
    assert "headline" in lower, "a body heading went missing"
    assert "$1.79" in text, "a table cell went missing"
    assert "not investment advice" in lower, "the disclaimer must always ship"


@pytest.mark.parametrize("renderer", ["chrome", "fpdf2"])
def test_a_short_report_is_one_page_with_no_trailing_dead_band(
    monkeypatch, tmp_path, renderer
):
    """Both paths fit the page to the content: a near-empty second page is the
    defect this replaced."""
    paths = _render_with(monkeypatch, tmp_path, _SAMPLE, renderer)
    assert reports.page_count(paths["pdf"]) == 1


@pytest.mark.parametrize("renderer", ["chrome", "fpdf2"])
def test_a_long_report_paginates_and_its_cover_is_a_single_sheet(
    monkeypatch, tmp_path, renderer
):
    """A long report paginates, and its cover is one whole sheet — the summary
    infographic, not a slice of the document."""
    paths = _render_with(monkeypatch, tmp_path, _LONG, renderer)
    pages = reports.page_count(paths["pdf"])
    assert pages > 1, f"{renderer} did not paginate a long report"
    assert paths["cover"] == "infographic"

    import pypdfium2 as pdfium
    from PIL import Image

    cover_pdf = Path(paths["pdf"]).with_name(Path(paths["pdf"]).stem + "-cover.pdf")
    # pypdfium2's scale is pixels per POINT, so the image is page-height-in-points
    # times the scale — not the CSS pixel height.
    expected = pdfium.PdfDocument(str(cover_pdf))[0].get_size()[1] * reports.render_scale()
    assert Image.open(paths["png"]).height == pytest.approx(expected, rel=0.02)


def test_the_fallback_is_used_only_when_chrome_is_absent(monkeypatch, tmp_path):
    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORTS_DIR", str(tmp_path))
    monkeypatch.setenv("FINANCIAL_RESEARCH_CHROME", "/nonexistent/chrome")
    paths = reports.render(reports.build_html(**_SAMPLE), "x", content=_SAMPLE)
    assert paths["renderer"] == "fpdf2"


def test_the_fallback_cannot_run_without_the_raw_content(monkeypatch, tmp_path):
    """`render()` is given HTML; the browser-free path needs the fields themselves,
    so a caller that omits them gets HTML only rather than a silent blank sheet."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORTS_DIR", str(tmp_path))
    monkeypatch.setenv("FINANCIAL_RESEARCH_CHROME", "/nonexistent/chrome")
    assert set(reports.render(reports.build_html(**_SAMPLE), "x")) == {"html"}


# --- font handling in the fallback ----------------------------------------------


def test_typographic_characters_are_transliterated_not_fatal():
    """The core PDF fonts are Latin-1 only and RAISE on an em dash — losing the
    dash beats losing the render."""
    out = reports._ascii("EPS — $1.84 · ▼ 2.64% ✓ → €")
    assert "—" not in out and "▼" not in out
    out.encode("latin-1")  # must not raise


def test_a_bold_face_is_only_claimed_when_a_real_one_exists(monkeypatch, tmp_path):
    """fpdf2 does not synthesize bold for an embedded TTF, so registering the
    regular file under "B" renders **bold** as plain body text."""
    regular = tmp_path / "Some-Regular.ttf"
    regular.write_bytes(b"x")
    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORT_FONT", str(regular))
    assert reports._unicode_font() == (str(regular), "")

    bold = tmp_path / "Some-Bold.ttf"
    bold.write_bytes(b"x")
    assert reports._unicode_font() == (str(regular), str(bold))


def test_headings_do_not_inherit_fpdf2s_red_defaults():
    """Its stock heading colour is dark red; in this palette red means
    miss/critical, so every section heading would read as an alarm."""
    styles = reports._tag_styles("helvetica")
    assert styles, "tag styles must be applied"
    for tag in ("h1", "h2", "h3"):
        colour = styles[tag].color
        assert colour is not None
        rgb = (colour.r, colour.g, colour.b)
        assert not (rgb[0] > 0.4 and rgb[1] < 0.2 and rgb[2] < 0.2), f"{tag} is red"


# --- the vendored font ----------------------------------------------------------


def test_the_bundled_font_is_present_and_preferred(monkeypatch):
    """Relying on system fonts made output depend on the machine — Arial on macOS,
    DejaVu on Linux, transliterated ASCII in a slim container."""
    monkeypatch.delenv("FINANCIAL_RESEARCH_REPORT_FONT", raising=False)
    regular, bold = reports._bundled_font()
    assert regular.endswith("Inter-Regular.ttf") and bold.endswith("Inter-Bold.ttf")
    assert reports._unicode_font() == (regular, bold), "the bundled pair must win"


def test_the_font_licence_ships_beside_it():
    from pathlib import Path

    licence = Path(reports._bundled_font()[0]).with_name("Inter-LICENSE.txt")
    assert licence.exists()
    assert "SIL Open Font License" in licence.read_text(encoding="utf-8")


def test_an_explicit_font_still_overrides_the_bundled_one(monkeypatch, tmp_path):
    custom = tmp_path / "Custom-Regular.ttf"
    custom.write_bytes(b"x")
    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORT_FONT", str(custom))
    assert reports._unicode_font()[0] == str(custom)


def test_emoji_are_stripped_whatever_the_font():
    """No text face carries them, and a missing TTF glyph renders as an empty box
    rather than raising — silently ugly."""
    assert "🔔" not in reports._strip_emoji("🔔 alert fired")
    assert "🤖" not in reports._strip_emoji("🤖 task s1")


def test_typography_survives_with_the_bundled_font(monkeypatch, tmp_path):
    """The point of vendoring: em dashes and friends reach the PDF as themselves,
    not as hyphens, on any machine."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORTS_DIR", str(tmp_path))
    monkeypatch.setenv("FINANCIAL_RESEARCH_CHROME", "/nonexistent/chrome")
    payload = {"title": "Sheet — dashes", "markdown": "Body — with an em dash · and a middot."}
    paths = reports.render(reports.build_html(**payload), "font", content=payload)
    assert paths["renderer"] == "fpdf2"

    import pypdfium2 as pdfium

    text = pdfium.PdfDocument(paths["pdf"])[0].get_textpage().get_text_range()
    assert "—" in text, "the em dash was transliterated despite a Unicode font"
    assert "·" in text


# --- the multi-page infographic cover -------------------------------------------

_REPORT_MD = """## Top Holdings (68.9% concentration)

1. **VOO** – 29.3% ($12,150) | +$2,940 unrealized gain
2. **UNH** – 13.6% ($5,420) | +$1,210 unrealized gain
3. **AMZN** – 10.2% ($4,310) | +$1,180 unrealized gain
4. **NVO** – 6.8% ($2,875) | +$15 unrealized gain

## True Sector Exposure

- **Healthcare:** 28.4% — UNH, NVO, MOH
- **Technology:** 21.6% — NVDA, MSFT, AMZN
- **Consumer Cyclical:** 15.9% — AMZN, TSLA
- **Industrials:** 8.1% — CPRT, FISV

## Movers

- **MSFT** +28.55% ($388.84 → $499.86) — AI optimism
- **NVDA** +11.20% ($196.93 → $218.99) — AI demand
- **TSLA** -20.69% ($402.90 → $319.53) — headwinds
- **MOH** -17.61% ($232.90 → $191.88) — earnings miss

## Key Observations

- One hundred percent equities, with no fixed income or international exposure
- Healthcare is a large overweight relative to the benchmark weighting
- Hidden overlap: several names are held both directly and through VOO
"""


def test_series_are_extracted_from_the_shapes_a_report_actually_uses():
    """Ranked holdings, sector weights and signed movers — the three list shapes
    these reports produce."""
    series = reports.extract_series(_REPORT_MD)
    assert [s["title"] for s in series] == [
        "Top Holdings (68.9% concentration)", "True Sector Exposure", "Movers",
    ]
    assert series[0]["items"][0] == ("VOO", 29.3)
    assert series[1]["items"][0] == ("Healthcare", 28.4)
    assert series[0]["signed"] is False


def test_signed_values_mark_a_series_diverging():
    """Gains and losses must read as opposites, not as magnitudes."""
    movers = reports.extract_series(_REPORT_MD)[2]
    assert movers["signed"] is True
    assert ("TSLA", -20.69) in movers["items"]


def test_a_list_without_percentages_is_not_charted():
    assert reports.extract_series("## Steps\n\n- do a thing\n- do another\n- and more") == []


def test_prose_between_lists_separates_series():
    md = ("## A\n\n- x 10%\n- y 20%\n- z 30%\n\nSome prose here breaks the run.\n\n"
          "- p 40%\n- q 50%\n- r 60%\n")
    assert len(reports.extract_series(md)) == 2


def test_series_and_items_are_capped():
    md = "".join(
        f"## S{i}\n\n" + "".join(f"- item{j} {j}%\n" for j in range(1, 12)) + "\n"
        for i in range(6)
    )
    series = reports.extract_series(md)
    assert len(series) <= 3
    assert all(len(s["items"]) <= 8 for s in series)


def test_the_infographic_charts_every_series_and_says_what_it_summarises():
    html = reports.build_infographic_html(
        "Portfolio Analysis Report", _REPORT_MD,
        highlights="Portfolio Value | $38,420 | +$1,860", subtitle="As of August 6",
        pages=3,
    )
    assert "TOP HOLDINGS" in html.upper() and "MOVERS" in html.upper()
    assert "$38,420" in html
    assert "Summary of a 3-page report" in html, "the cover must not pose as the report"
    assert "Key observations" in html
    assert "#e34948" in html, "losses must use the diverging negative pole"


def test_an_odd_third_chart_spans_the_grid_rather_than_leaving_a_hole():
    html = reports.build_infographic_html("t", _REPORT_MD, pages=2)
    assert 'class="card wide"' in html


@pytest.mark.parametrize("renderer", ["chrome", "fpdf2"])
def test_a_multi_page_report_gets_an_infographic_cover(monkeypatch, tmp_path, renderer):
    """Page 1 of a five-page report is the masthead and whatever fitted — the least
    informative slice of the document."""
    monkeypatch.setattr(reports, "_SINGLE_MAX_H", 500)
    monkeypatch.setattr(reports, "_PAGE_H", 500)  # force several pages
    payload = {
        "title": "Portfolio Analysis Report", "markdown": _REPORT_MD,
        "highlights": "Portfolio Value | $38,420 | +$1,860\nvs SPY | -10.46 pp | since inception",
        "subtitle": "As of August 6, 2026",
    }
    paths = _render_with(monkeypatch, tmp_path, payload, renderer)
    assert reports.page_count(paths["pdf"]) > 1
    assert paths["cover"] == "infographic"

    import pypdfium2 as pdfium

    cover_pdf = tmp_path / (Path(paths["pdf"]).stem + "-cover.pdf")
    text = " ".join(
        pdfium.PdfDocument(str(cover_pdf))[0].get_textpage().get_text_range().split()
    )
    assert "VOO" in text and "Healthcare" in text and "MSFT" in text, (
        "the cover must carry the whole report's series, not one page of it"
    )
    assert "29.3" in text


def test_a_single_page_report_also_gets_the_infographic(monkeypatch, tmp_path):
    """The distilled figures read better than the document's first page at any
    length, so the cover is not reserved for long reports."""
    payload = {
        "title": "Short", "markdown": _REPORT_MD,
        "highlights": "Portfolio Value | $38,420 | +$1,860",
    }
    paths = _render_with(monkeypatch, tmp_path, payload, "fpdf2")
    assert reports.page_count(paths["pdf"]) == 1
    assert paths["cover"] == "infographic"


def test_a_report_with_nothing_to_chart_keeps_page_one(monkeypatch, tmp_path):
    """No tiles and no series means the sheet would be a title over some bullets,
    and page 1 — which for a one-page report IS the report — strictly beats it."""
    payload = {"title": "Prose only", "markdown": "## H\n\nJust narrative text.",
               "highlights": ""}
    assert reports._worth_charting(payload) is False
    paths = _render_with(monkeypatch, tmp_path, payload, "fpdf2")
    assert paths["cover"] == "page-1"


def test_tiles_alone_are_enough_to_be_worth_a_cover():
    assert reports._worth_charting({"highlights": "A | 1 | x", "markdown": "prose"}) is True


def test_a_failed_cover_falls_back_to_page_one(monkeypatch, tmp_path):
    """A cover is worth less than the report it introduces."""
    monkeypatch.setattr(reports, "_SINGLE_MAX_H", 500)
    monkeypatch.setattr(reports, "_PAGE_H", 500)
    monkeypatch.setattr(reports, "_build_cover", lambda *a, **k: None)
    payload = {"title": "P", "markdown": _REPORT_MD, "highlights": "A | 1 | x"}
    paths = _render_with(monkeypatch, tmp_path, payload, "fpdf2")
    assert paths["cover"] == "page-1" and "png" in paths


def test_the_model_is_told_which_shapes_become_charts():
    """The cover is only as good as the shapes in the body. The same figures in a
    paragraph produce no chart, so the convention has to be stated where the model
    reads it — both in the prompt and in the tool's own description."""
    from financial_research_assistant.graph import SYSTEM_PROMPT

    for text in (SYSTEM_PROMPT, reports.render_report.__doc__ or ""):
        lowered = text.lower()
        assert "breakdowns as lists" in lowered
        assert "28.4%" in text, "the example must show the exact shape"
        assert "signed" in lowered, "diverging bars need signed values"


def test_the_documented_example_shapes_actually_chart():
    """The convention taught to the model must be one the extractor recognises —
    otherwise the prompt is telling it to write something that produces nothing."""
    md = (
        "## Sector exposure\n\n"
        "- **Healthcare:** 28.4% — UNH, NVO, MOH\n"
        "- **Technology:** 21.6% — NVDA, MSFT\n"
        "- **Industrials:** 8.1% — CPRT\n\n"
        "## Holdings\n\n"
        "1. **VOO** – 29.3% ($12,150)\n2. **UNH** – 13.6% ($5,420)\n"
        "3. **AMZN** – 10.2% ($4,310)\n"
    )
    series = reports.extract_series(md)
    assert len(series) == 2
    assert series[0]["items"][0] == ("Healthcare", 28.4)
    assert series[1]["items"][0] == ("VOO", 29.3)


def test_prose_observations_that_quote_percentages_are_not_charted():
    """From a live run: the model wrote observations citing percentages, and they
    were charted as a series — a row of ellipsised sentences. Category labels are
    short; sentences are not."""
    md = (
        "## Sector breakdown\n\n"
        "- **Healthcare:** 28.4% — UNH\n- **Technology:** 21.6% — NVDA\n"
        "- **Consumer Cyclical:** 15.9% — AMZN\n\n"
        "## Key observations\n\n"
        "- Healthcare dominance (28.4%) creates concentration risk in a regulated sector\n"
        "- Technology and cyclical exposure together reach 41.4% of the portfolio\n"
        "- Significant concentration: top three holdings are 57.8% of total value\n"
    )
    series = reports.extract_series(md)
    assert [s["title"] for s in series] == ["Sector breakdown"]
    notes = reports.extract_notes(md, {"Sector breakdown"})
    assert any("Healthcare dominance" in n for n in notes), (
        "the prose belongs in observations, not on an axis"
    )


def test_a_long_category_name_is_still_charted():
    """The guard must not reject real labels — 'Communication Services' is 22."""
    md = ("## Sectors\n\n- **Communication Services:** 3.1%\n"
          "- **Consumer Defensive:** 1.4%\n- **Basic Materials:** 0.5%\n")
    assert len(reports.extract_series(md)) == 1


# --- what a real report actually contains ---------------------------------------


def test_several_tiles_written_on_one_line_are_recovered():
    """Observed: the model put all five tiles on a single line, and the parser
    folded everything after field two into one unreadable run-on note."""
    tiles = reports.parse_highlights(
        "Consensus EPS | $0.37 | Q2 estimate | Position Value | $1,178 | 100 shares | "
        "Beat Streak | 5 quarters | since Q3 2025"
    )
    assert [t["label"] for t in tiles] == ["Consensus EPS", "Position Value", "Beat Streak"]
    assert tiles[1]["value"] == "$1,178"


def test_a_markdown_table_becomes_a_chart():
    """An earnings preview keeps its numbers in tables — beat history, scenario
    ladders — and ignoring them produced a cover with no charts at all."""
    md = ("## Beat history\n\n"
          "| Quarter | Estimate to reported | Surprise |\n|---|---|---|\n"
          "| Q1 2026 | $0.19 to $0.23 | +24.0% |\n"
          "| Q4 2025 | $0.43 to $0.43 | +0.3% |\n"
          "| Q3 2025 | $0.47 to $0.49 | +5.1% |\n")
    series = reports.extract_series(md)
    assert len(series) == 1
    assert series[0]["unit"] == "%" and series[0]["signed"] is True
    assert series[0]["items"][0] == ("Q1 2026", 24.0)


def test_a_transition_cell_is_not_mistaken_for_a_measure():
    """`$0.19 to $0.23` holds two numbers; charting either end is arbitrary, so the
    percent column must win."""
    md = ("## T\n\n| Q | Move | Surprise |\n|---|---|---|\n"
          "| A | $1.00 to $2.00 | +5% |\n| B | $2.00 to $3.00 | +6% |\n"
          "| C | $3.00 to $4.00 | +7% |\n")
    assert reports.extract_series(md)[0]["items"][0] == ("A", 5.0)


def test_a_dollar_table_keeps_its_unit():
    md = ("## Levels\n\n| Level | Scenario | P/L |\n|---|---|---|\n"
          "| $13.58 (mean target) | Beat | +$180 |\n| $11.00 | Miss | -$78 |\n"
          "| $10.00 | Big miss | -$178 |\n")
    series = reports.extract_series(md)[0]
    assert series["unit"] == "$"
    assert series["items"][0] == ("$13.58", 180.0), "the parenthetical must be stripped"


def test_a_column_of_mixed_units_is_not_charted():
    """`$13.58` and `+15.3%` cannot share an axis."""
    md = ("## M\n\n| Metric | Value |\n|---|---|\n| Target | $13.58 |\n"
          "| Upside | +15.3% |\n| Yield | 5.8% |\n")
    assert reports.extract_series(md) == []


def test_values_are_labelled_in_their_own_unit():
    """A dollar P/L rendered as "180.0%" is a lie."""
    assert reports._fmt_value(180.0, "$", True) == "+$180"
    assert reports._fmt_value(-78.0, "$", True) == "-$78"
    assert reports._fmt_value(24.0, "%", True) == "+24.0%"
    assert reports._fmt_value(29.3, "%", False) == "29.3%"


def test_a_run_of_one_sign_is_not_zero_centred():
    """Five positive surprises centred on zero waste half the width and squeeze
    +0.3% into an invisible sliver."""
    all_up = {"signed": True, "items": [("a", 24.0), ("b", 0.3), ("c", 5.1)]}
    straddling = {"signed": True, "items": [("a", 24.0), ("b", -7.0)]}
    assert reports._is_diverging(all_up) is False
    assert reports._is_diverging(straddling) is True


def test_the_tile_grid_has_no_orphan_cell():
    """An unfilled grid cell shows the gap colour as a grey block, which reads as a
    missing tile rather than as empty space."""
    html = reports.build_infographic_html(
        "t", "body", highlights="\n".join(f"L{i} | {i} | note" for i in range(5)),
    )
    assert html.count('<div class="tile">') == 6, "the last row must be padded"


def test_a_breakdown_buried_in_a_sentence_is_still_charted():
    """Narrative reports bury their only real data inside prose. Requiring a list
    or table meant such a report charted nothing at all."""
    md = ("## Key revenue drivers\n\n"
          "- Frozen vegetables and prepared meals (core business)\n"
          "- Geographic diversification (North America ~65%, Europe ~35% of revenue)\n")
    series = reports.extract_series(md)
    assert len(series) == 1
    assert series[0]["title"] == "Geographic diversification"
    assert series[0]["items"] == [("North America", 65.0), ("Europe", 35.0)]


def test_a_lone_figure_in_prose_is_not_a_chart():
    """`potential 5-10% move higher` is a sentence, not an enumeration."""
    md = ("## Scenarios\n\n"
          "- Consensus beat: earnings beat plus raised guidance, a potential 5-10% move higher\n"
          "- Miss or guide down: risk of a 5-15% drawdown if margin commentary disappoints\n"
          "- Dividend cut: low probability but would hurt the total-return thesis\n")
    assert reports.extract_series(md) == []


def test_inline_breakdowns_never_displace_a_real_list():
    """They are the weakest signal, so they only fill space a list or table left."""
    md = ("## Sectors\n\n- **Healthcare:** 28.4%\n- **Technology:** 21.6%\n"
          "- **Industrials:** 8.1%\n- Split (North America ~65%, Europe ~35%)\n")
    series = reports.extract_series(md)
    assert series[0]["title"] == "Sectors"
    assert [s["title"] for s in series][1:] == ["Split"]


# --- refusing a chartless body ---------------------------------------------------


def test_a_body_with_nothing_chartable_is_refused_with_instructions(monkeypatch):
    """Shipping a cover of tiles and text is how a narrative report happens by
    accident. The refusal turns it into a decision."""
    called = []
    monkeypatch.setattr(reports, "render", lambda *a, **k: called.append(1) or {})
    out = reports.render_report(
        "Digest", "## Overview\n\nThe market will scrutinise revenue trends.",
        highlights="EPS | $0.37 | Aug 13",
    )
    assert out.startswith("NOT RENDERED")
    assert "allow_prose=True" in out, "the escape hatch must be offered"
    assert "28.4%" in out and "|" in out, "it must show the shapes that work"
    assert not called, "nothing should be rendered or delivered"


def test_allow_prose_renders_a_genuinely_narrative_report(monkeypatch, tmp_path):
    monkeypatch.setattr(reports, "render", lambda *a, **k: {"pdf": str(tmp_path / "r.pdf")})
    monkeypatch.setattr(reports, "page_count", lambda p: 1)
    out = reports.render_report("Digest", "Just prose.", deliver=False, allow_prose=True)
    assert out.startswith("Rendered")


def test_a_chartable_body_is_never_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(reports, "render", lambda *a, **k: {"pdf": str(tmp_path / "r.pdf")})
    monkeypatch.setattr(reports, "page_count", lambda p: 1)
    md = "## S\n\n- **A:** 10%\n- **B:** 20%\n- **C:** 30%\n"
    assert reports.render_report("t", md, deliver=False).startswith("Rendered")


def test_the_prompt_tells_the_model_about_the_refusal():
    from financial_research_assistant.graph import SYSTEM_PROMPT

    assert "allow_prose=True" in SYSTEM_PROMPT


def test_unrelated_metrics_under_one_heading_are_not_a_series():
    """From a live run: a bear-case list charted a drawdown, a geographic share and
    a volatility figure on one axis. They are all percentages and they measure
    entirely different things — one axis means comparable values."""
    md = ("## Bear case\n\n"
          "- Stock down 9.1% from the recent high\n"
          "- Max drawdown -43.7% over the past year\n"
          "- Geographic split (65% North America, 35% Europe) exposes FX risk\n"
          "- High volatility (29.3%) means a wide outcome distribution\n")
    titles = [s["title"] for s in reports.extract_series(md)]
    assert "Bear case" not in titles, "prose bullets are not a comparable series"


def test_one_over_long_label_disqualifies_the_run():
    """If a label has to be cut to fit an axis it was never a category. A majority
    rule let a half-prose run through."""
    md = ("## S\n\n- **Healthcare:** 28.4%\n- **Technology:** 21.6%\n"
          "- High volatility of 29.3% means a much wider outcome distribution\n")
    assert reports.extract_series(md) == []


def test_a_split_written_number_first_is_still_read():
    md = "## Drivers\n\n- Geographic split (65% North America, 35% Europe)\n"
    series = reports.extract_series(md)
    assert series[0]["items"] == [("North America", 65.0), ("Europe", 35.0)]


def test_a_percentage_followed_by_prose_yields_no_pair():
    """`29.3%) means a wide outcome` must not produce a label of 'means a wide'."""
    md = "## D\n\n- High volatility (29.3%) means a wide outcome distribution\n"
    assert reports.extract_series(md) == []


# --- theme ----------------------------------------------------------------------


def test_the_theme_defaults_to_light_and_is_selectable(monkeypatch):
    monkeypatch.delenv("FINANCIAL_RESEARCH_REPORT_THEME", raising=False)
    assert reports.report_theme() == "light"
    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORT_THEME", "dark")
    assert reports.report_theme() == "dark"
    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORT_THEME", "neon")
    assert reports.report_theme() == "light", "an unknown theme must not break rendering"


def test_dark_is_selected_not_an_inverted_light(monkeypatch):
    """Dark takes its own steps from the same ramps, chosen for the dark surface —
    flipping the light values would fail contrast against it."""
    light, dark = reports._THEMES["light"], reports._THEMES["dark"]
    assert dark["surface"] != light["surface"] and dark["ink"] != light["ink"]
    # the diverging poles are re-stepped, not reused
    assert dark["pos"] != light["pos"] and dark["neg"] != light["neg"]


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_both_themes_reach_the_rendered_sheet(monkeypatch, theme):
    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORT_THEME", theme)
    html = reports.build_infographic_html(
        "t", "## S\n\n- **A:** 10%\n- **B:** 20%\n- **C:** 30%\n", highlights="L | 1 | n",
    )
    surface = reports._THEMES[theme]["surface"]
    assert f"--surface:{surface}" in html
    assert reports._THEMES[theme]["pos"] in html, "bars must use the theme's pole"


def test_the_fallback_renderer_follows_the_theme(monkeypatch):
    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORT_THEME", "dark")
    assert reports._ink("surface") == (26, 26, 25)
    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORT_THEME", "light")
    assert reports._ink("surface") == (252, 252, 251)


# --- tile values ----------------------------------------------------------------


def test_a_long_tile_value_steps_down_instead_of_wrapping():
    """Observed: "Q1 2026: +24.0% surprise" wrapped onto a second line, making that
    tile taller than its neighbours. An earlier attempt rewrote the text and turned
    "Trailing P/E: 11.33" into the meaningless "Trailing P/E: 11" — the type steps
    down instead, so every character survives."""
    assert reports._tile_size("$11.78") == ""
    assert reports._tile_size("Q1 2026: +24.0% surprise") == " sm"
    assert reports._tile_size("a value far too long to sit on one line at all") == " xs"


def test_the_tile_value_is_never_rewritten():
    tiles = reports.parse_highlights("Valuation | Trailing P/E: 11.33 | Forward P/E 6.38")
    assert tiles[0]["value"] == "Trailing P/E: 11.33"


def test_the_size_class_reaches_the_html():
    html = reports.build_infographic_html(
        "t", "body", highlights="Beat | Q1 2026: +24.0% surprise | note",
    )
    assert 'class="val sm"' in html


def test_the_prompt_asks_for_a_figure_in_the_value_slot():
    from financial_research_assistant.graph import SYSTEM_PROMPT

    assert "A TILE `value` IS A FIGURE" in SYSTEM_PROMPT


# --- cover and document are themed independently ---------------------------------


def test_the_document_defaults_to_light_even_when_the_cover_is_dark(monkeypatch):
    """A dark PDF lays down a full page of ink when printed, so asking for a dark
    sheet on a phone must not quietly commit you to that."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORT_THEME", "dark")
    monkeypatch.delenv("FINANCIAL_RESEARCH_REPORT_PDF_THEME", raising=False)
    assert reports.cover_theme() == "dark"
    assert reports.pdf_theme() == "light"


def test_the_document_theme_can_be_set_too(monkeypatch):
    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORT_PDF_THEME", "dark")
    assert reports.pdf_theme() == "dark"


def test_the_theme_context_is_restored(monkeypatch):
    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORT_THEME", "light")
    with reports.use_theme("dark"):
        assert reports.report_theme() == "dark"
    assert reports.report_theme() == "light"


def test_a_dark_cover_ships_with_a_light_document(monkeypatch, tmp_path):
    """The regression this replaced: `render` took pre-built HTML, so the
    document's theme depended on whoever called it — a caller asking only for a
    dark cover silently got a dark PDF too."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORTS_DIR", str(tmp_path))
    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORT_THEME", "dark")
    monkeypatch.setenv("FINANCIAL_RESEARCH_CHROME", "/nonexistent/chrome")
    payload = {
        "title": "T", "highlights": "P | $1 | n",
        "markdown": "## S\n\n- **A:** 10%\n- **B:** 20%\n- **C:** 30%\n",
    }
    paths = reports.render(reports.build_html(**payload), "t", content=payload)

    from pathlib import Path

    saved = Path(paths["html"]).read_text(encoding="utf-8")
    light = reports._THEMES["light"]["surface"]
    assert f"--surface:{light}" in saved, "the document must be light"
    assert paths["cover"] == "infographic"

    import pypdfium2 as pdfium
    from PIL import Image

    def luma(img):
        return sum(img.convert("L").resize((24, 24)).get_flattened_data()) / 576

    doc_page = pdfium.PdfDocument(paths["pdf"])[0].render(scale=1).to_pil()
    cover = Image.open(paths["png"])
    assert luma(doc_page) > 200, "the document is not light"
    assert luma(cover) < 120, "the cover is not dark"
