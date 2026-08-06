"""Rendered reports: document building, stat tiles, and file delivery.

Offline — Chrome is never invoked (the one test that would is skipped without a
binary), and no channel sends anything real.
"""

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
    out = reports.render_report("FISV Q2", "body text", deliver=True)
    assert "Sent to: telegram" in out and "PNG" in out


def test_the_tool_says_so_when_nothing_can_receive_a_file(monkeypatch, tmp_path):
    monkeypatch.setattr(reports, "render", lambda html, name, content=None: {"png": str(tmp_path / "r.png")})
    monkeypatch.setattr(channels, "deliver_file", lambda p, caption="", prefer="", full_quality=False: ([], []))
    assert "No channel accepted a file" in reports.render_report("t", "b")


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
    out = reports.render_report("t", "b", deliver=False)
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
    reports.render_report("t", "b")
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
def test_a_long_report_paginates_and_the_image_is_page_one(
    monkeypatch, tmp_path, renderer
):
    """The requirement a screenshot could not express: the cover is page 1, not
    the whole scroll."""
    paths = _render_with(monkeypatch, tmp_path, _LONG, renderer)
    pages = reports.page_count(paths["pdf"])
    assert pages > 1, f"{renderer} did not paginate a long report"

    import pypdfium2 as pdfium
    from PIL import Image

    doc = pdfium.PdfDocument(paths["pdf"])
    # pypdfium2's scale is pixels per POINT, so the cover is page-height-in-points
    # times the scale — not the CSS pixel height.
    expected = doc[0].get_size()[1] * reports.render_scale()
    cover = Image.open(paths["png"])
    assert cover.height == pytest.approx(expected, rel=0.02), (
        "the cover image is not exactly one page tall"
    )


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
