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
    monkeypatch.setattr(reports, "render", lambda html, name: {
        "html": str(tmp_path / "r.html"), "png": str(tmp_path / "r.png"),
    })
    monkeypatch.setattr(channels, "deliver_file", lambda p, caption="", prefer="", full_quality=False: (["telegram"], []))
    out = reports.render_report("FISV Q2", "body text", deliver=True)
    assert "Sent to: telegram" in out and "PNG" in out


def test_the_tool_says_so_when_nothing_can_receive_a_file(monkeypatch, tmp_path):
    monkeypatch.setattr(reports, "render", lambda html, name: {"png": str(tmp_path / "r.png")})
    monkeypatch.setattr(channels, "deliver_file", lambda p, caption="", prefer="", full_quality=False: ([], []))
    assert "No channel accepted a file" in reports.render_report("t", "b")


def test_a_missing_chrome_still_returns_the_html(monkeypatch, tmp_path):
    """A finished analysis must never be lost to a rendering problem."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORTS_DIR", str(tmp_path))
    monkeypatch.setattr(reports, "chrome_path", lambda: "")
    paths = reports.render(reports.build_html("t", "b"), "t")
    assert set(paths) == {"html"}
    assert paths["html"].endswith(".html")

    monkeypatch.setattr(reports, "render", lambda html, name: paths)
    out = reports.render_report("t", "b", deliver=False)
    assert "no Chrome/Chromium found" in out


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
    monkeypatch.setattr(reports, "render", lambda html, name: {"png": str(tmp_path / "r.png")})
    monkeypatch.setattr(
        channels, "deliver_file",
        lambda p, caption="", prefer="", full_quality=False: (
            seen.append(full_quality), (["telegram"], []))[1],
    )
    reports.render_report("t", "b")
    assert seen == [True]
