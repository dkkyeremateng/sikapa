"""Rendered reports: document building, stat tiles, and file delivery.

Offline — Chrome is never invoked (the one test that would is skipped without a
binary), and no channel sends anything real.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
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


def test_an_unsigned_two_digit_table_cell_keeps_its_leading_digit():
    """`[+-−]` is a RANGE spanning every digit, not three literals, so the sign
    group ate the leading digit and `74.1%` charted as `4.1%` — a plausible wrong
    number on a sheet whose whole promise is that it cannot invent one. Signed and
    thousands-separated cells backtracked into the right answer, which is why only
    the unsigned two-digit case ever showed it."""
    assert reports._cell_number("74.1%") == (74.1, "%")
    assert reports._cell_number("11.65") == (11.65, "")
    assert reports._cell_number("+11.4%") == (11.4, "%")
    assert reports._cell_number("-4.8%") == (-4.8, "%")
    # The unit carries the magnitude now — see the scale test below. The VALUE
    # is what this test is about, and it is unchanged.
    assert reports._cell_number("$208.7B") == (208.7, "$B")
    assert reports._cell_number("1,234.5") == (1234.5, "")


def test_a_parenthesised_number_is_the_negative_it_means():
    """Accounting notation, and the default in anything transcribed off a financial
    statement. Read as positive it does not just misstate the size — it points the
    bar the wrong way, charting a cash burn as cash generated."""
    assert reports._cell_number("(2.30)") == (-2.3, "")
    assert reports._cell_number("$(84.0)") == (-84.0, "$")
    assert reports._cell_number("(1,204)") == (-1204.0, "")
    assert reports._cell_number("(12.5%)") == (-12.5, "%")


def test_a_parenthetical_aside_is_not_a_negative():
    """The narrow rule that keeps the fix from creating its own wrong numbers: the
    parentheses have to wrap the number and nothing else."""
    assert reports._cell_number("11.65x (trailing)") == (11.65, "")
    assert reports._cell_number("(4.2% of total)") == (4.2, "%")


def test_a_negative_value_makes_its_series_signed_however_it_was_written():
    """Keyed only off a leading +/- glyph, an accounting negative left the series
    "unsigned" — so the loss drew in the gain colour and its label lost the minus."""
    series = reports.extract_series(
        "## EPS\n| Q | EPS |\n|---|---|\n"
        "| Q2 | $1.84 |\n| Q1 | $(0.42) |\n| Q4 | $0.19 |\n"
    )[0]
    assert series["signed"] is True
    assert dict(series["items"])["Q1"] == -0.42


def test_money_bar_labels_keep_the_cents_that_carry_the_meaning():
    """Whole dollars rounded an EPS table into uselessness: $1.84 -> "$2", and
    $0.19 -> "$0", which says the opposite of what the source did."""
    assert reports._fmt_value(1.84, "$", False) == "$1.84"
    assert reports._fmt_value(0.19, "$", False) == "$0.19"
    assert reports._fmt_value(-0.42, "$", True) == "-$0.42"
    assert reports._fmt_value(1234.5, "$", False) == "$1,234.50"
    # Dropped only when they are literally ".00", so one chart never mixes "$180"
    # with "$1.84" the way a magnitude threshold would.
    assert reports._fmt_value(180.0, "$", True) == "+$180"


def test_no_extractor_invents_a_number_the_source_never_stated():
    """The invariant behind three separate bugs found in this file: a value the
    sheet shows must be a COMPLETE numeric token from its source, never a fragment
    of one. `74.1% -> 4.1%` violated exactly this."""
    import re as _re

    token = _re.compile(r"\d[\d,]*(?:\.\d+)?")

    def stated(text):
        return {float(t.replace(",", "")) for t in token.findall(text)}

    for cell in ("74.1%", "73.2%", "11.65", "$1,234.50", "(2.30)", "24.8x",
                 "-4.8%", "−4.8%", "+11.4%", "$208.7B", "100%", "0.9%"):
        parsed = reports._cell_number(cell)
        assert parsed is not None, cell
        assert abs(parsed[0]) in stated(cell), f"{cell} -> {parsed[0]} is not in it"


def test_a_charted_table_reports_the_figures_the_table_states():
    series = reports.extract_series(
        "## Guidance\n"
        "| Metric | Guided |\n|---|---|\n"
        "| Q4 revenue | 5.2% |\n| Gross margin | 74.1% |\n| Opex growth | 9.4% |\n"
    )
    assert series[0]["items"] == [
        ("Q4 revenue", 5.2), ("Gross margin", 74.1), ("Opex growth", 9.4)
    ]


# --- verdict figures promoted from headings -------------------------------------
#
# The NVO report that prompted these: five highlight tiles of context, and the
# one number the whole second half argued for — a fear price — written as a
# heading, so it reached the PDF body and never the cover image at all.


def test_a_verdict_written_as_a_heading_becomes_a_tile():
    figures = reports.heading_figures(
        "## Fear Price: $32.00 – $38.00\n\nScenario 1: earnings recession.\n"
    )
    assert figures == [{"label": "Fear Price", "value": "$32.00 – $38.00", "note": ""}]


def test_a_promoted_figure_keeps_its_parenthetical_as_the_note():
    figures = reports.heading_figures("**Fair Value: $61.40 (DCF, 9% WACC)**")
    assert figures[0]["value"] == "$61.40"
    assert figures[0]["note"] == "DCF, 9% WACC"


def test_a_heading_that_is_prose_is_not_promoted():
    """The anchors are what separate a tile value from a sentence that happens to
    contain a number. Without them "Coverage: 12 analysts" becomes a stat tile."""
    assert reports.heading_figures(
        "## Analyst Consensus\n"
        "## Coverage: 12 analysts\n"
        "## Q2 2026: A Massive Beat\n"
        "## Bottom Line\n"
    ) == []


def test_bullets_are_not_promoted_only_headings():
    """A number in a heading is a verdict; the same number in a bullet is one of
    many, and promoting those would fill the sheet with whatever came first."""
    assert reports.heading_figures("- Mean Target: $47.28\n- Median Target: $45.06\n") == []


def test_the_verdict_displaces_the_least_important_tile_when_all_six_are_full():
    tiles = reports.cover_tiles(
        "\n".join(f"L{i} | V{i}" for i in range(6)),
        "## Fear Price: $32.00 – $38.00\n",
    )
    assert len(tiles) == 6
    assert tiles[-1]["label"] == "Fear Price"
    assert [t["label"] for t in tiles[:5]] == [f"L{i}" for i in range(5)]
    assert "L5" not in {t["label"] for t in tiles}


def test_a_verdict_the_model_already_tiled_is_not_duplicated():
    tiles = reports.cover_tiles(
        "Fear Price | $32–$38 | capitulation zone",
        "## Fear Price: $32.00 – $38.00\n",
    )
    assert [t["label"] for t in tiles] == ["Fear Price"]
    assert tiles[0]["note"] == "capitulation zone", "the model's own wording wins"


def test_promotion_is_bounded_so_scenarios_cannot_evict_every_tile():
    tiles = reports.cover_tiles(
        "\n".join(f"L{i} | V{i}" for i in range(6)),
        "".join(f"## Scenario {i}: ${i}0.00\n" for i in range(5)),
    )
    assert len(tiles) == 6
    assert sum(1 for t in tiles if t["label"].startswith("Scenario")) == 2


def test_the_cover_and_the_document_both_show_the_verdict():
    """Same tiles on both surfaces: the image is what gets read on a phone, and a
    figure that differs between the two reads as one of them being wrong."""
    md = "## Fear Price: $32.00 – $38.00\n\n- **Healthcare:** 28.4%\n- **Tech:** 21.0%\n- **Energy:** 9.4%\n"
    hl = "Current Price | $47.20 | -24.2% YTD"
    for html in (reports.build_html("NVO", md, highlights=hl),
                 reports.build_infographic_html("NVO", md, highlights=hl)):
        # The tile markup, not the string: the heading is in the rendered body of
        # both documents either way, so a bare `"Fear Price" in html` passes with
        # the promotion removed entirely — it did, until this was tightened.
        assert '<div class="lab">Fear Price</div>' in html
        assert "$32.00 – $38.00" in html.split('<div class="body">')[0]


# --- the call: buy, sell or hold ------------------------------------------------


def test_a_stance_carries_its_tone_and_reason():
    assert reports.parse_stance("HOLD") == {"label": "HOLD", "tone": "hold", "note": ""}
    assert reports.parse_stance("buy")["tone"] == "pos"
    assert reports.parse_stance("underweight")["tone"] == "neg"
    got = reports.parse_stance("Strong Buy | franchise intact")
    assert got["label"] == "STRONG BUY", "the longest phrase wins over a prefix of it"
    assert got["note"] == "franchise intact"


def test_a_long_stance_reason_is_cut_at_a_word_not_mid_letter():
    """A hard slice ended a reason on a bare letter ("…the cheapest leverage i"),
    which reads as a rendering fault rather than as text that was shortened."""
    note = reports.parse_stance(
        "HOLD | cash drag is real but the float is still the cheapest leverage "
        "in finance and that has not changed this quarter or the last one"
    )["note"]
    assert note.endswith("…")
    assert not note.rstrip("…").endswith(" ")
    assert note.rstrip("…").split()[-1] in (
        "cash drag is real but the float is still the cheapest leverage in finance "
        "and that has not changed this quarter or the last one"
    ).split()


def test_a_stance_we_cannot_colour_is_refused_rather_than_guessed():
    """Badging the wrong tone on this field is worse than omitting it, so an
    unknown word produces no badge — and `render_report` says why."""
    assert reports.parse_stance("maybe") is None
    assert reports.parse_stance("") is None


def test_an_unrecognised_stance_is_reported_not_swallowed():
    out = reports.render_report(
        "t", "- **A:** 10.0%\n- **B:** 20.0%\n- **C:** 30.0%\n",
        stance="probably fine", deliver=False,
    )
    assert "No stance badge" in out and "probably fine" in out


def test_a_stance_heading_is_read_but_a_stance_bullet_is_not():
    """The NVO report carried `Rating: BUY` as a bullet under Analyst Consensus —
    the street's view, which it argued was stale before concluding HOLD. Reading
    bullets would badge the sheet with the opinion it existed to disagree with."""
    assert reports.stance_of("", "## Rating: HOLD (trim 50%)")["label"] == "HOLD"
    assert reports.stance_of("", "- Rating: BUY (consensus 2.43)") is None


def test_an_explicit_stance_beats_a_heading():
    assert reports.stance_of("SELL", "## Rating: BUY")["label"] == "SELL"


def test_the_badge_is_drawn_on_both_the_cover_and_the_document():
    md = "## Fear Price: $32.00\n\n- **A:** 10.0%\n- **B:** 20.0%\n- **C:** 30.0%\n"
    for html in (reports.build_html("NVO", md, stance="HOLD | trim 50% at $40"),
                 reports.build_infographic_html("NVO", md, stance="HOLD | trim 50% at $40")):
        assert '<span class="pill hold">HOLD</span>' in html
        assert "trim 50% at $40" in html.split('<div class="rule">')[0]


def test_the_emitted_stylesheet_survives_templating():
    """The badge's markup can be perfect while its CSS is dead. `_STANCE_CSS` is
    spliced in as a `.format()` VALUE, and a value is not re-processed — doubled
    braces reached the stylesheet as `.stance{{...}}`, so every rule was dropped
    and the badge rendered as unstyled text run together with its reason. Asserting
    on the markup alone passed throughout."""
    for html in (reports.build_html("t", "x", stance="BUY"),
                 reports.build_infographic_html("t", "x", stance="BUY")):
        assert ".stance{display:flex" in html
        assert ".stance .pill.pos{color:var(--pos)" in html
        assert "{{" not in html and "}}" not in html, "unprocessed format braces"


def test_the_badge_tone_follows_the_verdict():
    for verdict, tone in (("BUY", "pos"), ("SELL", "neg"), ("HOLD", "hold")):
        html = reports.build_infographic_html("t", "x", stance=verdict)
        assert f'<span class="pill {tone}">{verdict}</span>' in html


def test_a_stance_is_escaped_like_every_other_model_supplied_string():
    html = reports.build_infographic_html("t", "x", stance="BUY | <script>alert(1)</script>")
    # Scoped to the badge: the template ends with its own measuring <script>, so a
    # whole-document check would pass on that and prove nothing about the stance.
    badge = html.split('<div class="stance">')[1].split("</div>")[0]
    assert "<script>" not in badge and "&lt;script&gt;" in badge


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
    monkeypatch.setattr(reports, "render", lambda html, name, content=None, theme="", output="": {
        "html": str(tmp_path / "r.html"), "png": str(tmp_path / "r.png"),
    })
    monkeypatch.setattr(channels, "deliver_file", lambda p, caption="", prefer="", full_quality=False: (["telegram"], []))
    out = reports.render_report("FISV Q2", "body text", deliver=True, allow_prose=True)
    assert "Sent to: telegram" in out and "PNG" in out


def test_the_tool_says_so_when_nothing_can_receive_a_file(monkeypatch, tmp_path):
    monkeypatch.setattr(reports, "render", lambda html, name, content=None, theme="", output="": {"png": str(tmp_path / "r.png")})
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

    monkeypatch.setattr(reports, "render", lambda html, name, content=None, theme="", output="": paths)
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


# --- the renderer's sandbox ------------------------------------------------------
#
# The page Chrome renders is written by the MODEL, and a report routinely
# summarises web-search results and filing text — content this project treats as
# untrusted everywhere else. The escaping is what actually prevents injection; the
# sandbox is the layer that still holds if that escaping is ever missed.


def test_the_sandbox_is_on_by_default(monkeypatch):
    monkeypatch.delenv("FINANCIAL_RESEARCH_CHROME_NO_SANDBOX", raising=False)
    monkeypatch.setattr(os, "geteuid", lambda: 501, raising=False)
    assert "--no-sandbox" not in reports._chrome_flags("chrome")


def test_running_as_root_disables_it_without_configuration(monkeypatch):
    """Chrome refuses to start sandboxed as root — the usual container case. It is
    DETECTED rather than configured so a Docker run still renders without anyone
    having to discover the flag."""
    monkeypatch.delenv("FINANCIAL_RESEARCH_CHROME_NO_SANDBOX", raising=False)
    monkeypatch.setattr(os, "geteuid", lambda: 0, raising=False)
    assert "--no-sandbox" in reports._chrome_flags("chrome")


def test_it_can_be_turned_off_explicitly(monkeypatch):
    """For a confined environment where the sandbox can't get its namespaces."""
    monkeypatch.setattr(os, "geteuid", lambda: 501, raising=False)
    for value in ("1", "true", "on", "YES"):
        monkeypatch.setenv("FINANCIAL_RESEARCH_CHROME_NO_SANDBOX", value)
        assert "--no-sandbox" in reports._chrome_flags("chrome"), value
    monkeypatch.setenv("FINANCIAL_RESEARCH_CHROME_NO_SANDBOX", "0")
    assert "--no-sandbox" not in reports._chrome_flags("chrome")


def test_both_chrome_passes_share_the_flags(monkeypatch, tmp_path):
    """The measuring pass and the print pass load the same page, so a sandbox
    setting that applied to only one of them would be no setting at all."""
    seen: list[list[str]] = []
    monkeypatch.setattr(reports, "_run_chrome", lambda args: seen.append(args))
    monkeypatch.setattr(reports, "_chrome_flags", lambda c: [c, "--sentinel"])

    html = tmp_path / "sheet.html"
    html.write_text("<html><head><style>@page{size:1080px 1500px}</style></head></html>")
    reports._measure_height("chrome", html.as_uri())
    reports._chrome_pdf("chrome", html, tmp_path / "sheet.pdf")

    assert len(seen) == 3, "one measuring pass each, plus the print"
    assert all("--sentinel" in args for args in seen), seen


def test_the_sheet_is_sent_uncompressed(monkeypatch, tmp_path):
    """sendPhoto re-encodes to JPEG and downscales, which turns dense body text to
    mush regardless of the render resolution. The sheet must go as a document."""
    seen: list[bool] = []
    monkeypatch.setattr(reports, "render", lambda html, name, content=None, theme="", output="": {"png": str(tmp_path / "r.png")})
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


# --- generation modes ------------------------------------------------------------


def test_the_renderer_can_be_pinned(monkeypatch, tmp_path):
    """Pinning matters for a container or for reproducing a bug; the only way to
    force the browser-free path used to be aiming the Chrome variable at a path
    that does not exist."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORTS_DIR", str(tmp_path))
    monkeypatch.delenv("FINANCIAL_RESEARCH_CHROME", raising=False)
    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORT_RENDERER", "fpdf2")
    payload = {"title": "T", "markdown": "## S\n\n- **A:** 10%\n- **B:** 20%\n- **C:** 30%\n"}
    assert reports.render(reports.build_html(**payload), "t", content=payload)["renderer"] == "fpdf2"


def test_a_pinned_chrome_does_not_silently_fall_back(monkeypatch, tmp_path):
    """Pinning means pinning: falling back would hide the very failure being
    reproduced."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORTS_DIR", str(tmp_path))
    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORT_RENDERER", "chrome")
    monkeypatch.setattr(reports, "chrome_path", lambda: "")
    payload = {"title": "T", "markdown": "## S\n\n- **A:** 10%\n"}
    assert "pdf" not in reports.render(reports.build_html(**payload), "t", content=payload)


def test_an_unknown_mode_falls_back_rather_than_breaking(monkeypatch):
    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORT_RENDERER", "inkscape")
    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORT_OUTPUT", "hologram")
    assert reports.renderer_mode() == "auto"
    assert reports.output_mode() == "both"


def test_pdf_only_output_skips_the_cover(monkeypatch, tmp_path):
    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORTS_DIR", str(tmp_path))
    monkeypatch.setenv("FINANCIAL_RESEARCH_CHROME", "/nonexistent/chrome")
    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORT_OUTPUT", "pdf")
    payload = {"title": "T", "markdown": "## S\n\n- **A:** 10%\n- **B:** 20%\n- **C:** 30%\n"}
    paths = reports.render(reports.build_html(**payload), "t", content=payload)
    assert "pdf" in paths and "png" not in paths


def test_output_mode_selects_what_is_delivered(monkeypatch, tmp_path):
    sent: list[str] = []
    monkeypatch.setattr(reports, "render", lambda *a, **k: {
        "png": str(tmp_path / "r.png"), "pdf": str(tmp_path / "r.pdf"), "renderer": "chrome",
    })
    monkeypatch.setattr(reports, "page_count", lambda p: 1)
    monkeypatch.setattr(
        channels, "deliver_file",
        lambda p, caption="", prefer="", full_quality=False: (
            sent.append(p.rsplit(".", 1)[-1]), (["telegram"], []))[1],
    )
    md = "## S\n\n- **A:** 10%\n- **B:** 20%\n- **C:** 30%\n"
    reports.render_report("t", md, output="image")
    assert sent == ["png"]
    sent.clear()
    reports.render_report("t", md, output="pdf")
    assert sent == ["pdf"]


def test_the_cover_only_promises_a_pdf_when_one_is_delivered():
    """Under `output="image"` the document stays on disk and never reaches the
    reader, so a footer pointing at an attachment points at nothing."""
    assert reports._cover_tail(3, True) == "Summary of a 3-page report — full PDF attached"
    # The page count survives — it says how much was distilled — the promise does not.
    assert reports._cover_tail(3, False) == "Summary of a 3-page report"
    # One page with nothing attached leaves the line saying nothing at all.
    assert reports._cover_tail(1, False) == ""


@pytest.mark.parametrize("renderer", ["chrome", "fpdf2"])
def test_an_image_only_render_drops_the_attachment_line(monkeypatch, tmp_path, renderer):
    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORT_OUTPUT", "image")
    monkeypatch.setattr(reports, "_SINGLE_MAX_H", 500)
    monkeypatch.setattr(reports, "_PAGE_H", 500)  # force several pages
    payload = {
        "title": "Portfolio Analysis Report", "markdown": _REPORT_MD,
        "highlights": "Portfolio Value | $38,420 | +$1,860",
    }
    paths = _render_with(monkeypatch, tmp_path, payload, renderer)
    assert paths["cover"] == "infographic"

    import pypdfium2 as pdfium

    cover_pdf = tmp_path / (Path(paths["pdf"]).stem + "-cover.pdf")
    text = " ".join(
        pdfium.PdfDocument(str(cover_pdf))[0].get_textpage().get_text_range().split()
    )
    assert "PDF attached" not in text
    assert "Summary of a" in text, "the page count is still worth stating"


def test_a_per_report_theme_reaches_the_render_as_an_argument(monkeypatch, tmp_path):
    """One report's theme travels down the call, not through the environment: the
    configured default is never written, so it cannot be read back."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORT_THEME", "light")
    seen: list[dict] = []
    monkeypatch.setattr(reports, "render", lambda *a, **k: seen.append(k) or {})
    reports.render_report("t", "## S\n\n- **A:** 10%\n- **B:** 20%\n- **C:** 30%\n",
                          deliver=False, theme="dark")
    assert seen[0]["theme"] == "dark"
    assert reports.cover_theme("dark") == "dark"      # the override wins where used
    assert reports.cover_theme() == "light"           # the default is untouched


def test_one_reports_theme_cannot_bleed_into_a_concurrent_one(monkeypatch, tmp_path):
    """The adapter runs tools in threads, so two render_report calls overlap. The
    dark one used to publish its theme process-wide, and the light one drew its
    cover in dark — the bug this argument-passing exists to prevent."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORTS_DIR", str(tmp_path))
    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORT_THEME", "light")
    md = "## S\n\n- **A:** 10%\n- **B:** 20%\n- **C:** 30%\n"
    seen: list[tuple[str, str]] = []
    barrier = threading.Barrier(2)

    def fake_pdf(pdf, *a, **k):
        # Rendezvous #1: neither call resolves its cover theme until BOTH are
        # inside a render, so a theme announced process-wide is guaranteed to be
        # in force while the other call reads it.
        barrier.wait(timeout=5)
        Path(pdf).write_bytes(b"%PDF-1.4")
        return True

    def fake_build_cover(out_dir, stem, content, pages, chrome, attached=True):
        # Rendezvous #2: neither call may finish (and undo a process-wide theme)
        # until both have resolved theirs.
        barrier.wait(timeout=5)
        seen.append((content["title"], reports.report_theme()))
        return None

    monkeypatch.setattr(reports, "_build_cover", fake_build_cover)
    monkeypatch.setattr(reports, "_chrome_pdf", lambda *a, **k: False)
    monkeypatch.setattr(reports, "_fpdf_pdf", fake_pdf)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(reports.render_report, "dark one", md, deliver=False, theme="dark"),
            pool.submit(reports.render_report, "light one", md, deliver=False, theme=""),
        ]
        for f in futures:
            f.result()
    assert dict(seen) == {"dark one": "dark", "light one": "light"}


def test_a_render_without_pypdfium2_does_not_claim_zero_pages(monkeypatch, tmp_path):
    """`page_count` needs an optional dependency. Reporting "Rendered 0 page(s)"
    for a PDF that is on disk and about to be delivered reads as a failure."""
    monkeypatch.setattr(reports, "render", lambda *a, **k: {
        "pdf": str(tmp_path / "r.pdf"), "renderer": "fpdf2",
    })
    monkeypatch.setattr(reports, "page_count", lambda p: 0)  # no pypdfium2
    out = reports.render_report("t", "Just prose.", deliver=False, allow_prose=True)
    assert "0 page(s)" not in out
    assert out.startswith("Rendered a PDF with fpdf2")


def test_two_reports_rendered_in_the_same_second_do_not_overwrite(monkeypatch, tmp_path):
    """The stem is title + timestamp to the second, so a batch that renders two
    reports on one subject used to write both over the same files — the first
    report's PDF replaced by the second's while the caller was told it was saved."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORTS_DIR", str(tmp_path))
    monkeypatch.setattr(reports, "chrome_path", lambda: "")
    monkeypatch.setattr(reports, "_fpdf_pdf",
                        lambda pdf, *a, **k: bool(Path(pdf).write_bytes(b"%PDF-1.4")) or True)
    payload = {"title": "NVDA", "markdown": "## S\n\n- **A:** 10%\n"}
    first = reports.render(reports.build_html(**payload), "NVDA", content=payload)
    second = reports.render(reports.build_html(**payload), "NVDA", content=payload)
    assert first["html"] != second["html"]
    assert first["pdf"] != second["pdf"]
    assert Path(first["pdf"]).exists() and Path(second["pdf"]).exists()


def test_a_series_whose_labels_are_all_the_same_is_not_charted():
    """Bullets written `- **Problem:** ...` all reduce to the same label, because
    the split takes the text before the colon — which is where the label lives in
    the shape this was built for. The result charted six bars reading "Problem":
    looks like six categories, names none of them, and the reader cannot tell
    which figure belongs to what. Observed on a delivered sheet."""
    series = reports.extract_series(
        "## What the alignment review shows\n"
        "- **Problem:** Tech concentration 77.0% above target\n"
        "- **Problem:** Defensive sleeve -20.0% versus policy\n"
        "- **Problem:** Cash buffer drift 3.7%\n"
        "- **Problem:** Top-5 concentration 64.2%\n"
    )
    assert series == [], "a one-category axis carries no information"


def test_repeated_labels_in_a_table_are_rejected_too():
    assert reports.extract_series(
        "## Risks\n| Item | Value |\n|---|---|\n"
        "| Risk | 10.0% |\n| Risk | 20.0% |\n| Risk | 30.0% |\n"
    ) == []


def test_distinct_labels_still_chart():
    """The guard must not cost the ordinary case."""
    series = reports.extract_series(
        "## Weights\n- **VOO:** 27.2%\n- **UNH:** 12.4%\n- **AMZN:** 10.0%\n"
    )
    assert [l for l, _v in series[0]["items"]] == ["VOO", "UNH", "AMZN"]


def test_a_commentary_section_is_not_charted():
    """Bullets under "Key Observations" each quote whatever figure their sentence
    is about, so charting the run put a month's return, a position weight and a
    two-month total on one axis — four bars sharing nothing but the % sign. It
    also collided with the sheet's own "Key observations" block, printing that
    heading twice. Observed on a delivered sheet."""
    md = (
        "## Key Observations\n"
        "- **April carried the year:** best month was April with +17.87%\n"
        "- **Concentration risk:** VOO alone is 27.2% of the book\n"
        "- **Crypto hedge modest:** Bitcoin allocation 5.5% added little\n"
        "- **Recent flatness:** June and July returned only +0.61%\n"
    )
    assert reports.extract_series(md) == []
    # The content is not lost — it belongs in the observations block, whole.
    assert len(reports.extract_notes(md, set())) == 4
    assert reports.build_infographic_html("t", md).count("<h2>Key observations</h2>") == 1


def test_a_real_breakdown_under_a_data_heading_still_charts():
    """The denylist is on prose headings only; it must not cost a real series."""
    series = reports.extract_series(
        "## Sector Exposure\n- **Healthcare:** 27.9%\n- **Technology:** 21.3%\n"
        "- **Industrials:** 7.8%\n"
    )
    assert series and series[0]["title"] == "Sector Exposure"


def test_a_figure_in_a_parenthetical_aside_is_not_the_items_measure():
    """`- Sharpe ratio: 1.06 (risk-free rate = 0%)` charted as 0.0%: the first
    percent sign belongs to the aside, and a ratio is not a percentage at all. The
    bar read as the metric while showing a number from its own footnote. Observed
    on a delivered risk profile."""
    series = reports.extract_series(
        "## Risk Profile\n"
        "- Annualised volatility: 16.0%\n"
        "- Sharpe ratio: 1.06 (risk-free rate = 0%)\n"
        "- Beta vs SPY: 0.91 (less volatile than the market)\n"
        "- Max drawdown: -20.2% (measured over 2025-08-11 to 2026-08-10)\n"
    )
    # Sharpe and beta carry no percentage of their own, so only two items remain —
    # below the minimum. No chart is the right answer: volatility, a ratio and a
    # drawdown were never a comparable set.
    assert series == []


def test_an_aside_does_not_stop_a_real_breakdown_charting():
    series = reports.extract_series(
        "## Sector Exposure\n"
        "- **Healthcare (US listed):** 27.9% — UNH, NVO\n"
        "- **Technology:** 21.3% (mega-cap weighted)\n"
        "- **Industrials:** 7.8%\n"
    )
    assert [v for _l, v in series[0]["items"]] == [27.9, 21.3, 7.8]


# --- the portfolio-review guard --------------------------------------------------

_REVIEW_HL = (
    "YTD Time-Weighted Return | +19.35% | deposit-independent\n"
    "Portfolio Value | $41,265.40 | current NAV\n"
    "Unrealised P/L | +$7,412.60 | since purchase"
)
_CHARTABLE = "## Mix\n- **A:** 10.0%\n- **B:** 20.0%\n- **C:** 30.0%\n"


def test_a_hand_built_performance_review_is_refused():
    """Routing to `render_review` was asked for in the system prompt and in this
    module's docstring, and three runs with both loaded ignored both. The last did
    not merely retype a figure — it invented the whole monthly series (January
    "+5.2%" against a real +0.10%, April "+1.9%" against a real +17.87%) and put
    the trailing-twelve-month return on the sheet as year-to-date. A prompt
    requests; a refusal decides."""
    out = reports.render_report(
        "Year-to-Date Portfolio Review 2026", _CHARTABLE,
        highlights=_REVIEW_HL, subtitle="As of August 10, 2026", deliver=False,
    )
    assert out.startswith("NOT RENDERED")
    assert "render_review" in out


def test_render_review_itself_is_not_refused():
    with reports.reviewing():
        out = reports.render_report(
            "Portfolio Performance — 2026 year to date", _CHARTABLE,
            highlights=_REVIEW_HL, deliver=False,
        )
    assert not out.startswith("NOT RENDERED")


def test_a_single_stock_report_is_not_a_portfolio_review():
    """The guard must cost nothing to every other report on the system."""
    out = reports.render_report(
        "NVO — Novo Nordisk | Biotech Under Pressure", _CHARTABLE,
        highlights="Current Price | $47.20 | -24.2% YTD", deliver=False,
    )
    assert not out.startswith("NOT RENDERED")


def test_a_portfolio_allocation_sheet_is_not_a_performance_review():
    """A portfolio sheet that is not a REVIEW — allocation, risk, tax lots — has no
    review word and different tiles, and keeps rendering."""
    for title, highlights in (
        ("Portfolio Allocation Breakdown",
         "Total value | $41,265.40 | 11 positions\nTop-5 | 64.2% | concentration"),
        ("Portfolio Risk Profile",
         "Volatility | 16.0% | annualised\nBeta vs SPY | 0.91 | trailing"),
    ):
        out = reports.render_report(title, _CHARTABLE, highlights=highlights,
                                    deliver=False)
        assert not out.startswith("NOT RENDERED"), title


def test_the_guard_needs_two_performance_tiles_not_one():
    """One return figure on a portfolio-titled sheet is not enough to call it a
    review — the bar is deliberately above a single coincidence."""
    out = reports.render_report(
        "Portfolio Review 2026", _CHARTABLE,
        highlights="Total value | $41,265.40 | 11 positions", deliver=False,
    )
    assert not out.startswith("NOT RENDERED")


def test_an_undelivered_report_says_it_was_not_sent():
    """With deliver=False the result was "Rendered … Saved: …" and nothing else —
    indistinguishable from a delivered one. A report that never left the machine
    could therefore be reported to the user as sent, which is what they noticed."""
    out = reports.render_report("T", _CHARTABLE, highlights="X | 1 |", deliver=False)
    assert "NOT SENT" in out
    assert "deliver=True" in out, "and it says how to actually send it"


def test_an_inline_label_is_not_a_sentence_cut_mid_phrase():
    """"Healthcare is 27.9%" yields the label "Healthcare is", which charts as a
    fragment. The list path refuses those via `_is_label`; this path had no
    equivalent, and shipped bars reading "Healthcare is" and "Consumer Cyclical
    at"."""
    got = reports._inline_series(
        "Healthcare is 27.9% and Consumer Cyclical at 14.8% of the book.", "h"
    )
    assert [l for l, _v in got["items"]] == ["Healthcare", "Consumer Cyclical"]


def test_prose_in_a_commentary_section_is_not_charted_as_a_breakdown():
    """An inline breakdown names itself after the SENTENCE it came from, so a
    commentary section slipped past the prose-heading check that already governs
    list and table series. It shipped a chart headed "Concentration risk is
    material and rising. VOO a" — a sentence cut at the title limit — with bars
    reading "VOO alone" and "AMZN together represent"."""
    assert reports.extract_series(
        "## Observations\n"
        "- Concentration risk is material and rising. VOO alone is 27.2% and "
        "AMZN together represent 24.0% of the book.\n"
    ) == []


def test_a_real_inline_breakdown_under_a_data_heading_still_charts():
    series = reports.extract_series(
        "## Revenue Mix\n- Split by region (North America ~65%, Europe ~35%)\n"
    )
    assert series[0]["items"] == [("North America", 65.0), ("Europe", 35.0)]


def test_a_long_observation_is_clipped_onto_the_cover_not_dropped():
    """A hard 150-character ceiling excluded every real observation the agent
    writes — "April carried the year: a +17.87% month against seven others near
    flat or negative…" runs to 218 — so the analysis reached the PDF while the
    cover, the thing actually read on a phone, carried none of it."""
    long_note = (
        "April carried the year: a +17.87% month against seven others near flat "
        "or negative. All other monthly moves net to -8.49%, showing the "
        "portfolio's gains are concentrated in one volatile month."
    )
    assert len(long_note) > 150
    notes = reports.extract_notes(f"## Observations\n- {long_note}\n", set())
    assert len(notes) == 1
    assert notes[0].startswith("April carried the year")
    assert len(notes[0]) <= reports._NOTE_DISPLAY + 1  # +1 for the ellipsis


def test_a_bullet_too_short_to_be_an_observation_is_still_skipped():
    assert reports.extract_notes("## Observations\n- Too short\n", set()) == []


def test_a_paragraph_is_not_an_observation():
    """Past a point a "bullet" is a paragraph, and the cover is a summary sheet."""
    assert reports.extract_notes("## Notes\n- " + "word " * 200 + "\n", set()) == []


def test_a_magnitude_suffix_survives_onto_the_bar_label():
    """Dropped, "$115.2B" charted as "$115.20" — a figure a billion times smaller
    than the source, and one that reads as perfectly ordinary."""
    assert reports._cell_number("$115.2B") == (115.2, "$B")
    assert reports._fmt_value(115.2, "$B", False) == "$115.20B"
    # "bn" normalises to one letter, or the formatter cannot see the suffix.
    assert reports._cell_number("€38.6bn")[1] == "B"
    assert reports._fmt_value(38.6, "B", False) == "38.60B"
    for cell, unit in (("980M", "M"), ("1.2T", "T")):
        assert reports._cell_number(cell)[1] == unit


def test_a_capital_letter_in_prose_is_not_a_magnitude():
    """Anchored to the number's end, so "$4.98 Beat" is not read as billions."""
    assert reports._cell_number("$4.98 Beat") == (4.98, "$")
    assert reports._cell_number("2.5yr coverage") == (2.5, "")


def test_a_billions_table_charts_with_its_scale():
    series = reports.extract_series(
        "## Segment Revenue\n| Segment | Revenue |\n|---|---|\n"
        "| Data centre | $115.2B |\n| Gaming | $11.4B |\n| Automotive | $1.7B |\n"
    )[0]
    assert series["unit"] == "$B"
    assert reports._fmt_value(series["items"][0][1], series["unit"], False) == "$115.20B"


def test_mixed_magnitudes_in_one_column_do_not_share_an_axis():
    """$980M drawn beside $115.2B would be the longer bar. Refusing is the honest
    outcome — the existing mixed-unit rule now sees the scale too."""
    assert reports.extract_series(
        "## Revenue\n| Segment | Revenue |\n|---|---|\n"
        "| Data centre | $115.2B |\n| Gaming | $980.0M |\n| Automotive | $1.7B |\n"
    ) == []
