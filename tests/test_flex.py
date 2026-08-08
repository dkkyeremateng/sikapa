"""IBKR Flex sync — the two-step fetch and where the statement lands.

The saved XML is the single most sensitive file this project writes: account
numbers, every position, every trade. It gets the same 0600 treatment as the
credential store.
"""

import stat

import pytest

from financial_research_assistant import flex

STATEMENT = (
    '<FlexQueryResponse queryName="Activity">'
    '<AccountInformation accountId="U1234567" name="Jane Doe"/>'
    "</FlexQueryResponse>"
)


@pytest.fixture(autouse=True)
def _tmp_flex_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("FINANCIAL_RESEARCH_FLEX_DIR", str(tmp_path / "flex"))


def test_the_saved_statement_is_0600():
    dest = flex.save_flex_xml(STATEMENT, "123456")
    mode = stat.S_IMODE(dest.stat().st_mode)
    assert mode == 0o600, f"flex statement is {oct(mode)}"
    assert dest.read_text(encoding="utf-8") == STATEMENT


def test_flex_sync_saves_privately(monkeypatch):
    """The permission has to hold on the path a user actually reaches, not only
    on the helper called directly."""
    monkeypatch.setattr(flex, "fetch_flex_xml", lambda *a, **k: STATEMENT)
    out = flex.flex_sync(query_id="123456", token="t0ken")

    written = list(flex.flex_dir().glob("flex-*.xml"))
    assert len(written) == 1
    assert stat.S_IMODE(written[0].stat().st_mode) == 0o600
    assert str(written[0]) in out


def test_no_temporary_file_survives_the_write():
    """`write_private` stages in the destination directory so the rename is
    atomic rather than a cross-filesystem copy; nothing may be left behind."""
    flex.save_flex_xml(STATEMENT, "123456")
    assert [p.name for p in flex.flex_dir().iterdir() if p.name.startswith(".flex-")] == []
