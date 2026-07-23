"""Document-upload RAG tests — fully offline.

The JSONL store is redirected to a temp dir via FINANCIAL_RESEARCH_DOCS_DIR (an
autouse fixture), so ingest/ask/list/forget never touch the user's real store.
Embeddings are unavailable in tests (no endpoint), so retrieval exercises the
keyword path by default; one test monkeypatches ``embed_query``/``embed_texts`` to
exercise the semantic path. PDF support is probed via a monkeypatched pypdf import.
"""

import sys

import pytest

from financial_research_assistant import documents, tools


@pytest.fixture(autouse=True)
def _tmp_store(tmp_path, monkeypatch):
    monkeypatch.setenv("FINANCIAL_RESEARCH_DOCS_DIR", str(tmp_path / "docs"))
    yield


# --- chunking ----------------------------------------------------------------
def test_chunk_short_text_single_chunk():
    assert documents._chunk_text("A short note about revenue growth.") == \
        ["A short note about revenue growth."]


def test_chunk_long_text_overlaps_and_breaks_on_boundaries():
    para = ("Revenue rose sharply this year. " * 60).strip()  # ~1800 chars
    chunks = documents._chunk_text(para, size=600, overlap=100)
    assert len(chunks) >= 3
    assert all(len(c) <= 620 for c in chunks)          # roughly bounded
    # Consecutive chunks share overlapping context.
    assert chunks[0][-30:] in (chunks[1][:200] + chunks[0][-30:])


# --- extraction --------------------------------------------------------------
def test_extract_text_and_html(tmp_path):
    txt = tmp_path / "note.txt"
    txt.write_text("Net income was $5B in fiscal 2025.", encoding="utf-8")
    assert documents._extract_pages(txt) == [(0, "Net income was $5B in fiscal 2025.")]

    html = tmp_path / "page.html"
    html.write_text("<html><body><p>Gross margin expanded to 46%.</p>"
                    "<script>ignore()</script></body></html>", encoding="utf-8")
    pages = documents._extract_pages(html)
    assert "Gross margin expanded to 46%." in pages[0][1]
    assert "ignore" not in pages[0][1]


def test_extract_empty_file_raises(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n  ", encoding="utf-8")
    with pytest.raises(ValueError):
        documents._extract_pages(empty)


# --- ingest / ask roundtrip (keyword mode) -----------------------------------
def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return str(p)


def test_ingest_then_ask_keyword(tmp_path):
    body = ("Section 1. The company operates three segments: cloud, devices and "
            "advertising.\n\n"
            "Section 2. Total revenue for fiscal 2025 was 96 billion dollars, up "
            "12 percent year over year, driven by cloud.\n\n"
            "Section 3. The board authorized a 20 billion dollar share repurchase "
            "program and raised the quarterly dividend.")
    out = documents.ingest_document(_write(tmp_path, "annual.txt", body))
    assert "Ingested 'annual'" in out
    assert "keyword" in out.lower()

    ans = documents.ask_document("How much was the share repurchase authorization?")
    assert "repurchase" in ans.lower() and "20 billion" in ans
    assert "[annual]" in ans          # citation tag present
    assert "Synthesize" in ans        # findings-return framing


def test_ask_scoped_to_missing_doc(tmp_path):
    documents.ingest_document(_write(tmp_path, "a.txt", "Revenue grew in every region this year."))
    out = documents.ask_document("revenue", doc="nope")
    assert "No ingested document named 'nope'" in out
    assert "Available:" in out


def test_ask_before_any_ingest():
    assert "No documents have been ingested" in documents.ask_document("anything")


def test_reingest_replaces_chunks(tmp_path):
    documents.ingest_document(_write(tmp_path, "d.txt", "Old text about widgets and gadgets galore."))
    documents.ingest_document(_write(tmp_path, "d.txt", "New text about revenue and margins entirely."))
    recs = documents._load_index()
    assert all("widgets" not in r["text"] for r in recs)
    assert any("revenue" in r["text"] for r in recs)


# --- list / forget -----------------------------------------------------------
def test_list_and_forget(tmp_path):
    documents.ingest_document(_write(tmp_path, "rep.txt", "Cash flow from operations reached a record high."))
    assert "rep" in documents.list_documents()
    assert "keyword-only" in documents.list_documents()
    out = documents.forget_document("rep")
    assert "Removed 'rep'" in out
    assert "No documents ingested" in documents.list_documents()
    assert "No ingested document named 'gone'" in documents.forget_document("gone")


# --- semantic path (mocked embeddings) ---------------------------------------
def test_semantic_retrieval_prefers_cosine(tmp_path, monkeypatch):
    from financial_research_assistant import embeddings

    # Two chunks with NO shared keywords with the query; embeddings decide.
    body = ("Alpha. The logistics network spans forty distribution centers worldwide.\n\n"
            "Beta. Management emphasized supply-chain resilience and inventory buffers.")
    # Deterministic fake vectors: the query aligns with the "Beta" chunk.
    def fake_texts(texts):
        return [[1.0, 0.0] if "resilience" in t else [0.0, 1.0] for t in texts]
    monkeypatch.setattr(embeddings, "embed_texts", fake_texts)
    documents.ingest_document(_write(tmp_path, "s.txt", body))
    assert any(r.get("vec") for r in documents._load_index())

    monkeypatch.setattr(embeddings, "embed_query", lambda q: [1.0, 0.0])
    ans = documents.ask_document("How robust is the operation to disruption?", max_passages=1)
    assert "resilience" in ans.lower()      # semantic match, not keyword


# --- PDF support gate --------------------------------------------------------
def test_pdf_without_pypdf_raises_support_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pypdf", None)  # force ModuleNotFoundError
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    out = documents.ingest_document(str(pdf))
    assert "documents" in out and "pip install" in out


def test_ingest_missing_file():
    assert "File not found" in documents.ingest_document("/no/such/file.txt")


# --- registration ------------------------------------------------------------
def test_document_tools_registered():
    names = {getattr(t, "name", getattr(t, "__name__", "")) for t in tools.TOOLS}
    for n in ("ingest_document", "ask_document", "list_documents", "forget_document"):
        assert n in names
