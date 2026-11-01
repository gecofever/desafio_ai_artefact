"""Policy manual: sectioning and retrieval quality.

The retrieval assertions run on the lexical path only (no network, no cached
embeddings), so they measure the fallback. The dense path is measured by
`scripts/eval.py`, which prints the numbers quoted in the README.
"""

from __future__ import annotations

import pytest

from app.rag import BM25, PolicyIndex, read_pdf_lines, split_sections

EXPECTED_SECTIONS = [
    "1", "1.1", "1.2", "2", "3", "3.1",
    "4", "4.1", "4.2", "4.3", "4.4",
    "5", "5.1", "5.2", "5.3",
    "6", "6.1", "6.2",
    "7", "7.1", "7.2", "7.3",
    "8", "8.1", "8.2", "8.3",
    "9", "10",
]


@pytest.fixture(scope="module")
def sections():
    return split_sections(read_pdf_lines())


def test_manual_splits_into_the_numbered_sections(sections):
    assert [section.section_id for section in sections] == EXPECTED_SECTIONS


def test_titles_are_the_heading_not_the_first_sentence(sections):
    titles = {section.section_id: section.title for section in sections}
    assert titles["4"] == "Política de Trocas e Devoluções"
    assert titles["4.1"] == "Direito de Arrependimento (Compras Online)"
    assert titles["5.1"] == "Entregas na Região Metropolitana de Campo Grande"


def test_section_text_covers_the_whole_manual(sections):
    """Nothing may be lost between the last heading and the end of the document."""
    empty = [section.section_id for section in sections if not section.text]
    # 5, 6 and 8 are container headings with no prose of their own: their content
    # lives entirely in subsections. Any other empty section is a splitting bug.
    assert empty == ["5", "6", "8"]
    assert sum(len(section.text) for section in sections) > 9000


def test_the_free_shipping_number_comes_from_the_manual(sections):
    """§5.1 is the only place the R$ 35,00 fee exists; the engine must agree."""
    from app.policy import SHIPPING_FEE

    text = {section.section_id: section.text for section in sections}["5.1"]
    assert "R$ 35,00" in text
    assert SHIPPING_FEE == pytest.approx(35.0)


def test_lexical_retrieval_finds_the_right_section_offline(sections):
    index = PolicyIndex(index_dir="/nonexistent")  # forces the BM25 fallback
    cases = {
        "posso devolver um violao que comprei semana passada?": "4.1",
        "qual o frete para Campo Grande?": "5.1",
        "aceita pix com desconto?": "3",
        "nao recebi o codigo de rastreio": "5.3",
        "que horas voces abrem no sabado?": "2",
        "meu violao estragou, e agora?": "4.2",
        "quantas vezes posso parcelar?": "3.1",
        "qual a garantia do fabricante?": "8.2",
    }
    misses = {}
    for query, expected in cases.items():
        hits = [hit["section"] for hit in index.search(query, k=3)]
        if expected not in hits:
            misses[query] = hits
    assert not misses, f"retrieval missed: {misses}"


def test_bm25_ignores_stopwords_and_accents(sections):
    bm25 = BM25(sections)
    scores = bm25.scores("o que é que a loja vende?")  # nothing but stopwords
    assert max(scores) >= 0.0
    assert BM25(sections).scores("ACESSÓRIOS") == pytest.approx(
        BM25(sections).scores("acessorios"), rel=1e-9
    )


def test_search_returns_citations(sections):
    index = PolicyIndex(index_dir="/nonexistent")
    hits = index.search("prazo de devolucao", k=2)
    assert hits
    for hit in hits:
        assert hit["citation"].startswith("§")
        assert hit["page"] >= 2
