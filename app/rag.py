"""Retrieval over the store policy manual.

Why RAG here
------------
The manual is only eight pages, so the *context budget* is not the reason to
retrieve. The reasons are:

1. **Citation.** Every answer that leans on a policy has to name the section it
   came from ("§4.1"), and a section id only exists if the document was split
   into sections somewhere.
2. **Isolation.** Sending one section instead of ten pages keeps the prompt
   small and stops unrelated rules (privacy, loyalty) from bleeding into an
   answer about freight.
3. **Change.** Policies change by section. A section-indexed corpus can be
   re-indexed per version and diffed; a blob cannot.

What RAG is *not* allowed to do: produce numbers. Windows, thresholds and
percentages come from `app/policy.py`, which computes them. Retrieval returns
prose and a citation; the engine returns the arithmetic. If retrieval fails,
the agent still answers correctly from the engine — that is the degradation
path, and `retrieve()` never raises.

Retrieval strategy: hybrid. Dense embeddings (OpenAI, cached to disk) plus a
lexical score that survives accents and plurals, because Brazilian Portuguese
questions ("troca", "trocas", "devolução") rarely match manual wording
verbatim. Lexical-only works with no API key and is the fallback.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from app.text import fold
from app.config import (
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    POLICY_INDEX_DIR,
    POLICY_PDF,
)

HEADING_LINE = re.compile(r"^\s*(\d{1,2})(?:\.(\d{1,2}))?\.?\s+(\S.{2,90})$")
PAGE_FOOTER = re.compile(r"P[áa]gina\s*\d+", re.IGNORECASE)
RUNNING_HEADER = re.compile(
    r"Emp[óo]rio da M[úu]sica\s+Manual de Pol[íi]ticas e Procedimentos",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PolicySection:
    section_id: str
    title: str
    page: int
    text: str

    @property
    def citation(self) -> str:
        return f"§{self.section_id} {self.title}"


_fold = fold  # one folding rule for SQL and for the retriever


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def read_pdf_lines(pdf_path: Path | str | None = None) -> list[tuple[int, list[str]]]:
    """Return `(page_number, non-empty lines)` using PDF layout mode.

    Layout mode matters: the default extraction flattens the manual into a
    single run of words, and a heading becomes indistinguishable from a
    sentence. In layout mode every heading sits on its own line, which is what
    makes deterministic section splitting possible.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path or POLICY_PDF))
    pages: list[tuple[int, list[str]]] = []
    for number, page in enumerate(reader.pages, start=1):
        raw = page.extract_text(extraction_mode="layout") or ""
        lines: list[str] = []
        for line in raw.splitlines():
            clean = _squash(line)
            if not clean:
                continue
            if RUNNING_HEADER.search(clean):
                clean = _squash(RUNNING_HEADER.sub("", clean))
                if not clean:
                    continue
            if PAGE_FOOTER.fullmatch(clean):
                continue
            lines.append(clean)
        pages.append((number, lines))
    return pages


def _valid_sequence(candidates: list[tuple[str, str, int, int]]) -> list[tuple[str, str, int, int]]:
    """Keep only candidates that form an increasing section numbering.

    A real manual numbers its sections in order (1, 1.1, 1.2, 2, ...), so any
    line that would move the counter backwards is body text (a price, a CNPJ, a
    time range), not a heading.
    """
    kept: list[tuple[str, str, int, int]] = []
    top, sub = 0, 0
    for section_id, title, position, page in candidates:
        major, _, minor = section_id.partition(".")
        major_i = int(major)
        if minor:
            if major_i != top or int(minor) <= sub:
                continue
            sub = int(minor)
        else:
            if major_i <= top:
                continue
            top, sub = major_i, 0
        kept.append((section_id, title, position, page))
    return kept


def split_sections(pages: Iterable[tuple[int, list[str]]] | None = None) -> list[PolicySection]:
    """Split the manual into numbered sections, each with page and citation."""
    pages = list(pages if pages is not None else read_pdf_lines())

    # Flatten into one list of (page, line) so a section can span pages.
    flat: list[tuple[int, str]] = [(page, line) for page, lines in pages for line in lines]

    candidates: list[tuple[str, str, int, int]] = []
    for position, (page, line) in enumerate(flat):
        match = HEADING_LINE.match(line)
        if not match:
            continue
        major, minor, title = match.group(1), match.group(2), _squash(match.group(3))
        if len(title.split()) > 10 or title.endswith((".", ":", ";")):
            continue
        section_id = f"{major}.{minor}" if minor else major
        candidates.append((section_id, title, position, page))

    sections = _valid_sequence(candidates)

    result: list[PolicySection] = []
    for index, (section_id, title, start, page) in enumerate(sections):
        end = sections[index + 1][2] if index + 1 < len(sections) else len(flat)
        body_lines = [
            text for position, text in flat[start:end] if not HEADING_LINE.match(text)
        ]
        result.append(
            PolicySection(
                section_id=section_id,
                title=title,
                page=page,
                text=_squash(" ".join(body_lines)),
            )
        )
    return result


STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "da", "do", "das", "dos",
    "em", "no", "na", "nos", "nas", "por", "para", "pra", "com", "sem", "que", "se",
    "eu", "voce", "voces", "ele", "ela", "meu", "minha", "me", "te", "lhe", "ja",
    "nao", "sim", "mais", "menos", "muito", "pouco", "como", "quando", "onde", "qual",
    "quais", "quanto", "quantos", "e", "ou", "mas", "tambem", "tem", "ter", "foi",
    "ser", "sou", "esta", "estou", "aqui", "ai", "la", "so", "todo", "toda", "isso",
    "este", "esse", "aquele", "pelo", "pela", "ao", "aos", "as", "sua", "seu",
}

# Customer vocabulary -> manual vocabulary. Brazilian customers ask "posso
# devolver?" and "que horas abre?", and the manual answers "direito de
# arrependimento" and "horário de funcionamento". A lexical retriever that does
# not bridge that gap returns the wrong section, so the bridge is explicit,
# versioned in code and covered by tests instead of being left to luck.
CONCEPT_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "devolver": ("devolucao", "arrependimento", "recebimento", "reembolso"),
    "devolucao": ("arrependimento", "recebimento"),
    "trocar": ("troca", "defeito", "elegiveis"),
    "troca": ("defeito", "elegiveis", "preferencia"),
    "estragou": ("defeito", "garantia", "fabricante"),
    "defeito": ("troca", "garantia", "fabricante"),
    "frete": ("entrega", "correios", "sedex", "jadlog", "cep", "prazo"),
    "entrega": ("frete", "prazo", "correios", "motoboy"),
    "chegar": ("entrega", "prazo"),
    "abre": ("horario", "funcionamento", "atendimento"),
    "abrem": ("horario", "funcionamento", "atendimento"),
    "fecha": ("horario", "funcionamento"),
    "funcionamento": ("horario", "atendimento"),
    "horario": ("funcionamento", "atendimento"),
    "rastreio": ("rastreamento", "codigo", "despachado"),
    "rastrear": ("rastreamento", "codigo"),
    "codigo": ("rastreamento", "despachado"),
    "palheta": ("acessorios", "corda", "cabo", "case"),
    "corda": ("acessorios", "palheta", "cabo"),
    "cabo": ("acessorios", "palheta", "pedal"),
    "pedal": ("acessorios", "amplificador"),
    "amplificador": ("acessorios", "pedal"),
    "parcela": ("parcelamento", "cartao", "credito"),
    "parcelar": ("parcelamento", "cartao", "credito"),
    "vezes": ("parcelamento", "credito"),
    "pix": ("desconto", "pagamento", "cumulativa"),
    "desconto": ("promocao", "pix", "cumulativa"),
    "promocao": ("desconto", "promocoes", "ativa"),
    "garantia": ("fabricante", "defeito", "cobre"),
    "bateria": ("instrumento", "catalogo"),
}


def _tokens(text: str) -> list[str]:
    return [t for t in _fold(text).split() if len(t) > 2 and t not in STOPWORDS]


def _expand(terms: list[str]) -> list[tuple[str, float]]:
    """Return (term, weight) pairs: original query terms plus lexicon bridges."""
    weighted: dict[str, float] = {}
    for term in terms:
        weighted[term] = max(weighted.get(term, 0.0), 1.0)
        for extra in CONCEPT_EXPANSIONS.get(term, ()):
            if extra not in weighted:
                weighted[extra] = 0.6
    return list(weighted.items())


class BM25:
    """Okapi BM25 over the section corpus: IDF + term saturation + length norm.

    Implemented here rather than pulled from a library because it is thirty
    lines, has no dependency, and is the offline fallback that keeps the tool
    answering when the embedding API is unavailable.
    """

    def __init__(self, sections: list[PolicySection], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs: list[dict[str, int]] = []
        self.lengths: list[int] = []
        for section in sections:
            counts: dict[str, int] = {}
            for token in _tokens(f"{section.title} {section.title} {section.text}"):
                counts[token] = counts.get(token, 0) + 1
            self.docs.append(counts)
            self.lengths.append(sum(counts.values()))
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 1.0
        document_frequency: dict[str, int] = {}
        for counts in self.docs:
            for token in counts:
                document_frequency[token] = document_frequency.get(token, 0) + 1
        self.idf = {
            token: math.log(1 + (len(self.docs) - freq + 0.5) / (freq + 0.5))
            for token, freq in document_frequency.items()
        }

    def scores(self, query: str) -> list[float]:
        terms = _expand(_tokens(query))
        results = [0.0] * len(self.docs)
        for position, counts in enumerate(self.docs):
            length = self.lengths[position] or 1
            total = 0.0
            for term, weight in terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    # Prefix tolerance: "devolv" matches "devolucao".
                    if len(term) > 4:
                        partial = sum(
                            value for token, value in counts.items() if token.startswith(term)
                        )
                        frequency = partial
                if not frequency:
                    continue
                idf = self.idf.get(term, math.log(1 + len(self.docs) / 1))
                denominator = frequency + self.k1 * (1 - self.b + self.b * length / self.avg_length)
                total += weight * idf * (frequency * (self.k1 + 1)) / denominator
            results[position] = total
        return results


def lexical_score(query: str, section: PolicySection) -> float:
    """Single-document BM25 score (kept for tests and ad-hoc lookups)."""
    return BM25([section]).scores(query)[0]


# --------------------------------------------------------------------------- #
# Dense index
# --------------------------------------------------------------------------- #


def _index_paths(index_dir: Path | None = None) -> tuple[Path, Path]:
    directory = Path(index_dir or POLICY_INDEX_DIR)
    return directory / "sections.json", directory / "embeddings.json"


def build_index(
    index_dir: Path | None = None,
    sections: list[PolicySection] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Extract sections and cache their embeddings. Returns a small report."""
    sections = sections or split_sections(read_pdf_lines())
    sections_path, embeddings_path = _index_paths(index_dir)
    sections_path.parent.mkdir(parents=True, exist_ok=True)

    payload = [asdict(section) for section in sections]
    sections_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report: dict[str, Any] = {"sections": len(sections), "embedded": 0, "model": None}
    if not OPENAI_API_KEY:
        report["skipped"] = "no OPENAI_API_KEY set; lexical retrieval only"
        return report

    from langchain_openai import OpenAIEmbeddings

    embedding_model = model or OPENAI_EMBEDDING_MODEL
    embedder = OpenAIEmbeddings(model=embedding_model, api_key=OPENAI_API_KEY)
    vectors = embedder.embed_documents([f"{s.citation}\n{s.text}" for s in sections])
    embeddings_path.write_text(
        json.dumps({"model": embedding_model, "vectors": vectors}), encoding="utf-8"
    )
    report["embedded"] = len(vectors)
    report["model"] = embedding_model
    return report


class PolicyIndex:
    """Hybrid retriever over the manual sections. Never raises on retrieval."""

    def __init__(self, index_dir: Path | None = None) -> None:
        self.sections_path, self.embeddings_path = _index_paths(index_dir)
        self._sections: list[PolicySection] | None = None
        self._vectors: list[list[float]] | None = None
        self._embedder: Any = None
        self._bm25: BM25 | None = None

    # -- lazy loaders ------------------------------------------------------- #
    @property
    def sections(self) -> list[PolicySection]:
        if self._sections is None:
            if self.sections_path.exists():
                raw = json.loads(self.sections_path.read_text(encoding="utf-8"))
                self._sections = [PolicySection(**item) for item in raw]
            else:
                self._sections = split_sections(read_pdf_lines())
        return self._sections

    @property
    def vectors(self) -> list[list[float]] | None:
        if self._vectors is None and self.embeddings_path.exists():
            self._vectors = json.loads(self.embeddings_path.read_text(encoding="utf-8"))["vectors"]
        return self._vectors

    def _embed_query(self, query: str) -> list[float] | None:
        if not OPENAI_API_KEY:
            return None
        if self._embedder is None:
            from langchain_openai import OpenAIEmbeddings

            self._embedder = OpenAIEmbeddings(
                model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY
            )
        try:
            return self._embedder.embed_query(query)
        except Exception:  # network/credential problem: fall back to lexical
            return None

    # -- search ------------------------------------------------------------- #
    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
        return dot / norm if norm else 0.0

    @property
    def bm25(self) -> BM25:
        if self._bm25 is None:
            self._bm25 = BM25(self.sections)
        return self._bm25

    def search(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []
        lexical_scores = self.bm25.scores(query)
        best_lexical = max(lexical_scores) if lexical_scores else 0.0
        query_vector = self._embed_query(query) if self.vectors else None

        scored: list[tuple[float, int]] = []
        for position, section in enumerate(self.sections):
            lexical = lexical_scores[position] / best_lexical if best_lexical else 0.0
            dense = 0.0
            if query_vector and self.vectors and position < len(self.vectors):
                dense = max(0.0, self._cosine(query_vector, self.vectors[position]))
            # Dense dominates when available; lexical keeps exact-term hits alive
            # and carries the whole answer when embeddings are unavailable.
            score = (0.65 * dense + 0.35 * lexical) if query_vector else lexical
            if score > 0:
                scored.append((score, position))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "section": self.sections[position].section_id,
                "title": self.sections[position].title,
                "page": self.sections[position].page,
                "citation": self.sections[position].citation,
                "score": round(score, 4),
                "text": self.sections[position].text,
            }
            for score, position in scored[:k]
        ]


_DEFAULT_INDEX: PolicyIndex | None = None


def get_index() -> PolicyIndex:
    global _DEFAULT_INDEX
    if _DEFAULT_INDEX is None:
        _DEFAULT_INDEX = PolicyIndex()
    return _DEFAULT_INDEX
