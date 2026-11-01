"""Text normalisation shared by the database, the queries and the retriever.

Brazilian customers type "violao", "violão", "violões" and "violoes" for the
same thing, and the CSV spells it only one way. Doing this in one place means
the SQL index and the retrieval index cannot disagree about what a word is.
"""

from __future__ import annotations

import re
import unicodedata


def fold(text: str | None) -> str:
    """Lowercase, strip accents, collapse plurals, keep letters/digits/spaces."""
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    folded = "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()
    folded = folded.replace("oes", "ao").replace("aes", "ao")
    folded = re.sub(r"[^a-z0-9 ]+", " ", folded)
    words = [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in folded.split()]
    return " ".join(words)
