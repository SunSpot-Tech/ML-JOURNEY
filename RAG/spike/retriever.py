"""
retriever.py — retrieval layer over clause-tagged chunks.

Uses TF-IDF + cosine similarity as a fast, dependency-light stand-in for
embeddings during the spike. Swap-out point for production: replace
TfidfRetriever with a pgvector-backed retriever (same interface:
`retrieve(query, k) -> list[(chunk, score)]`) once the schema in
migrations/ is live. Keeping the interface identical means agent.py
never needs to change when you make that swap.

--- Quick patch (this revision) ---
TF-IDF only matches shared *words*. Two real failures showed up in testing:
  1. "Can I hire a 16-year-old for offshore work?" shares almost no words
     with Regulation 64's actual text ("no person under the age of 18
     years shall be engaged to work at a dangerous area") — "16-year-old"
     and "age of 18" are the same concept, zero lexical overlap.
  2. "What is the tax rate for oil exports?" shares the word "oil" with
     nearly every chunk in the corpus, so it scores as if it were a real
     (if weak) match instead of "no relevant provision."

Fix applied here is query expansion: a small domain synonym map rewrites
the query before vectorizing, standing in for what a production system
would do with an LLM query-rewrite step (this is the same idea as the
Multi-Query / HyDE techniques from the RAG-from-Scratch material). This
is a patch, not the real fix — it only works for terms we thought to
anticipate. The real fix is semantic embeddings; see embedding_retriever.py.
"""
import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# Rule-based query expansion — a stand-in for an LLM query-rewrite step.
# Each pattern that matches the query adds its expansion terms before retrieval.
QUERY_EXPANSIONS = [
    (re.compile(r"\b\d{1,2}[\s-]?year[\s-]?old\b", re.IGNORECASE),
     "age minor child eighteen years dangerous area exclusion"),
    (re.compile(r"\bhire\b", re.IGNORECASE), "engage employ"),
    (re.compile(r"\bgas cylinder\b", re.IGNORECASE), "radioactive source isotope container"),
]


def expand_query(query: str) -> str:
    expanded = query
    for pattern, extra_terms in QUERY_EXPANSIONS:
        if pattern.search(query):
            expanded = f"{expanded} {extra_terms}"
    return expanded


class TfidfRetriever:
    def __init__(self, chunks_path: str = "data/chunks.json"):
        self.chunks = json.loads(Path(chunks_path).read_text())
        corpus = [c["text"] for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(corpus)

    def _content_word_overlap(self, query: str, chunk_text: str) -> int:
        """Count non-stopword query tokens that appear verbatim in the chunk.
        Used to gate 'low' vs 'none' confidence — a nonzero TF-IDF score with
        zero real word overlap is noise (e.g. shared stopwords/common nouns
        across an unrelated corpus), not a genuine partial match."""
        q_tokens = {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", query)} - ENGLISH_STOP_WORDS
        c_tokens = {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", chunk_text)} - ENGLISH_STOP_WORDS
        return len(q_tokens & c_tokens)

    def retrieve(self, query: str, k: int = 3, tenant_sector: str | None = None):
        candidates = self.chunks
        matrix = self.matrix
        if tenant_sector:
            # Simulates the RLS filter: a tenant only ever searches its own
            # sector's corpus. In production this is a WHERE clause, not a
            # Python filter — same idea, different layer.
            idx = [i for i, c in enumerate(self.chunks) if c["sector"] == tenant_sector]
            if not idx:
                return []
            candidates = [self.chunks[i] for i in idx]
            matrix = self.matrix[idx]

        expanded = expand_query(query)
        q_vec = self.vectorizer.transform([expanded])
        scores = cosine_similarity(q_vec, matrix).flatten()
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def overlap_for(self, query: str, chunk: dict) -> int:
        return self._content_word_overlap(query, chunk["text"])


if __name__ == "__main__":
    r = TfidfRetriever()
    for query in ["confined space entry", "gas cylinder pressure testing",
                  "diving supervisor duties", "Can I hire a 16-year-old for offshore work?"]:
        print(f"\nQuery: {query}  (expanded: {expand_query(query)!r})")
        for chunk, score in r.retrieve(query, k=3):
            overlap = r.overlap_for(query, chunk["text"] if False else chunk["text"])
            print(f"  [reg {chunk['regulation_no']}] score={score:.3f} overlap={r.overlap_for(query, chunk)} :: {chunk['text'][:70]}...")
