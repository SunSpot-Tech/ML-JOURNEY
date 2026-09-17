"""
embedding_retriever.py — semantic retrieval, the real fix for TF-IDF's
homonym/synonym blind spot (see retriever.py's docstring for the failures
this addresses: "16-year-old" vs "age of 18", and "tax rate" vs "corrosion
rate" — same words, different meaning, which TF-IDF cannot tell apart).

Requires internet access on first run only, to download the model once
(cached locally afterward at ~/.cache/torch or similar — no per-query
network calls after that).

Install:
    pip install sentence-transformers

Same interface as retriever.TfidfRetriever, so agent.py only needs a
one-line import change to swap this in.
"""
import json
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# all-MiniLM-L6-v2: small (~80MB), fast on CPU, good general-purpose quality.
# Swap for a larger model later if retrieval quality needs it once you have
# a real eval set (Phase 5) to measure against.
MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingRetriever:
    def __init__(self, chunks_path: str = "data/chunks.json"):
        self.chunks = json.loads(Path(chunks_path).read_text())
        self.model = SentenceTransformer(MODEL_NAME)
        corpus = [c["text"] for c in self.chunks]
        # encode once at startup; in production this happens at ingest time
        # and gets stored in pgvector, not recomputed on every process start.
        self.embeddings = self.model.encode(corpus, convert_to_tensor=True)

    def _content_word_overlap(self, query: str, chunk_text: str) -> int:
        q_tokens = {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", query)} - ENGLISH_STOP_WORDS
        c_tokens = {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", chunk_text)} - ENGLISH_STOP_WORDS
        return len(q_tokens & c_tokens)

    def retrieve(self, query: str, k: int = 3, tenant_sector: str | None = None):
        candidates = self.chunks
        embeddings = self.embeddings
        if tenant_sector:
            idx = [i for i, c in enumerate(self.chunks) if c["sector"] == tenant_sector]
            if not idx:
                return []
            candidates = [self.chunks[i] for i in idx]
            embeddings = self.embeddings[idx]

        q_emb = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, embeddings).flatten().tolist()
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def overlap_for(self, query: str, chunk: dict) -> int:
        return self._content_word_overlap(query, chunk["text"])


if __name__ == "__main__":
    r = EmbeddingRetriever()
    test_queries = [
        "confined space entry",
        "Can I hire a 16-year-old for offshore work?",
        "What is the tax rate for oil exports?",  # should now score noticeably
                                                    # lower / less confidently
                                                    # than genuine matches,
                                                    # since embeddings should
                                                    # separate "tax rate" from
                                                    # "corrosion rate" by meaning
    ]
    for query in test_queries:
        print(f"\nQuery: {query}")
        for chunk, score in r.retrieve(query, k=3):
            print(f"  [reg {chunk['regulation_no']}] score={score:.3f} :: {chunk['text'][:70]}...")
