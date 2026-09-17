"""
agent.py — the regulatory-intelligence agent's answer contract.

answer(query, tenant_sector) -> {
    "query": ...,
    "confidence": "high" | "low" | "none",
    "answer": ...,          # plain-language synthesis
    "citations": [...],     # clause-level, always present when confidence != none
}

This spike's synthesis step is a deterministic template (no LLM call — the
sandbox has no API key wired). The retrieval + citation + confidence logic
below is the actual moat and works identically once an LLM is plugged in
at LLM_SYNTHESIZE(). Swapping in Claude is a single function body change,
not a redesign — the contract above does not change.
"""
from retriever import TfidfRetriever

CONFIDENCE_THRESHOLD_HIGH = 0.30
CONFIDENCE_THRESHOLD_LOW = 0.08


def LLM_SYNTHESIZE(query: str, top_chunks: list) -> str:
    """
    Plug point for production: replace this body with a call to Claude,
    passing `query` and `top_chunks` as the ONLY grounding context, with a
    system prompt that forbids answering outside the provided chunks.
    Left deterministic here so the spike runs with zero external dependencies.
    """
    lead = top_chunks[0]
    return (f"Based on Regulation {lead['regulation_no']} of the "
            f"{lead['act']}, the applicable requirement concerns: "
            f"{lead['part_title']}.")


def answer(query: str, tenant_sector: str = "oil_and_gas", k: int = 3):
    retriever = TfidfRetriever()
    results = retriever.retrieve(query, k=k, tenant_sector=tenant_sector)

    if not results or results[0][1] < CONFIDENCE_THRESHOLD_LOW:
        return {
            "query": query,
            "confidence": "none",
            "answer": ("No matching provision found in the ingested corpus for "
                       "this jurisdiction/sector. This is not a legal conclusion "
                       "of 'no requirement exists' — it means the corpus does not "
                       "yet cover this question. Escalate to a human HSE reviewer."),
            "citations": [],
        }

    # Overlap gate: a nonzero TF-IDF score with zero real content-word overlap
    # is noise (shared common nouns like "oil"/"operation" across an unrelated
    # corpus), not a genuine partial match. Without this gate, "tax rate for
    # oil exports" was returning a "low confidence" citation instead of
    # admitting the corpus has no answer — a false compliance signal.
    top_overlap = retriever.overlap_for(query, results[0][0])
    if top_overlap == 0:
        return {
            "query": query,
            "confidence": "none",
            "answer": ("No matching provision found in the ingested corpus for "
                       "this jurisdiction/sector. This is not a legal conclusion "
                       "of 'no requirement exists' — it means the corpus does not "
                       "yet cover this question. Escalate to a human HSE reviewer."),
            "citations": [],
        }

    top_chunks = [c for c, _ in results]
    top_score = results[0][1]
    confidence = "high" if top_score >= CONFIDENCE_THRESHOLD_HIGH else "low"

    citations = [{
        "act": c["act"],
        "part": f"Part {c['part_number']} — {c['part_title']}",
        "regulation_no": c["regulation_no"],
        "source_url": c["source_url"],
        "match_score": round(score, 3),
    } for c, score in results]

    return {
        "query": query,
        "confidence": confidence,
        "answer": LLM_SYNTHESIZE(query, top_chunks),
        "citations": citations,
    }


if __name__ == "__main__":
    import json
    for q in [
        "Can a worker enter a confined space alone?",
        "What are the pipeline pressure testing requirements?",
        "Can I hire a 16-year-old for offshore work?",
        "What is the tax rate for oil exports?",  # deliberately unanswerable
    ]:
        result = answer(q)
        print(json.dumps(result, indent=2))
        print("-" * 60)
