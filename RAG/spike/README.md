# Spike: Regulatory-Intelligence Agent (Day 0 Proof)

Real source, real code, no notebooks: ingests the actual **Upstream Petroleum
Safety Regulations 2022** (NUPRC) and answers queries with clause-level
citations and a confidence tag.

## Files
- `data/nuprc_upstream_petroleum_safety_regulations_2022.txt` — real regulation text (fetched from nuprc.gov.ng)
- `ingest.py` — clause-aware chunker/tagger (CLI, not a notebook)
- `retriever.py` — TF-IDF retrieval (stand-in for embeddings; same interface pgvector will use later)
- `agent.py` — the answer contract: `{query, confidence, answer, citations}`

## Run it
```bash
python ingest.py --file data/nuprc_upstream_petroleum_safety_regulations_2022.txt \
                  --act "Upstream Petroleum Safety Regulations 2022" --sector oil_and_gas \
                  --source-url "https://www.nuprc.gov.ng/wp-content/uploads/2022/08/Upstream-Petroleum-Safety-Regulations.pdf" \
                  --out data/chunks.json
python agent.py
```

## What worked
- Clause-aware chunking correctly isolated 28 regulations with Part/section metadata intact.
- Retrieval correctly surfaced Regulation 33 for "confined space entry" and Regulation 75/78 for diving duties — high match scores, right answer.

## What broke — and why it matters
- **"Can I hire a 16-year-old for offshore work?"** should hit Regulation 64 (exclusion of minors from dangerous areas) with high confidence. It came back **low confidence**, and Regulation 64 ranked *second*, beaten by Regulation 13 (work-at-height) purely on keyword overlap with "offshore."
- **"What is the tax rate for oil exports?"** — a question this corpus has no answer to — should have returned `confidence: "none"`. It returned `"low"` with an irrelevant citation instead of abstaining.

This is the actual argument for Phase 5 (evaluation) in the 16-day plan, and for real embeddings over TF-IDF in production: TF-IDF matches vocabulary, not meaning, so it can't tell "no relevant law exists" apart from "the wording didn't overlap." A semantic embedding model plus a stricter, eval-tuned abstention threshold is what turns "low confidence" into a trustworthy signal instead of a guess. Shipping the "none" case wrong is worse than a slow answer — it's a false compliance signal.

## Next swap-in points (no redesign needed)
- `TfidfRetriever` → pgvector-backed retriever, same `.retrieve()` interface
- `LLM_SYNTHESIZE()` → real Claude call, same input/output contract
- `tenant_sector` filter → real RLS policy in Postgres
