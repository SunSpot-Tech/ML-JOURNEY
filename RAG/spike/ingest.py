"""
ingest.py — Clause-aware ingestion for Nigerian regulatory text.

Usage:
    python ingest.py --file data/nuprc_upstream_petroleum_safety_regulations_2022.txt \
                      --act "Upstream Petroleum Safety Regulations 2022" \
                      --sector "oil_and_gas" \
                      --out data/chunks.json

Design note: generic fixed-size chunking destroys legal citation accuracy
(a 500-char window can straddle two unrelated clauses). This parser instead
splits on regulation-number boundaries ("17 -", "44 -", etc.) and keeps the
enclosing PART as context, so every chunk is citable down to the regulation
number.
"""
import argparse
import json
import re
from pathlib import Path

# Matches lines like "17 - (a) The blow-out preventer..." or "1 - A Company..."
REGULATION_START = re.compile(r"^(\d{1,3})\s*-\s*", re.MULTILINE)
PART_HEADER = re.compile(r"^PART\s+([IVX]+)\s*-\s*(.+)$", re.MULTILINE)


def split_into_parts(text: str):
    """Split the raw text into (part_number, part_title, body_text) tuples."""
    matches = list(PART_HEADER.finditer(text))
    parts = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        parts.append({
            "part_number": m.group(1),
            "part_title": m.group(2).strip(),
            "body": text[start:end],
        })
    if not parts:
        parts = [{"part_number": None, "part_title": None, "body": text}]
    return parts


def split_into_clauses(body: str):
    """Split a PART's body into individual numbered regulations (clauses)."""
    matches = list(REGULATION_START.finditer(body))
    clauses = []
    for i, m in enumerate(matches):
        reg_no = m.group(1)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        clause_text = body[start:end].strip()
        if len(clause_text) < 15:
            continue
        clauses.append({"regulation_no": reg_no, "text": clause_text})
    return clauses


def make_chunks(raw_text: str, act: str, sector: str, source_url: str = ""):
    parts = split_into_parts(raw_text)
    chunks = []
    chunk_id = 0
    for part in parts:
        for clause in split_into_clauses(part["body"]):
            chunk_id += 1
            chunks.append({
                "chunk_id": f"{act.replace(' ', '_')}_{chunk_id:04d}",
                "act": act,
                "sector": sector,
                "part_number": part["part_number"],
                "part_title": part["part_title"],
                "regulation_no": clause["regulation_no"],
                "text": clause["text"],
                "source_url": source_url,
            })
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--act", required=True)
    ap.add_argument("--sector", default="oil_and_gas")
    ap.add_argument("--source-url", default="")
    ap.add_argument("--out", default="data/chunks.json")
    ap.add_argument("--append", action="store_true",
                     help="Append to an existing chunk store instead of overwriting it. "
                          "Without this flag, --out is replaced fresh every run.")
    args = ap.parse_args()

    raw_text = Path(args.file).read_text()
    # Strip our own header lines (SOURCE:/URL:) before parsing clauses
    body_start = raw_text.find("PART I")
    raw_text = raw_text[body_start:] if body_start != -1 else raw_text

    chunks = make_chunks(raw_text, args.act, args.sector, args.source_url)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if args.append and out_path.exists():
        existing = json.loads(out_path.read_text())
    existing.extend(chunks)
    out_path.write_text(json.dumps(existing, indent=2))

    print(f"Parsed {len(chunks)} clause-tagged chunks from '{args.act}'.")
    print(f"Sample chunk metadata: part={chunks[0]['part_number']} "
          f"({chunks[0]['part_title']}), regulation_no={chunks[0]['regulation_no']}")
    print(f"Written to {out_path} (total chunks in file: {len(existing)})")


if __name__ == "__main__":
    main()
