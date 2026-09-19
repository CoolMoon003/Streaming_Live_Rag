import json
from pathlib import Path


REQUIRED_FIELDS = ("chunk_id", "doc_id", "section", "text", "source")


def load_chunks(chunks_path):
    """
    Load chunks.jsonl defensively.

    Skips blank lines and malformed JSON, and normalizes missing
    metadata fields so downstream retrievers never hit a KeyError.
    """
    path = Path(chunks_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {path} "
            f"(resolved: {path.resolve()}). "
            "Generate it with: python -m scripts.ingest"
        )

    chunks = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            if not line:
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                print(f"[chunk_loader] Skipping malformed JSON on line {line_number}")
                continue

            if not isinstance(chunk, dict):
                print(f"[chunk_loader] Skipping non-object record on line {line_number}")
                continue

            text = str(chunk.get("text") or "").strip()

            if not text:
                print(f"[chunk_loader] Skipping chunk with empty text on line {line_number}")
                continue

            normalized = dict(chunk)
            normalized["text"] = text
            normalized["chunk_id"] = str(
                chunk.get("chunk_id") or f"CHUNK_{line_number:04d}"
            )
            normalized["doc_id"] = str(chunk.get("doc_id") or "UNKNOWN_DOC")
            normalized["section"] = str(chunk.get("section") or "General")
            normalized["source"] = str(chunk.get("source") or "unknown")

            chunks.append(normalized)

    if not chunks:
        raise ValueError(
            f"No usable chunks loaded from {path.resolve()}. "
            "Re-run ingestion: python -m scripts.ingest"
        )

    return chunks