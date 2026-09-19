import json
from pathlib import Path

from backend.app.ingestion.loader import load_documents
from backend.app.ingestion.chunker import chunk_markdown


RAW_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/processed/chunks.jsonl")


def main():
    documents = load_documents(str(RAW_DIR))

    all_chunks = []

    for document in documents:
        path = Path(document["path"])

        if path.suffix.lower() not in {".txt", ".md"}:
            print(f"Skipping unsupported parser for now: {path.name}")
            continue

        text = path.read_text(encoding="utf-8")

        chunks = chunk_markdown(
            text=text,
            filename=path.name,
        )

        all_chunks.extend(chunks)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(
                json.dumps(
                    chunk.model_dump(),
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Documents discovered: {len(documents)}")
    print(f"Chunks created: {len(all_chunks)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()