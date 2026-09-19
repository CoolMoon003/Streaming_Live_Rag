import re

from backend.app.models.chunk import DocumentChunk


def make_doc_id(filename: str) -> str:
    name = filename.rsplit(".", 1)[0]
    name = re.sub(r"[^A-Za-z0-9]+", "_", name)
    return f"DOC_{name.upper()}"


def chunk_markdown(text: str, filename: str):
    doc_id = make_doc_id(filename)

    lines = text.splitlines()

    current_section = "General"
    current_text = []

    chunks = []
    chunk_number = 1

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith("#"):
            if current_text:
                chunk_id = f"{doc_id}_C{chunk_number:03d}"

                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        section=current_section,
                        text=" ".join(current_text),
                        source=filename,
                    )
                )

                chunk_number += 1
                current_text = []

            current_section = stripped.lstrip("#").strip()

        else:
            current_text.append(stripped)

    if current_text:
        chunk_id = f"{doc_id}_C{chunk_number:03d}"

        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                section=current_section,
                text=" ".join(current_text),
                source=filename,
            )
        )

    return chunks