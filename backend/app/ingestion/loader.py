from pathlib import Path


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def load_documents(input_dir: str):
    input_path = Path(input_dir)

    documents = []

    for file_path in input_path.rglob("*"):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        documents.append({
            "filename": file_path.name,
            "path": str(file_path),
            "extension": file_path.suffix.lower(),
        })

    return documents