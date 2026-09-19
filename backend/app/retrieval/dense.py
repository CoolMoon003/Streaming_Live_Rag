from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from backend.app.retrieval.chunk_loader import load_chunks


class DenseRetriever:
    def __init__(
        self,
        chunks_path: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.chunks_path = Path(chunks_path)

        self.chunks = load_chunks(self.chunks_path)

        print(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)

        texts = [chunk["text"] for chunk in self.chunks]

        print("Creating embeddings...")

        embeddings = np.asarray(
            self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype="float32",
        )

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        self.embeddings = embeddings

        dimension = int(embeddings.shape[1])

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        print(
            f"FAISS index ready: "
            f"{len(self.chunks)} chunks, "
            f"dimension={dimension}"
        )

    def search(self, query: str, top_k: int = 10):
        if self.index is None or self.index.ntotal == 0:
            return []

        query_embedding = np.asarray(
            self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ),
            dtype="float32",
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # FAISS pads with -1 when k > ntotal; clamp instead
        limit = max(1, min(int(top_k), self.index.ntotal))

        scores, indices = self.index.search(
            query_embedding,
            limit,
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            index = int(index)

            if index < 0 or index >= len(self.chunks):
                continue

            results.append(
                {
                    "chunk": self.chunks[index],
                    "score": float(score),
                    # rank derived from kept results so ranks stay contiguous
                    "rank": len(results) + 1,
                }
            )

        return results