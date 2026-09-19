from pathlib import Path

from rank_bm25 import BM25Okapi

from backend.app.retrieval.chunk_loader import load_chunks


class BM25Retriever:
    def __init__(self, chunks_path: str):
        self.chunks_path = Path(chunks_path)
        self.chunks = load_chunks(self.chunks_path)

        corpus = [
            chunk["text"].lower().split()
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 10):
        tokens = str(query or "").lower().split()

        if not tokens or not self.chunks:
            return []

        scores = self.bm25.get_scores(tokens)

        # top_k may exceed the corpus size
        limit = max(1, min(int(top_k), len(scores)))

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:limit]

        return [
            {
                "chunk": self.chunks[i],
                "score": float(scores[i]),
                "rank": rank + 1,
            }
            for rank, i in enumerate(ranked_indices)
        ]