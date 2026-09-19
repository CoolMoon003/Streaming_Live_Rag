import numpy as np
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        print(f"Loading reranker: {model_name}")

        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list,
        top_k: int = 5,
    ):
        if not results:
            return []

        candidates = []
        pairs = []

        for result in results:
            if not isinstance(result, dict):
                continue

            chunk = result.get("chunk")

            if not isinstance(chunk, dict):
                continue

            text = str(chunk.get("text") or "").strip()

            if not text:
                continue

            candidates.append(result)
            pairs.append([query, text])

        if not pairs:
            return []

        # predict() returns a scalar for a single pair
        scores = np.atleast_1d(
            np.asarray(self.model.predict(pairs), dtype="float32")
        )

        reranked = []

        for result, score in zip(candidates, scores):
            item = dict(result)
            item["reranker_score"] = float(score)
            reranked.append(item)

        reranked.sort(
            key=lambda x: x["reranker_score"],
            reverse=True,
        )

        limit = max(1, min(int(top_k), len(reranked)))

        final = reranked[:limit]

        for rank, result in enumerate(final, start=1):
            result["rank"] = rank

        return final