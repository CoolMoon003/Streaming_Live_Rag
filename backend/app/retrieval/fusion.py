from collections import defaultdict


def reciprocal_rank_fusion(
    result_lists,
    k: int = 60,
    top_k: int = 10,
):
    scores = defaultdict(float)
    chunks = {}

    for results in result_lists or []:
        if not results:
            continue

        for position, result in enumerate(results, start=1):
            if not isinstance(result, dict):
                continue

            chunk = result.get("chunk")

            if not isinstance(chunk, dict):
                continue

            chunk_id = chunk.get("chunk_id")

            if not chunk_id:
                continue

            # fall back to list position if rank is missing
            rank = result.get("rank") or position

            scores[chunk_id] += 1.0 / (k + rank)

            chunks[chunk_id] = chunk

    if not scores:
        return []

    limit = max(1, int(top_k))

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:limit]

    return [
        {
            "chunk": chunks[chunk_id],
            "score": score,
            "rrf_score": score,
            "rank": rank,
        }
        for rank, (chunk_id, score) in enumerate(
            ranked,
            start=1,
        )
    ]