from backend.app.retrieval.streaming_retriever import StreamingRetriever
from backend.app.query.refinement_query_builder import RefinementQueryBuilder


class RefinementRetriever:
    """
    Performs targeted retrieval for an additive refinement
    while preserving the existing retrieval pipeline.
    """

    def __init__(self, retriever: StreamingRetriever, final_top_k: int = 5):
        self.retriever = retriever
        self.final_top_k = final_top_k
        self.query_builder = RefinementQueryBuilder()

    def retrieve_refinement(
        self,
        previous_query: str,
        refinement_query: str,
        existing_results: list[dict],
    ) -> dict:

        query_info = self.query_builder.build(
            previous_query=previous_query,
            refinement_query=refinement_query,
            refinement_type="ADDITIVE",
        )

        contextual_query = query_info["query"]

        retrieval = self.retriever.retrieve(contextual_query)

        new_results = retrieval["reranked_results"]

        merged = []
        seen_chunk_ids = set()

        for result in new_results:
            chunk_id = result["chunk"]["chunk_id"]

            if chunk_id in seen_chunk_ids:
                continue

            merged.append(result)
            seen_chunk_ids.add(chunk_id)

            if len(merged) >= self.final_top_k:
                break

        if len(merged) < self.final_top_k:
            for result in existing_results:
                chunk_id = result["chunk"]["chunk_id"]

                if chunk_id in seen_chunk_ids:
                    continue

                merged.append(result)
                seen_chunk_ids.add(chunk_id)

                if len(merged) >= self.final_top_k:
                    break

        return {
            "original_refinement": refinement_query,
            "contextual_query": contextual_query,
            "query_method": query_info["method"],
            "new_results": new_results,
            "merged_results": merged,
        }