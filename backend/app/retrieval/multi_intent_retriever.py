import asyncio
from typing import Any

from backend.app.retrieval.async_streaming_retriever import (
    AsyncStreamingRetriever,
)


class MultiIntentRetriever:
    """
    Executes multiple independent subqueries concurrently.

    Pipeline:

        Multi-intent subqueries
                 |
                 v
        asyncio.gather()
          /           \
    Intent 1       Intent 2
       |              |
       v              v
    BM25 + Dense + RRF + Reranker
       |              |
       +-------> Intent-aware evidence
    """

    def __init__(
        self,
        chunks_path: str = "",
        retriever: AsyncStreamingRetriever | None = None,
    ):
        if retriever is not None:
            self.retriever = retriever
        else:
            self.retriever = AsyncStreamingRetriever(chunks_path)

    async def retrieve(
        self,
        subqueries: list[str],
        generation_id: int,
    ) -> dict[str, Any]:

        if not subqueries:
            return {
                "generation_id": generation_id,
                "subqueries": [],
                "results": [],
            }

        print(
            f"[MULTI-INTENT RETRIEVAL START] "
            f"generation={generation_id} | "
            f"subqueries={len(subqueries)}"
        )

        # --------------------------------------------------
        # Run all subqueries concurrently
        # --------------------------------------------------

        tasks = [
            self.retriever.retrieve(
                query=subquery,
                generation_id=generation_id,
            )
            for subquery in subqueries
        ]

        raw_results = await asyncio.gather(
            *tasks
        )

        print(
            f"[MULTI-INTENT RETRIEVAL COMPLETE] "
            f"generation={generation_id}"
        )

        # --------------------------------------------------
        # Normalize result structure
        #
        # AsyncStreamingRetriever may return either:
        #
        # 1. {
        #       "reranked_results": [...]
        #    }
        #
        # OR
        #
        # 2. {
        #       "results": {
        #           "reranked_results": [...]
        #       }
        #
        # This handles both safely.
        # --------------------------------------------------

        intent_results = []

        for intent_id, (subquery, raw_result) in enumerate(
            zip(subqueries, raw_results),
            start=1,
        ):

            retrieval_result = self._extract_retrieval(
                raw_result
            )

            intent_results.append(
                {
                    "intent_id": intent_id,
                    "query": subquery,
                    "results": retrieval_result,
                }
            )

        # --------------------------------------------------
        # Merge evidence while preserving intent metadata
        # --------------------------------------------------

        merged_results = self._merge_results(
            intent_results
        )

        return {
            "generation_id": generation_id,
            "subqueries": intent_results,
            "results": merged_results,
        }

    @staticmethod
    def _extract_retrieval(
        raw_result: dict[str, Any],
    ) -> dict[str, Any]:

        # Direct retrieval structure
        if "reranked_results" in raw_result:
            return raw_result

        # Wrapped retrieval structure
        if (
            "results" in raw_result
            and isinstance(
                raw_result["results"],
                dict,
            )
        ):
            nested = raw_result["results"]

            if "reranked_results" in nested:
                return nested

        # Unexpected structure
        print(
            "[WARNING] Unexpected retrieval result structure:"
        )
        print(
            raw_result.keys()
        )

        return {
            "reranked_results": []
        }

    @staticmethod
    def _merge_results(
        subquery_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        merged = []
        seen = set()

        for intent_result in subquery_results:

            intent_id = intent_result[
                "intent_id"
            ]

            subquery = intent_result[
                "query"
            ]

            retrieval = intent_result[
                "results"
            ]

            reranked_results = retrieval.get(
                "reranked_results",
                [],
            )

            for item in reranked_results:

                chunk = item.get(
                    "chunk",
                    {},
                )

                chunk_id = chunk.get(
                    "chunk_id"
                )

                if not chunk_id:
                    continue

                # Deduplicate evidence appearing
                # under multiple intents.
                if chunk_id in seen:
                    continue

                seen.add(chunk_id)

                # Preserve original retrieval result.
                enriched_item = item.copy()

                # Add intent provenance.
                enriched_item[
                    "intent_id"
                ] = intent_id

                enriched_item[
                    "subquery"
                ] = subquery

                merged.append(
                    enriched_item
                )

        return merged


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    CHUNKS_PATH = (
        "data/processed/chunks.jsonl"
    )

    retriever = MultiIntentRetriever(
        CHUNKS_PATH
    )

    async def test():

        result = await retriever.retrieve(
            subqueries=[
                "What approval is needed for international travel?",
                "What documentation is needed for reimbursement?",
            ],
            generation_id=1,
        )

        print()
        print("=" * 80)
        print("MULTI-INTENT RETRIEVAL TEST")
        print("=" * 80)

        print(
            f"SUBQUERIES: "
            f"{len(result['subqueries'])}"
        )

        print(
            f"MERGED RESULTS: "
            f"{len(result['results'])}"
        )

        # --------------------------------------------------
        # Individual intent results
        # --------------------------------------------------

        for intent_result in result[
            "subqueries"
        ]:

            intent_id = intent_result[
                "intent_id"
            ]

            query = intent_result[
                "query"
            ]

            retrieval = intent_result[
                "results"
            ]

            reranked = retrieval.get(
                "reranked_results",
                [],
            )

            print()
            print(
                f"INTENT {intent_id}: "
                f"{query}"
            )

            print(
                f"  RESULTS: "
                f"{len(reranked)}"
            )

            for rank, item in enumerate(
                reranked,
                start=1,
            ):

                chunk = item.get(
                    "chunk",
                    {},
                )

                score = item.get(
                    "reranker_score",
                    0.0,
                )

                print(
                    f"  {rank}. "
                    f"{chunk.get('chunk_id')} "
                    f"| score={score:.4f}"
                )

        # --------------------------------------------------
        # Intent-aware merged evidence
        # --------------------------------------------------

        print()
        print("=" * 80)
        print("INTENT-AWARE MERGED EVIDENCE")
        print("=" * 80)

        for rank, item in enumerate(
            result["results"],
            start=1,
        ):

            chunk = item.get(
                "chunk",
                {},
            )

            print(
                f"{rank}. "
                f"[Intent {item['intent_id']}] "
                f"{chunk.get('chunk_id')} "
                f"| score="
                f"{item.get('reranker_score', 0.0):.4f}"
            )

            print(
                f"   Query: "
                f"{item['subquery']}"
            )

            print(
                f"   Section: "
                f"{chunk.get('section', '')}"
            )

        # --------------------------------------------------
        # Final status
        # --------------------------------------------------

        print()
        print("=" * 80)
        print("MULTI-INTENT TEST COMPLETE")
        print("=" * 80)

        print(
            f"Generation ID : "
            f"{result['generation_id']}"
        )

        print(
            f"Intents       : "
            f"{len(result['subqueries'])}"
        )

        print(
            f"Merged chunks : "
            f"{len(result['results'])}"
        )

        if result["results"]:
            print(
                "Intent metadata: PRESERVED"
            )
            print(
                "Status: PASS"
            )
        else:
            print(
                "Intent metadata: NOT AVAILABLE"
            )
            print(
                "Status: FAIL"
            )

    asyncio.run(test())