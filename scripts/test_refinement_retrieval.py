from backend.app.query.refinement import (
    QueryRefinementAnalyzer,
    RefinementType,
)

from backend.app.query.refinement_retriever import (
    RefinementRetriever,
)

from backend.app.retrieval.streaming_retriever import (
    StreamingRetriever,
)


CHUNKS_PATH = "data/processed/chunks.jsonl"


def print_results(
    title: str,
    results: list[dict],
):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    for result in results:

        chunk = result["chunk"]

        print(
            f"{result['rank']}. "
            f"{chunk['chunk_id']} | "
            f"{chunk['section']}"
        )

        print(
            f"   {chunk['text']}"
        )


def main():

    print()
    print("=" * 80)
    print("SELECTIVE REFINEMENT RETRIEVAL TEST")
    print("=" * 80)

    analyzer = QueryRefinementAnalyzer()

    base_retriever = StreamingRetriever(
        CHUNKS_PATH
    )

    refinement_retriever = RefinementRetriever(
        retriever=base_retriever
    )

    # --------------------------------------------------
    # Step 1 — Original query
    # --------------------------------------------------

    previous_query = (
        "What are the international travel rules?"
    )

    print()
    print("STEP 1 — ORIGINAL QUERY")
    print(
        f"Query: {previous_query}"
    )

    base_retrieval = base_retriever.retrieve(
        previous_query
    )

    existing_results = (
        base_retrieval["reranked_results"]
    )

    print_results(
        "ORIGINAL EVIDENCE",
        existing_results,
    )

    # --------------------------------------------------
    # Step 2 — New transcript
    # --------------------------------------------------

    refinement_query = (
        "And what about late bookings?"
    )

    print()
    print("STEP 2 — NEW TRANSCRIPT")
    print(
        f"Transcript: {refinement_query}"
    )

    decision = analyzer.analyze(
        previous_query=previous_query,
        new_query=refinement_query,
    )

    print()
    print(
        f"REFINEMENT TYPE: "
        f"{decision.refinement_type.value}"
    )

    print(
        f"REASON: "
        f"{decision.reason}"
    )

    # --------------------------------------------------
    # Step 3 — Only refine if additive
    # --------------------------------------------------

    if (
        decision.refinement_type
        != RefinementType.ADDITIVE
    ):

        print()
        print(
            "ERROR: Expected ADDITIVE refinement."
        )

        return

    refinement = refinement_retriever.retrieve_refinement(
        previous_query=previous_query,
        refinement_query=refinement_query,
        existing_results=existing_results,
    )

    # --------------------------------------------------
    # Step 4 — Show targeted retrieval
    # --------------------------------------------------

    print_results(
        "NEW REFINEMENT EVIDENCE",
        refinement["new_results"],
    )

    # --------------------------------------------------
    # Step 5 — Show merged state
    # --------------------------------------------------

    print_results(
        "MERGED SESSION EVIDENCE",
        refinement["merged_results"],
    )

    print()
    print("=" * 80)
    print("REFINEMENT TEST COMPLETE")
    print("=" * 80)

    print(
        f"Previous evidence count: "
        f"{len(existing_results)}"
    )

    print(
        f"New refinement evidence count: "
        f"{len(refinement['new_results'])}"
    )

    print(
        f"Merged evidence count: "
        f"{len(refinement['merged_results'])}"
    )


if __name__ == "__main__":
    main()