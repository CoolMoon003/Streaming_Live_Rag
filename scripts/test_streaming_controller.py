from backend.app.controller.retrieval_controller import (
    RetrievalController,
    RetrievalAction,
)

from backend.app.retrieval.streaming_retriever import (
    StreamingRetriever,
)

from backend.app.models.session import (
    SessionState,
)


CHUNKS_PATH = "data/processed/chunks.jsonl"


def print_retrieval_results(results):

    print()
    print("=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)

    for result in results:

        chunk = result["chunk"]

        print(
            f"\n#{result['rank']} "
            f"{chunk['chunk_id']}"
        )

        print(
            f"Section: {chunk['section']}"
        )

        if "reranker_score" in result:

            print(
                f"Reranker Score: "
                f"{result['reranker_score']:.4f}"
            )

        print(
            f"Text: {chunk['text']}"
        )


def main():

    controller = RetrievalController()

    retriever = StreamingRetriever(
        CHUNKS_PATH
    )

    session = SessionState(
        session_id="demo-session-001"
    )

    transcript_versions = [

        (
            "I need to",
            False,
        ),

        (
            "I need to know the rules for",
            False,
        ),

        (
            "I need to know the rules for international travel",
            False,
        ),

        (
            "I need to know the rules for international travel and late bookings",
            False,
        ),

        (
            "Repeat that in two bullets",
            True,
        ),
    ]

    for version, (
        transcript,
        is_final,
    ) in enumerate(
        transcript_versions,
        start=1,
    ):

        print()
        print("#" * 80)
        print(
            f"TRANSCRIPT v{version}: "
            f"{transcript}"
        )
        print("#" * 80)

        decision = controller.decide(
            transcript=transcript,
            previous_transcript=session.current_query,
            is_final=is_final,
        )

        print()
        print(
            f"ACTION: {decision.action.value}"
        )

        print(
            f"REASON: {decision.reason}"
        )

        print(
            f"CONFIDENCE: "
            f"{decision.confidence:.2f}"
        )

        # ---------------------------------------------------------
        # WAIT
        # ---------------------------------------------------------

        if decision.action == RetrievalAction.WAIT:

            print()
            print(
                "→ Waiting for more transcript..."
            )

            continue

        # ---------------------------------------------------------
        # SUPPRESS
        # ---------------------------------------------------------

        if decision.action == RetrievalAction.SUPPRESS:

            print()
            print(
                "→ Retrieval suppressed."
            )

            print(
                "→ Reusing existing session retrieval state."
            )

            continue

        # ---------------------------------------------------------
        # RETRIEVE
        # ---------------------------------------------------------

        if decision.action == RetrievalAction.RETRIEVE:

            session.previous_query = (
                session.current_query
            )

            session.current_query = transcript

            session.query_version += 1

            print()
            print(
                f"→ Starting retrieval "
                f"for query version "
                f"v{session.query_version}"
            )

            retrieval = retriever.retrieve(
                transcript
            )

            session.latest_results = (
                retrieval["reranked_results"]
            )

            print_retrieval_results(
                session.latest_results
            )

            print()
            print(
                f"→ Session now contains "
                f"{len(session.latest_results)} "
                f"reranked results."
            )


if __name__ == "__main__":
    main()