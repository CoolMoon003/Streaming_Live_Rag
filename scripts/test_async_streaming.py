import asyncio

from backend.app.models.session import (
    SessionState,
)

from backend.app.retrieval.async_streaming_retriever import (
    AsyncStreamingRetriever,
)


CHUNKS_PATH = "data/processed/chunks.jsonl"


async def main():

    print()
    print("=" * 80)
    print("REAL ASYNC STREAMING RETRIEVAL TEST")
    print("=" * 80)

    session = SessionState(
        session_id="async-demo-001"
    )

    retriever = AsyncStreamingRetriever(
        CHUNKS_PATH
    )

    # --------------------------------------------------
    # Query v1
    # --------------------------------------------------

    query_1 = (
        "What are the international travel rules?"
    )

    generation_1 = session.start_new_query(
        query_1
    )

    print()
    print(
        f"Started query v1 | "
        f"generation={generation_1}"
    )

    task_1 = asyncio.create_task(
        retriever.retrieve(
            query=query_1,
            generation_id=generation_1,
        )
    )

    # --------------------------------------------------
    # Simulate a newer transcript arriving
    # --------------------------------------------------

    await asyncio.sleep(0.05)

    query_2 = (
        "What are the international travel rules "
        "and what about late bookings?"
    )

    generation_2 = session.start_new_query(
        query_2
    )

    print()
    print(
        f"Started query v2 | "
        f"generation={generation_2}"
    )

    task_2 = asyncio.create_task(
        retriever.retrieve(
            query=query_2,
            generation_id=generation_2,
        )
    )

    # --------------------------------------------------
    # Wait for both real retrieval pipelines
    # --------------------------------------------------

    result_1, result_2 = await asyncio.gather(
        task_1,
        task_2,
    )

    # --------------------------------------------------
    # Process generation 1
    # --------------------------------------------------

    print()
    print(
        f"Processing generation "
        f"{result_1['generation_id']}"
    )

    accepted = session.accept_results(
        generation_id=result_1["generation_id"],
        query=result_1["query"],
        results=result_1["results"]["reranked_results"],
    )

    if accepted:
        print(
            f"Generation {result_1['generation_id']}: "
            f"ACCEPTED"
        )
    else:
        print(
            f"Generation {result_1['generation_id']}: "
            f"DISCARDED AS STALE"
        )

    # --------------------------------------------------
    # Process generation 2
    # --------------------------------------------------

    print()
    print(
        f"Processing generation "
        f"{result_2['generation_id']}"
    )

    accepted = session.accept_results(
        generation_id=result_2["generation_id"],
        query=result_2["query"],
        results=result_2["results"]["reranked_results"],
    )

    if accepted:
        print(
            f"Generation {result_2['generation_id']}: "
            f"ACCEPTED"
        )
    else:
        print(
            f"Generation {result_2['generation_id']}: "
            f"DISCARDED AS STALE"
        )

    # --------------------------------------------------
    # Show final session state
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL SESSION STATE")
    print("=" * 80)

    print(
        f"Active generation: "
        f"{session.active_generation_id}"
    )

    print(
        f"Query version: "
        f"{session.query_version}"
    )

    print(
        f"Current query: "
        f"{session.current_query}"
    )

    print(
        f"Latest evidence chunks: "
        f"{len(session.latest_results)}"
    )

    print(
        f"Retrieval history entries: "
        f"{len(session.retrieval_history)}"
    )

    print()

    for result in session.latest_results:

        chunk = result["chunk"]

        print(
            f"{result['rank']}. "
            f"{chunk['chunk_id']} | "
            f"{chunk['section']}"
        )


if __name__ == "__main__":
    asyncio.run(main()) 