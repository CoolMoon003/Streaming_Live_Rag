import asyncio

from backend.app.models.session import SessionState


async def simulated_retrieval(
    query: str,
    generation_id: int,
    delay: float,
):
    print(
        f"Retrieval started | "
        f"generation={generation_id} | "
        f"delay={delay}s | "
        f"query='{query}'"
    )

    await asyncio.sleep(delay)

    results = [
        {
            "chunk_id": f"RESULT_FROM_GENERATION_{generation_id}",
            "query": query,
        }
    ]

    print(
        f"Retrieval finished | "
        f"generation={generation_id}"
    )

    return generation_id, query, results


async def main():

    session = SessionState(
        session_id="generation-demo-001"
    )

    print()
    print("=" * 80)
    print("STALE RESULT PROTECTION TEST")
    print("=" * 80)

    # --------------------------------------------------
    # Query v1
    # --------------------------------------------------

    generation_1 = session.start_new_query(
        "What are the international travel rules?"
    )

    print()
    print(
        f"Started query v1 "
        f"with generation {generation_1}"
    )

    task_1 = asyncio.create_task(
        simulated_retrieval(
            query=session.current_query,
            generation_id=generation_1,
            delay=2.0,
        )
    )

    # --------------------------------------------------
    # Query v2 arrives before v1 finishes
    # --------------------------------------------------

    await asyncio.sleep(0.2)

    generation_2 = session.start_new_query(
        "What are the international travel rules "
        "and what about late bookings?"
    )

    print()
    print(
        f"Started query v2 "
        f"with generation {generation_2}"
    )

    task_2 = asyncio.create_task(
        simulated_retrieval(
            query=session.current_query,
            generation_id=generation_2,
            delay=0.5,
        )
    )

    # --------------------------------------------------
    # Wait for both retrievals
    # --------------------------------------------------

    result_1 = await task_1
    result_2 = await task_2

    # --------------------------------------------------
    # Try to accept v1
    # --------------------------------------------------

    generation_id, query, results = result_1

    accepted = session.accept_results(
        generation_id=generation_id,
        query=query,
        results=results,
    )

    print()

    if accepted:
        print(
            f"Generation {generation_id}: ACCEPTED"
        )
    else:
        print(
            f"Generation {generation_id}: "
            f"DISCARDED AS STALE"
        )

    # --------------------------------------------------
    # Try to accept v2
    # --------------------------------------------------

    generation_id, query, results = result_2

    accepted = session.accept_results(
        generation_id=generation_id,
        query=query,
        results=results,
    )

    print()

    if accepted:
        print(
            f"Generation {generation_id}: ACCEPTED"
        )
    else:
        print(
            f"Generation {generation_id}: "
            f"DISCARDED AS STALE"
        )

    # --------------------------------------------------
    # Final state
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
        f"Latest result: "
        f"{session.latest_results}"
    )

    print(
        f"Retrieval history entries: "
        f"{len(session.retrieval_history)}"
    )


if __name__ == "__main__":
    asyncio.run(main())