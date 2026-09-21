import asyncio

from backend.app.orchestration.streaming_rag_orchestrator import StreamingRagOrchestrator


QUERY = (
    "What approval is needed for international travel and "
    "what documentation is needed for reimbursement?"
)


async def main():
    print("=" * 90)
    print("REAL END-TO-END Q016 TEST")
    print("=" * 90)

    orchestrator = StreamingRagOrchestrator(
        chunks_path="data/processed/phase6_chunks.jsonl",
        model="llama3.2:3b",
    )

    print("\n--- COMMIT ---")

    events = []
    async for event in orchestrator.process_commit(QUERY):
        events.append(event)

        if isinstance(event, dict):
            event_type = event.get("type") or event.get("event")
            if event_type in {
                "controller_decision",
                "retrieval_complete",
                "answer_complete",
                "generation_complete",
                "citation_validation",
            }:
                print(event)

    print("\n--- ALL EVENTS ---")
    for event in events:
        if isinstance(event, dict):
            print(event)

    print("\n" + "=" * 90)
    print("EVENT COUNT:", len(events))
    print("=" * 90)


if __name__ == "__main__":
    asyncio.run(main())
