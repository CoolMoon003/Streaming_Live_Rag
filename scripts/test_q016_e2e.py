import asyncio
from backend.app.orchestration.streaming_rag_orchestrator import StreamingRagOrchestrator
from backend.app.models.session import SessionState

QUERY = (
    "What approval is needed for international travel and "
    "what documentation is needed for reimbursement?"
)

async def main():
    print("REAL END-TO-END Q016 TEST")
    print("=" * 70)

    orchestrator = StreamingRagOrchestrator(
        chunks_path="data/processed/phase6_chunks.jsonl",
        model="llama3.2:3b",
    )

    session = SessionState(session_id="q016-e2e-test")

    print("\n--- COMMIT ---")

    events = []
    async for event in orchestrator.process_commit(session, QUERY):
        events.append(event)

        event_type = event.get("event")

        if event_type == "answer_started":
            print("\nANSWER STARTED")
            print("action:", event.get("action"))
            print("multi-intent:", event.get("is_multi_intent"))

        elif event_type == "answer_token":
            print(event.get("token", ""), end="", flush=True)

        elif event_type == "answer_completed":
            print("\n\nANSWER COMPLETED")
            print("action:", event.get("action"))
            print("evidence_sufficient:", event.get("evidence_sufficient"))
            print("citations:", event.get("citations"))
            print("is_multi_intent:", event.get("is_multi_intent"))
            print("intents:", event.get("intents"))

    print("\n" + "=" * 70)
    print("SESSION CHECKS")
    print("retrieval history:", len(session.retrieval_history))
    print("latest results:", len(session.latest_results))
    print("answer version:", session.answer_version)

    answer_events = [
        e for e in events
        if e.get("event") == "answer_completed"
    ]

    if answer_events:
        answer = answer_events[-1].get("answer", "")

        print("\nFINAL ANSWER:")
        print(answer)

        print("\nVALIDATION")
        print("Intent 1 present:", "Intent 1:" in answer)
        print("Intent 2 present:", "Intent 2:" in answer)
        print("Travel citation:", "[DOC_TRAVEL_POLICY §2. International Travel]" in answer)
        print("Reimbursement citation:", "[DOC_EXPENSE_REIMBURSEMENT_POLICY §2. International Expenses]" in answer)
        print("Mojibake:", "Â§" in answer)

asyncio.run(main())
