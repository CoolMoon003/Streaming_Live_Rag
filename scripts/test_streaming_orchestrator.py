import asyncio
import time
from backend.app.models.session import SessionState
from backend.app.orchestration.streaming_rag_orchestrator import StreamingRagOrchestrator


CHUNKS_PATH = "data/processed/chunks.jsonl"


async def main():
    print()
    print("=" * 80)
    print("STREAMING LIVE RAG ORCHESTRATOR - END-TO-END VERIFICATION")
    print("=" * 80)

    orchestrator = StreamingRagOrchestrator(chunks_path=CHUNKS_PATH)
    session = SessionState(session_id="orch-e2e-demo-001")

    # =========================================================================
    # SCENARIO 1: Incremental Partials -> Early Retrieval -> Commit
    # =========================================================================
    print()
    print("=" * 80)
    print("SCENARIO 1: Incremental Partials -> Early Retrieval -> Commit")
    print("=" * 80)

    partials = [
        "I need to",
        "I need to know the rules for",
        "I need to know the rules for international travel",
    ]

    for idx, partial in enumerate(partials, start=1):
        print(f"\n[PARTIAL v{idx}] '{partial}'")
        res = await orchestrator.process_partial(session, partial)
        print(f"  Action: {res.get('action')} | Event: {res.get('event')} | Reason: {res.get('reason')}")
        if res.get("action") == "RETRIEVE":
            print(f"  --> Early Retrieval initiated! Chunks retrieved: {res.get('chunks_retrieved')} (No LLM called)")

    # Now COMMIT arrives
    commit_text = "I need to know the rules for international travel"
    print(f"\n[COMMIT] '{commit_text}'")
    print("  Streaming tokens: ", end="", flush=True)

    final_event = None
    token_count = 0
    t0 = time.perf_counter()

    async for event in orchestrator.process_commit(session, commit_text):
        if event.get("event") == "answer_token":
            print(event.get("token"), end="", flush=True)
            token_count += 1
        elif event.get("event") == "answer_completed":
            final_event = event

    elapsed = time.perf_counter() - t0
    print()
    print(f"\n  Answer Version : {final_event['answer_version']}")
    print(f"  Citations      : {final_event['citations']}")
    print(f"  Citation Valid : {final_event['citation_valid']}")
    print(f"  Stream time    : {elapsed:.2f}s (Tokens: {token_count})")

    assert final_event["citation_valid"] is True, "Citation validation failed for Scenario 1"
    assert final_event["answer_version"] == 1, "Expected answer_version 1"

    # =========================================================================
    # SCENARIO 2: Additive Refinement (State Preservation)
    # =========================================================================
    print()
    print("=" * 80)
    print("SCENARIO 2: Additive Refinement (State Preservation)")
    print("=" * 80)

    refinement_partial = "And what about late bookings?"
    print(f"\n[PARTIAL] '{refinement_partial}'")
    res = await orchestrator.process_partial(session, refinement_partial)
    print(f"  Action: {res.get('action')} | Event: {res.get('event')} | Reason: {res.get('reason')}")

    print(f"\n[COMMIT] '{refinement_partial}'")
    print("  Streaming tokens: ", end="", flush=True)

    final_event_2 = None
    async for event in orchestrator.process_commit(session, refinement_partial):
        if event.get("event") == "answer_token":
            print(event.get("token"), end="", flush=True)
        elif event.get("event") == "answer_completed":
            final_event_2 = event

    print()
    print(f"\n  Answer Version : {final_event_2['answer_version']}")
    print(f"  Citations      : {final_event_2['citations']}")
    print(f"  Citation Valid : {final_event_2['citation_valid']}")

    assert final_event_2["answer_version"] == 2, "Expected answer_version 2"

    # =========================================================================
    # SCENARIO 3: Presentation Suppression
    # =========================================================================
    print()
    print("=" * 80)
    print("SCENARIO 3: Presentation Suppression (No Retrieval)")
    print("=" * 80)

    pres_text = "Repeat that in two bullets"
    print(f"\n[PARTIAL] '{pres_text}'")
    res = await orchestrator.process_partial(session, pres_text)
    print(f"  Action: {res.get('action')} | Event: {res.get('event')} | Reason: {res.get('reason')}")
    assert res.get("action") == "SUPPRESS", "Expected SUPPRESS action on presentation request"

    print(f"\n[COMMIT] '{pres_text}'")
    final_event_3 = None
    async for event in orchestrator.process_commit(session, pres_text):
        if event.get("event") == "answer_completed":
            final_event_3 = event

    print(f"  Action         : {final_event_3.get('action')}")
    print(f"  Answer Version : {final_event_3.get('answer_version')}")
    print(f"  Answer Reused  : {final_event_3.get('answer')[:70]}...")
    assert final_event_3["answer_version"] == 2, "Answer version should remain 2"

    # =========================================================================
    # SCENARIO 4: Missing Numeric Fact Gating
    # =========================================================================
    print()
    print("=" * 80)
    print("SCENARIO 4: Missing Numeric Fact Gating")
    print("=" * 80)

    numeric_query = "What is the maximum international airfare amount an employee can claim?"
    print(f"\n[COMMIT] '{numeric_query}'")
    gate_event = None
    async for event in orchestrator.process_commit(session, numeric_query):
        if event.get("event") == "uncertainty_emitted":
            gate_event = event

    print(f"  Event          : {gate_event.get('event')}")
    print(f"  Reason         : {gate_event.get('reason')}")
    print(f"  Answer Output  : {gate_event.get('answer')}")
    print(f"  Sufficient     : {gate_event.get('evidence_sufficient')}")
    assert gate_event is not None, "Expected uncertainty_emitted event"
    assert gate_event["evidence_sufficient"] is False, "Expected evidence_sufficient to be False"

    # =========================================================================
    # SCENARIO 5: Multi-Intent Compound Query
    # =========================================================================
    print()
    print("=" * 80)
    print("SCENARIO 5: Multi-Intent Compound Query (Parallel Retrieval)")
    print("=" * 80)

    multi_query = "What approval is needed for international travel and what documentation is needed for reimbursement?"
    print(f"\n[PARTIAL] '{multi_query}'")
    res = await orchestrator.process_partial(session, multi_query)
    print(f"  Action: {res.get('action')} | Multi-Intent: {res.get('is_multi_intent')} | Chunks: {res.get('chunks_retrieved')}")
    assert res.get("is_multi_intent") is True, "Expected multi-intent to be True"

    print(f"\n[COMMIT] '{multi_query}'")
    print("  Streaming tokens: ", end="", flush=True)
    multi_final = None
    async for event in orchestrator.process_commit(session, multi_query):
        if event.get("event") == "answer_token":
            print(event.get("token"), end="", flush=True)
        elif event.get("event") == "answer_completed":
            multi_final = event

    print()
    print(f"\n  Answer Version : {multi_final['answer_version']}")
    print(f"  Citations      : {multi_final['citations']}")
    print(f"  Citation Valid : {multi_final['citation_valid']}")

    # ---------------------------------------------------------------------
    # Multi-intent completeness. A compound query must not be answered by
    # covering only the first intent, so assert per-intent evidence coverage
    # and per-intent citations rather than just "an answer was produced".
    # ---------------------------------------------------------------------
    intents = multi_final.get("intents", [])

    print(f"  Multi-Intent   : {multi_final.get('is_multi_intent')}")
    for intent in intents:
        print(
            f"    intent {intent['intent_id']}: "
            f"supported={intent['supported']} "
            f"| retrieved={intent['retrieved']} "
            f"| top_score={intent['top_score']} "
            f"| reason={intent['reason']} "
            f"| chunks={intent['chunk_ids']}"
        )

    assert multi_final.get("is_multi_intent") is True, "Expected a multi-intent answer"
    assert len(intents) == 2, f"Expected 2 intents, got {len(intents)}"

    for intent in intents:
        assert intent["supported"], (
            f"Intent {intent['intent_id']} ({intent['query']}) was not supported "
            f"[reason={intent['reason']}, retrieved={intent['retrieved']}, "
            f"top_score={intent['top_score']}]"
        )
        assert intent["chunk_ids"], f"Intent {intent['intent_id']} selected no evidence"

    # The two intents must be grounded in different documents, so neither is
    # answered using the other's evidence.
    intent_docs = [
        {chunk_id.rsplit("_C", 1)[0] for chunk_id in intent["chunk_ids"]}
        for intent in intents
    ]
    assert intent_docs[0] != intent_docs[1], (
        f"Both intents selected the same source document(s): {intent_docs}"
    )

    # Every intent must be visible in the answer, and every supported intent
    # must have contributed at least one validated citation.
    answer_text = multi_final["answer"]
    citations = multi_final["citations"]

    assert multi_final["citation_valid"], "Multi-intent citations failed validation"

    for intent in intents:
        assert f"Intent {intent['intent_id']}" in answer_text, (
            f"Answer does not address intent {intent['intent_id']}"
        )

        intent_doc_ids = {
            chunk_id.rsplit("_C", 1)[0] for chunk_id in intent["chunk_ids"]
        }
        assert any(
            any(doc_id in citation for doc_id in intent_doc_ids)
            for citation in citations
        ), (
            f"No validated citation from intent {intent['intent_id']} "
            f"(expected one of {sorted(intent_doc_ids)}, got {citations})"
        )

    print("  Multi-intent completeness: VERIFIED")

    print()
    print("=" * 80)
    print("ALL STREAMING ORCHESTRATOR TESTS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())