"""
Focused tests for early-retrieval reuse on an evolving transcript.

Part 1 (decision layer) uses only backend.app.models.session and needs no
models. Part 2 drives the real orchestrator and therefore needs the real
retrieval stack and a live LLM, exactly like scripts/test_streaming_orchestrator.py.
"""

import asyncio
import sys

from backend.app.models.session import (
    SessionState,
    build_delta_query,
    delta_needs_retrieval,
    merge_evidence,
)


CHUNKS_PATH = "data/processed/chunks.jsonl"

EARLY = "I need to know the rules for international travel"


def make_session(query, results, generation_id=None, active=None):
    """Session with one accepted retrieval, built through the real API."""
    session = SessionState(session_id="reuse-test")
    gen = session.start_new_query(query)
    session.accept_results(generation_id=gen, query=query, results=results)

    if generation_id is not None:
        session.retrieval_history[-1].generation_id = generation_id
    if active is not None:
        session.active_generation_id = active

    return session


def chunk(chunk_id, text, section="1. Section", doc="DOC_TEST"):
    return {
        "chunk": {
            "chunk_id": chunk_id,
            "doc_id": doc,
            "section": section,
            "text": text,
            "source": "test.md",
        },
        "reranker_score": 5.0,
        "rank": 1,
    }


RESULTS = [
    chunk(
        "DOC_TRAVEL_POLICY_C002",
        "International travel requires prior approval from the appropriate manager.",
    ),
    chunk(
        "DOC_TRAVEL_POLICY_C003",
        "Travel should normally be booked through the approved booking process. "
        "Late bookings may require additional approval.",
    ),
]


def check(label, got, expected):
    status = "PASS" if got == expected else "FAIL"
    print(f"  [{status}] {label}: {got}")
    return got == expected


def decision_tests():
    print()
    print("=" * 78)
    print("PART 1 - REUSE DECISION (no models required)")
    print("=" * 78)

    ok = []

    # A. exact match -> reuse
    print("\nA. exact match")
    d = make_session(EARLY, RESULTS).find_reusable_retrieval(EARLY)
    ok.append(check("mode", d.mode, "exact"))
    ok.append(check("results carried", len(d.retrieval.results), 2))

    print("\nA2. exact match ignoring punctuation/case")
    d = make_session(EARLY, RESULTS).find_reusable_retrieval(
        "  I NEED to know the rules for international travel?  "
    )
    ok.append(check("mode", d.mode, "exact"))

    # B. meaningful extension -> reuse
    print("\nB. meaningful transcript extension")
    d = make_session(EARLY, RESULTS).find_reusable_retrieval(
        EARLY + " and late bookings"
    )
    ok.append(check("mode", d.mode, "extension"))
    ok.append(check("added tokens", d.added_tokens, ["and", "late", "bookings"]))

    # C. unrelated query -> fresh retrieval
    print("\nC. unrelated query")
    d = make_session(EARLY, RESULTS).find_reusable_retrieval(
        "What documentation is needed for reimbursement claims?"
    )
    ok.append(check("mode", d.mode, "none"))
    ok.append(check("reason", d.reason, "not_a_prefix"))

    print("\nC2. shared words but not a prefix")
    d = make_session(EARLY, RESULTS).find_reusable_retrieval(
        "the rules for international travel that I need to know about workshops"
    )
    ok.append(check("mode", d.mode, "none"))

    # D. very short / low-coverage prefix -> fresh retrieval
    print("\nD. very short prefix")
    d = make_session("what are the rules", RESULTS).find_reusable_retrieval(
        "what are the rules for international travel approval"
    )
    ok.append(check("mode", d.mode, "none"))
    ok.append(check("reason", d.reason, "prefix_too_short"))

    print("\nD2. prefix too small a share of the commit")
    d = make_session("what are the travel rules", RESULTS).find_reusable_retrieval(
        "what are the travel rules for international workshops and "
        "reimbursement deadlines and catering limits"
    )
    ok.append(check("mode", d.mode, "none"))
    ok.append(check("reason", d.reason, "prefix_coverage_too_low"))

    # E. stale generation -> NEVER reuse
    print("\nE. stale generation (exact text match)")
    stale = make_session(EARLY, RESULTS, generation_id=1, active=2)
    d = stale.find_reusable_retrieval(EARLY)
    ok.append(check("mode", d.mode, "none"))
    ok.append(check("reason", d.reason, "stale_generation"))

    print("\nE2. stale generation (extension)")
    d = stale.find_reusable_retrieval(EARLY + " and late bookings")
    ok.append(check("mode", d.mode, "none"))
    ok.append(check("reason", d.reason, "stale_generation"))

    print("\nE3. empty accepted results")
    d = make_session(EARLY, []).find_reusable_retrieval(EARLY)
    ok.append(check("mode", d.mode, "none"))

    # Delta helpers
    print("\nG. delta-query helpers")
    ok.append(check("strips connector", build_delta_query(["and", "late", "bookings"]), "late bookings"))
    ok.append(
        check(
            "covered tail needs no retrieval",
            delta_needs_retrieval("late bookings", RESULTS),
            False,
        )
    )
    ok.append(
        check(
            "new tail needs retrieval",
            delta_needs_retrieval("workshop catering", RESULTS),
            True,
        )
    )
    ok.append(
        check("filler tail needs no retrieval", delta_needs_retrieval("please", RESULTS), False)
    )
    merged = merge_evidence(RESULTS, [RESULTS[0], chunk("DOC_WORKSHOP_POLICY_C003", "Catering.")])
    ok.append(check("merge dedupes", [r["chunk"]["chunk_id"] for r in merged],
                    ["DOC_TRAVEL_POLICY_C002", "DOC_TRAVEL_POLICY_C003", "DOC_WORKSHOP_POLICY_C003"]))

    return all(ok)


async def orchestrator_tests():
    from backend.app.orchestration.streaming_rag_orchestrator import (
        StreamingRagOrchestrator,
    )

    print()
    print("=" * 78)
    print("PART 2 - ORCHESTRATOR (real retrieval + live LLM required)")
    print("=" * 78)

    orchestrator = StreamingRagOrchestrator(chunks_path=CHUNKS_PATH)
    ok = []

    real_retrieve = orchestrator.async_retriever.retrieve
    calls = []

    async def spy(query, generation_id):
        calls.append(query)
        return await real_retrieve(query=query, generation_id=generation_id)

    orchestrator.async_retriever.retrieve = spy

    async def commit(session, text):
        calls.clear()
        final = None
        async for event in orchestrator.process_commit(session, text):
            if event.get("event") in ("answer_completed", "uncertainty_emitted"):
                final = event
        return final

    # A. exact match -> no new retrieval
    print("\nA. exact commit after early retrieval")
    session = SessionState(session_id="orch-exact")
    await orchestrator.process_partial(session, EARLY)
    early_ids = [r["chunk"]["chunk_id"] for r in session.latest_results]
    await commit(session, EARLY)
    ok.append(check("retrieval calls", calls[:], []))
    ok.append(check("evidence unchanged",
                    [r["chunk"]["chunk_id"] for r in session.latest_results], early_ids))

    # B1. extension whose tail is already covered -> reuse, no new retrieval
    print("\nB1. extension with already-covered tail")
    session = SessionState(session_id="orch-ext-covered")
    await orchestrator.process_partial(session, EARLY)
    early_ids = [r["chunk"]["chunk_id"] for r in session.latest_results]
    await commit(session, EARLY + " and late bookings")
    ok.append(check("retrieval calls", calls[:], []))
    ok.append(check("early evidence kept",
                    [r["chunk"]["chunk_id"] for r in session.latest_results], early_ids))
    ok.append(check("history query is commit text",
                    session.retrieval_history[-1].query, EARLY + " and late bookings"))

    # B2. extension with genuinely new content -> delta retrieval + merge
    print("\nB2. extension introducing new content")
    session = SessionState(session_id="orch-ext-new")
    await orchestrator.process_partial(session, EARLY)
    early_ids = [r["chunk"]["chunk_id"] for r in session.latest_results]
    await commit(session, EARLY + " and workshop catering")
    merged_ids = [r["chunk"]["chunk_id"] for r in session.latest_results]
    ok.append(check("delta query retrieved", calls[:], ["workshop catering"]))
    ok.append(check("early evidence preserved",
                    all(cid in merged_ids for cid in early_ids), True))
    ok.append(check("new evidence added", len(merged_ids) > len(early_ids), True))
    ok.append(check("no duplicate chunks", len(merged_ids), len(set(merged_ids))))

    # C. unrelated commit -> fresh retrieval
    print("\nC. unrelated commit")
    session = SessionState(session_id="orch-unrelated")
    await orchestrator.process_partial(session, EARLY)
    unrelated = "What documentation is needed for reimbursement claims?"
    await commit(session, unrelated)
    ok.append(check("fresh retrieval issued", calls[:], [unrelated]))
    ok.append(check("history query", session.retrieval_history[-1].query, unrelated))

    # E. stale generation -> never reuse
    print("\nE. commit arriving while a newer partial is in flight")
    session = SessionState(session_id="orch-stale")
    await orchestrator.process_partial(session, EARLY)
    stale_ids = [r["chunk"]["chunk_id"] for r in session.latest_results]
    newer = "What documentation is needed for reimbursement claims?"
    session.start_new_query(newer)  # newer partial bumps generation, result not back yet
    await commit(session, newer)
    ok.append(check("fresh retrieval issued", calls[:], [newer]))
    ok.append(check("stale evidence not reused",
                    [r["chunk"]["chunk_id"] for r in session.latest_results] != stale_ids, True))

    # F. additive refinement still works
    print("\nF. additive refinement after a committed answer")
    session = SessionState(session_id="orch-additive")
    await orchestrator.process_partial(session, EARLY)
    first = await commit(session, EARLY)
    refinement = "And what about late bookings?"
    await orchestrator.process_partial(session, refinement)
    second = await commit(session, refinement)
    ok.append(check("first answer version", first["answer_version"], 1))
    ok.append(check("second answer version", second["answer_version"], 2))
    ok.append(check("additive refinement type", second["refinement_type"], "ADDITIVE"))

    return all(ok)


async def main():
    decisions_ok = decision_tests()

    if "--decision-only" in sys.argv:
        orchestrator_ok = True
        print("\n(skipping orchestrator tests: --decision-only)")
    else:
        orchestrator_ok = await orchestrator_tests()

    print()
    print("=" * 78)
    if decisions_ok and orchestrator_ok:
        print("EARLY REUSE TESTS: ALL PASSED")
    else:
        print("EARLY REUSE TESTS: FAILURES PRESENT")
        sys.exit(1)
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())