"""
Phase 4 - multi-intent answer completeness.

Part 1 (selection / gating / prompt / orchestrator) exercises the real
EvidenceSelector, EvidenceGate, GroundedAnswerGenerator, CitationValidator and
StreamingRagOrchestrator against a stub retriever and a stub LLM, so no
Hugging Face models and no Ollama are needed. The stub LLM can only cite
evidence that actually reached the prompt, so a dropped intent fails the test
rather than being papered over.

Part 2 drives the real orchestrator with the real retrieval stack and a live
LLM, exactly like scripts/test_streaming_orchestrator.py. It is skipped (not
failed) when models or Ollama are unavailable.
"""

import asyncio
import re
import sys

from backend.app.models.session import SessionState
from backend.app.retrieval.evidence_selector import EvidenceSelector
from backend.app.retrieval.evidence_gate import EvidenceGate
from backend.app.llm.grounded_generator import GroundedAnswerGenerator
from backend.app.orchestration.streaming_rag_orchestrator import (
    StreamingRagOrchestrator,
)


CHUNKS_PATH = "data/processed/chunks.jsonl"

MULTI_QUERY = (
    "What approval is needed for international travel "
    "and what documentation is needed for reimbursement?"
)

SINGLE_QUERY = "What approval is needed for international travel?"


# =========================================================================
# FIXTURES
# =========================================================================


def chunk(chunk_id, doc_id, section, text, score):
    return {
        "chunk": {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "section": section,
            "text": text,
            "source": f"{doc_id.lower()}.md",
        },
        "reranker_score": score,
        "rank": 1,
    }


def chunk_with_rrf(chunk_id, doc_id, section, text, rrf_score, score):
    """
    Same as chunk(), but also carries the RRF-fused score the real
    CrossEncoderReranker preserves alongside reranker_score (see
    reranker.py: it copies the fused-stage dict before adding
    reranker_score). Needed to exercise
    EvidenceSelector._recover_by_retrieval_agreement, which reads
    "rrf_score"/"score" - plain chunk() fixtures never carry it, so they
    never trigger the fallback.
    """
    entry = chunk(chunk_id, doc_id, section, text, score)
    entry["score"] = rrf_score
    entry["rrf_score"] = rrf_score
    return entry


# Exact shape of the real live bug: intent 2's own CrossEncoder scores are
# ALL negative (best is -0.35757, for the wrong chunk), so the plain
# per-intent floor (intent_min_score=0.0) selects nothing. Retrieval-stage
# (BM25/Dense/RRF) scores are the real reported RRF values and agree
# strongly on the reimbursement cluster; DOC_TRAVEL_POLICY_C002 leaked in
# from a different intent's topic with a far lower RRF score and a
# catastrophic CrossEncoder score, and must stay rejected.
REAL_BUG_REIMBURSEMENT_RESULTS = [
    chunk_with_rrf(
        "DOC_REIMBURSEMENT_POLICY_C003", "DOC_REIMBURSEMENT_POLICY", "3. Submission Deadline",
        "Expense claims should be submitted within the required reimbursement period.",
        0.0317540, -0.35757,
    ),
    chunk_with_rrf(
        "DOC_REIMBURSEMENT_POLICY_C001", "DOC_REIMBURSEMENT_POLICY", "1. Eligible Expenses",
        "Employees may request reimbursement for eligible business expenses supported by valid receipts.",
        0.0327869, -0.53476,
    ),
    chunk_with_rrf(
        "DOC_REIMBURSEMENT_POLICY_C004", "DOC_REIMBURSEMENT_POLICY", "4. Exceptions",
        "Exceptions to normal reimbursement rules require appropriate authorization.",
        0.0310096, -1.27502,
    ),
    chunk_with_rrf(
        "DOC_REIMBURSEMENT_POLICY_C002", "DOC_REIMBURSEMENT_POLICY", "2. International Expenses",
        "International business expenses must include appropriate supporting documentation.",
        0.0317460, -2.63567,
    ),
    chunk_with_rrf(
        "DOC_TRAVEL_POLICY_C002", "DOC_TRAVEL_POLICY", "2. International Travel",
        "International travel requires prior approval from the appropriate manager.",
        0.0153846, -10.41589,
    ),
]

# A subquery where retrieval itself found nothing meaningful: a single weak
# candidate with no competing signal to "agree" with. Must stay rejected -
# the fallback must not recover a candidate just because it is alone in the
# pool (see EvidenceSelector._recover_by_retrieval_agreement).
NO_AGREEMENT_RESULTS = [
    chunk_with_rrf(
        "DOC_WORKSHOP_POLICY_C003", "DOC_WORKSHOP_POLICY", "3. Catering",
        "Workshop catering arrangements are handled by the facilities team.",
        0.0164, -6.4,
    ),
]


# Intent 1 (travel) outscores intent 2 (reimbursement) on its own subquery and
# has enough chunks to exhaust max_chunks=3 by itself. This is the shape that
# used to starve the second intent under a single global selection.
TRAVEL_RESULTS = [
    chunk("DOC_TRAVEL_POLICY_C002", "DOC_TRAVEL_POLICY", "2. International Travel",
          "International travel requires prior approval from the appropriate manager.", 8.9),
    chunk("DOC_TRAVEL_POLICY_C003", "DOC_TRAVEL_POLICY", "3. Booking",
          "Travel should normally be booked through the approved booking process. "
          "Late bookings may require additional approval.", 7.4),
    chunk("DOC_TRAVEL_POLICY_C001", "DOC_TRAVEL_POLICY", "1. Domestic Travel",
          "Employees travelling within the country may claim eligible transportation expenses.", 6.2),
]

REIMBURSEMENT_RESULTS = [
    chunk("DOC_REIMBURSEMENT_POLICY_C002", "DOC_REIMBURSEMENT_POLICY", "2. International Expenses",
          "International business expenses must include appropriate supporting documentation.", 5.6),
    chunk("DOC_REIMBURSEMENT_POLICY_C001", "DOC_REIMBURSEMENT_POLICY", "1. Eligible Expenses",
          "Employees may request reimbursement for eligible business expenses supported by valid receipts.", 4.1),
]

# The correct evidence for this intent, at a modest cross-encoder score. This
# is the live-failure shape: the chunk is rank #1 for its own subquery but
# scores below the 0.5 whole-query floor.
LOW_SCORE_REIMBURSEMENT_RESULTS = [
    chunk("DOC_REIMBURSEMENT_POLICY_C002", "DOC_REIMBURSEMENT_POLICY", "2. International Expenses",
          "International business expenses must include appropriate supporting documentation.", 0.21),
    chunk("DOC_REIMBURSEMENT_POLICY_C003", "DOC_REIMBURSEMENT_POLICY", "3. Submission Deadline",
          "Expense claims should be submitted within the required reimbursement period.", 0.05),
]

# Nothing in the corpus supports the second intent.
UNSUPPORTED_RESULTS = [
    chunk("DOC_WORKSHOP_POLICY_C003", "DOC_WORKSHOP_POLICY", "3. Catering",
          "Workshop catering arrangements are handled by the facilities team.", -6.4),
]


def intent_entry(intent_id, query, results):
    """One MultiIntentRetriever subquery entry."""
    return {
        "intent_id": intent_id,
        "query": query,
        "results": {"reranked_results": results},
    }


# =========================================================================
# STUBS
# =========================================================================


class StubSyncRetriever:
    """Stands in for StreamingRetriever (used by RefinementRetriever)."""

    def __init__(self, table):
        self.table = table

    def retrieve(self, query):
        return {"query": query, "reranked_results": self.table(query)}


class StubAsyncRetriever:
    """Stands in for AsyncStreamingRetriever."""

    def __init__(self, table):
        self.table = table
        self.retriever = StubSyncRetriever(table)
        self.queries = []

    async def retrieve(self, query, generation_id):
        self.queries.append(query)
        return {
            "generation_id": generation_id,
            "query": query,
            "results": {"reranked_results": self.table(query)},
        }


class StubLLM:
    """
    Stands in for OllamaClient.

    Answers strictly from the prompt it is handed: for every "INTENT n:" block
    it emits one line citing the citations present in that block. It cannot
    invent evidence, so an intent whose evidence never reached the prompt
    simply does not appear in the answer.
    """

    INTENT_BLOCK = re.compile(
        r"^INTENT (\d+): (.*?)$(.*?)(?=^INTENT \d+: |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    CITATION = re.compile(r"\[[A-Za-z0-9_]+ §[^\]]+\]")

    def __init__(self):
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self._answer(prompt)

    def generate_stream(self, prompt):
        self.prompts.append(prompt)
        for token in self._answer(prompt).split(" "):
            yield token + " "

    def _answer(self, prompt):
        # Everything after the ANSWER: marker is ours to write; look only at
        # the evidence section of the prompt.
        blocks = self.INTENT_BLOCK.findall(prompt)

        if not blocks:
            # Cite only from the EVIDENCE section, never from the rules text.
            evidence_section = prompt.split("EVIDENCE:", 1)[-1]
            citations = self.CITATION.findall(evidence_section)
            return f"Single intent answer. {citations[0]}" if citations else "No evidence."

        lines = []

        for intent_id, subquery, body in blocks:
            if "EVIDENCE FOR INTENT" in body and "NONE" not in body.split("\n")[1]:
                citations = self.CITATION.findall(body)
                cited = " ".join(sorted(set(citations)))
                lines.append(f"Intent {intent_id}: Grounded answer. {cited}")
            else:
                lines.append(
                    f"Intent {intent_id}: "
                    f"{GroundedAnswerGenerator.INTENT_INSUFFICIENT_MSG}"
                )

        return "\n".join(lines)

    def get_last_metrics(self):
        return {"ttft_ms": 0.0, "latency_ms": 0.0}


def build_orchestrator(table, multi_table):
    """Real orchestrator, stub retrieval + stub LLM."""
    async_retriever = StubAsyncRetriever(table)

    class StubMultiIntentRetriever:
        def __init__(self):
            self.calls = []

        async def retrieve(self, subqueries, generation_id):
            self.calls.append(list(subqueries))
            intents = [
                intent_entry(index, subquery, multi_table(subquery))
                for index, subquery in enumerate(subqueries, start=1)
            ]
            merged = []
            seen = set()
            for intent in intents:
                for item in intent["results"]["reranked_results"]:
                    chunk_id = item["chunk"]["chunk_id"]
                    if chunk_id in seen:
                        continue
                    seen.add(chunk_id)
                    enriched = dict(item)
                    enriched["intent_id"] = intent["intent_id"]
                    enriched["subquery"] = intent["query"]
                    merged.append(enriched)
            return {
                "generation_id": generation_id,
                "subqueries": intents,
                "results": merged,
            }

    llm = StubLLM()

    orchestrator = StreamingRagOrchestrator(
        chunks_path=CHUNKS_PATH,
        async_retriever=async_retriever,
        multi_intent_retriever=StubMultiIntentRetriever(),
        llm_client=llm,
    )

    return orchestrator, async_retriever, llm


def travel_table(query):
    return TRAVEL_RESULTS


def both_supported_table(subquery):
    lowered = subquery.lower()
    if "reimbursement" in lowered:
        return REIMBURSEMENT_RESULTS
    return TRAVEL_RESULTS


def low_score_table(subquery):
    lowered = subquery.lower()
    if "reimbursement" in lowered:
        return LOW_SCORE_REIMBURSEMENT_RESULTS
    return TRAVEL_RESULTS


def one_unsupported_table(subquery):
    lowered = subquery.lower()
    if "reimbursement" in lowered:
        return UNSUPPORTED_RESULTS
    return TRAVEL_RESULTS


def real_bug_table(subquery):
    """Reproduces the exact live-failure numbers reported for Phase 4."""
    lowered = subquery.lower()
    if "reimbursement" in lowered:
        return REAL_BUG_REIMBURSEMENT_RESULTS
    return TRAVEL_RESULTS


def no_agreement_table(subquery):
    lowered = subquery.lower()
    if "reimbursement" in lowered:
        return NO_AGREEMENT_RESULTS
    return TRAVEL_RESULTS


async def commit(orchestrator, session, text):
    final = None
    async for event in orchestrator.process_commit(session, text):
        if event.get("event") in ("answer_completed", "uncertainty_emitted"):
            final = event
    return final


def check(label, got, expected):
    status = "PASS" if got == expected else "FAIL"
    print(f"  [{status}] {label}: {got}")
    return got == expected


def check_true(label, got):
    return check(label, bool(got), True)


# =========================================================================
# TEST 1 - PER-INTENT EVIDENCE SELECTION / PROVENANCE
# =========================================================================


def selection_tests():
    print()
    print("=" * 78)
    print("TEST 1 - PER-INTENT EVIDENCE SELECTION (no models required)")
    print("=" * 78)

    ok = []
    selector = EvidenceSelector()

    intents = [
        intent_entry(1, "What approval is needed for international travel?", TRAVEL_RESULTS),
        intent_entry(2, "What documentation is needed for reimbursement?", REIMBURSEMENT_RESULTS),
    ]

    # Baseline: the old global path over the merged pool starves intent 2.
    merged = []
    for intent in intents:
        for item in intent["results"]["reranked_results"]:
            enriched = dict(item)
            enriched["intent_id"] = intent["intent_id"]
            merged.append(enriched)

    global_selected = selector.select(merged, MULTI_QUERY)
    print("\nA. baseline global selection (documents the bug)")
    ok.append(
        check(
            "intents covered by select()",
            sorted({r["intent_id"] for r in global_selected}),
            [1],
        )
    )

    print("\nB. per-intent selection")
    per_intent = selector.select_per_intent(intents)
    ok.append(check("intent count", len(per_intent), 2))
    ok.append(
        check(
            "intent 1 chunks",
            [r["chunk"]["chunk_id"] for r in per_intent[0]["selected"]],
            ["DOC_TRAVEL_POLICY_C002", "DOC_TRAVEL_POLICY_C003", "DOC_TRAVEL_POLICY_C001"],
        )
    )
    ok.append(
        check(
            "intent 2 chunks",
            [r["chunk"]["chunk_id"] for r in per_intent[1]["selected"]],
            ["DOC_REIMBURSEMENT_POLICY_C002", "DOC_REIMBURSEMENT_POLICY_C001"],
        )
    )
    ok.append(
        check(
            "provenance preserved",
            [(r["intent_id"], r["subquery"] == intents[1]["query"]) for r in per_intent[1]["selected"]],
            [(2, True), (2, True)],
        )
    )
    ok.append(
        check(
            "per-intent max_chunks still enforced",
            all(len(entry["selected"]) <= selector.max_chunks for entry in per_intent),
            True,
        )
    )

    print("\nC0. modestly-scoring intent still selected (live-failure shape)")
    live_shape = [
        intent_entry(1, "What approval is needed for international travel?", TRAVEL_RESULTS),
        intent_entry(2, "what documentation is needed for reimbursement?",
                     LOW_SCORE_REIMBURSEMENT_RESULTS),
    ]
    # The whole-query floor would reject it outright...
    ok.append(
        check(
            "whole-query floor rejects it",
            selector.select(LOW_SCORE_REIMBURSEMENT_RESULTS, "reimbursement documentation?"),
            [],
        )
    )
    # ...the per-intent floor keeps it.
    live_selected = selector.select_per_intent(live_shape)
    ok.append(
        check(
            "per-intent floor keeps it",
            [r["chunk"]["chunk_id"] for r in live_selected[1]["selected"]],
            ["DOC_REIMBURSEMENT_POLICY_C002", "DOC_REIMBURSEMENT_POLICY_C003"],
        )
    )
    live_support = EvidenceGate().check_per_intent(live_selected)
    ok.append(check("both intents supported", [e["supported"] for e in live_support], [True, True]))

    print("\nC0b. real bug numbers: ALL CrossEncoder scores negative, retrieval-agreement recovers")
    real_bug_shape = [
        intent_entry(1, "What approval is needed for international travel?", TRAVEL_RESULTS),
        intent_entry(2, "what documentation is needed for reimbursement?",
                     REAL_BUG_REIMBURSEMENT_RESULTS),
    ]
    # The per-intent floor alone (no fallback) rejects everything, exactly
    # as reported live: best CrossEncoder score for intent 2 is -0.35757.
    ok.append(
        check(
            "per-intent floor alone rejects it",
            selector.select(REAL_BUG_REIMBURSEMENT_RESULTS, "documentation reimbursement?",
                             min_score=selector.intent_min_score),
            [],
        )
    )
    bug_selected = selector.select_per_intent(real_bug_shape)
    bug_intent2_ids = [r["chunk"]["chunk_id"] for r in bug_selected[1]["selected"]]
    ok.append(
        check_true(
            "DOC_REIMBURSEMENT_POLICY_C002 recovered via retrieval agreement",
            "DOC_REIMBURSEMENT_POLICY_C002" in bug_intent2_ids,
        )
    )
    ok.append(
        check_true(
            "leaked travel chunk NOT recovered into intent 2",
            "DOC_TRAVEL_POLICY_C002" not in bug_intent2_ids,
        )
    )
    ok.append(
        check_true(
            "recovered evidence is flagged",
            all(r.get("recovered_via_retrieval_agreement") for r in bug_selected[1]["selected"]),
        )
    )
    bug_support = EvidenceGate().check_per_intent(bug_selected)
    ok.append(check("both intents supported", [e["supported"] for e in bug_support], [True, True]))
    ok.append(
        check(
            "intent 2 reason reflects the fallback",
            bug_support[1]["decision"].reason,
            "sufficient_evidence_retrieval_agreement",
        )
    )

    print("\nC0c. unrelated candidate rejection: no competing retrieval agreement -> stays unsupported")
    no_agreement_shape = [
        intent_entry(1, "What approval is needed for international travel?", TRAVEL_RESULTS),
        intent_entry(2, "what documentation is needed for reimbursement?", NO_AGREEMENT_RESULTS),
    ]
    no_agreement_selected = selector.select_per_intent(no_agreement_shape)
    ok.append(
        check(
            "lone weak candidate not recovered just because its CE score is negative",
            no_agreement_selected[1]["selected"],
            [],
        )
    )
    no_agreement_support = EvidenceGate().check_per_intent(no_agreement_selected)
    ok.append(
        check(
            "intent 2 correctly stays unsupported",
            no_agreement_support[1]["supported"],
            False,
        )
    )

    print("\nC. duplicate evidence across intents is kept once")
    shared = [
        intent_entry(1, "travel approval?", TRAVEL_RESULTS[:1]),
        intent_entry(2, "travel documentation?", TRAVEL_RESULTS[:1]),
    ]
    shared_selected = selector.select_per_intent(shared)
    ok.append(check("intent 1 keeps it", len(shared_selected[0]["selected"]), 1))
    ok.append(check("intent 2 does not repeat it", len(shared_selected[1]["selected"]), 0))

    print("\nD. select() itself is unchanged for single-intent input")
    single = selector.select(TRAVEL_RESULTS, SINGLE_QUERY)
    ok.append(
        check(
            "single-intent selection",
            [r["chunk"]["chunk_id"] for r in single],
            ["DOC_TRAVEL_POLICY_C002", "DOC_TRAVEL_POLICY_C003", "DOC_TRAVEL_POLICY_C001"],
        )
    )

    print("\nE. per-intent gating builds a support map")
    gate = EvidenceGate()
    support = gate.check_per_intent(selector.select_per_intent(intents))
    ok.append(check("both supported", [e["supported"] for e in support], [True, True]))

    mixed = [
        intent_entry(1, "What approval is needed for international travel?", TRAVEL_RESULTS),
        intent_entry(2, "What documentation is needed for reimbursement?", UNSUPPORTED_RESULTS),
    ]
    support = gate.check_per_intent(selector.select_per_intent(mixed))
    ok.append(check("mixed support map", [e["supported"] for e in support], [True, False]))
    # The unsupported intent selects no evidence at all, so its own gate call
    # reports no_evidence rather than a global "sufficient" verdict.
    ok.append(check("unsupported selects nothing", len(support[1]["evidence"]), 0))
    ok.append(check("unsupported reason", support[1]["decision"].reason, "no_evidence"))

    return all(ok)


# =========================================================================
# TEST 2 - MULTI-INTENT PROMPT
# =========================================================================


def prompt_tests():
    print()
    print("=" * 78)
    print("TEST 2 - MULTI-INTENT PROMPT (no models required)")
    print("=" * 78)

    ok = []
    generator = GroundedAnswerGenerator(StubLLM())

    intents = [
        {
            "intent_id": 1,
            "query": "What approval is needed for international travel?",
            "evidence": TRAVEL_RESULTS[:1],
            "supported": True,
        },
        {
            "intent_id": 2,
            "query": "What documentation is needed for reimbursement?",
            "evidence": REIMBURSEMENT_RESULTS[:1],
            "supported": True,
        },
    ]

    prompt, allowed = generator._build_multi_intent_prompt(MULTI_QUERY, intents)

    print("\nA. both intents present in the prompt")
    ok.append(check_true("intent 1 block", "INTENT 1:" in prompt))
    ok.append(check_true("intent 2 block", "INTENT 2:" in prompt))
    ok.append(
        check_true(
            "intent 1 evidence text",
            "International travel requires prior approval" in prompt,
        )
    )
    ok.append(
        check_true(
            "intent 2 evidence text",
            "must include appropriate supporting documentation" in prompt,
        )
    )
    ok.append(
        check(
            "allowed citations",
            allowed,
            [
                "[DOC_TRAVEL_POLICY §2. International Travel]",
                "[DOC_REIMBURSEMENT_POLICY §2. International Expenses]",
            ],
        )
    )
    ok.append(check_true("coverage requirement stated", "Answering only one is a failure" in prompt))

    print("\nB. unsupported intent gets the deterministic sentence")
    intents[1] = {
        "intent_id": 2,
        "query": "What documentation is needed for reimbursement?",
        "evidence": [],
        "supported": False,
    }
    prompt, allowed = generator._build_multi_intent_prompt(MULTI_QUERY, intents)

    ok.append(check_true("insufficiency text in prompt", generator.INTENT_INSUFFICIENT_MSG in prompt))
    ok.append(check_true("no evidence marker", "EVIDENCE FOR INTENT 2: NONE" in prompt))
    ok.append(
        check(
            "unsupported intent adds no citations",
            allowed,
            ["[DOC_TRAVEL_POLICY §2. International Travel]"],
        )
    )

    print("\nC. single-intent prompt path untouched")
    single_prompt, single_allowed = generator._build_prompt(SINGLE_QUERY, TRAVEL_RESULTS[:1])
    ok.append(check_true("no intent blocks", "INTENT 1:" not in single_prompt))
    ok.append(check_true("original header", "You are a strict corpus-grounded assistant." in single_prompt))
    ok.append(check("single allowed citations", single_allowed, ["[DOC_TRAVEL_POLICY §2. International Travel]"]))

    return all(ok)


# =========================================================================
# TEST 3 - ORCHESTRATOR END TO END (stub retrieval + stub LLM)
# =========================================================================


async def orchestrator_tests():
    print()
    print("=" * 78)
    print("TEST 3 - ORCHESTRATOR (stub retrieval + stub LLM)")
    print("=" * 78)

    ok = []

    # ---------------- CASE A: two supported intents ----------------
    print("\nCASE A. two supported intents")
    orchestrator, _, llm = build_orchestrator(travel_table, both_supported_table)
    session = SessionState(session_id="phase4-a")
    final = await commit(orchestrator, session, MULTI_QUERY)

    ok.append(check("event", final["event"], "answer_completed"))
    ok.append(check("multi-intent flagged", final["is_multi_intent"], True))
    ok.append(check("intents answered", [i["supported"] for i in final["intents"]], [True, True]))
    ok.append(
        check(
            "intent 1 evidence reached the prompt",
            "International travel requires prior approval" in llm.prompts[0] if llm.prompts else False,
            True,
        )
    )
    ok.append(
        check(
            "intent 2 evidence reached the prompt",
            "must include appropriate supporting documentation" in llm.prompts[0] if llm.prompts else False,
            True,
        )
    )
    ok.append(check_true("answer covers intent 1", "Intent 1:" in final["answer"]))
    ok.append(check_true("answer covers intent 2", "Intent 2:" in final["answer"]))
    ok.append(
        check_true(
            "travel citation validated",
            any("DOC_TRAVEL_POLICY" in c for c in final["citations"]),
        )
    )
    ok.append(
        check_true(
            "reimbursement citation validated",
            any("DOC_REIMBURSEMENT_POLICY" in c for c in final["citations"]),
        )
    )
    ok.append(check("citations verified", final["citation_valid"], True))

    # ---------------- CASE B: one supported, one unsupported ----------------
    print("\nCASE B. one supported + one unsupported intent")
    orchestrator, _, llm = build_orchestrator(travel_table, one_unsupported_table)
    session = SessionState(session_id="phase4-b")
    final = await commit(orchestrator, session, MULTI_QUERY)

    ok.append(check("event", final["event"], "answer_completed"))
    ok.append(check("support map", [i["supported"] for i in final["intents"]], [True, False]))
    ok.append(check_true("supported intent answered", "Intent 1:" in final["answer"]))
    ok.append(
        check_true(
            "unsupported intent declared",
            GroundedAnswerGenerator.INTENT_INSUFFICIENT_MSG in final["answer"],
        )
    )
    ok.append(
        check(
            "unsupported evidence never reached the prompt",
            "Workshop catering arrangements" in llm.prompts[0],
            False,
        )
    )
    ok.append(
        check(
            "no fabricated citations",
            [c for c in final["citations"] if "WORKSHOP" in c],
            [],
        )
    )
    ok.append(check("unsupported intent reports retrieval happened", final["intents"][1]["retrieved"], 1))
    ok.append(check("unsupported intent reports its top score", final["intents"][1]["top_score"], -6.4))

    # ---------------- CASE A2: live-failure shape ----------------
    print("\nCASE A2. second intent with modestly-scoring evidence (live failure)")
    orchestrator, _, llm = build_orchestrator(travel_table, low_score_table)
    session = SessionState(session_id="phase4-a2")
    final = await commit(orchestrator, session, MULTI_QUERY)

    ok.append(check("support map", [i["supported"] for i in final["intents"]], [True, True]))
    ok.append(
        check(
            "intent 2 chunk selected",
            final["intents"][1]["chunk_ids"],
            ["DOC_REIMBURSEMENT_POLICY_C002", "DOC_REIMBURSEMENT_POLICY_C003"],
        )
    )
    ok.append(check("intent 2 retrieved count reported", final["intents"][1]["retrieved"], 2))
    ok.append(check("intent 2 top score reported", final["intents"][1]["top_score"], 0.21))
    ok.append(
        check_true(
            "reimbursement citation present",
            any("DOC_REIMBURSEMENT_POLICY" in c for c in final["citations"]),
        )
    )

    # ---------------- CASE A3: exact real-bug numbers, end to end ----------------
    print("\nCASE A3. real bug reproduction: negative CE scores across the board")
    orchestrator, _, llm = build_orchestrator(travel_table, real_bug_table)
    session = SessionState(session_id="phase4-a3")
    final = await commit(orchestrator, session, MULTI_QUERY)

    ok.append(check("event", final["event"], "answer_completed"))
    ok.append(check("two intents detected", len(final["intents"]), 2))
    ok.append(check("both intents supported", [i["supported"] for i in final["intents"]], [True, True]))
    ok.append(
        check_true(
            "travel evidence selected",
            any(
                cid.startswith("DOC_TRAVEL_POLICY")
                for intent in final["intents"]
                for cid in intent["chunk_ids"]
            ),
        )
    )
    ok.append(
        check_true(
            "reimbursement evidence selected (specifically C002)",
            "DOC_REIMBURSEMENT_POLICY_C002" in final["intents"][1]["chunk_ids"],
        )
    )
    ok.append(
        check_true(
            "leaked travel chunk excluded from intent 2 evidence",
            "DOC_TRAVEL_POLICY_C002" not in final["intents"][1]["chunk_ids"],
        )
    )
    ok.append(check_true("answer covers intent 1", "Intent 1:" in final["answer"]))
    ok.append(check_true("answer covers intent 2", "Intent 2:" in final["answer"]))
    ok.append(check("citations verified", final["citation_valid"], True))
    ok.append(
        check_true(
            "travel citation present",
            any("DOC_TRAVEL_POLICY" in c for c in final["citations"]),
        )
    )
    ok.append(
        check_true(
            "reimbursement citation present",
            any("DOC_REIMBURSEMENT_POLICY" in c for c in final["citations"]),
        )
    )

    # ---------------- CASE A4: no retrieval-stage agreement -> refusal ----------------
    print("\nCASE A4. negative CE scores with NO retrieval agreement -> intent 2 stays unsupported")
    orchestrator, _, llm = build_orchestrator(travel_table, no_agreement_table)
    session = SessionState(session_id="phase4-a4")
    final = await commit(orchestrator, session, MULTI_QUERY)

    ok.append(check("support map", [i["supported"] for i in final["intents"]], [True, False]))
    ok.append(
        check(
            "no fabricated reimbursement citations",
            [c for c in final["citations"] if "REIMBURSEMENT" in c],
            [],
        )
    )

    # ---------------- CASE C: single intent regression ----------------
    print("\nCASE C. single-intent regression")
    orchestrator, async_retriever, llm = build_orchestrator(travel_table, both_supported_table)
    session = SessionState(session_id="phase4-c")
    final = await commit(orchestrator, session, SINGLE_QUERY)

    ok.append(check("event", final["event"], "answer_completed"))
    ok.append(check("not multi-intent", final["is_multi_intent"], False))
    ok.append(check("no intent payload", final["intents"], []))
    ok.append(check("single retrieval call", async_retriever.queries, [SINGLE_QUERY]))
    ok.append(check_true("no intent blocks in prompt", "INTENT 1:" not in llm.prompts[0]))
    ok.append(check("citations verified", final["citation_valid"], True))

    # ---------------- CASE D: presentation suppression ----------------
    print("\nCASE D. presentation suppression unchanged")
    orchestrator, async_retriever, llm = build_orchestrator(travel_table, both_supported_table)
    session = SessionState(session_id="phase4-d")
    await commit(orchestrator, session, SINGLE_QUERY)
    before = len(llm.prompts)
    final = await commit(orchestrator, session, "can you summarize that")

    ok.append(check("suppressed", final["action"], "SUPPRESS"))
    ok.append(check("no new generation", len(llm.prompts), before))

    # ---------------- CASE E: numeric gate unchanged ----------------
    print("\nCASE E. missing-numeric-fact gate unchanged")
    orchestrator, _, _ = build_orchestrator(travel_table, both_supported_table)
    session = SessionState(session_id="phase4-e")
    final = await commit(
        orchestrator,
        session,
        "What is the maximum airfare amount an employee can claim?",
    )

    ok.append(check("event", final["event"], "uncertainty_emitted"))
    ok.append(check("reason", final["reason"], "missing_numeric_fact"))

    # ---------------- CASE F: early retrieval reuse unchanged ----------------
    print("\nCASE F. early-retrieval reuse unchanged")
    orchestrator, async_retriever, _ = build_orchestrator(travel_table, both_supported_table)
    session = SessionState(session_id="phase4-f")

    await orchestrator.process_partial(session, SINGLE_QUERY)
    calls_after_partial = len(async_retriever.queries)
    final = await commit(orchestrator, session, SINGLE_QUERY)

    ok.append(check("reused, no second retrieval", len(async_retriever.queries), calls_after_partial))
    ok.append(check("answer produced", final["event"], "answer_completed"))

    # ---------------- CASE G: all intents unsupported ----------------
    print("\nCASE G. no intent supported -> refusal, no fabrication")

    def all_unsupported(subquery):
        return UNSUPPORTED_RESULTS

    orchestrator, _, _ = build_orchestrator(travel_table, all_unsupported)
    session = SessionState(session_id="phase4-g")
    final = await commit(orchestrator, session, MULTI_QUERY)

    ok.append(check("event", final["event"], "uncertainty_emitted"))
    ok.append(check("no citations", final["citations"], []))

    return all(ok)


# =========================================================================
# TEST 4 - REAL MODELS + LIVE LLM
# =========================================================================


async def live_tests():
    print()
    print("=" * 78)
    print("TEST 4 - REAL RETRIEVAL + LIVE OLLAMA")
    print("=" * 78)

    try:
        orchestrator = StreamingRagOrchestrator(chunks_path=CHUNKS_PATH)
    except Exception as error:  # noqa: BLE001 - environment probe
        print(f"  [SKIPPED] retrieval stack unavailable: {error}")
        return None

    session = SessionState(session_id="phase4-live")

    try:
        final = await commit(orchestrator, session, MULTI_QUERY)
    except Exception as error:  # noqa: BLE001 - environment probe
        print(f"  [SKIPPED] live LLM unavailable: {error}")
        return None

    print()
    print("ANSWER:")
    print(final["answer"])
    print()
    print(f"CITATIONS: {final['citations']}")
    for intent in final["intents"]:
        print(
            f"  intent {intent['intent_id']}: supported={intent['supported']} "
            f"| retrieved={intent['retrieved']} "
            f"| top_score={intent['top_score']} "
            f"| reason={intent['reason']} "
            f"| chunks={intent.get('chunk_ids', [])}"
        )

    ok = []
    answer = final["answer"]

    ok.append(check("multi-intent flagged", final["is_multi_intent"], True))
    ok.append(check("two intents detected", len(final["intents"]), 2))
    ok.append(
        check_true(
            "intent 1 evidence selected",
            any(
                cid.startswith("DOC_TRAVEL_POLICY")
                for intent in final["intents"]
                for cid in intent.get("chunk_ids", [])
            ),
        )
    )
    ok.append(
        check_true(
            "intent 2 evidence selected",
            any(
                cid.startswith("DOC_REIMBURSEMENT_POLICY")
                for intent in final["intents"]
                for cid in intent.get("chunk_ids", [])
            ),
        )
    )
    ok.append(check_true("answer mentions intent 1", "Intent 1" in answer))
    ok.append(check_true("answer mentions intent 2", "Intent 2" in answer))
    ok.append(check("citations verified", final["citation_valid"], True))
    ok.append(
        check_true(
            "travel citation present",
            any("DOC_TRAVEL_POLICY" in c for c in final["citations"]),
        )
    )
    ok.append(
        check_true(
            "reimbursement citation present",
            any("DOC_REIMBURSEMENT_POLICY" in c for c in final["citations"]),
        )
    )

    return all(ok)


# =========================================================================
# MAIN
# =========================================================================


async def main():
    results = {
        "selection": selection_tests(),
        "prompt": prompt_tests(),
        "orchestrator": await orchestrator_tests(),
    }

    live = await live_tests()

    print()
    print("=" * 78)
    print("PHASE 4 SUMMARY")
    print("=" * 78)

    for name, passed in results.items():
        print(f"  {name:<14} {'PASS' if passed else 'FAIL'}")

    if live is None:
        print(f"  {'live':<14} SKIPPED (models or Ollama unavailable)")
    else:
        print(f"  {'live':<14} {'PASS' if live else 'FAIL'}")

    offline_ok = all(results.values())
    overall = offline_ok and (live is None or live)

    print()
    print("STATUS:", "PASS" if overall else "FAIL")

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))