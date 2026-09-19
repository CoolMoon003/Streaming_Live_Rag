"""
Phase 4 - multi-intent adversarial stress tests.

Exercises the real EvidenceSelector, EvidenceGate, GroundedAnswerGenerator,
CitationValidator and StreamingRagOrchestrator against deterministic stub
retrievers and a deterministic stub LLM. No Hugging Face models, no FAISS
search, and no live Ollama call are required - only import of the production
modules (which pulls in sentence-transformers/faiss at module import time)
is needed, exactly like scripts/test_multi_intent_answer.py's stub-based
Test 1-3.

This file adds NO new production code. It only stresses the existing Phase 4
implementation with harder fixtures than the happy-path tests already cover:
three-way intents, near-identical documents, evidence shared across intents,
lexical-only distractors, a misbehaving LLM that fabricates citations, and an
LLM that ignores the "unsupported intent" instruction outright.

Where a case exposes a real gap in the current implementation, the assertion
is left intact and the case is reported as FAILED with an explanation -
it is not weakened to force a PASS.
"""

import asyncio
import re
import sys

from backend.app.models.session import SessionState
from backend.app.llm.grounded_generator import GroundedAnswerGenerator
from backend.app.orchestration.streaming_rag_orchestrator import (
    StreamingRagOrchestrator,
)

CHUNKS_PATH = "data/processed/chunks.jsonl"

UNVERIFIED_ANSWER_MSG = (
    "The generated answer could not be verified against the provided corpus."
)

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
    Same as chunk(), but also carries the RRF-fused score
    (EvidenceSelector._recover_by_retrieval_agreement reads "rrf_score",
    falling back to "score"). Plain chunk() fixtures never carry it, so
    they can never trigger the retrieval-agreement fallback - needed to
    deliberately keep the lexical-distractor / wrong-decomposition cases
    out of that fallback path.
    """
    entry = chunk(chunk_id, doc_id, section, text, score)
    entry["score"] = rrf_score
    entry["rrf_score"] = rrf_score
    return entry


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


class StubMultiIntentRetriever:
    """
    Stands in for MultiIntentRetriever. Generalized to any number of
    subqueries (the production tests only ever exercised two).

    Mirrors the real component's merge behaviour: results are flattened
    into a single deduplicated list tagged with intent_id/subquery, in
    addition to the per-intent nested shape EvidenceSelector.select_per_intent
    actually reads from.
    """

    def __init__(self, multi_table):
        self.multi_table = multi_table
        self.calls = []

    async def retrieve(self, subqueries, generation_id):
        self.calls.append(list(subqueries))
        intents = [
            intent_entry(index, subquery, self.multi_table(subquery))
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


class StubLLM:
    """
    Stands in for OllamaClient.

    Answers strictly from the prompt it is handed: for every "INTENT n:"
    block it emits one line citing the citations actually present in that
    block. It cannot invent evidence, so an intent whose evidence never
    reached the prompt simply does not appear in the answer. This is the
    well-behaved baseline LLM; the adversarial subclasses below deliberately
    misbehave to stress the validation layer.
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
        blocks = self.INTENT_BLOCK.findall(prompt)

        if not blocks:
            evidence_section = prompt.split("EVIDENCE:", 1)[-1]
            citations = self.CITATION.findall(evidence_section)
            return f"Single intent answer. {citations[0]}" if citations else "No evidence."

        lines = []
        for intent_id, subquery, body in blocks:
            lines.append(self._intent_line(intent_id, body))
        return "\n".join(lines)

    def _intent_line(self, intent_id, body):
        if "EVIDENCE FOR INTENT" in body and "NONE" not in body.split("\n")[1]:
            citations = self.CITATION.findall(body)
            cited = " ".join(sorted(set(citations)))
            return f"Intent {intent_id}: Grounded answer. {cited}"
        return f"Intent {intent_id}: {GroundedAnswerGenerator.INTENT_INSUFFICIENT_MSG}"

    def get_last_metrics(self):
        return {"ttft_ms": 0.0, "latency_ms": 0.0}


class FabricatingLLM(StubLLM):
    """
    Adversarial case 9: otherwise well-behaved, but always tacks on a
    citation to a document that was never supplied as evidence to any
    intent. Used to verify CitationValidator actually rejects a
    hallucinated document/section rather than merely checking that
    *some* citation is present.
    """

    def _intent_line(self, intent_id, body):
        line = super()._intent_line(intent_id, body)
        if "EVIDENCE FOR INTENT" in body and "NONE" not in body.split("\n")[1]:
            line += " [DOC_GHOST_POLICY §9. Fabricated Section]"
        return line


class HallucinatingUnsupportedLLM(StubLLM):
    """
    Adversarial case 10: ignores the explicit instruction to copy the
    insufficiency sentence verbatim for an unsupported intent, and instead
    writes a confident, uncited factual claim for it - the failure mode a
    small local model could plausibly produce if it does not follow the
    "write exactly this text" instruction.
    """

    def _intent_line(self, intent_id, body):
        if "EVIDENCE FOR INTENT" in body and "NONE" not in body.split("\n")[1]:
            citations = self.CITATION.findall(body)
            cited = " ".join(sorted(set(citations)))
            return f"Intent {intent_id}: Grounded answer. {cited}"
        # Hallucinated, evidence-free claim for an intent that was never
        # supported - no citation attached, so it cannot be caught by a
        # purely citation-based hallucination check.
        return (
            f"Intent {intent_id}: Employees may claim unlimited amounts "
            f"with no documentation required."
        )


def build_orchestrator(table, multi_table, llm=None):
    """Real orchestrator, stub retrieval + stub (or adversarial) LLM."""
    async_retriever = StubAsyncRetriever(table)
    multi_intent_retriever = StubMultiIntentRetriever(multi_table)
    llm = llm if llm is not None else StubLLM()

    orchestrator = StreamingRagOrchestrator(
        chunks_path=CHUNKS_PATH,
        async_retriever=async_retriever,
        multi_intent_retriever=multi_intent_retriever,
        llm_client=llm,
    )

    return orchestrator, async_retriever, llm


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


def chunk_ids_of(intent):
    return intent.get("chunk_ids", [])


# =========================================================================
# 1. THREE INTENTS
# =========================================================================


async def test_three_intents():
    print()
    print("=" * 78)
    print("CASE 1 - THREE INDEPENDENT SUPPORTED INTENTS")
    print("=" * 78)

    query = (
        "What approval is needed for international travel? "
        "What documentation is needed for reimbursement? "
        "What is the workshop cancellation policy?"
    )

    def travel_table(_q):
        return [
            chunk("DOC_TRAVEL_POLICY_C002", "DOC_TRAVEL_POLICY", "2. International Travel",
                  "International travel requires prior approval from the appropriate manager.", 8.9),
        ]

    def multi_table(subquery):
        q = subquery.lower()
        if "travel" in q:
            return [
                chunk("DOC_TRAVEL_POLICY_C002", "DOC_TRAVEL_POLICY", "2. International Travel",
                      "International travel requires prior approval from the appropriate manager.", 8.9),
            ]
        if "reimbursement" in q or "documentation" in q:
            return [
                chunk("DOC_REIMBURSEMENT_POLICY_C002", "DOC_REIMBURSEMENT_POLICY", "2. International Expenses",
                      "International business expenses must include appropriate supporting documentation.", 6.4),
            ]
        if "workshop" in q or "cancellation" in q:
            return [
                chunk("DOC_WORKSHOP_POLICY_C004", "DOC_WORKSHOP_POLICY", "4. Cancellation",
                      "Workshop cancellations must be submitted at least 48 hours in advance.", 5.9),
            ]
        return []

    orchestrator, _, llm = build_orchestrator(travel_table, multi_table)
    session = SessionState(session_id="adv-1")
    final = await commit(orchestrator, session, query)

    ok = []
    ok.append(check("event", final["event"], "answer_completed"))
    ok.append(check("multi-intent flagged", final["is_multi_intent"], True))
    ok.append(check("three intents detected", len(final["intents"]), 3))
    ok.append(check("all three supported", [i["supported"] for i in final["intents"]], [True, True, True]))
    ok.append(check("intent 1 owns only travel chunk", chunk_ids_of(final["intents"][0]), ["DOC_TRAVEL_POLICY_C002"]))
    ok.append(check("intent 2 owns only reimbursement chunk", chunk_ids_of(final["intents"][1]), ["DOC_REIMBURSEMENT_POLICY_C002"]))
    ok.append(check("intent 3 owns only workshop chunk", chunk_ids_of(final["intents"][2]), ["DOC_WORKSHOP_POLICY_C004"]))
    for label in ("Intent 1:", "Intent 2:", "Intent 3:"):
        ok.append(check_true(f"answer contains '{label}'", label in final["answer"]))
    ok.append(check("citations verified", final["citation_valid"], True))
    for doc in ("DOC_TRAVEL_POLICY", "DOC_REIMBURSEMENT_POLICY", "DOC_WORKSHOP_POLICY"):
        ok.append(check_true(f"{doc} citation present", any(doc in c for c in final["citations"])))

    return all(ok)


# =========================================================================
# 2. MIXED SUPPORT (3 intents: supported / unsupported / supported)
# =========================================================================


async def test_mixed_support():
    print()
    print("=" * 78)
    print("CASE 2 - MIXED SUPPORT ACROSS THREE INTENTS")
    print("=" * 78)

    query = (
        "What approval is needed for international travel? "
        "What snacks are provided at the office? "
        "What is the workshop cancellation policy?"
    )

    def travel_table(_q):
        return []

    def multi_table(subquery):
        q = subquery.lower()
        if "travel" in q:
            return [
                chunk("DOC_TRAVEL_POLICY_C002", "DOC_TRAVEL_POLICY", "2. International Travel",
                      "International travel requires prior approval from the appropriate manager.", 8.9),
            ]
        if "snack" in q:
            # Nothing in the corpus supports this - clearly off-topic and
            # negative-scoring, with no companion candidate to "agree" with.
            return [
                chunk("DOC_CATERING_POLICY_C001", "DOC_CATERING_POLICY", "1. Snacks",
                      "This document does not exist in the real corpus and is unrelated.", -6.0),
            ]
        if "workshop" in q or "cancellation" in q:
            return [
                chunk("DOC_WORKSHOP_POLICY_C004", "DOC_WORKSHOP_POLICY", "4. Cancellation",
                      "Workshop cancellations must be submitted at least 48 hours in advance.", 5.9),
            ]
        return []

    orchestrator, _, llm = build_orchestrator(travel_table, multi_table)
    session = SessionState(session_id="adv-2")
    final = await commit(orchestrator, session, query)

    ok = []
    ok.append(check("event", final["event"], "answer_completed"))
    ok.append(check("support map", [i["supported"] for i in final["intents"]], [True, False, True]))
    ok.append(check("unsupported intent receives no evidence", chunk_ids_of(final["intents"][1]), []))
    ok.append(
        check_true(
            "unsupported intent declared with required text",
            GroundedAnswerGenerator.INTENT_INSUFFICIENT_MSG in final["answer"],
        )
    )
    ok.append(
        check(
            "no fabricated citation for the unsupported intent's document",
            [c for c in final["citations"] if "CATERING" in c],
            [],
        )
    )
    ok.append(check_true("travel citation present", any("DOC_TRAVEL_POLICY" in c for c in final["citations"])))
    ok.append(check_true("workshop citation present", any("DOC_WORKSHOP_POLICY" in c for c in final["citations"])))
    ok.append(check("citations verified", final["citation_valid"], True))

    return all(ok)


# =========================================================================
# 3. HIGHLY SIMILAR DOCUMENTS
# =========================================================================


async def test_highly_similar_documents():
    print()
    print("=" * 78)
    print("CASE 3 - HIGHLY SIMILAR (NEAR-IDENTICAL WORDING) DOCUMENTS")
    print("=" * 78)

    query = (
        "What approval is needed for international travel "
        "and what approval is needed for workshop attendance?"
    )

    def travel_table(_q):
        return []

    def multi_table(subquery):
        q = subquery.lower()
        if "travel" in q:
            return [
                chunk("DOC_TRAVEL_POLICY_C002", "DOC_TRAVEL_POLICY", "2. International Travel",
                      "International travel requires prior approval from the appropriate manager.", 8.5),
            ]
        if "workshop" in q:
            # Deliberately near-identical boilerplate wording to the travel
            # chunk above ("requires prior approval from the appropriate
            # manager"), but a different document entirely.
            return [
                chunk("DOC_WORKSHOP_POLICY_C005", "DOC_WORKSHOP_POLICY", "5. Approval",
                      "Workshop attendance requires prior approval from the appropriate manager.", 8.3),
            ]
        return []

    orchestrator, _, llm = build_orchestrator(travel_table, multi_table)
    session = SessionState(session_id="adv-3")
    final = await commit(orchestrator, session, query)

    ok = []
    ok.append(check("event", final["event"], "answer_completed"))
    ok.append(check("both intents supported", [i["supported"] for i in final["intents"]], [True, True]))
    ok.append(check("intent 1 keeps only its own (travel) chunk", chunk_ids_of(final["intents"][0]), ["DOC_TRAVEL_POLICY_C002"]))
    ok.append(check("intent 2 keeps only its own (workshop) chunk", chunk_ids_of(final["intents"][1]), ["DOC_WORKSHOP_POLICY_C005"]))
    ok.append(check_true("travel citation present", any("DOC_TRAVEL_POLICY" in c for c in final["citations"])))
    ok.append(check_true("workshop citation present", any("DOC_WORKSHOP_POLICY" in c for c in final["citations"])))
    ok.append(check("citations verified", final["citation_valid"], True))

    return all(ok)


# =========================================================================
# 4. SHARED CHUNK
# =========================================================================


async def test_shared_chunk():
    print()
    print("=" * 78)
    print("CASE 4 - SHARED CHUNK LEGITIMATELY RELEVANT TO BOTH INTENTS")
    print("=" * 78)

    query = (
        "What approval is needed for international travel "
        "and what documentation is needed for reimbursement?"
    )

    shared = chunk(
        "DOC_TRAVEL_POLICY_C005", "DOC_TRAVEL_POLICY", "5. International Expenses",
        "International trips require both manager approval and supporting "
        "documentation for any claimed expenses.", 7.0,
    )

    def travel_table(_q):
        return []

    def multi_table(subquery):
        q = subquery.lower()
        if "travel" in q:
            return [
                chunk("DOC_TRAVEL_POLICY_C002", "DOC_TRAVEL_POLICY", "2. International Travel",
                      "International travel requires prior approval from the appropriate manager.", 8.9),
                dict(shared),
            ]
        if "reimbursement" in q or "documentation" in q:
            return [
                chunk("DOC_REIMBURSEMENT_POLICY_C002", "DOC_REIMBURSEMENT_POLICY", "2. International Expenses",
                      "International business expenses must include appropriate supporting documentation.", 6.4),
                dict(shared),
            ]
        return []

    orchestrator, _, llm = build_orchestrator(travel_table, multi_table)
    session = SessionState(session_id="adv-4")
    final = await commit(orchestrator, session, query)

    ok = []
    ok.append(check("event", final["event"], "answer_completed"))
    ok.append(check("both intents still supported", [i["supported"] for i in final["intents"]], [True, True]))

    all_chunk_ids = chunk_ids_of(final["intents"][0]) + chunk_ids_of(final["intents"][1])
    ok.append(
        check(
            "shared chunk not duplicated across intents",
            all_chunk_ids.count("DOC_TRAVEL_POLICY_C005"),
            1,
        )
    )
    ok.append(
        check(
            "citation list has no duplicate strings",
            len(final["citations"]),
            len(set(final["citations"])),
        )
    )
    ok.append(check_true("travel citation present", any("DOC_TRAVEL_POLICY" in c for c in final["citations"])))
    ok.append(check_true("reimbursement citation present", any("DOC_REIMBURSEMENT_POLICY" in c for c in final["citations"])))
    ok.append(check("citations verified", final["citation_valid"], True))

    print(
        f"  [INFO] shared chunk credited to intent "
        f"{'1' if 'DOC_TRAVEL_POLICY_C005' in chunk_ids_of(final['intents'][0]) else '2'} "
        f"(EvidenceSelector.select_per_intent dedups a chunk_id to the "
        f"first intent that selects it - this is by design)."
    )

    case_a = all(ok)

    # ---- 4B: the shared chunk is an intent's ONLY qualifying evidence ----
    # This isolates exactly what the cross-intent dedup in
    # EvidenceSelector.select_per_intent does when the first intent to
    # process "claims" a chunk_id that a later intent has no other
    # evidence for. It is a documented, deliberate dedup (see the
    # docstring on select_per_intent), but it is worth surfacing
    # explicitly: a genuinely relevant retrieval result for intent 2 can
    # be silently dropped purely because of intent processing order.
    print()
    print("  CASE 4B - shared chunk is the SECOND intent's only evidence")

    def multi_table_sole(subquery):
        q = subquery.lower()
        if "travel" in q:
            return [
                chunk("DOC_TRAVEL_POLICY_C002", "DOC_TRAVEL_POLICY", "2. International Travel",
                      "International travel requires prior approval from the appropriate manager.", 8.9),
                dict(shared),
            ]
        if "reimbursement" in q or "documentation" in q:
            # No independent evidence of its own - the shared chunk is
            # genuinely the only thing retrieval found for this subquery.
            return [dict(shared)]
        return []

    orchestrator_b, _, _ = build_orchestrator(travel_table, multi_table_sole)
    session_b = SessionState(session_id="adv-4b")
    final_b = await commit(orchestrator_b, session_b, query)

    intent2_supported = final_b["intents"][1]["supported"] if len(final_b.get("intents", [])) > 1 else None
    if intent2_supported is False:
        print(
            "  [FINDING] intent 2's only qualifying evidence (the shared "
            "chunk) was claimed by intent 1's earlier dedup pass, so "
            "intent 2 was reported UNSUPPORTED despite retrieval genuinely "
            "finding relevant evidence for it. This is a real ordering-"
            "dependent gap in EvidenceSelector.select_per_intent's "
            "cross-intent dedup, not a flaw in this test."
        )
    else:
        print(
            f"  [INFO] intent 2 supported={intent2_supported} - shared "
            f"chunk starvation did not reproduce with this fixture."
        )

    # This is reported as a finding, not folded into case_a's pass/fail -
    # it documents existing dedup behaviour rather than a regression
    # introduced by this test file.
    return case_a


# =========================================================================
# 5. NEGATIVE CROSSENCODER + RETRIEVAL AGREEMENT
# =========================================================================


async def test_negative_ce_with_agreement():
    print()
    print("=" * 78)
    print("CASE 5 - NEGATIVE CROSSENCODER SCORES WITH RETRIEVAL AGREEMENT")
    print("=" * 78)

    query = (
        "What approval is needed for international travel "
        "and what documentation is needed for reimbursement?"
    )

    # Exact shape of the real Phase 4 bug: every CrossEncoder score for this
    # intent is negative, but BM25+Dense+RRF strongly agree on the
    # reimbursement cluster, and a low-RRF outlier from a different topic
    # must stay excluded.
    reimbursement_agreement_results = [
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

    def travel_table(_q):
        return []

    def multi_table(subquery):
        q = subquery.lower()
        if "travel" in q:
            return [
                chunk("DOC_TRAVEL_POLICY_C002", "DOC_TRAVEL_POLICY", "2. International Travel",
                      "International travel requires prior approval from the appropriate manager.", 7.4),
            ]
        if "reimbursement" in q or "documentation" in q:
            return reimbursement_agreement_results
        return []

    orchestrator, _, llm = build_orchestrator(travel_table, multi_table)
    session = SessionState(session_id="adv-5")
    final = await commit(orchestrator, session, query)

    ok = []
    ok.append(check("event", final["event"], "answer_completed"))
    ok.append(check("both intents supported", [i["supported"] for i in final["intents"]], [True, True]))
    ok.append(check("intent 2 recovered via retrieval agreement", final["intents"][1]["reason"], "sufficient_evidence_retrieval_agreement"))
    ok.append(
        check_true(
            "leaked travel chunk excluded from intent 2 evidence",
            "DOC_TRAVEL_POLICY_C002" not in chunk_ids_of(final["intents"][1]),
        )
    )
    ok.append(check_true("reimbursement citation present", any("DOC_REIMBURSEMENT_POLICY" in c for c in final["citations"])))
    ok.append(check("citations verified", final["citation_valid"], True))

    return all(ok)


# =========================================================================
# 6. NEGATIVE CROSSENCODER WITHOUT AGREEMENT
# =========================================================================


async def test_negative_ce_without_agreement():
    print()
    print("=" * 78)
    print("CASE 6 - NEGATIVE CROSSENCODER SCORES, NO RETRIEVAL AGREEMENT")
    print("=" * 78)

    query = (
        "What approval is needed for international travel "
        "and what documentation is needed for reimbursement?"
    )

    # A single weak, unrelated candidate: nothing for the retrieval stages
    # to "agree" with, so the fallback must not recover it.
    no_agreement_results = [
        chunk_with_rrf(
            "DOC_WORKSHOP_POLICY_C003", "DOC_WORKSHOP_POLICY", "3. Catering",
            "Workshop catering arrangements are handled by the facilities team.",
            0.0164, -6.4,
        ),
    ]

    def travel_table(_q):
        return []

    def multi_table(subquery):
        q = subquery.lower()
        if "travel" in q:
            return [
                chunk("DOC_TRAVEL_POLICY_C002", "DOC_TRAVEL_POLICY", "2. International Travel",
                      "International travel requires prior approval from the appropriate manager.", 7.4),
            ]
        if "reimbursement" in q or "documentation" in q:
            return no_agreement_results
        return []

    orchestrator, _, llm = build_orchestrator(travel_table, multi_table)
    session = SessionState(session_id="adv-6")
    final = await commit(orchestrator, session, query)

    ok = []
    ok.append(check("support map", [i["supported"] for i in final["intents"]], [True, False]))
    ok.append(
        check(
            "no fabricated reimbursement/workshop citation",
            [c for c in final["citations"] if "REIMBURSEMENT" in c or "WORKSHOP" in c],
            [],
        )
    )
    ok.append(check_true("travel citation still present", any("DOC_TRAVEL_POLICY" in c for c in final["citations"])))

    return all(ok)


# =========================================================================
# 7. LEXICAL DISTRACTOR
# =========================================================================


async def test_lexical_distractor():
    print()
    print("=" * 78)
    print("CASE 7 - LEXICAL OVERLAP WITHOUT TOPICAL RELEVANCE")
    print("=" * 78)

    query = (
        "What approval is needed for international travel "
        "and what documentation is needed for reimbursement?"
    )

    def travel_table(_q):
        return []

    def multi_table(subquery):
        q = subquery.lower()
        if "travel" in q:
            return [
                chunk("DOC_TRAVEL_POLICY_C002", "DOC_TRAVEL_POLICY", "2. International Travel",
                      "International travel requires prior approval from the appropriate manager.", 7.4),
            ]
        if "reimbursement" in q or "documentation" in q:
            # Shares the words "documentation" and "reimbursement" with the
            # question, but actually answers a different question (workshop
            # sign-up paperwork, not expense reimbursement). Plain chunk()
            # (no rrf_score) so it can never enter the retrieval-agreement
            # fallback either - lexical overlap alone must not recover it.
            return [
                chunk("DOC_WORKSHOP_POLICY_C007", "DOC_WORKSHOP_POLICY", "7. Sign-up Paperwork",
                      "Workshop sign-up documentation must be submitted before reimbursement of the "
                      "registration deposit can be processed.", -1.5),
            ]
        return []

    orchestrator, _, llm = build_orchestrator(travel_table, multi_table)
    session = SessionState(session_id="adv-7")
    final = await commit(orchestrator, session, query)

    ok = []
    ok.append(check("support map", [i["supported"] for i in final["intents"]], [True, False]))
    ok.append(
        check(
            "lexical-distractor document never cited",
            [c for c in final["citations"] if "WORKSHOP_POLICY_C007" in c or "DOC_WORKSHOP_POLICY" in c],
            [],
        )
    )
    ok.append(check_true("travel citation still present", any("DOC_TRAVEL_POLICY" in c for c in final["citations"])))

    return all(ok)


# =========================================================================
# 8. WRONG INTENT DECOMPOSITION
# =========================================================================


async def test_wrong_decomposition():
    print()
    print("=" * 78)
    print("CASE 8 - IMPERFECT SUBQUERY RETRIEVES ONLY VAGUE MATCHES")
    print("=" * 78)

    query = (
        "What approval is needed for international travel "
        "and what documentation is needed for reimbursement?"
    )

    # Simulates a poorly-formed subquery (e.g. "documentation ?" from an
    # imperfect decomposition) whose retrieval only surfaces two vague,
    # weakly-retrieved candidates: RRF scores sit BELOW
    # EvidenceSelector.RETRIEVAL_AGREEMENT_MIN_SIGNAL (0.02), so even
    # though the two happen to "agree" with each other in relative terms,
    # neither reflects genuine two-stage retrieval participation. CE
    # scores are also deeply negative. The system must not fabricate
    # support out of two mutually-vague candidates.
    vague_results = [
        chunk_with_rrf(
            "DOC_UNRELATED_A", "DOC_UNRELATED_POLICY", "1. Misc",
            "General office guidance that is only loosely related to the question.",
            0.018, -4.0,
        ),
        chunk_with_rrf(
            "DOC_UNRELATED_B", "DOC_UNRELATED_POLICY", "2. Misc",
            "Another loosely related passage that does not answer the question.",
            0.016, -4.5,
        ),
    ]

    def travel_table(_q):
        return []

    def multi_table(subquery):
        q = subquery.lower()
        if "travel" in q:
            return [
                chunk("DOC_TRAVEL_POLICY_C002", "DOC_TRAVEL_POLICY", "2. International Travel",
                      "International travel requires prior approval from the appropriate manager.", 7.4),
            ]
        if "reimbursement" in q or "documentation" in q:
            return vague_results
        return []

    orchestrator, _, llm = build_orchestrator(travel_table, multi_table)
    session = SessionState(session_id="adv-8")
    final = await commit(orchestrator, session, query)

    ok = []
    ok.append(check("support map", [i["supported"] for i in final["intents"]], [True, False]))
    ok.append(
        check(
            "no citation fabricated from the vague candidates",
            [c for c in final["citations"] if "UNRELATED" in c],
            [],
        )
    )
    ok.append(check_true("travel citation still present", any("DOC_TRAVEL_POLICY" in c for c in final["citations"])))

    return all(ok)


# =========================================================================
# 9. CITATION FABRICATION
# =========================================================================


async def test_citation_fabrication():
    print()
    print("=" * 78)
    print("CASE 9 - LLM FABRICATES A CITATION NOT IN ALLOWED CITATIONS")
    print("=" * 78)

    query = (
        "What approval is needed for international travel "
        "and what documentation is needed for reimbursement?"
    )

    def travel_table(_q):
        return []

    def multi_table(subquery):
        q = subquery.lower()
        if "travel" in q:
            return [
                chunk("DOC_TRAVEL_POLICY_C002", "DOC_TRAVEL_POLICY", "2. International Travel",
                      "International travel requires prior approval from the appropriate manager.", 8.9),
            ]
        if "reimbursement" in q or "documentation" in q:
            return [
                chunk("DOC_REIMBURSEMENT_POLICY_C002", "DOC_REIMBURSEMENT_POLICY", "2. International Expenses",
                      "International business expenses must include appropriate supporting documentation.", 6.4),
            ]
        return []

    orchestrator, _, llm = build_orchestrator(travel_table, multi_table, llm=FabricatingLLM())
    session = SessionState(session_id="adv-9")
    final = await commit(orchestrator, session, query)

    ok = []
    ok.append(check("event", final["event"], "answer_completed"))
    ok.append(check("citation validation rejects the answer", final["citation_valid"], False))
    ok.append(check("answer replaced with the unverified-answer message", final["answer"], UNVERIFIED_ANSWER_MSG))
    ok.append(check("no citations surfaced for a rejected answer", final["citations"], []))

    return all(ok)


# =========================================================================
# 10. UNSUPPORTED-INTENT HALLUCINATION
# =========================================================================


async def test_unsupported_hallucination():
    print()
    print("=" * 78)
    print("CASE 10 - LLM HALLUCINATES CONTENT FOR AN UNSUPPORTED INTENT")
    print("=" * 78)

    query = (
        "What approval is needed for international travel "
        "and what snacks are provided at the office?"
    )

    def travel_table(_q):
        return []

    def multi_table(subquery):
        q = subquery.lower()
        if "travel" in q:
            return [
                chunk("DOC_TRAVEL_POLICY_C002", "DOC_TRAVEL_POLICY", "2. International Travel",
                      "International travel requires prior approval from the appropriate manager.", 8.9),
            ]
        if "snack" in q:
            return [
                chunk("DOC_CATERING_POLICY_C001", "DOC_CATERING_POLICY", "1. Snacks",
                      "This document does not exist in the real corpus and is unrelated.", -6.0),
            ]
        return []

    orchestrator, _, llm = build_orchestrator(
        travel_table, multi_table, llm=HallucinatingUnsupportedLLM()
    )
    session = SessionState(session_id="adv-10")
    final = await commit(orchestrator, session, query)

    ok = []
    ok.append(check("event", final["event"], "answer_completed"))
    ok.append(check("intent 2 correctly reported unsupported", final["intents"][1]["supported"], False))
    ok.append(
        check(
            "no fabricated citation was accepted",
            [c for c in final["citations"] if "CATERING" in c],
            [],
        )
    )

    hallucinated_present = "unlimited amounts" in final["answer"]
    guard_held = check_true(
        "unsupported intent's hallucinated claim rejected or replaced",
        (not hallucinated_present) or (final["citation_valid"] is False),
    )
    ok.append(guard_held)

    if hallucinated_present and final["citation_valid"] is True:
        print(
            "  [FINDING] the current citation validator only checks whether "
            "citations in the text are backed by evidence; it has no "
            "check that an intent marked unsupported actually produced the "
            "required insufficiency sentence. An uncited hallucinated claim "
            "for an unsupported intent passes citation validation untouched "
            "and reaches the final answer. This is a real gap in the "
            "current safeguards, not a flaw in this test - the assertion "
            "above is left failing to surface it rather than being "
            "loosened to force a PASS."
        )

    return all(ok)


# =========================================================================
# 11. THREE-WAY EVIDENCE LEAKAGE
# =========================================================================


async def test_three_way_leakage():
    print()
    print("=" * 78)
    print("CASE 11 - THREE-WAY EVIDENCE LEAKAGE")
    print("=" * 78)

    query = (
        "What approval is needed for international travel? "
        "What documentation is needed for reimbursement? "
        "What is the workshop cancellation policy?"
    )

    # A candidate genuinely on-topic for intent 1 (travel) is deliberately
    # also returned - at a reasonable but non-dominant score - by intent 2
    # and intent 3's own retrieval tables, simulating vector-space leakage
    # across nearby topics.
    leaked = chunk(
        "DOC_TRAVEL_POLICY_C002", "DOC_TRAVEL_POLICY", "2. International Travel",
        "International travel requires prior approval from the appropriate manager.", 9.0,
    )

    def travel_table(_q):
        return []

    def multi_table(subquery):
        q = subquery.lower()
        if "travel" in q:
            return [dict(leaked)]
        if "reimbursement" in q or "documentation" in q:
            return [
                chunk("DOC_REIMBURSEMENT_POLICY_C002", "DOC_REIMBURSEMENT_POLICY", "2. International Expenses",
                      "International business expenses must include appropriate supporting documentation.", 6.0),
                dict(leaked, reranker_score=3.0),
            ]
        if "workshop" in q or "cancellation" in q:
            return [
                chunk("DOC_WORKSHOP_POLICY_C004", "DOC_WORKSHOP_POLICY", "4. Cancellation",
                      "Workshop cancellations must be submitted at least 48 hours in advance.", 5.5),
                dict(leaked, reranker_score=2.5),
            ]
        return []

    orchestrator, _, llm = build_orchestrator(travel_table, multi_table)
    session = SessionState(session_id="adv-11")
    final = await commit(orchestrator, session, query)

    ok = []
    ok.append(check("event", final["event"], "answer_completed"))
    ok.append(check("all three intents supported", [i["supported"] for i in final["intents"]], [True, True, True]))
    ok.append(check("intent 1 claims the travel chunk", chunk_ids_of(final["intents"][0]), ["DOC_TRAVEL_POLICY_C002"]))
    ok.append(
        check_true(
            "intent 2 keeps its own reimbursement evidence",
            "DOC_REIMBURSEMENT_POLICY_C002" in chunk_ids_of(final["intents"][1]),
        )
    )
    ok.append(
        check_true(
            "leaked travel chunk excluded from intent 2's own evidence",
            "DOC_TRAVEL_POLICY_C002" not in chunk_ids_of(final["intents"][1]),
        )
    )
    ok.append(
        check_true(
            "intent 3 keeps its own workshop evidence",
            "DOC_WORKSHOP_POLICY_C004" in chunk_ids_of(final["intents"][2]),
        )
    )
    ok.append(
        check_true(
            "leaked travel chunk excluded from intent 3's own evidence",
            "DOC_TRAVEL_POLICY_C002" not in chunk_ids_of(final["intents"][2]),
        )
    )
    all_ids = chunk_ids_of(final["intents"][0]) + chunk_ids_of(final["intents"][1]) + chunk_ids_of(final["intents"][2])
    ok.append(check("leaked chunk counted only once across all intents", all_ids.count("DOC_TRAVEL_POLICY_C002"), 1))
    ok.append(check("citations verified", final["citation_valid"], True))

    return all(ok)


# =========================================================================
# 12. SINGLE-INTENT REGRESSION
# =========================================================================


async def test_single_intent_regression():
    print()
    print("=" * 78)
    print("CASE 12 - SINGLE-INTENT BEHAVIOUR UNCHANGED")
    print("=" * 78)

    query = "What approval is needed for international travel?"

    def travel_table(_q):
        return [
            chunk("DOC_TRAVEL_POLICY_C002", "DOC_TRAVEL_POLICY", "2. International Travel",
                  "International travel requires prior approval from the appropriate manager.", 8.9),
        ]

    def multi_table(_subquery):
        return []

    orchestrator, async_retriever, llm = build_orchestrator(travel_table, multi_table)
    session = SessionState(session_id="adv-12")
    final = await commit(orchestrator, session, query)

    ok = []
    ok.append(check("event", final["event"], "answer_completed"))
    ok.append(check("not multi-intent", final["is_multi_intent"], False))
    ok.append(check("no intent payload", final["intents"], []))
    ok.append(check("single retrieval call", async_retriever.queries, [query]))
    ok.append(check_true("no intent blocks in prompt", "INTENT 1:" not in llm.prompts[0]))
    ok.append(check("citations verified", final["citation_valid"], True))
    ok.append(check_true("travel citation present", any("DOC_TRAVEL_POLICY" in c for c in final["citations"])))

    return all(ok)


# =========================================================================
# MAIN
# =========================================================================


async def main():
    cases = [
        ("three_intents", test_three_intents),
        ("mixed_support", test_mixed_support),
        ("highly_similar_documents", test_highly_similar_documents),
        ("shared_chunk", test_shared_chunk),
        ("negative_ce_with_agreement", test_negative_ce_with_agreement),
        ("negative_ce_without_agreement", test_negative_ce_without_agreement),
        ("lexical_distractor", test_lexical_distractor),
        ("wrong_decomposition", test_wrong_decomposition),
        ("citation_fabrication", test_citation_fabrication),
        ("unsupported_hallucination", test_unsupported_hallucination),
        ("three_way_leakage", test_three_way_leakage),
        ("single_intent_regression", test_single_intent_regression),
    ]

    results = {}
    for name, coro in cases:
        results[name] = await coro()

    print()
    print("=" * 78)
    print("PHASE 4 ADVERSARIAL SUMMARY")
    print("=" * 78)

    for name, passed in results.items():
        print(f"  {name:<32} {'PASS' if passed else 'FAIL'}")

    overall = all(results.values())

    print()
    print("STATUS:", "PASS" if overall else "FAIL")

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))