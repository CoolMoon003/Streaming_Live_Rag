"""
G5 - Session Refinement quantitative evaluator.

Proves quantitatively that the system supports STATE-PRESERVING refinement
(partial -> early retrieval -> Answer V1 -> late detail -> additive
refinement -> reuse existing evidence where sufficient -> delta retrieval
only for genuinely new information -> Answer V2) rather than restarting the
whole RAG pipeline on every turn.

This evaluator drives the REAL production code for every layer that is not
an ML model call:

    - backend.app.models.session.SessionState (reuse decisions, delta
      helpers, generation-id/answer-version bookkeeping)
    - backend.app.query.refinement.QueryRefinementAnalyzer
    - backend.app.query.refinement_query_builder.RefinementQueryBuilder
    - backend.app.query.refinement_retriever.RefinementRetriever
    - backend.app.orchestration.streaming_rag_orchestrator.StreamingRagOrchestrator
      (process_partial / process_commit, unmodified)
    - backend.app.retrieval.evidence_selector.EvidenceSelector
    - backend.app.retrieval.evidence_gate.EvidenceGate
    - backend.app.retrieval.citation_validator.CitationValidator

The two pieces this checkpoint environment cannot reach are the dense
embedding model (sentence-transformers/all-MiniLM-L6-v2) and the local
Ollama LLM -- both require network access to Hugging Face / a running
Ollama server that this environment does not have. StreamingRagOrchestrator
already accepts both as constructor-injected dependencies (see
scripts/test_early_reuse.py's own "--decision-only" split for exactly this
constraint), so this evaluator injects:

    - a small deterministic keyword-overlap fixture retriever standing in
      for BM25+Dense+RRF+CrossEncoder, and
    - a deterministic stub LLM client that always cites back the exact
      "Allowed Citations" it was given.

Everything else -- routing, reuse/delta decisions, evidence merging,
generation-id and answer-version bookkeeping, citation validation -- is the
unmodified production implementation. No production file is changed by
this script.

Usage:
    python -m scripts.evaluate_session_refinement
    python -m scripts.evaluate_session_refinement --json session_refinement_report.json
    python -m scripts.evaluate_session_refinement --strict-g5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any

from backend.app.models.session import SessionState, normalize_query_tokens
from backend.app.orchestration.streaming_rag_orchestrator import (
    StreamingRagOrchestrator,
)
from backend.app.retrieval.citation_validator import CitationValidator


# =============================================================================
# FIXTURE CORPUS (deterministic, keyword-overlap scored -- no ML models)
# =============================================================================

def _chunk(chunk_id, doc_id, section, text, source):
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "section": section,
        "text": text,
        "source": source,
    }


FIXTURE_CHUNKS = [
    {
        "chunk": _chunk(
            "T_C002",
            "DOC_TRAVEL_POLICY",
            "2. International Travel",
            "International travel requires prior approval from the manager "
            "before any flights are booked for the trip.",
            "travel_policy.md",
        ),
        "keywords": {"international", "travel", "approval", "manager"},
    },
    {
        "chunk": _chunk(
            "T_C003",
            "DOC_TRAVEL_POLICY",
            "3. Booking Process",
            "Late bookings require additional approval from the travel desk "
            "and may delay confirmation for the whole team.",
            "travel_policy.md",
        ),
        "keywords": {"late", "bookings", "booking", "approval", "travel", "desk"},
    },
    {
        "chunk": _chunk(
            "R_C001",
            "DOC_REIMBURSEMENT_POLICY",
            "1. Eligible Expenses",
            "Reimbursement claims must include original receipts for all "
            "business expenses submitted.",
            "reimbursement_policy.md",
        ),
        "keywords": {"reimbursement", "claims", "receipts", "expenses", "business"},
    },
    {
        "chunk": _chunk(
            "R_C002",
            "DOC_REIMBURSEMENT_POLICY",
            "2. Overseas Expenses",
            "Expense claims for overseas business trips require an "
            "additional currency conversion form attached.",
            "reimbursement_policy.md",
        ),
        "keywords": {
            "expense", "expenses", "claims", "currency", "conversion",
            "form", "overseas", "trips",
        },
    },
    {
        "chunk": _chunk(
            "W_C001",
            "DOC_WORKSHOP_POLICY",
            "1. Catering Arrangements",
            "Workshop catering must be arranged with the venue team well "
            "before the scheduled event.",
            "workshop_policy.md",
        ),
        "keywords": {"workshop", "catering", "venue", "event"},
    },
]


def _score(query: str, keywords: set) -> float:
    tokens = set(normalize_query_tokens(query))
    overlap = len(tokens & keywords)
    if overlap == 0:
        return -5.0
    return 2.0 + float(overlap)


class FixtureSyncRetriever:
    """Stands in for StreamingRetriever.retrieve() with deterministic
    keyword-overlap scoring instead of BM25+Dense+RRF+CrossEncoder.
    Only chunks with at least one keyword hit are returned, mirroring a
    real top-k pipeline that would not surface totally unrelated chunks.
    """

    def __init__(self, chunks=FIXTURE_CHUNKS, final_top_k: int = 5):
        self.chunks = chunks
        self.final_top_k = final_top_k
        self.call_log: list[str] = []

    def retrieve(self, query: str) -> dict:
        self.call_log.append(query)

        scored = []
        for entry in self.chunks:
            score = _score(query, entry["keywords"])
            if score > 0:
                scored.append((score, entry["chunk"]))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        scored = scored[: self.final_top_k]

        reranked = [
            {"chunk": chunk, "reranker_score": score, "rrf_score": score, "rank": i + 1}
            for i, (score, chunk) in enumerate(scored)
        ]

        return {
            "query": query,
            "bm25_results": [],
            "dense_results": [],
            "fused_results": reranked,
            "reranked_results": reranked,
        }


class FixtureAsyncRetriever:
    """Matches AsyncStreamingRetriever's interface (.retriever + async
    .retrieve(query, generation_id)) using the fixture sync retriever.
    """

    def __init__(self):
        self.retriever = FixtureSyncRetriever()
        self.call_log: list[str] = []

    async def retrieve(self, query: str, generation_id: int) -> dict:
        self.call_log.append(query)
        results = self.retriever.retrieve(query)
        return {"generation_id": generation_id, "query": query, "results": results}


class StubLLMClient:
    """Deterministic stand-in for OllamaClient. Never makes a network call.
    Extracts every citation block already present in the prompt (the
    "Allowed Citations" the generator computed from real, gated evidence)
    and cites all of them back, so CitationValidator always finds a
    grounded, valid answer -- exactly like a well-behaved model would,
    without spending an actual LLM call on session-layer plumbing tests.
    """

    def __init__(self):
        self.call_log: list[str] = []

    @staticmethod
    def _citations_in(prompt: str) -> list[str]:
        found = CitationValidator.CITATION_BLOCK_PATTERN.findall(prompt)
        seen = []
        for doc_id, section in found:
            citation = f"[{doc_id} \u00a7{section}]"
            if citation not in seen:
                seen.append(citation)
        return seen

    def generate(self, prompt: str) -> str:
        self.call_log.append(prompt)
        citations = self._citations_in(prompt)
        if not citations:
            return "The provided corpus does not contain enough evidence to answer this."
        return "Grounded refinement answer. " + " ".join(citations)

    def generate_stream(self, prompt: str):
        yield self.generate(prompt)

    def get_last_metrics(self) -> dict:
        return {"stub": True, "note": "no live LLM used in G5 evaluator"}


def make_orchestrator() -> tuple[StreamingRagOrchestrator, FixtureAsyncRetriever, StubLLMClient]:
    async_retriever = FixtureAsyncRetriever()
    llm_client = StubLLMClient()
    orchestrator = StreamingRagOrchestrator(
        chunks_path="unused-fixture-path",
        async_retriever=async_retriever,
        llm_client=llm_client,
    )
    return orchestrator, async_retriever, llm_client


# =============================================================================
# CASE RESULT MODEL
# =============================================================================

@dataclass
class CaseResult:
    case_id: str
    category: str
    initial_query: str
    refinement_query: str = ""
    final_query: str = ""
    expected_refinement_type: str = ""
    actual_refinement_type: str = ""
    expected_retrieval_behavior: str = ""
    actual_retrieval_behavior: str = ""
    retrieval_calls: int = 0
    generation_ids: list[int] = field(default_factory=list)
    answer_versions: list[int] = field(default_factory=list)
    initial_chunk_ids: list[str] = field(default_factory=list)
    final_chunk_ids: list[str] = field(default_factory=list)
    preserved_chunk_ids: list[str] = field(default_factory=list)
    newly_added_chunk_ids: list[str] = field(default_factory=list)
    duplicate_chunk_ids: list[str] = field(default_factory=list)
    state_continuous: bool = True
    answer_version_progression_ok: bool = True
    applicable_to_evidence_preservation: bool = False
    passed: bool = True
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "initial_query": self.initial_query,
            "refinement_query": self.refinement_query,
            "final_query": self.final_query,
            "expected_refinement_type": self.expected_refinement_type,
            "actual_refinement_type": self.actual_refinement_type,
            "expected_retrieval_behavior": self.expected_retrieval_behavior,
            "actual_retrieval_behavior": self.actual_retrieval_behavior,
            "retrieval_calls": self.retrieval_calls,
            "generation_ids": self.generation_ids,
            "answer_versions": self.answer_versions,
            "initial_chunk_ids": self.initial_chunk_ids,
            "final_chunk_ids": self.final_chunk_ids,
            "preserved_chunk_ids": self.preserved_chunk_ids,
            "newly_added_chunk_ids": self.newly_added_chunk_ids,
            "duplicate_chunk_ids": self.duplicate_chunk_ids,
            "state_continuous": self.state_continuous,
            "answer_version_progression_ok": self.answer_version_progression_ok,
            "passed": self.passed,
            "notes": self.notes,
        }


def _chunk_ids(results: list[dict]) -> list[str]:
    return [r["chunk"]["chunk_id"] for r in results]


async def _drain_commit(orchestrator, session, text):
    """Run process_commit to completion and return the terminal event."""
    final_event = None
    async for event in orchestrator.process_commit(session, text):
        if event.get("event") in ("answer_completed", "uncertainty_emitted"):
            final_event = event
    return final_event


# =============================================================================
# INITIAL QUERIES / FIXTURE TEXT USED ACROSS CASES
# =============================================================================

Q_TRAVEL = "What are the international travel approval rules"
Q_UNRELATED = "Please describe the weekend parking policy"


# =============================================================================
# CASES
# =============================================================================

async def case_1_exact_reuse() -> CaseResult:
    r = CaseResult(
        case_id="G5-01",
        category="exact_reuse",
        initial_query=Q_TRAVEL,
        final_query=Q_TRAVEL,
        expected_refinement_type="NEW",
        expected_retrieval_behavior="reuse_exact",
    )

    orchestrator, async_retriever, _ = make_orchestrator()
    session = SessionState(session_id="g5-01")

    await orchestrator.process_partial(session, Q_TRAVEL)
    r.initial_chunk_ids = _chunk_ids(session.latest_results)
    gen_after_partial = session.active_generation_id

    async_retriever.call_log.clear()
    async_retriever.retriever.call_log.clear()

    final = await _drain_commit(orchestrator, session, Q_TRAVEL)

    r.retrieval_calls = len(async_retriever.retriever.call_log)
    r.actual_refinement_type = final["refinement_type"] if final else "NONE"
    r.actual_retrieval_behavior = "reuse_exact" if r.retrieval_calls == 0 else "unexpected_retrieval"
    r.generation_ids = [gen_after_partial, session.active_generation_id]
    r.answer_versions = [final["answer_version"]] if final else []
    r.final_chunk_ids = _chunk_ids(session.latest_results)
    r.preserved_chunk_ids = [c for c in r.initial_chunk_ids if c in r.final_chunk_ids]
    r.duplicate_chunk_ids = [c for c in r.final_chunk_ids if r.final_chunk_ids.count(c) > 1]

    r.state_continuous = session.active_generation_id == gen_after_partial
    r.answer_version_progression_ok = r.answer_versions == [1]

    r.passed = (
        r.actual_refinement_type == "NEW"
        and r.retrieval_calls == 0
        and set(r.final_chunk_ids) == set(r.initial_chunk_ids)
        and not r.duplicate_chunk_ids
        and r.state_continuous
        and r.answer_version_progression_ok
    )
    return r


async def case_2_extension_covered() -> CaseResult:
    refinement = Q_TRAVEL + " for the whole team"
    r = CaseResult(
        case_id="G5-02",
        category="extension_tail_already_covered",
        initial_query=Q_TRAVEL,
        final_query=refinement,
        expected_refinement_type="NEW",
        expected_retrieval_behavior="reuse_extension_no_delta",
    )

    orchestrator, async_retriever, _ = make_orchestrator()
    session = SessionState(session_id="g5-02")

    await orchestrator.process_partial(session, Q_TRAVEL)
    r.initial_chunk_ids = _chunk_ids(session.latest_results)

    async_retriever.call_log.clear()
    async_retriever.retriever.call_log.clear()

    final = await _drain_commit(orchestrator, session, refinement)

    r.retrieval_calls = len(async_retriever.retriever.call_log)
    r.actual_refinement_type = final["refinement_type"] if final else "NONE"
    r.actual_retrieval_behavior = (
        "reuse_extension_no_delta" if r.retrieval_calls == 0 else "unexpected_delta_retrieval"
    )
    r.answer_versions = [final["answer_version"]] if final else []
    r.final_chunk_ids = _chunk_ids(session.latest_results)
    r.preserved_chunk_ids = [c for c in r.initial_chunk_ids if c in r.final_chunk_ids]
    r.duplicate_chunk_ids = [c for c in r.final_chunk_ids if r.final_chunk_ids.count(c) > 1]
    r.state_continuous = True

    r.passed = (
        r.actual_refinement_type == "NEW"
        and r.retrieval_calls == 0
        and set(r.final_chunk_ids) == set(r.initial_chunk_ids)
        and not r.duplicate_chunk_ids
    )
    return r


async def case_3_extension_new_retrieval() -> CaseResult:
    refinement = Q_TRAVEL + " and workshop catering arrangements"
    r = CaseResult(
        case_id="G5-03",
        category="extension_requires_new_retrieval",
        initial_query=Q_TRAVEL,
        final_query=refinement,
        expected_refinement_type="NEW",
        expected_retrieval_behavior="reuse_extension_with_delta",
        applicable_to_evidence_preservation=True,
    )

    orchestrator, async_retriever, _ = make_orchestrator()
    session = SessionState(session_id="g5-03")

    await orchestrator.process_partial(session, Q_TRAVEL)
    r.initial_chunk_ids = _chunk_ids(session.latest_results)

    async_retriever.call_log.clear()
    async_retriever.retriever.call_log.clear()

    final = await _drain_commit(orchestrator, session, refinement)

    r.retrieval_calls = len(async_retriever.retriever.call_log)
    r.actual_refinement_type = final["refinement_type"] if final else "NONE"
    r.actual_retrieval_behavior = (
        "reuse_extension_with_delta" if r.retrieval_calls >= 1 else "missing_delta_retrieval"
    )
    r.answer_versions = [final["answer_version"]] if final else []
    r.final_chunk_ids = _chunk_ids(session.latest_results)
    r.preserved_chunk_ids = [c for c in r.initial_chunk_ids if c in r.final_chunk_ids]
    r.newly_added_chunk_ids = [c for c in r.final_chunk_ids if c not in r.initial_chunk_ids]
    r.duplicate_chunk_ids = [c for c in r.final_chunk_ids if r.final_chunk_ids.count(c) > 1]

    r.passed = (
        r.actual_refinement_type == "NEW"
        and r.retrieval_calls == 1
        and set(r.initial_chunk_ids).issubset(set(r.final_chunk_ids))
        and len(r.newly_added_chunk_ids) > 0
        and not r.duplicate_chunk_ids
    )
    return r


async def case_4_additive_refinement() -> CaseResult:
    refinement = "Also what about reimbursement claims for these trips"
    r = CaseResult(
        case_id="G5-04",
        category="additive_refinement_after_answer",
        initial_query=Q_TRAVEL,
        refinement_query=refinement,
        expected_refinement_type="ADDITIVE",
        expected_retrieval_behavior="contextual_merge_retrieval",
        applicable_to_evidence_preservation=True,
    )

    orchestrator, async_retriever, _ = make_orchestrator()
    session = SessionState(session_id="g5-04")

    await orchestrator.process_partial(session, Q_TRAVEL)
    first = await _drain_commit(orchestrator, session, Q_TRAVEL)
    v1_chunk_ids = _chunk_ids(session.latest_results)
    gen_v1 = session.active_generation_id

    async_retriever.call_log.clear()
    async_retriever.retriever.call_log.clear()

    second = await _drain_commit(orchestrator, session, refinement)

    r.retrieval_calls = len(async_retriever.retriever.call_log)
    r.actual_refinement_type = second["refinement_type"] if second else "NONE"
    r.actual_retrieval_behavior = (
        "contextual_merge_retrieval" if r.retrieval_calls == 1 else "unexpected_retrieval_count"
    )
    r.generation_ids = [gen_v1, session.active_generation_id]
    r.answer_versions = [first["answer_version"], second["answer_version"]] if first and second else []
    r.initial_chunk_ids = v1_chunk_ids
    r.final_chunk_ids = _chunk_ids(session.latest_results)
    r.preserved_chunk_ids = [c for c in v1_chunk_ids if c in r.final_chunk_ids]
    r.newly_added_chunk_ids = [c for c in r.final_chunk_ids if c not in v1_chunk_ids]
    r.duplicate_chunk_ids = [c for c in r.final_chunk_ids if r.final_chunk_ids.count(c) > 1]

    r.state_continuous = session.active_generation_id > gen_v1
    r.answer_version_progression_ok = r.answer_versions == [1, 2]

    r.passed = (
        r.actual_refinement_type == "ADDITIVE"
        and r.retrieval_calls == 1
        and set(v1_chunk_ids).issubset(set(r.final_chunk_ids))
        and len(r.newly_added_chunk_ids) > 0
        and not r.duplicate_chunk_ids
        and r.state_continuous
        and r.answer_version_progression_ok
    )
    return r


async def case_5_multi_step_refinement() -> CaseResult:
    refinement_1 = "Also what about reimbursement claims for these trips"
    refinement_2 = "Also what about workshop catering arrangements"
    r = CaseResult(
        case_id="G5-05",
        category="multi_step_refinement_v1_v2_v3",
        initial_query=Q_TRAVEL,
        refinement_query=f"{refinement_1} | {refinement_2}",
        expected_refinement_type="ADDITIVE,ADDITIVE",
        expected_retrieval_behavior="contextual_merge_retrieval_x2",
        applicable_to_evidence_preservation=True,
    )

    orchestrator, async_retriever, _ = make_orchestrator()
    session = SessionState(session_id="g5-05")

    await orchestrator.process_partial(session, Q_TRAVEL)
    v1 = await _drain_commit(orchestrator, session, Q_TRAVEL)
    v1_ids = _chunk_ids(session.latest_results)
    gen1 = session.active_generation_id

    v2 = await _drain_commit(orchestrator, session, refinement_1)
    v2_ids = _chunk_ids(session.latest_results)
    gen2 = session.active_generation_id

    v3 = await _drain_commit(orchestrator, session, refinement_2)
    v3_ids = _chunk_ids(session.latest_results)
    gen3 = session.active_generation_id

    r.actual_refinement_type = f"{v2['refinement_type']},{v3['refinement_type']}"
    r.generation_ids = [gen1, gen2, gen3]
    r.answer_versions = [v1["answer_version"], v2["answer_version"], v3["answer_version"]]
    r.initial_chunk_ids = v1_ids
    r.final_chunk_ids = v3_ids
    r.preserved_chunk_ids = [c for c in v1_ids if c in v3_ids] + [
        c for c in v2_ids if c in v3_ids and c not in v1_ids
    ]
    r.newly_added_chunk_ids = [c for c in v3_ids if c not in v1_ids and c not in v2_ids]
    r.duplicate_chunk_ids = [c for c in v3_ids if v3_ids.count(c) > 1]

    monotonic_growth = (
        set(v1_ids).issubset(set(v2_ids)) and set(v2_ids).issubset(set(v3_ids))
    )
    r.state_continuous = gen1 < gen2 < gen3
    r.answer_version_progression_ok = r.answer_versions == [1, 2, 3]
    r.actual_retrieval_behavior = (
        "contextual_merge_retrieval_x2" if monotonic_growth else "evidence_not_monotonically_preserved"
    )

    r.passed = (
        v2["refinement_type"] == "ADDITIVE"
        and v3["refinement_type"] == "ADDITIVE"
        and monotonic_growth
        and not r.duplicate_chunk_ids
        and r.state_continuous
        and r.answer_version_progression_ok
    )
    return r


async def case_6_replacement() -> CaseResult:
    replacement = "No, I meant reimbursement claims for business trips"
    r = CaseResult(
        case_id="G5-06",
        category="replacement_refinement",
        initial_query=Q_TRAVEL,
        refinement_query=replacement,
        expected_refinement_type="REPLACEMENT",
        expected_retrieval_behavior="fresh_retrieval_no_merge",
    )

    orchestrator, async_retriever, _ = make_orchestrator()
    session = SessionState(session_id="g5-06")

    await orchestrator.process_partial(session, Q_TRAVEL)
    first = await _drain_commit(orchestrator, session, Q_TRAVEL)
    v1_ids = _chunk_ids(session.latest_results)
    gen1 = session.active_generation_id

    async_retriever.call_log.clear()
    async_retriever.retriever.call_log.clear()

    second = await _drain_commit(orchestrator, session, replacement)

    r.retrieval_calls = len(async_retriever.retriever.call_log)
    r.actual_refinement_type = second["refinement_type"] if second else "NONE"
    r.actual_retrieval_behavior = (
        "fresh_retrieval_no_merge" if r.retrieval_calls == 1 else "unexpected_retrieval_count"
    )
    r.generation_ids = [gen1, session.active_generation_id]
    r.answer_versions = [first["answer_version"], second["answer_version"]] if first and second else []
    r.initial_chunk_ids = v1_ids
    r.final_chunk_ids = _chunk_ids(session.latest_results)
    # For REPLACEMENT, old evidence is EXPECTED to be dropped, not preserved.
    r.preserved_chunk_ids = [c for c in v1_ids if c in r.final_chunk_ids]
    r.newly_added_chunk_ids = [c for c in r.final_chunk_ids if c not in v1_ids]
    r.duplicate_chunk_ids = [c for c in r.final_chunk_ids if r.final_chunk_ids.count(c) > 1]

    r.state_continuous = session.active_generation_id > gen1
    r.answer_version_progression_ok = r.answer_versions == [1, 2]

    r.passed = (
        r.actual_refinement_type == "REPLACEMENT"
        and r.retrieval_calls == 1
        and not r.preserved_chunk_ids  # old evidence correctly NOT carried over
        and len(r.final_chunk_ids) > 0
        and not r.duplicate_chunk_ids
        and r.state_continuous
        and r.answer_version_progression_ok
    )
    r.notes = "REPLACEMENT is expected to discard prior evidence; empty preserved_chunk_ids is a PASS signal here."
    return r


async def case_7_presentation_only() -> CaseResult:
    presentation = "Repeat that in two bullets"
    r = CaseResult(
        case_id="G5-07",
        category="presentation_only_turn",
        initial_query=Q_TRAVEL,
        refinement_query=presentation,
        expected_refinement_type="PRESENTATION",
        expected_retrieval_behavior="suppressed_no_retrieval",
    )

    orchestrator, async_retriever, _ = make_orchestrator()
    session = SessionState(session_id="g5-07")

    await orchestrator.process_partial(session, Q_TRAVEL)
    first = await _drain_commit(orchestrator, session, Q_TRAVEL)
    v1_ids = _chunk_ids(session.latest_results)
    gen1 = session.active_generation_id
    answer_v1 = first["answer"]

    async_retriever.call_log.clear()
    async_retriever.retriever.call_log.clear()

    events = []
    async for event in orchestrator.process_commit(session, presentation):
        events.append(event)

    completed = next((e for e in events if e.get("event") == "answer_completed"), None)

    r.retrieval_calls = len(async_retriever.retriever.call_log)
    r.actual_refinement_type = completed["refinement_type"] if completed else "NONE"
    r.actual_retrieval_behavior = (
        "suppressed_no_retrieval" if r.retrieval_calls == 0 else "unexpected_retrieval"
    )
    r.generation_ids = [gen1, session.active_generation_id]
    r.answer_versions = [first["answer_version"], completed["answer_version"]] if completed else []
    r.initial_chunk_ids = v1_ids
    r.final_chunk_ids = _chunk_ids(session.latest_results)
    r.preserved_chunk_ids = list(v1_ids)

    r.state_continuous = session.active_generation_id == gen1  # unchanged: no new generation
    r.answer_version_progression_ok = r.answer_versions == [1, 1]  # unchanged: suppressed, no new version

    echoed_correctly = bool(completed) and completed.get("answer") == answer_v1

    r.passed = (
        r.actual_refinement_type == "PRESENTATION"
        and completed is not None
        and completed.get("action") == "SUPPRESS"
        and r.retrieval_calls == 0
        and echoed_correctly
        and r.state_continuous
        and r.answer_version_progression_ok
    )
    return r


def case_8_stale_generation_protection() -> CaseResult:
    """
    Exercises SessionState.accept_results() directly -- the single choke
    point every orchestrator retrieval-acceptance call site (process_partial
    and every branch of process_commit) goes through. A retrieval result
    that arrives for a generation_id that is no longer the session's active
    generation must never be allowed to silently corrupt state.
    """
    r = CaseResult(
        case_id="G5-08",
        category="stale_generation_protection",
        initial_query="query A",
        refinement_query="query B (newer partial, arrives before stale gen-1 retrieval lands)",
        expected_refinement_type="N/A",
        expected_retrieval_behavior="stale_result_rejected",
    )

    session = SessionState(session_id="g5-08")

    gen1 = session.start_new_query("query A")
    results_a = [FIXTURE_CHUNKS[0]["chunk"]]
    accepted_a = session.accept_results(
        generation_id=gen1,
        query="query A",
        results=[{"chunk": results_a[0], "reranker_score": 5.0, "rank": 1}],
    )
    ids_after_a = _chunk_ids(session.latest_results)

    # A newer partial starts (transcript kept growing) -- bumps the active
    # generation before gen1's (hypothetical, slow) retrieval has returned.
    gen2 = session.start_new_query("query B")

    # The stale gen-1 retrieval now lands late.
    results_stale = [FIXTURE_CHUNKS[2]["chunk"]]
    accepted_stale = session.accept_results(
        generation_id=gen1,
        query="query A",
        results=[{"chunk": results_stale[0], "reranker_score": 5.0, "rank": 1}],
    )
    ids_after_stale_attempt = _chunk_ids(session.latest_results)

    r.generation_ids = [gen1, gen2]
    r.initial_chunk_ids = ids_after_a
    r.final_chunk_ids = ids_after_stale_attempt
    r.actual_retrieval_behavior = (
        "stale_result_rejected" if not accepted_stale else "stale_result_incorrectly_accepted"
    )
    r.state_continuous = (
        accepted_a
        and not accepted_stale
        and session.active_generation_id == gen2
        and ids_after_stale_attempt == ids_after_a  # state untouched by the stale attempt
    )
    r.answer_version_progression_ok = True  # no answers involved in this case

    r.passed = r.state_continuous and r.actual_retrieval_behavior == "stale_result_rejected"
    return r


async def case_9_empty_early_retrieval() -> CaseResult:
    r = CaseResult(
        case_id="G5-09",
        category="empty_early_retrieval",
        initial_query=Q_UNRELATED,
        final_query=Q_UNRELATED,
        expected_refinement_type="NEW",
        expected_retrieval_behavior="fresh_retrieval_after_empty_partial",
    )

    orchestrator, async_retriever, _ = make_orchestrator()
    session = SessionState(session_id="g5-09")

    await orchestrator.process_partial(session, Q_UNRELATED)
    r.initial_chunk_ids = _chunk_ids(session.latest_results)  # expected: []

    async_retriever.call_log.clear()
    async_retriever.retriever.call_log.clear()

    final = await _drain_commit(orchestrator, session, Q_UNRELATED)

    r.retrieval_calls = len(async_retriever.retriever.call_log)
    r.actual_refinement_type = final["refinement_type"] if final else "NONE"
    r.actual_retrieval_behavior = (
        "fresh_retrieval_after_empty_partial" if r.retrieval_calls == 1 else "unexpected_reuse_of_empty_evidence"
    )
    r.answer_versions = [final["answer_version"]] if final else []
    r.final_chunk_ids = _chunk_ids(session.latest_results)
    r.duplicate_chunk_ids = []
    r.state_continuous = True
    r.answer_version_progression_ok = r.answer_versions == [1]

    r.passed = (
        r.initial_chunk_ids == []
        and r.actual_refinement_type == "NEW"
        and r.retrieval_calls == 1  # correctly re-attempted retrieval rather than reusing empty evidence
        and final is not None
        and final.get("evidence_sufficient") is False
    )
    r.notes = "Empty early retrieval correctly triggers a fresh commit-time attempt rather than a false 'exact' reuse."
    return r


async def case_10_unrelated_followup() -> CaseResult:
    r = CaseResult(
        case_id="G5-10",
        category="unrelated_followup",
        initial_query=Q_TRAVEL,
        refinement_query=Q_UNRELATED,
        expected_refinement_type="NEW",
        expected_retrieval_behavior="fresh_retrieval_no_merge",
    )

    orchestrator, async_retriever, _ = make_orchestrator()
    session = SessionState(session_id="g5-10")

    await orchestrator.process_partial(session, Q_TRAVEL)
    first = await _drain_commit(orchestrator, session, Q_TRAVEL)
    v1_ids = _chunk_ids(session.latest_results)
    gen1 = session.active_generation_id

    async_retriever.call_log.clear()
    async_retriever.retriever.call_log.clear()

    second = await _drain_commit(orchestrator, session, Q_UNRELATED)

    r.retrieval_calls = len(async_retriever.retriever.call_log)
    r.actual_refinement_type = second["refinement_type"] if second else "NONE"
    r.actual_retrieval_behavior = (
        "fresh_retrieval_no_merge" if r.retrieval_calls == 1 else "unexpected_retrieval_count"
    )
    r.generation_ids = [gen1, session.active_generation_id]
    r.answer_versions = [first["answer_version"], second["answer_version"]] if first and second else []
    r.initial_chunk_ids = v1_ids
    r.final_chunk_ids = _chunk_ids(session.latest_results)
    r.preserved_chunk_ids = [c for c in v1_ids if c in r.final_chunk_ids]
    r.duplicate_chunk_ids = [c for c in r.final_chunk_ids if r.final_chunk_ids.count(c) > 1]

    r.state_continuous = session.active_generation_id > gen1
    r.answer_version_progression_ok = r.answer_versions == [1, 2]

    r.passed = (
        r.actual_refinement_type == "NEW"
        and r.retrieval_calls == 1
        and not r.preserved_chunk_ids  # unrelated topic must NOT inherit old evidence
        and not r.duplicate_chunk_ids
        and r.state_continuous
        and r.answer_version_progression_ok
    )
    return r


# =============================================================================
# RUN ALL CASES + METRICS
# =============================================================================

async def run_all_cases() -> list[CaseResult]:
    results = []
    results.append(await case_1_exact_reuse())
    results.append(await case_2_extension_covered())
    results.append(await case_3_extension_new_retrieval())
    results.append(await case_4_additive_refinement())
    results.append(await case_5_multi_step_refinement())
    results.append(await case_6_replacement())
    results.append(await case_7_presentation_only())
    results.append(case_8_stale_generation_protection())
    results.append(await case_9_empty_early_retrieval())
    results.append(await case_10_unrelated_followup())
    return results


def compute_metrics(cases: list[CaseResult]) -> dict:
    total = len(cases)

    def rate(numerator, denominator):
        return round(numerator / denominator, 4) if denominator else None

    refinement_type_matches = sum(
        1
        for c in cases
        if c.expected_refinement_type != "N/A"
        and c.actual_refinement_type == c.expected_refinement_type.split(",")[0]
    )
    # G5-05 has a compound expected type; count it correctly.
    g5_05 = next(c for c in cases if c.case_id == "G5-05")
    refinement_type_matches = sum(
        1
        for c in cases
        if c.case_id != "G5-05"
        and c.expected_refinement_type != "N/A"
        and c.actual_refinement_type == c.expected_refinement_type
    )
    refinement_type_matches += 1 if g5_05.actual_refinement_type == g5_05.expected_refinement_type else 0
    refinement_type_denominator = sum(1 for c in cases if c.expected_refinement_type != "N/A")

    retrieval_behavior_matches = sum(
        1 for c in cases if c.actual_retrieval_behavior == c.expected_retrieval_behavior
    )

    state_continuity_matches = sum(1 for c in cases if c.state_continuous)
    answer_version_matches = sum(1 for c in cases if c.answer_version_progression_ok)

    preservation_cases = [c for c in cases if c.applicable_to_evidence_preservation]
    preservation_matches = sum(
        1
        for c in preservation_cases
        if c.initial_chunk_ids and set(c.initial_chunk_ids).issubset(set(c.final_chunk_ids))
    )

    duplicate_cases = sum(1 for c in cases if c.duplicate_chunk_ids)

    stale_case = next(c for c in cases if c.category == "stale_generation_protection")
    stale_protection_rate = 1.0 if stale_case.passed else 0.0

    presentation_case = next(c for c in cases if c.category == "presentation_only_turn")
    presentation_suppression_rate = 1.0 if presentation_case.passed else 0.0

    return {
        "refinement_classification_accuracy": {
            "value": rate(refinement_type_matches, refinement_type_denominator),
            "matches": refinement_type_matches,
            "denominator": refinement_type_denominator,
        },
        "correct_reuse_or_delta_behavior": {
            "value": rate(retrieval_behavior_matches, total),
            "matches": retrieval_behavior_matches,
            "denominator": total,
        },
        "state_continuity_rate": {
            "value": rate(state_continuity_matches, total),
            "matches": state_continuity_matches,
            "denominator": total,
        },
        "answer_version_continuity_rate": {
            "value": rate(answer_version_matches, total),
            "matches": answer_version_matches,
            "denominator": total,
        },
        "evidence_preservation_rate": {
            "value": rate(preservation_matches, len(preservation_cases)),
            "matches": preservation_matches,
            "denominator": len(preservation_cases),
        },
        "duplicate_evidence_rate": {
            "value": rate(duplicate_cases, total),
            "matches": duplicate_cases,
            "denominator": total,
        },
        "stale_generation_protection_rate": {
            "value": stale_protection_rate,
            "matches": 1 if stale_case.passed else 0,
            "denominator": 1,
        },
        "presentation_suppression_rate": {
            "value": presentation_suppression_rate,
            "matches": 1 if presentation_case.passed else 0,
            "denominator": 1,
        },
    }


def strict_g5_pass(cases: list[CaseResult], metrics: dict) -> bool:
    if not cases:
        return False
    if metrics["state_continuity_rate"]["value"] != 1.0:
        return False
    if metrics["stale_generation_protection_rate"]["value"] != 1.0:
        return False
    if metrics["presentation_suppression_rate"]["value"] != 1.0:
        return False
    reuse_delta = metrics["correct_reuse_or_delta_behavior"]["value"]
    if reuse_delta is None or reuse_delta < 0.90:
        return False
    return True


def print_report(cases: list[CaseResult], metrics: dict, strict_pass: bool | None):
    print()
    print("=" * 90)
    print("G5 SESSION REFINEMENT EVALUATION")
    print("=" * 90)

    for c in cases:
        status = "PASS" if c.passed else "FAIL"
        print()
        print(f"[{status}] {c.case_id} - {c.category}")
        print(f"    initial_query:      {c.initial_query}")
        if c.refinement_query:
            print(f"    refinement_query:   {c.refinement_query}")
        if c.final_query:
            print(f"    final_query:        {c.final_query}")
        print(f"    refinement_type:    expected={c.expected_refinement_type} actual={c.actual_refinement_type}")
        print(f"    retrieval_behavior: expected={c.expected_retrieval_behavior} actual={c.actual_retrieval_behavior}")
        print(f"    retrieval_calls:    {c.retrieval_calls}")
        print(f"    generation_ids:     {c.generation_ids}")
        print(f"    answer_versions:    {c.answer_versions}")
        print(f"    initial_chunks:     {c.initial_chunk_ids}")
        print(f"    final_chunks:       {c.final_chunk_ids}")
        print(f"    preserved_chunks:   {c.preserved_chunk_ids}")
        print(f"    newly_added_chunks: {c.newly_added_chunk_ids}")
        print(f"    duplicate_chunks:   {c.duplicate_chunk_ids}")
        print(f"    state_continuous:   {c.state_continuous}")
        print(f"    version_progression_ok: {c.answer_version_progression_ok}")
        if c.notes:
            print(f"    notes:              {c.notes}")

    print()
    print("=" * 90)
    print("AGGREGATE METRICS")
    print("=" * 90)
    for name, m in metrics.items():
        print(f"  {name}: {m['value']}  ({m['matches']}/{m['denominator']})")

    total = len(cases)
    passed = sum(1 for c in cases if c.passed)
    print()
    print(f"CASES: {passed}/{total} passed")

    if strict_pass is not None:
        print()
        print(f"STRICT G5 GATE: {'PASS' if strict_pass else 'FAIL'}")
    print("=" * 90)


async def main_async(args):
    cases = await run_all_cases()
    metrics = compute_metrics(cases)

    strict_pass = None
    if args.strict_g5:
        strict_pass = strict_g5_pass(cases, metrics)

    print_report(cases, metrics, strict_pass)

    if args.json:
        report = {
            "cases": [c.to_dict() for c in cases],
            "metrics": metrics,
            "totals": {
                "total": len(cases),
                "passed": sum(1 for c in cases if c.passed),
                "failed": sum(1 for c in cases if not c.passed),
            },
        }
        if args.strict_g5:
            report["strict_g5_pass"] = strict_pass
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote JSON report to {args.json}")

    all_cases_passed = all(c.passed for c in cases)

    if args.strict_g5:
        sys.exit(0 if strict_pass else 1)
    else:
        sys.exit(0 if all_cases_passed else 1)


def main():
    parser = argparse.ArgumentParser(description="G5 Session Refinement evaluator")
    parser.add_argument("--json", type=str, default=None, help="Write a JSON report to this path")
    parser.add_argument("--strict-g5", action="store_true", help="Apply strict G5 pass/fail gates")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()