"""Deterministic offline benchmark for Samsung PRISM Theme 04 G2:
"Early retrieval before final transcript completion".

Usage:
    python -m scripts.evaluate_early_retrieval
    python -m scripts.evaluate_early_retrieval --strict-g2
    python -m scripts.evaluate_early_retrieval --json early_retrieval_report.json

This evaluator drives ONLY the decision layer (backend.app.controller
.retrieval_controller.RetrievalController). It never imports retrieval,
embeddings, the LLM, FAISS, sentence-transformers, or Ollama, and it makes
no production code changes. It is stdlib-only.

G2 target (informal, not an official acceptance test):
    Early retrieval should occur before final transcript completion for
    >=80% of eligible queries, with a false-trigger rate <=20%.

Definitions used by this script:

  * "Early retrieval": a RETRIEVE decision returned for any NON-FINAL
    partial transcript, strictly before the final transcript is
    submitted. A RETRIEVE decision on the final transcript itself does
    NOT count as early retrieval. A SUPPRESS decision is never counted
    as early retrieval, regardless of when it occurs.

  * "Eligible": a case whose utterance contains a substantive
    information-seeking request. Presentation-only requests (category D)
    and non-query conversational continuations (category H) are
    deliberately excluded from the eligible denominator -- they are not
    early-retrieval opportunities at all. A case is NOT excluded merely
    because the controller finds it hard; eligibility is a property of
    the benchmark case, decided by category, not of the controller's
    measured behavior on it.

  * "False trigger": a case labeled expected_early_retrieval=False (i.e.
    a presentation-only or non-query case) for which the controller
    nonetheless returns RETRIEVE on a non-final partial. This includes
    the case where that partial is itself explicitly a deliberately
    incomplete/non-query continuation.

Nothing here changes RetrievalController thresholds, StreamingRagOrchestrator
behavior, or evaluation metrics used elsewhere in the project.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.controller.retrieval_controller import (  # noqa: E402
    RetrievalAction,
    RetrievalController,
)

BENCHMARK_NAME = "G2 Early Retrieval (Streaming Live RAG, Samsung PRISM Theme 04)"


# ===========================================================================
# 1. Benchmark dataset
# ===========================================================================
#
# Every case was traced against the real, unmodified RetrievalController
# before being written down here, so `expected_final_action` and the
# reasoning in each description reflect actually-observed controller
# behavior, not a guess. `expected_early_retrieval` is the benchmark's
# design-intent label (would an ideal streaming system retrieve early for
# this kind of utterance?) and is what false-trigger detection is checked
# against; it is deliberately independent of whether today's controller
# happens to achieve it, so this benchmark can also surface real gaps
# (see categories E and G below) instead of only ever passing.


@dataclass
class BenchmarkCase:
    case_id: str
    description: str
    category: str
    partial_transcripts: list[str]
    final_transcript: str
    eligible_for_early_retrieval: bool
    expected_early_retrieval: bool
    expected_final_action: str  # "RETRIEVE" or "SUPPRESS" (or "WAIT" for
                                 # non-eligible conversational continuations
                                 # that never resolve into a query)


BENCHMARK: list[BenchmarkCase] = [

    # -----------------------------------------------------------------
    # A. Clear information queries
    # -----------------------------------------------------------------
    BenchmarkCase(
        case_id="G2-A01",
        description="Straightforward travel-rules question, builds up word by word.",
        category="clear_query",
        partial_transcripts=[
            "I need to",
            "I need to know the rules for",
            "I need to know the rules for international travel",
        ],
        final_transcript="I need to know the rules for international travel",
        eligible_for_early_retrieval=True,
        expected_early_retrieval=True,
        expected_final_action="RETRIEVE",
    ),
    BenchmarkCase(
        case_id="G2-A02",
        description="Short 'what are the rules' question, resolves in one extension.",
        category="clear_query",
        partial_transcripts=[
            "What are the",
            "What are the international travel rules",
        ],
        final_transcript="What are the international travel rules?",
        eligible_for_early_retrieval=True,
        expected_early_retrieval=True,
        expected_final_action="RETRIEVE",
    ),
    BenchmarkCase(
        case_id="G2-A03",
        description="'Who approves' question about domestic travel.",
        category="clear_query",
        partial_transcripts=[
            "Who approves",
            "Who approves domestic travel requests for staff",
        ],
        final_transcript="Who approves domestic travel requests for staff?",
        eligible_for_early_retrieval=True,
        expected_early_retrieval=True,
        expected_final_action="RETRIEVE",
    ),

    # -----------------------------------------------------------------
    # B. Multi-intent queries
    # -----------------------------------------------------------------
    BenchmarkCase(
        case_id="G2-B01",
        description="First clause is a complete standalone query before 'and' extends it.",
        category="multi_intent",
        partial_transcripts=[
            "What approval is needed for",
            "What approval is needed for international travel",
            "What approval is needed for international travel and",
        ],
        final_transcript=(
            "What approval is needed for international travel and what "
            "documentation is needed for reimbursement?"
        ),
        eligible_for_early_retrieval=True,
        expected_early_retrieval=True,
        expected_final_action="RETRIEVE",
    ),
    BenchmarkCase(
        case_id="G2-B02",
        description="Workshop rules clause is already retrievable before phishing clause arrives.",
        category="multi_intent",
        partial_transcripts=[
            "What are the workshop safety rules",
            "What are the workshop safety rules and",
        ],
        final_transcript=(
            "What are the workshop safety rules and what is the phishing "
            "reporting process?"
        ),
        eligible_for_early_retrieval=True,
        expected_early_retrieval=True,
        expected_final_action="RETRIEVE",
    ),
    BenchmarkCase(
        case_id="G2-B03",
        description="Reimbursement documentation clause is complete before the late-bookings clause.",
        category="multi_intent",
        partial_transcripts=[
            "What documentation is needed for reimbursement claims",
            "What documentation is needed for reimbursement claims and",
        ],
        final_transcript=(
            "What documentation is needed for reimbursement claims and "
            "who approves late bookings?"
        ),
        eligible_for_early_retrieval=True,
        expected_early_retrieval=True,
        expected_final_action="RETRIEVE",
    ),

    # -----------------------------------------------------------------
    # C. Late-detail queries (a qualifying detail arrives after the
    #    core question has already been retrievable)
    # -----------------------------------------------------------------
    BenchmarkCase(
        case_id="G2-C01",
        description="Core travel question retrieves; 'late bookings' detail added afterward.",
        category="late_detail",
        partial_transcripts=[
            "I need to know the rules for international travel",
        ],
        final_transcript=(
            "I need to know the rules for international travel and late bookings"
        ),
        eligible_for_early_retrieval=True,
        expected_early_retrieval=True,
        expected_final_action="RETRIEVE",
    ),
    BenchmarkCase(
        case_id="G2-C02",
        description="Remote-work eligibility question retrieves; equipment stipend detail added afterward.",
        category="late_detail",
        partial_transcripts=[
            "What is the policy for remote work eligibility",
        ],
        final_transcript=(
            "What is the policy for remote work eligibility and equipment stipends"
        ),
        eligible_for_early_retrieval=True,
        expected_early_retrieval=True,
        expected_final_action="RETRIEVE",
    ),
    BenchmarkCase(
        case_id="G2-C03",
        description=(
            "Bare 'who approves domestic travel requests' first WAITs as "
            "not-yet-stable, then an additive clause pushes it to RETRIEVE "
            "before the (identical) final commit."
        ),
        category="late_detail",
        partial_transcripts=[
            "Who approves domestic travel requests",
        ],
        final_transcript=(
            "Who approves domestic travel requests especially for last minute trips"
        ),
        eligible_for_early_retrieval=True,
        expected_early_retrieval=True,
        expected_final_action="RETRIEVE",
    ),

    # -----------------------------------------------------------------
    # D. Presentation-only requests (NOT eligible)
    # -----------------------------------------------------------------
    BenchmarkCase(
        case_id="G2-D01",
        description="'Repeat that in two bullets' — pure presentation restructure.",
        category="presentation_only",
        partial_transcripts=[
            "Can you",
            "Can you repeat that",
        ],
        final_transcript="Can you repeat that in two bullets",
        eligible_for_early_retrieval=False,
        expected_early_retrieval=False,
        expected_final_action="SUPPRESS",
    ),
    BenchmarkCase(
        case_id="G2-D02",
        description="'Make it shorter please' — presentation restructure.",
        category="presentation_only",
        partial_transcripts=[
            "Make it",
        ],
        final_transcript="Make it shorter please",
        eligible_for_early_retrieval=False,
        expected_early_retrieval=False,
        expected_final_action="SUPPRESS",
    ),
    BenchmarkCase(
        case_id="G2-D03",
        description="'Say that again please' — presentation restructure.",
        category="presentation_only",
        partial_transcripts=[
            "Say that",
        ],
        final_transcript="Say that again please",
        eligible_for_early_retrieval=False,
        expected_early_retrieval=False,
        expected_final_action="SUPPRESS",
    ),

    # -----------------------------------------------------------------
    # E. Incomplete utterances that eventually resolve into a real query
    # -----------------------------------------------------------------
    BenchmarkCase(
        case_id="G2-E01",
        description=(
            "Resolves to RETRIEVE one partial before the word 'travel' is "
            "even spoken, because 'rules' already signals query intent at "
            ">=6 words."
        ),
        category="incomplete_utterance",
        partial_transcripts=[
            "I need to know the rules for",
            "I need to know the rules for international",
        ],
        final_transcript="I need to know the rules for international travel",
        eligible_for_early_retrieval=True,
        expected_early_retrieval=True,
        expected_final_action="RETRIEVE",
    ),
    BenchmarkCase(
        case_id="G2-E02",
        description=(
            "'I was wondering about the...' stays WAIT through both partials; "
            "only the final (parental leave) transcript retrieves. A genuine "
            "controller gap, kept in the benchmark rather than hidden."
        ),
        category="incomplete_utterance",
        partial_transcripts=[
            "I was wondering",
            "I was wondering about the",
        ],
        final_transcript="I was wondering about the policy for parental leave",
        eligible_for_early_retrieval=True,
        expected_early_retrieval=True,
        expected_final_action="RETRIEVE",
    ),
    BenchmarkCase(
        case_id="G2-E03",
        description=(
            "'I need some information about the...' never matches an explicit "
            "query-intent phrase, so even the final transcript only retrieves "
            "via the generic final_transcript fallback, not early. A second "
            "genuine controller gap."
        ),
        category="incomplete_utterance",
        partial_transcripts=[
            "I need",
            "I need some information about the",
        ],
        final_transcript=(
            "I need some information about the training certification process"
        ),
        eligible_for_early_retrieval=True,
        expected_early_retrieval=True,
        expected_final_action="RETRIEVE",
    ),

    # -----------------------------------------------------------------
    # F. Corrections / replacements
    # -----------------------------------------------------------------
    BenchmarkCase(
        case_id="G2-F01",
        description=(
            "Original domestic-travel question retrieves early; user then "
            "corrects to international."
        ),
        category="correction",
        partial_transcripts=[
            "I need to know the domestic travel rules",
        ],
        final_transcript="No, I meant the international travel rules",
        eligible_for_early_retrieval=True,
        expected_early_retrieval=True,
        expected_final_action="RETRIEVE",
    ),
    BenchmarkCase(
        case_id="G2-F02",
        description="Domestic travel policy question retrieves early; user corrects with 'actually'.",
        category="correction",
        partial_transcripts=[
            "What is the policy for domestic travel",
        ],
        final_transcript="Actually I meant international travel policy",
        eligible_for_early_retrieval=True,
        expected_early_retrieval=True,
        expected_final_action="RETRIEVE",
    ),
    BenchmarkCase(
        case_id="G2-F03",
        description="Workshop schedule question retrieves early; user replaces it with phishing question.",
        category="correction",
        partial_transcripts=[
            "Tell me about the workshop schedule",
        ],
        final_transcript="No not that, tell me about the phishing reporting steps",
        eligible_for_early_retrieval=True,
        expected_early_retrieval=True,
        expected_final_action="RETRIEVE",
    ),

    # -----------------------------------------------------------------
    # G. Short / noisy speech before the real question
    # -----------------------------------------------------------------
    BenchmarkCase(
        case_id="G2-G01",
        description="Filler ('um so like') precedes a clear travel-rules question that still retrieves early.",
        category="noisy_short",
        partial_transcripts=[
            "um",
            "um so like",
            "um so like what are the travel rules",
        ],
        final_transcript="um so like what are the travel rules?",
        eligible_for_early_retrieval=True,
        expected_early_retrieval=True,
        expected_final_action="RETRIEVE",
    ),
    BenchmarkCase(
        case_id="G2-G02",
        description="Filler ('uh well so') precedes a parental-leave policy question that still retrieves early.",
        category="noisy_short",
        partial_transcripts=[
            "uh",
            "uh well",
            "uh well so what is the policy for parental leave",
        ],
        final_transcript="uh well so what is the policy for parental leave?",
        eligible_for_early_retrieval=True,
        expected_early_retrieval=True,
        expected_final_action="RETRIEVE",
    ),
    BenchmarkCase(
        case_id="G2-G03",
        description=(
            "Filler ('hmm let's see') precedes an 'what about' additive query "
            "that only resolves at the final transcript, not early. Kept as "
            "an honest noisy-speech failure case."
        ),
        category="noisy_short",
        partial_transcripts=[
            "hmm",
            "hmm let's see",
        ],
        final_transcript="hmm let's see okay what about equipment loss reporting",
        eligible_for_early_retrieval=True,
        expected_early_retrieval=True,
        expected_final_action="RETRIEVE",
    ),

    # -----------------------------------------------------------------
    # H. Non-query conversational continuation (NOT eligible)
    # -----------------------------------------------------------------
    BenchmarkCase(
        case_id="G2-H01",
        description="'okay thanks that's helpful' — closing remark, never a query.",
        category="nonquery_continuation",
        partial_transcripts=[
            "okay",
            "okay thanks",
        ],
        final_transcript="okay thanks that's helpful",
        eligible_for_early_retrieval=False,
        expected_early_retrieval=False,
        expected_final_action="WAIT",
    ),
    BenchmarkCase(
        case_id="G2-H02",
        description="'fine that works' — short acknowledgement, never a query.",
        category="nonquery_continuation",
        partial_transcripts=[
            "fine",
        ],
        final_transcript="fine that works",
        eligible_for_early_retrieval=False,
        expected_early_retrieval=False,
        expected_final_action="WAIT",
    ),
    BenchmarkCase(
        case_id="G2-H03",
        description="'sure sounds good' — short acknowledgement, never a query.",
        category="nonquery_continuation",
        partial_transcripts=[
            "sure",
        ],
        final_transcript="sure sounds good",
        eligible_for_early_retrieval=False,
        expected_early_retrieval=False,
        expected_final_action="WAIT",
    ),
]


# ===========================================================================
# 2. Deterministic dataset assertions
# ===========================================================================

def validate_benchmark(cases: list[BenchmarkCase]) -> list[str]:
    """Returns a list of failure messages; empty list means the benchmark
    dataset itself is well-formed. This does NOT run the controller."""

    failures: list[str] = []

    if not cases:
        failures.append("benchmark is empty")
        return failures

    seen_ids: set[str] = set()
    for c in cases:
        if c.case_id in seen_ids:
            failures.append(f"duplicate case_id: {c.case_id}")
        seen_ids.add(c.case_id)

        if not c.partial_transcripts:
            failures.append(f"{c.case_id}: has no partial_transcripts")
        if not c.final_transcript or not c.final_transcript.strip():
            failures.append(f"{c.case_id}: missing final_transcript")
        if c.expected_final_action not in ("RETRIEVE", "SUPPRESS", "WAIT"):
            failures.append(
                f"{c.case_id}: invalid expected_final_action "
                f"{c.expected_final_action!r}"
            )
        if not c.eligible_for_early_retrieval and c.expected_early_retrieval:
            failures.append(
                f"{c.case_id}: expected_early_retrieval=True but marked "
                f"ineligible; ineligible cases must be expected_early_retrieval=False"
            )

    return failures


# ===========================================================================
# 3. Running a case through the real controller
# ===========================================================================

@dataclass
class PartialTrace:
    index: int  # 1-based
    text: str
    action: str
    reason: str
    confidence: float


@dataclass
class CaseResult:
    case: BenchmarkCase
    partial_trace: list[PartialTrace] = field(default_factory=list)
    final_action: str = ""
    final_reason: str = ""
    final_confidence: float = 0.0
    first_early_retrieve_index: Optional[int] = None
    early_retrieved: bool = False
    false_trigger: bool = False
    suppress_count: int = 0

    def to_dict(self) -> dict:
        return {
            "case_id": self.case.case_id,
            "category": self.case.category,
            "description": self.case.description,
            "eligible": self.case.eligible_for_early_retrieval,
            "expected_early_retrieval": self.case.expected_early_retrieval,
            "expected_final_action": self.case.expected_final_action,
            "final_transcript": self.case.final_transcript,
            "partial_transcripts": self.case.partial_transcripts,
            "trace": [
                {
                    "index": t.index,
                    "text": t.text,
                    "action": t.action,
                    "reason": t.reason,
                    "confidence": t.confidence,
                }
                for t in self.partial_trace
            ],
            "final_action": self.final_action,
            "final_reason": self.final_reason,
            "final_confidence": self.final_confidence,
            "first_early_retrieve_index": self.first_early_retrieve_index,
            "early_retrieved": self.early_retrieved,
            "false_trigger": self.false_trigger,
            "suppress_count": self.suppress_count,
        }


def run_case(case: BenchmarkCase) -> CaseResult:
    """Feeds every partial transcript (is_final=False) into a *fresh*
    RetrievalController, then feeds the final transcript (is_final=True).
    Does not touch the orchestrator, retrievers, or any production
    retrieval/generation code."""

    controller = RetrievalController()
    result = CaseResult(case=case)

    previous: Optional[str] = None

    for i, partial in enumerate(case.partial_transcripts, start=1):
        decision = controller.decide(
            transcript=partial,
            previous_transcript=previous,
            is_final=False,
        )
        result.partial_trace.append(
            PartialTrace(
                index=i,
                text=partial,
                action=decision.action.value,
                reason=decision.reason,
                confidence=decision.confidence,
            )
        )

        if decision.action == RetrievalAction.SUPPRESS:
            result.suppress_count += 1

        # Early retrieval = RETRIEVE on a non-final partial, first
        # occurrence only. SUPPRESS is never early retrieval.
        if (
            decision.action == RetrievalAction.RETRIEVE
            and result.first_early_retrieve_index is None
        ):
            result.first_early_retrieve_index = i
            result.early_retrieved = True

            if not case.expected_early_retrieval:
                result.false_trigger = True

        previous = partial

    final_decision = controller.decide(
        transcript=case.final_transcript,
        previous_transcript=previous,
        is_final=True,
    )
    result.final_action = final_decision.action.value
    result.final_reason = final_decision.reason
    result.final_confidence = final_decision.confidence

    return result


# ===========================================================================
# 4. Aggregate metrics
# ===========================================================================

@dataclass
class Report:
    benchmark: str
    total_cases: int
    eligible_cases: int
    early_retrieval_cases: int
    early_retrieval_rate: float
    false_trigger_cases: int
    false_trigger_rate: float
    final_retrieve_rate: float
    cases: list[dict]
    by_category: dict
    first_retrieval_index_histogram: dict
    reason_counts: dict
    wait_reason_counts: dict
    suppress_count: int

    def to_dict(self) -> dict:
        return {
            "benchmark": self.benchmark,
            "total_cases": self.total_cases,
            "eligible_cases": self.eligible_cases,
            "early_retrieval_cases": self.early_retrieval_cases,
            "early_retrieval_rate": self.early_retrieval_rate,
            "false_trigger_cases": self.false_trigger_cases,
            "false_trigger_rate": self.false_trigger_rate,
            "final_retrieve_rate": self.final_retrieve_rate,
            "cases": self.cases,
            "by_category": self.by_category,
            "first_retrieval_index_histogram": self.first_retrieval_index_histogram,
            "reason_counts": self.reason_counts,
            "wait_reason_counts": self.wait_reason_counts,
            "suppress_count": self.suppress_count,
        }


def build_report(cases: list[BenchmarkCase]) -> tuple[Report, list[CaseResult]]:
    results = [run_case(c) for c in cases]

    total_cases = len(results)
    eligible = [r for r in results if r.case.eligible_for_early_retrieval]
    eligible_cases = len(eligible)

    early_retrieval_cases = sum(1 for r in eligible if r.early_retrieved)
    early_retrieval_rate = (
        early_retrieval_cases / eligible_cases if eligible_cases else 0.0
    )

    false_trigger_cases = sum(1 for r in results if r.false_trigger)
    # False-trigger rate is measured against the NON-eligible population,
    # since only non-eligible (expected_early_retrieval=False) cases can
    # produce a false trigger by definition.
    non_eligible = [r for r in results if not r.case.eligible_for_early_retrieval]
    false_trigger_rate = (
        false_trigger_cases / len(non_eligible) if non_eligible else 0.0
    )

    final_retrieve_cases = sum(1 for r in results if r.final_action == "RETRIEVE")
    final_retrieve_rate = final_retrieve_cases / total_cases if total_cases else 0.0

    by_category: dict = {}
    cats = sorted({r.case.category for r in results})
    for cat in cats:
        cat_results = [r for r in results if r.case.category == cat]
        cat_eligible = [r for r in cat_results if r.case.eligible_for_early_retrieval]
        cat_early = sum(1 for r in cat_eligible if r.early_retrieved)
        by_category[cat] = {
            "total": len(cat_results),
            "eligible": len(cat_eligible),
            "early_retrieval_cases": cat_early,
            "early_retrieval_rate": (
                cat_early / len(cat_eligible) if cat_eligible else None
            ),
            "false_triggers": sum(1 for r in cat_results if r.false_trigger),
        }

    histogram: Counter = Counter()
    for r in results:
        if r.first_early_retrieve_index is not None:
            histogram[str(r.first_early_retrieve_index)] += 1

    reason_counts: Counter = Counter()
    for r in results:
        for t in r.partial_trace:
            if t.action == "RETRIEVE":
                reason_counts[t.reason] += 1

    wait_reason_counts: Counter = Counter()
    for r in results:
        for t in r.partial_trace:
            if t.action == "WAIT":
                wait_reason_counts[t.reason] += 1

    suppress_total = sum(r.suppress_count for r in results)

    report = Report(
        benchmark=BENCHMARK_NAME,
        total_cases=total_cases,
        eligible_cases=eligible_cases,
        early_retrieval_cases=early_retrieval_cases,
        early_retrieval_rate=round(early_retrieval_rate, 4),
        false_trigger_cases=false_trigger_cases,
        false_trigger_rate=round(false_trigger_rate, 4),
        final_retrieve_rate=round(final_retrieve_rate, 4),
        cases=[r.to_dict() for r in results],
        by_category=by_category,
        first_retrieval_index_histogram=dict(sorted(histogram.items())),
        reason_counts=dict(reason_counts),
        wait_reason_counts=dict(wait_reason_counts),
        suppress_count=suppress_total,
    )

    return report, results


# ===========================================================================
# 5. Additional deterministic post-run assertions
# ===========================================================================

def validate_report(report: Report, results: list[CaseResult]) -> list[str]:
    failures: list[str] = []

    for r in results:
        for t in r.partial_trace:
            if t.action not in ("WAIT", "RETRIEVE", "SUPPRESS"):
                failures.append(f"{r.case.case_id}: unknown action {t.action!r}")

        if r.first_early_retrieve_index is not None:
            if r.first_early_retrieve_index > len(r.partial_trace):
                failures.append(
                    f"{r.case.case_id}: early retrieve index out of range"
                )
            # "Strictly before final" -- by construction the early index
            # only ever refers to a partial, and the final transcript is
            # evaluated separately afterward, so this is structurally
            # guaranteed; assert it explicitly anyway.
            if r.first_early_retrieve_index > len(r.case.partial_transcripts):
                failures.append(
                    f"{r.case.case_id}: early retrieve index not strictly "
                    f"before final transcript"
                )

    for rate_name, rate in (
        ("early_retrieval_rate", report.early_retrieval_rate),
        ("false_trigger_rate", report.false_trigger_rate),
        ("final_retrieve_rate", report.final_retrieve_rate),
    ):
        if not (0.0 <= rate <= 1.0):
            failures.append(f"{rate_name} out of [0,1] range: {rate}")

    try:
        json.dumps(report.to_dict())
    except (TypeError, ValueError) as exc:
        failures.append(f"JSON serialization failed: {exc}")

    return failures


# ===========================================================================
# 6. Printing
# ===========================================================================

def print_case_trace(r: CaseResult) -> None:
    print(f"CASE {r.case.case_id}")
    print(f"category: {r.case.category}")
    print(f"eligible: {r.case.eligible_for_early_retrieval}")
    print(f"expected_early: {r.case.expected_early_retrieval}")
    print(f"first_early_retrieve_index: {r.first_early_retrieve_index}")
    print(f"early_retrieved: {r.early_retrieved}")
    print(f"final_action: {r.final_action}")
    if r.false_trigger:
        print("FALSE TRIGGER")
    print("trace:")
    for t in r.partial_trace:
        print(f"{t.index} {t.action} {t.reason}")
    print(f"FINAL {r.final_action} {r.final_reason}")
    print()


def print_report(report: Report) -> None:
    print("=" * 78)
    print(report.benchmark)
    print("=" * 78)
    print()
    print(f"total_cases          : {report.total_cases}")
    print(f"eligible_cases        : {report.eligible_cases}")
    print(f"early_retrieval_cases : {report.early_retrieval_cases}")
    print(f"early_retrieval_rate  : {report.early_retrieval_rate:.4f}")
    print(f"false_trigger_cases   : {report.false_trigger_cases}")
    print(f"false_trigger_rate    : {report.false_trigger_rate:.4f}")
    print(f"final_retrieve_rate   : {report.final_retrieve_rate:.4f}")
    print(f"suppress_count        : {report.suppress_count}")
    print()

    print("-" * 78)
    print("BY CATEGORY")
    print("-" * 78)
    for cat, stats in report.by_category.items():
        rate = stats["early_retrieval_rate"]
        rate_str = f"{rate:.4f}" if rate is not None else "n/a (no eligible cases)"
        print(
            f"  {cat:26} total={stats['total']:<3} eligible={stats['eligible']:<3} "
            f"early={stats['early_retrieval_cases']:<3} rate={rate_str:<22} "
            f"false_triggers={stats['false_triggers']}"
        )
    print()

    print("-" * 78)
    print("FIRST EARLY-RETRIEVE PARTIAL INDEX HISTOGRAM")
    print("-" * 78)
    if report.first_retrieval_index_histogram:
        for idx, count in report.first_retrieval_index_histogram.items():
            print(f"  partial index {idx}: {count} case(s)")
    else:
        print("  (no case retrieved early)")
    print()

    print("-" * 78)
    print("RETRIEVE REASON COUNTS (across all partials, all cases)")
    print("-" * 78)
    for reason, count in sorted(report.reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason:35} {count}")
    print()

    print("-" * 78)
    print("WAIT REASON COUNTS (across all partials, all cases)")
    print("-" * 78)
    for reason, count in sorted(report.wait_reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason:35} {count}")
    print()


# ===========================================================================
# 7. CLI
# ===========================================================================

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic, stdlib-only benchmark for G2 early retrieval "
            "(decision-layer only, no LLM/embeddings/retrieval)."
        )
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="Write the machine-readable report to this JSON file.",
    )
    parser.add_argument(
        "--strict-g2",
        action="store_true",
        help=(
            "Exit non-zero if eligible_cases == 0, early_retrieval_rate < 0.80, "
            "or false_trigger_rate > 0.20. This is only the automated local "
            "gate corresponding to the Samsung G2 target -- it is NOT an "
            "official or complete acceptance test."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-case trace output (summary tables still print).",
    )
    args = parser.parse_args(argv)

    dataset_failures = validate_benchmark(BENCHMARK)
    if dataset_failures:
        print("BENCHMARK DATASET IS INVALID:")
        for f in dataset_failures:
            print(f"  - {f}")
        return 1

    report, results = build_report(BENCHMARK)

    report_failures = validate_report(report, results)

    if not args.quiet:
        for r in results:
            print_case_trace(r)

    print_report(report)

    if report_failures:
        print("=" * 78)
        print("DETERMINISTIC ASSERTION FAILURES")
        print("=" * 78)
        for f in report_failures:
            print(f"  - {f}")
        print()
        return 1

    print("DETERMINISTIC ASSERTIONS: ALL PASSED")
    print(
        "(non-empty benchmark, unique case ids, every case has partials + "
        "final, partials evaluated is_final=False, final evaluated "
        "is_final=True, early index strictly before final, rates in "
        "[0,1], JSON-serializable)"
    )
    print()

    if args.json:
        out_path = Path(args.json)
        out_path.write_text(
            json.dumps(report.to_dict(), indent=2),
            encoding="utf-8",
        )
        print(f"JSON report written to: {out_path}")
        print()

    if args.strict_g2:
        print("=" * 78)
        print("--strict-g2 GATE")
        print("=" * 78)
        print(
            "NOTE: this is only the automated local gate corresponding to "
            "the Samsung G2 target. It is not an official or complete "
            "acceptance test."
        )
        gate_failures = []
        if report.eligible_cases == 0:
            gate_failures.append("eligible_cases == 0")
        if report.early_retrieval_rate < 0.80:
            gate_failures.append(
                f"early_retrieval_rate {report.early_retrieval_rate:.4f} < 0.80"
            )
        if report.false_trigger_rate > 0.20:
            gate_failures.append(
                f"false_trigger_rate {report.false_trigger_rate:.4f} > 0.20"
            )

        if gate_failures:
            print("STRICT-G2: FAILED")
            for f in gate_failures:
                print(f"  - {f}")
            print()
            return 1

        print("STRICT-G2: PASSED")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())