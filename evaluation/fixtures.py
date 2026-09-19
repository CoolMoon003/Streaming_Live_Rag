"""
Phase 5 - synthetic evidence fixtures for the architecture/unit evaluation.

These fixtures build the same result-dict shape the real retrieval pipeline
produces (see backend.app.retrieval.reranker.CrossEncoderReranker.rerank and
backend.app.retrieval.fusion.reciprocal_rank_fusion), but with hand-picked
scores so EvidenceSelector / EvidenceGate / CitationValidator can be
exercised deterministically, without Hugging Face models or FAISS.

This mirrors the fixture style already used in scripts/test_multi_intent_answer.py.
"""

from typing import Any


def chunk(
    chunk_id: str,
    doc_id: str,
    section: str,
    text: str,
    score: float,
) -> dict[str, Any]:
    """One reranked result, shaped like CrossEncoderReranker output."""
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


def chunk_with_rrf(
    chunk_id: str,
    doc_id: str,
    section: str,
    text: str,
    rrf_score: float,
    reranker_score: float,
) -> dict[str, Any]:
    """
    Same as chunk(), but also carries the RRF-fused score
    (EvidenceSelector._recover_by_retrieval_agreement reads "rrf_score",
    falling back to "score" - see evidence_selector.py).
    """
    entry = chunk(chunk_id, doc_id, section, text, reranker_score)
    entry["score"] = rrf_score
    entry["rrf_score"] = rrf_score
    return entry


def intent_entry(intent_id: int, query: str, results: list[dict]) -> dict:
    """One MultiIntentRetriever subquery entry (nested results shape)."""
    return {
        "intent_id": intent_id,
        "query": query,
        "results": {"reranked_results": results},
    }


# ---------------------------------------------------------------------
# EVIDENCE SELECTION fixtures (Phase 5 requirement 2)
# ---------------------------------------------------------------------

# Normal single-intent query: one clearly best chunk.
SELECTION_NORMAL_SINGLE_INTENT = [
    chunk(
        "DOC_TRAVEL_POLICY_C002", "DOC_TRAVEL_POLICY", "2. International Travel",
        "International travel requires prior approval from the appropriate manager.", 8.5,
    ),
]

# Multiple relevant chunks within score_margin of each other: all should be kept.
SELECTION_MULTIPLE_RELEVANT_CHUNKS = [
    chunk(
        "DOC_TRAVEL_POLICY_C002", "DOC_TRAVEL_POLICY", "2. International Travel",
        "International travel requires prior approval from the appropriate manager.", 8.5,
    ),
    chunk(
        "DOC_TRAVEL_POLICY_C003", "DOC_TRAVEL_POLICY", "3. Booking Requirements",
        "Travel should normally be booked through the approved booking process.", 7.0,
    ),
    chunk(
        "DOC_TRAVEL_POLICY_C001", "DOC_TRAVEL_POLICY", "1. Domestic Travel",
        "Employees travelling within the country may claim eligible transportation expenses.", 5.2,
    ),
]

# Competing documents: a workshop chunk scores close to a travel chunk; only
# the genuinely close-scoring competitor should survive the score_margin cut.
SELECTION_COMPETING_DOCUMENTS = [
    chunk(
        "DOC_TRAVEL_POLICY_C002", "DOC_TRAVEL_POLICY", "2. International Travel",
        "International travel requires prior approval from the appropriate manager.", 6.0,
    ),
    chunk(
        "DOC_WORKSHOP_POLICY_C004", "DOC_WORKSHOP_POLICY", "4. External Participants",
        "Workshops involving external participants may require additional organizational approval.", 5.0,
    ),
    chunk(
        "DOC_WORKSHOP_POLICY_C002", "DOC_WORKSHOP_POLICY", "2. Cancellation",
        "Venue cancellation terms depend on the booking agreement.", -3.0,
    ),
]

# Weak but correct evidence: below the whole-query floor (0.5) but the best
# the corpus has for this specific subquery - must be selectable via the
# per-intent floor, not the whole-query floor.
SELECTION_WEAK_BUT_CORRECT = [
    chunk(
        "DOC_REIMBURSEMENT_POLICY_C002", "DOC_REIMBURSEMENT_POLICY", "2. International Expenses",
        "International business expenses must include appropriate supporting documentation.", 0.21,
    ),
    chunk(
        "DOC_REIMBURSEMENT_POLICY_C003", "DOC_REIMBURSEMENT_POLICY", "3. Submission Deadline",
        "Expense claims should be submitted within the required reimbursement period.", 0.05,
    ),
]

# Irrelevant evidence: nothing here should ever be selected.
SELECTION_IRRELEVANT = [
    chunk(
        "DOC_WORKSHOP_POLICY_C003", "DOC_WORKSHOP_POLICY", "3. Catering",
        "Workshop catering arrangements are handled by the facilities team.", -6.4,
    ),
]

# Multi-intent: intent 1 (travel) is strong enough to exhaust max_chunks=3 on
# its own; intent 2 (reimbursement) must still get its own evidence budget.
SELECTION_MULTI_INTENT_TRAVEL = [
    chunk("DOC_TRAVEL_POLICY_C002", "DOC_TRAVEL_POLICY", "2. International Travel",
          "International travel requires prior approval from the appropriate manager.", 8.9),
    chunk("DOC_TRAVEL_POLICY_C003", "DOC_TRAVEL_POLICY", "3. Booking Requirements",
          "Travel should normally be booked through the approved booking process. "
          "Late bookings may require additional approval.", 7.4),
    chunk("DOC_TRAVEL_POLICY_C001", "DOC_TRAVEL_POLICY", "1. Domestic Travel",
          "Employees travelling within the country may claim eligible transportation expenses.", 6.2),
]

SELECTION_MULTI_INTENT_REIMBURSEMENT = [
    chunk("DOC_REIMBURSEMENT_POLICY_C002", "DOC_REIMBURSEMENT_POLICY", "2. International Expenses",
          "International business expenses must include appropriate supporting documentation.", 5.6),
    chunk("DOC_REIMBURSEMENT_POLICY_C001", "DOC_REIMBURSEMENT_POLICY", "1. Eligible Expenses",
          "Employees may request reimbursement for eligible business expenses supported by valid receipts.", 4.1),
]

# Retrieval-agreement recovery: CrossEncoder scores are all negative for the
# correct intent, but BM25/Dense/RRF agree strongly on the reimbursement
# cluster (real Phase 4 bug shape).
SELECTION_RETRIEVAL_AGREEMENT_RECOVERY = [
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
        "DOC_TRAVEL_POLICY_C002", "DOC_TRAVEL_POLICY", "2. International Travel",
        "International travel requires prior approval from the appropriate manager.",
        0.0153846, -10.41589,
    ),
]

# No retrieval agreement: a single weak candidate with nothing to agree with.
SELECTION_NO_RETRIEVAL_AGREEMENT = [
    chunk_with_rrf(
        "DOC_WORKSHOP_POLICY_C003", "DOC_WORKSHOP_POLICY", "3. Catering",
        "Workshop catering arrangements are handled by the facilities team.",
        0.0164, -6.4,
    ),
]

# Nothing supports this intent at all.
SELECTION_UNSUPPORTED = [
    chunk("DOC_WORKSHOP_POLICY_C003", "DOC_WORKSHOP_POLICY", "3. Catering",
          "Workshop catering arrangements are handled by the facilities team.", -6.4),
]


# ---------------------------------------------------------------------
# CITATION VALIDATION fixtures (Phase 5 requirement 4)
# ---------------------------------------------------------------------

CITATION_EVIDENCE_TRAVEL = [
    chunk("DOC_TRAVEL_POLICY_C002", "DOC_TRAVEL_POLICY", "2. International Travel",
          "International travel requires prior approval from the appropriate manager.", 8.0),
]

CITATION_EVIDENCE_REIMBURSEMENT = [
    chunk("DOC_REIMBURSEMENT_POLICY_C001", "DOC_REIMBURSEMENT_POLICY", "1. Eligible Expenses",
          "Employees may request reimbursement for eligible business expenses supported by valid receipts.", 7.0),
    chunk("DOC_REIMBURSEMENT_POLICY_C003", "DOC_REIMBURSEMENT_POLICY", "3. Submission Deadline",
          "Expense claims should be submitted within the required reimbursement period.", 6.5),
    chunk("DOC_REIMBURSEMENT_POLICY_C002", "DOC_REIMBURSEMENT_POLICY", "2. International Expenses",
          "International business expenses must include appropriate supporting documentation.", 6.0),
]