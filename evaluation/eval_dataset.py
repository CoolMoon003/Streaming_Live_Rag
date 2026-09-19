"""
Evaluation datasets for Streaming-Live-RAG.

Phase 5:
    Uses the small fixed baseline corpus and its hand-reviewed labels.

Phase 6:
    Uses the larger generated enterprise-policy corpus:
        data/processed/phase6_chunks.jsonl
        data/processed/phase6_eval_queries.json

The architecture/unit fixtures remain separate and are not affected by this
module.
"""

import json
from pathlib import Path


# =========================================================================
# PHASE 5 BASELINE DATASET
# =========================================================================

CORPUS_SIZE_NOTE = (
    "Phase 5 baseline corpus has only 11 chunks across 3 documents. "
    "Its retrieval metrics describe behavior on this small fixed corpus, "
    "not a general benchmark."
)

RETRIEVAL_EVAL_QUERIES: list[dict] = [
    {
        "query": "What approval is needed for international travel?",
        "expected_chunk_ids": ["DOC_TRAVEL_POLICY_C002"],
    },
    {
        "query": "What are the rules for domestic travel expenses?",
        "expected_chunk_ids": ["DOC_TRAVEL_POLICY_C001"],
    },
    {
        "query": "What happens if a trip is booked late?",
        "expected_chunk_ids": ["DOC_TRAVEL_POLICY_C003"],
    },
    {
        "query": "What documentation is needed for international business expenses?",
        "expected_chunk_ids": ["DOC_REIMBURSEMENT_POLICY_C002"],
    },
    {
        "query": "What expenses can employees request reimbursement for?",
        "expected_chunk_ids": ["DOC_REIMBURSEMENT_POLICY_C001"],
    },
    {
        "query": "What is the deadline for submitting an expense claim?",
        "expected_chunk_ids": ["DOC_REIMBURSEMENT_POLICY_C003"],
    },
    {
        "query": "What are the exceptions to the reimbursement rules?",
        "expected_chunk_ids": ["DOC_REIMBURSEMENT_POLICY_C004"],
    },
    {
        "query": "How much venue capacity is required for a workshop?",
        "expected_chunk_ids": ["DOC_WORKSHOP_POLICY_C001"],
    },
    {
        "query": "What are the cancellation terms for a workshop venue?",
        "expected_chunk_ids": ["DOC_WORKSHOP_POLICY_C002"],
    },
    {
        "query": "What catering arrangements are needed for a workshop?",
        "expected_chunk_ids": ["DOC_WORKSHOP_POLICY_C003"],
    },
    {
        "query": "Do workshops with external participants need extra approval?",
        "expected_chunk_ids": ["DOC_WORKSHOP_POLICY_C004"],
    },
]

MULTI_INTENT_EVAL_QUERIES: list[dict] = [
    {
        "query": (
            "What approval is needed for international travel and what "
            "documentation is needed for reimbursement?"
        ),
        "expected_intent_chunk_ids": [
            ["DOC_TRAVEL_POLICY_C002"],
            ["DOC_REIMBURSEMENT_POLICY_C002", "DOC_REIMBURSEMENT_POLICY_C001"],
        ],
    },
    {
        "query": (
            "What are the cancellation terms for a workshop venue and what "
            "catering arrangements are needed?"
        ),
        "expected_intent_chunk_ids": [
            ["DOC_WORKSHOP_POLICY_C002"],
            ["DOC_WORKSHOP_POLICY_C003"],
        ],
    },
    {
        "query": (
            "What are the domestic travel rules and what is the deadline "
            "for submitting an expense claim?"
        ),
        "expected_intent_chunk_ids": [
            ["DOC_TRAVEL_POLICY_C001"],
            ["DOC_REIMBURSEMENT_POLICY_C003"],
        ],
    },
]


# =========================================================================
# PHASE 6 DATASET LOADER
# =========================================================================

PHASE6_QUERIES_PATH = Path("data/processed/phase6_eval_queries.json")


def load_phase6_eval_queries(
    path: str | Path = PHASE6_QUERIES_PATH,
) -> tuple[list[dict], list[dict]]:
    """
    Load the Phase 6 generated evaluation dataset.

    Expected JSON structure:

    {
        "retrieval_queries": [...],
        "multi_intent_queries": [...]
    }

    Returns:
        (retrieval_queries, multi_intent_queries)
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Phase 6 evaluation dataset not found: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected Phase 6 evaluation JSON object, got {type(data).__name__}"
        )

    retrieval_queries = data.get("retrieval_queries", [])
    multi_intent_queries = data.get("multi_intent_queries", [])

    if not isinstance(retrieval_queries, list):
        raise ValueError("'retrieval_queries' must be a list")

    if not isinstance(multi_intent_queries, list):
        raise ValueError("'multi_intent_queries' must be a list")

    return retrieval_queries, multi_intent_queries


def phase6_corpus_note(
    chunks_path: str | Path = "data/processed/phase6_chunks.jsonl",
) -> str:
    """
    Return a corpus-size note for Phase 6 without hardcoding the number
    of chunks.
    """
    path = Path(chunks_path)

    if not path.exists():
        return (
            f"Phase 6 corpus not found at {path}. "
            "Run scripts/generate_phase6_corpus.py first."
        )

    with path.open("r", encoding="utf-8") as f:
        chunk_count = sum(1 for line in f if line.strip())

    return (
        f"Phase 6 benchmark corpus contains {chunk_count} chunks. "
        "Metrics describe this generated evaluation corpus and should not "
        "be treated as universal retrieval quality."
    )