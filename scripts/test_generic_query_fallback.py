"""Generic-query fallback: one refined retry when the first gate fails.

Real orchestrator, real EvidenceSelector / EvidenceGate / validator, stub
retriever and stub LLM (no models, no Ollama). Run from the project root:

    .venv\\Scripts\\python scripts\\test_generic_query_fallback.py
"""

import asyncio
import sys

from backend.app.models.session import SessionState
from backend.app.query.generic_query_expander import GenericQueryExpander
from scripts.test_multi_intent_answer import (
    CHUNKS_PATH,
    build_orchestrator,
    chunk,
    check,
    check_true,
    commit,
)

STRONG = [
    chunk("DOC_TRAVEL_POLICY_C002", "DOC_TRAVEL_POLICY", "2. International Travel",
          "International business travel requires prior approval from the "
          "employee's department manager.", 8.9),
]
WEAK = [
    chunk("DOC_TRAVEL_POLICY_C002", "DOC_TRAVEL_POLICY", "2. International Travel",
          "International business travel requires prior approval from the "
          "employee's department manager.", 0.2),
]
NO_NUMBER = [
    chunk("DOC_TRAVEL_POLICY_C001", "DOC_TRAVEL_POLICY", "1. Domestic Travel",
          "Domestic travel normally requires manager approval before booking.", 8.0),
]

GENERIC = "What is the travel policy?"
GENERIC_2 = "Explain the travel policy"


def refined_only_table(query):
    """Weak for the user's wording, strong only once the query is expanded."""
    return STRONG if "international" in query.lower() else WEAK


def run(table, text):
    orchestrator, retriever, llm = build_orchestrator(table, table)
    final = asyncio.run(commit(orchestrator, SessionState(session_id="t"), text))
    return final, retriever, llm


def main():
    ok = True

    print("\nEXPANDER. deterministic, corpus-derived, generic queries only")
    expander = GenericQueryExpander(CHUNKS_PATH)
    for q in (GENERIC, GENERIC_2):
        refined = expander.expand(q) or ""
        ok &= check_true(f"{q!r} expands with corpus terms", "international" in refined and "approval" in refined)
    for q in ("What is the weather policy?",
              "What is the travel policy for astronauts?",
              "What approval is required for international travel?",
              "What is the maximum hotel allowance?"):
        ok &= check(f"{q!r} not expanded", expander.expand(q), None)

    for q in (GENERIC, GENERIC_2):
        print(f"\nCASE A. generic query rescued by refinement: {q!r}")
        final, retriever, llm = run(refined_only_table, q)
        ok &= check("event", final["event"], "answer_completed")
        ok &= check("two retrievals (original, then refined)", len(retriever.queries), 2)
        ok &= check("first retrieval is the original query", retriever.queries[0], q)
        ok &= check_true("second retrieval is a different (refined) query", retriever.queries[1] != q)
        ok &= check_true("citations verified", final.get("citation_valid"))
        prompt = llm.prompts[-1]
        ok &= check_true("generator saw the original question", q in prompt)
        ok &= check("refined query not exposed to generator", retriever.queries[1] in prompt, False)

    print("\nCASE B. refined retry still weak -> original refusal, gate not weakened")
    final, retriever, llm = run(lambda q: WEAK, GENERIC)
    ok &= check("event", final["event"], "uncertainty_emitted")
    ok &= check("two retrievals", len(retriever.queries), 2)
    ok &= check("no LLM call", len(llm.prompts), 0)
    ok &= check("no citations", final["citations"], [])

    print("\nCASE C. unsupported non-generic query -> no fallback")
    final, retriever, llm = run(lambda q: WEAK, "What is the weather policy?")
    ok &= check("event", final["event"], "uncertainty_emitted")
    ok &= check("single retrieval", len(retriever.queries), 1)

    print("\nCASE D. first retrieval sufficient -> no fallback")
    final, retriever, llm = run(lambda q: STRONG, GENERIC)
    ok &= check("event", final["event"], "answer_completed")
    ok &= check("single retrieval", len(retriever.queries), 1)

    print("\nCASE E. missing numeric fact -> unchanged, no fallback")
    final, retriever, llm = run(lambda q: NO_NUMBER, "What is the maximum travel allowance?")
    ok &= check("event", final["event"], "uncertainty_emitted")
    ok &= check("reason", final["reason"], "missing_numeric_fact")
    ok &= check("single retrieval", len(retriever.queries), 1)

    print("\nSTATUS:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())