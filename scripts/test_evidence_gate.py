from backend.app.retrieval.streaming_retriever import StreamingRetriever
from backend.app.retrieval.evidence_gate import EvidenceGate
from backend.app.retrieval.evidence_selector import EvidenceSelector


CHUNKS_PATH = "data/processed/chunks.jsonl"


def main():

    print()
    print("=" * 80)
    print("EVIDENCE SUFFICIENCY GATE TEST")
    print("=" * 80)

    retriever = StreamingRetriever(CHUNKS_PATH)
    selector = EvidenceSelector()
    gate = EvidenceGate()

    test_cases = [
        {
            "category": "1. CLEARLY ANSWERABLE QUERY",
            "query": "What are the rules for international travel?",
            "expected_sufficient": True,
        },
        {
            "category": "2. WEAKLY RELATED QUERY",
            "query": "Can an employee book a luxury submarine for domestic travel?",
            "expected_sufficient": False,
        },
        {
            "category": "3. MISSING NUMERIC FACT",
            "query": "What is the maximum international airfare amount an employee can claim?",
            "expected_sufficient": False,
        },
        {
            "category": "4. COMPLETELY UNRELATED QUERY",
            "query": "What is the capital of France?",
            "expected_sufficient": False,
        },
    ]

    all_passed = True

    for case in test_cases:
        query = case["query"]
        expected = case["expected_sufficient"]
        category = case["category"]

        print()
        print("-" * 80)
        print(f"CASE: {category}")
        print(f"QUERY: {query}")

        retrieval = retriever.retrieve(query)
        raw_results = retrieval["reranked_results"]

        selected = selector.select(raw_results, query=query)
        decision = gate.check(results=selected if selected else raw_results, query=query)

        print(f"SUFFICIENT: {decision.sufficient} (expected: {expected})")
        print(f"CONFIDENCE: {decision.confidence:.3f}")
        print(f"REASON: {decision.reason}")
        print(f"MISSING CONSTRAINTS: {decision.missing_constraints}")

        print("SUPPORTED CHUNKS:")
        for chunk_id in decision.supported_chunk_ids:
            print(f"  - {chunk_id}")

        passed = (decision.sufficient == expected)
        if not passed:
            all_passed = False
            print("--> RESULT: FAIL")
        else:
            print("--> RESULT: PASS")

    print()
    print("=" * 80)
    if all_passed:
        print("ALL EVIDENCE GATE TESTS PASSED")
    else:
        print("SOME EVIDENCE GATE TESTS FAILED")
    print("=" * 80)


if __name__ == "__main__":
    main()