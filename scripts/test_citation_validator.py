from backend.app.retrieval.streaming_retriever import StreamingRetriever
from backend.app.retrieval.evidence_gate import EvidenceGate
from backend.app.retrieval.citation_validator import CitationValidator


CHUNKS_PATH = "data/processed/chunks.jsonl"


def main():

    print()
    print("=" * 80)
    print("GROUNDED CITATION VALIDATOR TEST")
    print("=" * 80)

    retriever = StreamingRetriever(CHUNKS_PATH)
    gate = EvidenceGate()
    validator = CitationValidator()

    query = "What are the rules for international travel?"

    retrieval = retriever.retrieve(query)
    evidence = retrieval["reranked_results"]

    decision = gate.check(evidence)

    print()
    print(f"QUERY: {query}")
    print(f"EVIDENCE SUFFICIENT: {decision.sufficient}")

    if not decision.sufficient:

        print("UNCERTAINTY: TRUE")
        print(
            "Answer: I don't have enough evidence "
            "in the provided corpus to answer that."
        )

        return

    supported = [
        result
        for result in evidence
        if result["chunk"]["chunk_id"]
        in decision.supported_chunk_ids
    ]

    top = supported[0]["chunk"]

    answer = (
        "International travel requires prior approval "
        "from the appropriate manager. Employees may "
        "claim eligible airfare and accommodation "
        "expenses according to applicable travel limits. "
        f"[{top['doc_id']} §{top['section']}]"
    )

    result = validator.validate(
        answer=answer,
        evidence=supported,
    )

    print()
    print("ANSWER:")
    print(answer)

    print()
    print("CITATION VALIDATION:")
    print(f"VALID: {result['valid']}")

    print()
    print("CITATIONS FOUND:")

    for citation in result["citations_found"]:
        print(f"  - {citation}")

    print()
    print("INVALID CITATIONS:")

    for citation in result["invalid_citations"]:
        print(f"  - {citation}")

    print()
    print("=" * 80)
    print("CITATION VALIDATOR TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()