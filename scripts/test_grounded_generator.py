from backend.app.retrieval.streaming_retriever import StreamingRetriever
from backend.app.retrieval.evidence_gate import EvidenceGate
from backend.app.retrieval.citation_validator import CitationValidator

from backend.app.llm.ollama_client import OllamaClient
from backend.app.llm.grounded_generator import GroundedAnswerGenerator


CHUNKS_PATH = "data/processed/chunks.jsonl"


def main():

    print()
    print("=" * 80)
    print("GROUNDED LLM GENERATION TEST")
    print("=" * 80)

    retriever = StreamingRetriever(CHUNKS_PATH)
    gate = EvidenceGate()
    validator = CitationValidator()

    llm = OllamaClient(
        model="qwen3:8b"
    )

    generator = GroundedAnswerGenerator(llm)

    queries = [
        "What are the rules for international travel?",
        "What is the maximum international airfare amount?",
    ]

    for query in queries:

        print()
        print("-" * 80)
        print(f"QUERY: {query}")

        retrieval = retriever.retrieve(query)

        evidence = retrieval["reranked_results"]

        decision = gate.check(evidence, query=query)

        print(
            f"EVIDENCE SUFFICIENT: "
            f"{decision.sufficient}"
        )

        if not decision.sufficient:

            answer = (
                "The provided corpus does not contain "
                "enough evidence to answer this."
            )

            print()
            print("UNCERTAINTY: TRUE")
            print(f"ANSWER: {answer}")

            continue

        supported_evidence = [
            result
            for result in evidence
            if result["chunk"]["chunk_id"]
            in decision.supported_chunk_ids
        ]

        answer = generator.generate(
            query=query,
            evidence=supported_evidence,
        )

        metrics = llm.get_last_metrics()
        print(f"LATENCY: {metrics['latency_ms']:.1f}ms | TTFT: {metrics['ttft_ms']:.1f}ms")

        validation = validator.validate(
            answer=answer,
            evidence=supported_evidence,
        )

        print()
        print("ANSWER:")
        print(answer)

        print()
        print(
            f"CITATION VALID: "
            f"{validation['valid']}"
        )

        if validation["invalid_citations"]:
            print("INVALID CITATIONS:")

            for citation in validation["invalid_citations"]:
                print(f"  - {citation}")

    print()
    print("=" * 80)
    print("GROUNDED LLM TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()