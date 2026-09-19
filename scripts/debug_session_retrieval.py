from backend.app.retrieval.streaming_retriever import StreamingRetriever
from backend.app.retrieval.evidence_gate import EvidenceGate

CHUNKS_PATH = "data/processed/chunks.jsonl"

query = "What are the international travel rules?"

print("=" * 80)
print("DEBUG SESSION RETRIEVAL")
print("=" * 80)

retriever = StreamingRetriever(CHUNKS_PATH)

result = retriever.retrieve(query)

print("\n--- RERANKED RESULTS ---")

for i, item in enumerate(result["reranked_results"], start=1):
    chunk = item["chunk"]

    print(
        f"{i}. "
        f"{chunk['chunk_id']} | "
        f"{chunk['doc_id']} | "
        f"{chunk['section']} | "
        f"reranker_score={item.get('reranker_score')}"
    )

print("\n--- EVIDENCE GATE ---")

gate = EvidenceGate()

decision = gate.check(result["reranked_results"])

print(f"sufficient: {decision.sufficient}")
print(f"confidence: {decision.confidence}")
print(f"reason: {decision.reason}")
print(f"supported_chunk_ids: {decision.supported_chunk_ids}")