import asyncio

from backend.app.llm.grounded_generator import GroundedAnswerGenerator
from backend.app.llm.ollama_client import OllamaClient
from backend.app.retrieval.chunk_loader import load_chunks


async def main():
    chunks = load_chunks("data/processed/phase6_chunks.jsonl")
    by_id = {c["chunk_id"]: c for c in chunks}

    intents = [
        {
            "intent_id": 1,
            "query": "What approval is needed for international travel?",
            "evidence": [
                {"chunk": by_id["DOC_TRAVEL_POLICY_C002"]}
            ],
            "supported": True,
        },
        {
            "intent_id": 2,
            "query": "What documentation is needed for reimbursement?",
            "evidence": [
                {"chunk": by_id["DOC_EXPENSE_REIMBURSEMENT_POLICY_C002"]}
            ],
            "supported": True,
        },
    ]

    query = (
        "What approval is needed for international travel and "
        "what documentation is needed for reimbursement?"
    )

    llm = OllamaClient(model="llama3.2:3b")
    generator = GroundedAnswerGenerator(llm)

    print("=" * 80)
    print("REAL LLAMA3.2:3B MULTI-INTENT GENERATION — Q016")
    print("=" * 80)

    answer = generator.generate_multi_intent(
        query=query,
        intents=intents,
    )

    print("\nANSWER:")
    print(answer)

    print("\n" + "=" * 80)
    print("CHECKS")
    print("=" * 80)

    print("Intent 1 present:", "Intent 1:" in answer)
    print("Intent 2 present:", "Intent 2:" in answer)
    print(
        "Travel citation present:",
        "[DOC_TRAVEL_POLICY_C002 §" in answer,
    )
    print(
        "Reimbursement citation present:",
        "[DOC_EXPENSE_REIMBURSEMENT_POLICY_C002 §" in answer,
    )
    print("Mojibake Â§ present:", "Â§" in answer)


if __name__ == "__main__":
    asyncio.run(main())

