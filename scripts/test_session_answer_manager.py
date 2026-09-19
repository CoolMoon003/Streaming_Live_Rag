from backend.app.models.session import SessionState

from backend.app.retrieval.streaming_retriever import (
    StreamingRetriever,
)

from backend.app.llm.ollama_client import OllamaClient
from backend.app.llm.grounded_generator import (
    GroundedAnswerGenerator,
)

from backend.app.query.session_answer_manager import (
    SessionAnswerManager,
)


CHUNKS_PATH = "data/processed/chunks.jsonl"


def print_result(title: str, result: dict):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    print(f"ACTION: {result['action']}")
    print(
        f"REFINEMENT TYPE: "
        f"{result.get('refinement_type', '-')}"
    )
    print(
        f"ANSWER VERSION: "
        f"{result.get('answer_version', '-')}"
    )

    print()
    print("ANSWER:")
    print(result.get("answer", ""))

    print()
    print("CITATIONS:")

    for citation in result.get("citations", []):
        print(f"  - {citation}")


def main():

    print()
    print("=" * 80)
    print("SESSION-AWARE ANSWER MANAGER TEST")
    print("=" * 80)

    # --------------------------------------------------
    # Initialize components ONCE
    # --------------------------------------------------

    retriever = StreamingRetriever(
        CHUNKS_PATH
    )

    llm = OllamaClient(
        model="qwen3:8b"
    )

    generator = GroundedAnswerGenerator(
        llm
    )

    manager = SessionAnswerManager(
        retriever=retriever,
        generator=generator,
    )

    session = SessionState(
        session_id="demo-session-001"
    )

    # --------------------------------------------------
    # 1. NEW QUERY
    # --------------------------------------------------

    query_1 = (
        "What are the international travel rules?"
    )

    result_1 = manager.process(
        session=session,
        query=query_1,
    )

    print_result(
        "STEP 1 — NEW QUERY",
        result_1,
    )

    # --------------------------------------------------
    # 2. ADDITIVE REFINEMENT
    # --------------------------------------------------

    query_2 = (
        "And what about late bookings?"
    )

    result_2 = manager.process(
        session=session,
        query=query_2,
    )

    print_result(
        "STEP 2 — ADDITIVE REFINEMENT",
        result_2,
    )

    # --------------------------------------------------
    # 3. REPLACEMENT / CORRECTION
    # --------------------------------------------------

    query_3 = (
        "No, I meant the international "
        "reimbursement rules."
    )

    result_3 = manager.process(
        session=session,
        query=query_3,
    )

    print_result(
        "STEP 3 — REPLACEMENT",
        result_3,
    )

    # --------------------------------------------------
    # 4. PRESENTATION-ONLY REQUEST
    # --------------------------------------------------

    query_4 = (
        "Repeat that in two bullets."
    )

    result_4 = manager.process(
        session=session,
        query=query_4,
    )

    print_result(
        "STEP 4 — PRESENTATION SUPPRESSION",
        result_4,
    )

    # --------------------------------------------------
    # Session state
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL SESSION STATE")
    print("=" * 80)

    print(
        f"Session ID: "
        f"{session.session_id}"
    )

    print(
        f"Query Version: "
        f"{session.query_version}"
    )

    print(
        f"Answer Version: "
        f"{session.answer_version}"
    )

    print(
        f"Retrieval History: "
        f"{len(session.retrieval_history)}"
    )

    print(
        f"Answer History: "
        f"{len(session.answer_history)}"
    )

    print()
    print("LATEST ANSWER:")
    print(session.latest_answer)

    print()
    print("LATEST CITATIONS:")

    for citation in session.latest_citations:
        print(f"  - {citation}")

    print()
    print("=" * 80)
    print("SESSION ANSWER MANAGER TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()