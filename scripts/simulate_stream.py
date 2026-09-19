from backend.app.controller.retrieval_controller import (
    RetrievalController,
    RetrievalAction,
)


def print_decision(
    version: int,
    transcript: str,
    decision,
):
    print()
    print("-" * 80)
    print(f"TRANSCRIPT VERSION: v{version}")
    print(f'TEXT: "{transcript}"')
    print()
    print(f"ACTION:     {decision.action.value}")
    print(f"REASON:     {decision.reason}")
    print(f"CONFIDENCE: {decision.confidence:.2f}")
    print("-" * 80)


def main():

    controller = RetrievalController()

    # ---------------------------------------------------------
    # Scenario 1: Incremental query
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("SCENARIO 1 — INCREMENTAL QUERY")
    print("=" * 80)

    transcript_versions = [
        "I need to",
        "I need to know the rules for",
        "I need to know the rules for international travel",
        "I need to know the rules for international travel and late bookings",
    ]

    for version, transcript in enumerate(
        transcript_versions,
        start=1,
    ):

        decision = controller.decide(
            transcript=transcript,
            is_final=False,
        )

        print_decision(
            version,
            transcript,
            decision,
        )

    # ---------------------------------------------------------
    # Scenario 2: Presentation-only request
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("SCENARIO 2 — PRESENTATION SUPPRESSION")
    print("=" * 80)

    transcript = "Repeat that in two bullets"

    decision = controller.decide(
        transcript=transcript,
        is_final=True,
    )

    print_decision(
        1,
        transcript,
        decision,
    )

    # ---------------------------------------------------------
    # Scenario 3: Late additive constraint
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("SCENARIO 3 — ADDITIVE REFINEMENT")
    print("=" * 80)

    transcript_versions = [
        "What are the international travel rules?",
        "What are the international travel rules and what about late bookings?",
    ]

    for version, transcript in enumerate(
        transcript_versions,
        start=1,
    ):

        decision = controller.decide(
            transcript=transcript,
            is_final=False,
        )

        print_decision(
            version,
            transcript,
            decision,
        )

    # ---------------------------------------------------------
    # Scenario 4: Query correction
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("SCENARIO 4 — QUERY REPLACEMENT")
    print("=" * 80)

    transcript_versions = [
        "What are the international travel rules?",
        "No, I meant the international reimbursement rules",
    ]

    for version, transcript in enumerate(
        transcript_versions,
        start=1,
    ):

        decision = controller.decide(
            transcript=transcript,
            is_final=False,
        )

        print_decision(
            version,
            transcript,
            decision,
        )


if __name__ == "__main__":
    main()