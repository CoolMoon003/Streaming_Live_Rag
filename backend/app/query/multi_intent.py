from dataclasses import dataclass
import re


@dataclass
class MultiIntentDecision:
    is_multi_intent: bool
    reason: str
    confidence: float


@dataclass
class SubQuery:
    intent_id: int
    query: str


class MultiIntentDetector:
    """
    Lightweight deterministic detector for queries containing
    multiple independently answerable intents.

    This is a routing component, not a trained classifier.
    """

    QUESTION_START = (
        r"what|why|how|when|where|which|who|"
        r"can|could|does|do|is|are|should|must|"
        r"will|would"
    )

    def detect(self, query: str) -> MultiIntentDecision:

        text = query.strip()

        if not text:
            return MultiIntentDecision(
                False,
                "empty_query",
                1.0,
            )

        # ---------------------------------------------------------
        # Look for a coordination boundary followed by another
        # question/intent marker.
        #
        # Examples:
        #
        # "... and what ..."
        # "... and how ..."
        # "... and does ..."
        # "... also what ..."
        # ---------------------------------------------------------

        pattern = (
            rf"\s+(?:and|also|as well as)\s+"
            rf"(?:{self.QUESTION_START})\b"
        )

        if re.search(pattern, text, flags=re.IGNORECASE):

            return MultiIntentDecision(
                True,
                "second_question_after_coordination",
                0.94,
            )

        # ---------------------------------------------------------
        # Explicit "what about" normally introduces another intent.
        # ---------------------------------------------------------

        if re.search(
            r"\bwhat about\b",
            text,
            flags=re.IGNORECASE,
        ):

            return MultiIntentDecision(
                True,
                "what_about_secondary_intent",
                0.90,
            )

        # ---------------------------------------------------------
        # Multiple explicit question marks can indicate multiple
        # independently answerable questions.
        # ---------------------------------------------------------

        if text.count("?") >= 2:

            return MultiIntentDecision(
                True,
                "multiple_question_marks",
                0.95,
            )

        return MultiIntentDecision(
            False,
            "single_intent",
            0.90,
        )


class MultiIntentDecomposer:
    """
    Decomposes clearly independent questions.

    Avoids splitting normal coordinated phrases such as:
        "rules and requirements"

    Handles:
        "rules and what documentation..."
        "rules and also what..."
        "question? second question?"
    """

    CONNECTOR_PATTERN = re.compile(
        rf"\s+(?:and\s+also|and|also|as well as)\s+"
        rf"(?=(?:{MultiIntentDetector.QUESTION_START})\b)",
        flags=re.IGNORECASE,
    )

    QUESTION_BOUNDARY_PATTERN = re.compile(
        r"\?\s+"
        rf"(?=(?:{MultiIntentDetector.QUESTION_START})\b)",
        flags=re.IGNORECASE,
    )

    def decompose(self, query: str) -> list[SubQuery]:

        text = query.strip()

        if not text:
            return []

        # ---------------------------------------------------------
        # First split explicit independent questions:
        #
        # "What are the travel rules? What are the reimbursement
        # rules?"
        # ---------------------------------------------------------

        parts = self.QUESTION_BOUNDARY_PATTERN.split(text)

        # ---------------------------------------------------------
        # If there are no explicit question boundaries, look for
        # coordinated independent questions.
        # ---------------------------------------------------------

        if len(parts) == 1:

            parts = self.CONNECTOR_PATTERN.split(text)

        # ---------------------------------------------------------
        # Clean pieces
        # ---------------------------------------------------------

        cleaned_parts = []

        for part in parts:

            cleaned = part.strip()

            # Remove accidental leading connectors
            cleaned = re.sub(
                r"^(?:and\s+also|and|also|as well as)\s+",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )

            cleaned = cleaned.strip(" ,?.")

            if cleaned:
                cleaned_parts.append(cleaned)

        # ---------------------------------------------------------
        # Build SubQuery objects
        # ---------------------------------------------------------

        if len(cleaned_parts) <= 1:

            return [
                SubQuery(
                    intent_id=1,
                    query=text,
                )
            ]

        return [
            SubQuery(
                intent_id=index,
                query=(
                    part if part.endswith("?")
                    else part + "?"
                ),
            )
            for index, part in enumerate(
                cleaned_parts,
                start=1,
            )
        ]

if __name__ == "__main__":

    detector = MultiIntentDetector()
    decomposer = MultiIntentDecomposer()

    tests = [

        # -----------------------------------------------------
        # SINGLE INTENT
        # -----------------------------------------------------

        "What are the international travel rules?",

        "Tell me about international travel.",

        "What are the rules and requirements for international travel?",

        "What are the rules and booking requirements for international travel?",

        # -----------------------------------------------------
        # MULTI INTENT
        # -----------------------------------------------------

        "What approval is needed for international travel and what documentation is needed for reimbursement?",

        "What is required for international travel and what happens with late bookings?",

        "What are the travel rules and also what are the reimbursement rules?",

        "What approval is needed for travel and does late booking require approval?",

        "What are the travel rules? What are the reimbursement rules?",
    ]

    print()
    print("=" * 80)
    print("MULTI-INTENT DETECTOR / DECOMPOSER TEST")
    print("=" * 80)

    for index, query in enumerate(tests, start=1):

        decision = detector.detect(query)

        print()
        print(f"TEST {index}")
        print("-" * 80)

        print(f"QUERY: {query}")
        print(f"MULTI: {decision.is_multi_intent}")
        print(f"REASON: {decision.reason}")
        print(f"CONFIDENCE: {decision.confidence:.2f}")

        if decision.is_multi_intent:

            subqueries = decomposer.decompose(query)

            print("SUBQUERIES:")

            for subquery in subqueries:
                print(
                    f"  {subquery.intent_id}. "
                    f"{subquery.query}"
                )