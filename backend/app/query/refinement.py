from enum import Enum
from dataclasses import dataclass
import re


class RefinementType(str, Enum):
    NEW = "NEW"
    ADDITIVE = "ADDITIVE"
    REPLACEMENT = "REPLACEMENT"
    PRESENTATION = "PRESENTATION"


@dataclass
class RefinementDecision:
    refinement_type: RefinementType
    reason: str
    confidence: float


class QueryRefinementAnalyzer:
    """
    Determines how a new transcript relates to the
    previous query in the active session.

    This is a lightweight routing component, not a
    trained ML classifier.
    """

    PRESENTATION_PATTERNS = [
        r"\brepeat\b",
        r"\bsay that again\b",
        r"\bagain\b",
        r"\bin two bullets\b",
        r"\bin bullet points\b",
        r"\bmake it shorter\b",
        r"\bshorten that\b",
        r"\bsummarize that\b",
        r"\bsummary of that\b",
        r"\brephrase that\b",
        r"\breword that\b",
        r"\bexplain that again\b",
    ]

    REPLACEMENT_PATTERNS = [
        r"\bno\b",
        r"\bi meant\b",
        r"\bactually\b",
        r"\binstead\b",
        r"\bnot that\b",
        r"\bforget that\b",
        r"\brather\b",
        r"\bcorrection\b",
    ]

    ADDITIVE_PATTERNS = [
        r"\band also\b",
        r"\balso\b",
        r"\bwhat about\b",
        r"\bespecially\b",
        r"\badditionally\b",
        r"\bin addition\b",
        r"\btoo\b",
    ]

    def analyze(
        self,
        previous_query: str | None,
        new_query: str,
    ) -> RefinementDecision:

        previous = (previous_query or "").strip()
        current = new_query.strip()

        # --------------------------------------------------
        # No previous query = new query
        # --------------------------------------------------

        if not previous:
            return RefinementDecision(
                refinement_type=RefinementType.NEW,
                reason="no_previous_query",
                confidence=1.0,
            )

        if not current:
            return RefinementDecision(
                refinement_type=RefinementType.NEW,
                reason="empty_new_query",
                confidence=0.95,
            )

        normalized = current.lower()

        # --------------------------------------------------
        # Presentation-only request
        # --------------------------------------------------

        for pattern in self.PRESENTATION_PATTERNS:
            if re.search(pattern, normalized):
                return RefinementDecision(
                    refinement_type=RefinementType.PRESENTATION,
                    reason="presentation_restructure",
                    confidence=0.95,
                )

        # --------------------------------------------------
        # Explicit correction / replacement
        # --------------------------------------------------

        for pattern in self.REPLACEMENT_PATTERNS:
            if re.search(pattern, normalized):
                return RefinementDecision(
                    refinement_type=RefinementType.REPLACEMENT,
                    reason="explicit_query_correction",
                    confidence=0.92,
                )

        # --------------------------------------------------
        # Explicit additive refinement
        # --------------------------------------------------

        for pattern in self.ADDITIVE_PATTERNS:
            if re.search(pattern, normalized):
                return RefinementDecision(
                    refinement_type=RefinementType.ADDITIVE,
                    reason="explicit_additive_constraint",
                    confidence=0.90,
                )

        # --------------------------------------------------
        # Detect continuation using shared wording
        # --------------------------------------------------

        # --------------------------------------------------
        # Otherwise treat as a new query
        # --------------------------------------------------

        return RefinementDecision(
            refinement_type=RefinementType.NEW,
            reason="independent_query",
            confidence=0.75,
        )


if __name__ == "__main__":

    analyzer = QueryRefinementAnalyzer()

    tests = [
        (
            None,
            "What are the international travel rules?",
        ),
        (
            "What are the international travel rules?",
            "And what about late bookings?",
        ),
        (
            "What are the international travel rules?",
            "No, I meant the international reimbursement rules.",
        ),
        (
            "What are the international travel rules?",
            "Repeat that in two bullets.",
        ),
        (
            "What are the international travel rules?",
            "What documents are needed for international expenses?",
        ),
        (
            "What are the international travel rules?",
            "Also, does the policy require manager approval?",
        ),
    ]

    print()
    print("=" * 80)
    print("QUERY REFINEMENT ANALYZER TEST")
    print("=" * 80)

    for index, (previous, current) in enumerate(
        tests,
        start=1,
    ):

        decision = analyzer.analyze(
            previous_query=previous,
            new_query=current,
        )

        print()
        print(f"TEST {index}")
        print("-" * 80)

        print(
            f"Previous: {previous}"
        )

        print(
            f"Current:  {current}"
        )

        print(
            f"TYPE:       "
            f"{decision.refinement_type.value}"
        )

        print(
            f"REASON:     "
            f"{decision.reason}"
        )

        print(
            f"CONFIDENCE: "
            f"{decision.confidence:.2f}"
        )