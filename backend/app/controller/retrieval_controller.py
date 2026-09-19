from enum import Enum
from dataclasses import dataclass
import re


class RetrievalAction(str, Enum):
    WAIT = "WAIT"
    RETRIEVE = "RETRIEVE"
    SUPPRESS = "SUPPRESS"


@dataclass
class ControllerDecision:
    action: RetrievalAction
    reason: str
    confidence: float


class RetrievalController:
    """
    Adaptive Query Router for Streaming Live RAG.

    WAIT:
        Transcript is too incomplete or unstable to retrieve.

    RETRIEVE:
        There is enough semantic intent to perform retrieval.

    SUPPRESS:
        Retrieval is unnecessary, usually for presentation-only
        follow-up requests.
    """

    SUPPRESS_PATTERNS = [
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
    ]

    INCOMPLETE_ENDINGS = {
        "a",
        "an",
        "the",
        "for",
        "to",
        "about",
        "and",
        "or",
        "with",
        "when",
        "where",
        "how",
        "what",
        "which",
        "that",
        "my",
        "does",
        "is",
        "are",
    }

    # Verbs/phrases that usually indicate the user is asking
    # for information rather than merely continuing a sentence.
    QUERY_INTENT_PATTERNS = [
        r"\bneed to know\b",
        r"\bwanted to know\b",
        r"\bwant to know\b",
        r"\blooking for\b",
        r"\bneed information\b",
        r"\btell me\b",
        r"\bexplain\b",
        r"\bshow me\b",
        r"\bfind out\b",
        r"\bcan i\b",
        r"\bcan an\b",
        r"\bcan we\b",
        r"\bwhat\b",
        r"\bwhy\b",
        r"\bhow\b",
        r"\bwhen\b",
        r"\bwhere\b",
        r"\bwhich\b",
        r"\bwho\b",
        r"\bdoes\b",
        r"\bdo\b",
        r"\bis\b",
        r"\bare\b",
        r"\bshould\b",
        r"\bcan\b",
        r"\bmust\b",
        r"\brequires?\b",
        r"\brules?\b",
        r"\bpolicy\b",
        r"\brequirements?\b",
    ]

    ADDITIVE_PATTERNS = [
        r"\band also\b",
        r"\balso\b",
        r"\bwhat about\b",
        r"\bespecially\b",
        r"\bwhile\b",
        r"\band\b.*\bdoes\b",
        r"\band\b.*\bcan\b",
        r"\band\b.*\bhow\b",
    ]

    def decide(
        self,
        transcript: str,
        previous_transcript: str | None = None,
        is_final: bool = False,
    ) -> ControllerDecision:

        text = transcript.strip()

        # ---------------------------------------------------------
        # 1. Empty transcript
        # ---------------------------------------------------------

        if not text:
            return ControllerDecision(
                action=RetrievalAction.WAIT,
                reason="empty_transcript",
                confidence=1.0,
            )

        normalized = text.lower()
        words = normalized.split()

        # ---------------------------------------------------------
        # 2. Presentation-only request
        # ---------------------------------------------------------

        for pattern in self.SUPPRESS_PATTERNS:
            if re.search(pattern, normalized):

                return ControllerDecision(
                    action=RetrievalAction.SUPPRESS,
                    reason="presentation_restructure",
                    confidence=0.95,
                )

        # ---------------------------------------------------------
        # 3. Explicit correction/replacement
        # ---------------------------------------------------------

        for pattern in self.REPLACEMENT_PATTERNS:
            if re.search(pattern, normalized):

                return ControllerDecision(
                    action=RetrievalAction.RETRIEVE,
                    reason="query_correction_or_replacement",
                    confidence=0.92,
                )

        # ---------------------------------------------------------
        # 4. Very short transcript
        # ---------------------------------------------------------

        if len(words) < 4:

            return ControllerDecision(
                action=RetrievalAction.WAIT,
                reason="insufficient_query_signal",
                confidence=0.90,
            )

        # ---------------------------------------------------------
        # 5. Obviously incomplete transcript
        #
        # Only apply this to non-final transcripts.
        # ---------------------------------------------------------

        if not is_final and words[-1] in self.INCOMPLETE_ENDINGS:

            return ControllerDecision(
                action=RetrievalAction.WAIT,
                reason="incomplete_transcript",
                confidence=0.88,
            )

        # ---------------------------------------------------------
        # 6. Additive refinement
        # ---------------------------------------------------------

        for pattern in self.ADDITIVE_PATTERNS:
            if re.search(pattern, normalized):

                return ControllerDecision(
                    action=RetrievalAction.RETRIEVE,
                    reason="additive_refinement",
                    confidence=0.90,
                )

        # ---------------------------------------------------------
        # 7. Detect semantic query intent
        # ---------------------------------------------------------

        has_query_intent = False

        for pattern in self.QUERY_INTENT_PATTERNS:
            if re.search(pattern, normalized):
                has_query_intent = True
                break

        # ---------------------------------------------------------
        # 8. Stable semantic query
        # ---------------------------------------------------------

        if has_query_intent and len(words) >= 6:

            return ControllerDecision(
                action=RetrievalAction.RETRIEVE,
                reason="stable_intent",
                confidence=0.90,
            )

        # ---------------------------------------------------------
        # 9. Final transcript
        # ---------------------------------------------------------

        if is_final and len(words) >= 5:

            return ControllerDecision(
                action=RetrievalAction.RETRIEVE,
                reason="final_transcript",
                confidence=0.85,
            )

        # ---------------------------------------------------------
        # 10. Default
        # ---------------------------------------------------------

        return ControllerDecision(
            action=RetrievalAction.WAIT,
            reason="intent_not_yet_stable",
            confidence=0.75,
        )


if __name__ == "__main__":
    controller = RetrievalController()

    tests = [
        "I need to",
        "I need to know the rules for",
        "I need to know the rules for international travel",
        "I need to know the rules for international travel and late bookings",
        "Repeat that in two bullets",
        "What are the international travel rules?",
        "What are the international travel rules and what about late bookings?",
        "No, I meant the international reimbursement rules",
    ]

    for text in tests:

        decision = controller.decide(
            transcript=text,
            is_final=False,
        )

        print(
            f"{decision.action.value:10} | "
            f"{decision.reason:35} | "
            f"{text}"
        )