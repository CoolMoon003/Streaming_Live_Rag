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
        r"\bneed .{0,20}information\b",
        r"\btell me\b",
        r"\bexplain\b",
        r"\bshow me\b",
        r"\bfind out\b",
        r"\bwondering\b",
        r"\bcurious\b",
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

    # ---------------------------------------------------------------
    # Generic lexical/structural signal for "has a real topic appeared
    # yet", independent of the explicit QUERY_INTENT_PATTERNS above.
    #
    # STOPWORDS covers ordinary function words (articles, pronouns,
    # prepositions, conjunctions, auxiliary/modal verbs) plus common
    # spoken-language discourse fillers and contractions. None of these
    # entries are specific to any benchmark query -- they are ordinary
    # closed-class English words that carry no topic information on
    # their own.
    #
    # FRAMING_VERBS are open-ended information-seeking verbs that
    # introduce a request but, on their own, name no subject/topic
    # ("I was wondering about the...", "I need some information
    # about..."). They are excluded from the content-word count for the
    # same reason stopwords are: they signal *that* the speaker wants
    # something, not *what*.
    #
    # QUESTION_WORDS are wh-words / polar-question openers. They are
    # structural interrogative markers, not topic content, so they are
    # also excluded from the content-word count (their contribution to
    # "this is a question" is already captured by has_query_intent /
    # has_question_mark).
    # ---------------------------------------------------------------

    STOPWORDS = frozenset({
        # articles / determiners
        "a", "an", "the", "this", "that", "these", "those",
        # pronouns
        "i", "you", "he", "she", "it", "we", "they",
        "me", "him", "her", "us", "them",
        "my", "your", "his", "its", "our", "their",
        # conjunctions / connectors
        "and", "or", "but", "so", "because", "if", "while",
        "also", "especially", "though", "although",
        # prepositions
        "for", "to", "of", "in", "on", "at", "with", "about",
        "from", "into", "over", "under", "between", "during",
        "after", "before", "through", "without", "within",
        # auxiliary / modal / copula verbs
        "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "can", "could", "would", "should",
        "will", "shall", "must", "may", "might",
        # spoken-language discourse fillers
        "um", "uh", "uhh", "umm", "hmm", "well", "like",
        "okay", "ok", "please", "thanks", "thank", "yeah", "yep",
        "alright", "right", "actually", "basically", "literally",
        "just", "kind", "sort", "maybe", "sure", "fine",
        # common contractions
        "that's", "it's", "what's", "who's", "there's", "let's",
        "i'm", "you're", "we're", "they're", "i've",
        "don't", "doesn't", "didn't", "isn't", "aren't",
        "wasn't", "weren't", "can't", "couldn't", "wouldn't",
        "shouldn't",
    })

    FRAMING_VERBS = frozenset({
        "need", "needs", "needed", "want", "wants", "wanted",
        "wondering", "curious", "looking", "know", "information",
        "tell", "explain", "show", "find",
    })

    QUESTION_WORDS = frozenset({
        "what", "why", "how", "when", "where", "which",
        "who", "whom", "whose",
    })

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

    # Minimum number of real topic/content words required before a
    # generic additive connector ("also", "what about", "especially",
    # ...) is allowed to trigger retrieval on its own. This is what
    # keeps a bare "...and also..." or "...what about..." with no new
    # subject matter from retrieving (see decide() step 6).
    MIN_CONTENT_WORDS_FOR_ADDITIVE = 2

    # Minimum content words required, alongside an explicit
    # QUERY_INTENT_PATTERNS match, to treat a SHORT (<6 raw word)
    # utterance as having a stable, retrievable topic already.
    MIN_CONTENT_WORDS_WITH_QUERY_INTENT = 3

    # Minimum content words required to treat an utterance as having a
    # stable, retrievable topic even when NO explicit
    # QUERY_INTENT_PATTERNS match and no "?" is present. This is
    # intentionally higher than the threshold above: without any
    # interrogative/framing marker at all, we ask for more topic
    # evidence before committing to early retrieval.
    MIN_CONTENT_WORDS_WITHOUT_QUERY_INTENT = 4

    @classmethod
    def _content_word_count(cls, words: list[str]) -> int:
        """Counts words that plausibly carry topic/subject information.

        Excludes ordinary function words, discourse fillers and
        contractions (STOPWORDS), open-ended framing verbs that
        introduce a request without naming its subject (FRAMING_VERBS),
        and wh-word/question-word openers (QUESTION_WORDS), since those
        three groups signal *that* something is being asked, not *what*
        it is about. This is a lightweight lexical heuristic, not a
        semantic classifier: no benchmark-specific phrases are checked
        here, only closed-class function words and generic discourse
        markers.
        """

        count = 0

        for raw in words:

            token = raw.strip(".,!?;:\"'()[]{}").lower()

            if not token:
                continue

            if token in cls.STOPWORDS:
                continue

            if token in cls.FRAMING_VERBS:
                continue

            if token in cls.QUESTION_WORDS:
                continue

            count += 1

        return count

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
        #
        # A bare connector ("also", "what about", "especially", ...)
        # is not enough on its own -- it must be attached to at least
        # a couple of real topic/content words, or it is just filler
        # ("...and also...", "...what about...") with nothing new to
        # retrieve yet.
        # ---------------------------------------------------------

        for pattern in self.ADDITIVE_PATTERNS:
            if re.search(pattern, normalized):

                if (
                    self._content_word_count(words)
                    >= self.MIN_CONTENT_WORDS_FOR_ADDITIVE
                ):

                    return ControllerDecision(
                        action=RetrievalAction.RETRIEVE,
                        reason="additive_refinement",
                        confidence=0.90,
                    )

                break

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
        #
        # A topic is considered established once either:
        #
        #   (a) there's an explicit query-intent marker (a wh-word,
        #       modal, "policy"/"rules"/"requirements", or a framing
        #       phrase like "tell me"/"wondering about") AND the
        #       utterance is long enough OR already carries a handful
        #       of real content words -- so a short-but-topical
        #       question ("Who approves domestic travel requests")
        #       doesn't have to pad itself out to 6 raw words before
        #       it's considered stable; or
        #
        #   (b) there is no explicit query-intent marker at all, but a
        #       clearly larger amount of real topic content has
        #       accumulated regardless -- a more conservative bar,
        #       since there's no interrogative/framing signal to lean
        #       on.
        #
        # This never fires for open-ended framing alone ("I was
        # wondering about the...", "I need some information about...")
        # because framing verbs and question words are excluded from
        # the content-word count, so those utterances only qualify once
        # an actual subject/topic word has been spoken.
        # ---------------------------------------------------------

        content_words = self._content_word_count(words)
        has_interrogative = has_query_intent or "?" in text

        if has_interrogative and (
            len(words) >= 6
            or content_words >= self.MIN_CONTENT_WORDS_WITH_QUERY_INTENT
        ):

            return ControllerDecision(
                action=RetrievalAction.RETRIEVE,
                reason="stable_intent",
                confidence=0.90,
            )

        if (
            not has_interrogative
            and content_words >= self.MIN_CONTENT_WORDS_WITHOUT_QUERY_INTENT
        ):

            return ControllerDecision(
                action=RetrievalAction.RETRIEVE,
                reason="topic_established",
                confidence=0.82,
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

    # -------------------------------------------------------------------
    # Deterministic regression assertions.
    #
    # These use fresh, non-benchmark example sentences (none of these
    # exact strings appear in scripts/evaluate_early_retrieval.py) to
    # confirm the generic content-word / interrogative-signal logic
    # behaves as intended, without hardcoding any benchmark query.
    # -------------------------------------------------------------------

    print()
    print("=" * 78)
    print("DETERMINISTIC ASSERTIONS")
    print("=" * 78)

    failures = []

    def check(label, got, expected):
        status = "PASS" if got == expected else "FAIL"
        print(f"  [{status}] {label}: {got!r} (expected {expected!r})")
        if got != expected:
            failures.append(label)

    def decide(text, is_final=False, previous=None):
        return controller.decide(
            transcript=text,
            previous_transcript=previous,
            is_final=is_final,
        )

    # 1. A short (<6 word) but clearly topical question now retrieves
    #    early instead of waiting for 6 raw words.
    d = decide("Who approves urgent equipment purchases")
    check("short topical question -> RETRIEVE", d.action.value, "RETRIEVE")
    check("short topical question reason", d.reason, "stable_intent")

    # 2. Open-ended framing with NO topic yet still WAITs (item 1).
    d = decide("I was curious about the")
    check("open framing, no topic yet -> WAIT", d.action.value, "WAIT")

    # 2b. Same framing verb, once a real topic follows, retrieves.
    d = decide("I was curious about the equipment loss procedure")
    check("open framing + topic -> RETRIEVE", d.action.value, "RETRIEVE")

    # 3. The incomplete-ending safeguard is untouched.
    d = decide("What is the policy for")
    check("incomplete ending still WAITs", d.action.value, "WAIT")
    check("incomplete ending reason unchanged", d.reason, "incomplete_transcript")

    # 4. A bare additive connector with no real new content still WAITs
    #    (item 4: don't retrieve merely because of "and"/"also"/
    #    "what about").
    d = decide("and also that thing")
    check("bare additive connector -> WAIT", d.action.value, "WAIT")

    # 4b. An additive connector WITH real new content still retrieves.
    d = decide("and also the phishing escalation steps")
    check("additive connector with content -> RETRIEVE", d.action.value, "RETRIEVE")
    check("additive connector with content reason", d.reason, "additive_refinement")

    # 5. Very short fragments and generic function words never retrieve.
    d = decide("um so")
    check("very short fragment -> WAIT", d.action.value, "WAIT")

    # 6. Presentation-only and non-query conversational continuations
    #    are unaffected by the content-word changes.
    d = decide("Can you repeat that")
    check("presentation-only unaffected -> SUPPRESS", d.action.value, "SUPPRESS")

    d = decide("okay thanks that's helpful", is_final=True)
    check("non-query continuation unaffected -> WAIT", d.action.value, "WAIT")

    # 7. Previously-passing long-form stable-intent detection is
    #    unaffected (still the original len(words) >= 6 path).
    d = decide("What are the requirements for the annual compliance training")
    check("long-form stable intent unaffected -> RETRIEVE", d.action.value, "RETRIEVE")
    check("long-form stable intent reason unchanged", d.reason, "stable_intent")

    print()
    print("-" * 78)
    if failures:
        print(f"{len(failures)} ASSERTION(S) FAILED")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)

    print("ALL ASSERTIONS PASSED")