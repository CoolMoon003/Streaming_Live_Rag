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

    # -------------------------------------------------------------
    # Deterministic shared-context propagation tables.
    #
    # A qualifier is only propagated from intent 1 to intent 2 when
    # ALL of the following hold:
    #
    #   * intent 1 contains exactly one qualifier from one group
    #   * intent 1 also contains an anchor noun for that group
    #     (proving the qualifier really scopes that domain)
    #   * intent 2 contains a compatible subject from the SAME group
    #   * intent 2 contains no blocker term for that group
    #   * intent 2 does not already carry a scope qualifier
    #
    # Anything else is treated as ambiguous and left untouched.
    # -------------------------------------------------------------

    QUALIFIER_GROUPS = (
        {
            "name": "travel_scope",
            "qualifiers": (
                "international",
                "domestic",
                "overseas",
                "foreign",
            ),
            "anchors": (
                "travel",
                "travels",
                "traveling",
                "travelling",
                "trip",
                "trips",
                "flight",
                "flights",
                "assignment",
                "assignments",
            ),
            "subjects": (
                "reimbursement",
                "reimbursements",
                "reimbursed",
                "expense",
                "expenses",
                "per diem",
                "receipt",
                "receipts",
                "claim",
                "claims",
                "allowance",
                "allowances",
                "mileage",
            ),
            "blockers": (
                "remote",
                "workshop",
                "phishing",
                "training",
                "certification",
                "equipment",
                "onboarding",
                "payroll",
                "parental",
                "sick",
            ),
        },
        {
            "name": "asset_scope",
            "qualifiers": (
                "company equipment",
                "company property",
                "company device",
                "company laptop",
                "company asset",
                "corporate equipment",
            ),
            "anchors": (),
            "subjects": (
                "incident",
                "incidents",
                "loss",
                "damage",
                "theft",
            ),
            "blockers": (
                "phishing",
                "password",
                "email",
                "security",
                "breach",
                "cyber",
                "personal",
                "travel",
                "reimbursement",
                "expense",
                "workshop",
                "training",
            ),
        },
    )

    # Any scope qualifier already present in intent 2 means intent 2
    # defines its own scope and must be left alone.
    EXISTING_SCOPE_TERMS = (
        "international",
        "domestic",
        "overseas",
        "foreign",
        "personal",
        "company",
        "corporate",
    )

    @staticmethod
    def _contains_term(text: str, term: str) -> bool:
        """Whole-word / whole-phrase containment check."""

        return re.search(
            rf"(?<!\w){re.escape(term)}(?!\w)",
            text,
            flags=re.IGNORECASE,
        ) is not None

    def _propagate_shared_qualifier(
        self,
        first_query: str,
        second_query: str,
    ) -> str:
        """
        Deterministically re-attach a clear contextual qualifier from
        the first intent onto the second intent.

        Returns the second query unchanged whenever the relationship
        is anything short of unambiguous.
        """

        if not first_query or not second_query:
            return second_query

        # Intent 2 already declares its own scope -> never touch it.
        for term in self.EXISTING_SCOPE_TERMS:

            if self._contains_term(second_query, term):
                return second_query

        matched_qualifier = None
        matched_group = None

        for group in self.QUALIFIER_GROUPS:

            present = [
                qualifier
                for qualifier in group["qualifiers"]
                if self._contains_term(first_query, qualifier)
            ]

            if not present:
                continue

            # Two competing qualifiers -> ambiguous.
            if len(present) > 1:
                return second_query

            anchors = group["anchors"]

            if anchors and not any(
                self._contains_term(first_query, anchor)
                for anchor in anchors
            ):
                continue

            # A qualifier matched in more than one group -> ambiguous.
            if matched_qualifier is not None:
                return second_query

            matched_qualifier = present[0]
            matched_group = group

        if matched_qualifier is None:
            return second_query

        # Blockers signal an unrelated domain in intent 2.
        for blocker in matched_group["blockers"]:

            if self._contains_term(second_query, blocker):
                return second_query

        # Intent 2 must name a compatible subject.
        subjects = [
            subject
            for subject in matched_group["subjects"]
            if self._contains_term(second_query, subject)
        ]

        if not subjects:
            return second_query

        # Longest subject wins, so "per diem" beats a bare token.
        target = max(subjects, key=len)

        pattern = re.compile(
            rf"(?<!\w){re.escape(target)}(?!\w)",
            flags=re.IGNORECASE,
        )

        rewritten, count = pattern.subn(
            f"{matched_qualifier} {target}",
            second_query,
            count=1,
        )

        if count != 1:
            return second_query

        # Intent 2 is now a standalone sentence, so restore sentence
        # casing the way the original query opened.
        if first_query[:1].isupper():
            rewritten = rewritten[:1].upper() + rewritten[1:]

        return rewritten

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

        # ---------------------------------------------------------
        # Context-aware repair.
        #
        # Splitting can strip a qualifier that scoped the whole
        # original question ("international travel ... and
        # reimbursement"). Re-attach it when, and only when, the
        # relationship is unambiguous.
        # ---------------------------------------------------------

        first_part = cleaned_parts[0]

        cleaned_parts = [first_part] + [
            self._propagate_shared_qualifier(first_part, part)
            for part in cleaned_parts[1:]
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

    # -----------------------------------------------------------------
    # DETERMINISTIC ASSERTIONS
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print("CONTEXT PROPAGATION ASSERTIONS")
    print("=" * 80)

    failures = []

    def check(label, condition):

        status = "PASS" if condition else "FAIL"

        if not condition:
            failures.append(label)

        print(f"[{status}] {label}")

    # --- Q016 -------------------------------------------------------

    q016 = (
        "What approval is needed for international travel "
        "and what documentation is needed for reimbursement?"
    )

    q016_subqueries = decomposer.decompose(q016)

    check(
        "Q016 produces two intents",
        len(q016_subqueries) == 2,
    )

    q016_second = q016_subqueries[1].query.lower()

    check(
        "Q016 second query contains 'international'",
        "international" in q016_second,
    )

    check(
        "Q016 second query contains 'reimbursement'",
        "reimbursement" in q016_second,
    )

    check(
        "Q016 first query unchanged",
        q016_subqueries[0].query
        == "What approval is needed for international travel?",
    )

    # --- Domestic ---------------------------------------------------

    domestic = decomposer.decompose(
        "What rules apply to domestic travel "
        "and what documents are needed for reimbursement?"
    )

    domestic_second = domestic[1].query.lower()

    check(
        "Domestic case propagates 'domestic'",
        "domestic" in domestic_second,
    )

    check(
        "Domestic case retains 'reimbursement'",
        "reimbursement" in domestic_second,
    )

    # --- Company equipment ------------------------------------------

    equipment = decomposer.decompose(
        "What happens if company equipment is lost "
        "and how should the incident be reported?"
    )

    equipment_second = equipment[1].query.lower()

    check(
        "Equipment case retains company/equipment context",
        "company equipment" in equipment_second,
    )

    check(
        "Equipment case retains 'incident'",
        "incident" in equipment_second,
    )

    # --- Negative case 1 --------------------------------------------

    remote = decomposer.decompose(
        "What approval is needed for international travel "
        "and who approves regular remote work?"
    )

    check(
        "Remote-work intent does NOT receive 'international'",
        "international" not in remote[1].query.lower(),
    )

    check(
        "Remote-work intent preserved",
        remote[1].query == "who approves regular remote work?",
    )

    # --- Negative case 2 --------------------------------------------

    phishing = decomposer.decompose(
        "What are the workshop rules "
        "and what is the phishing reporting process?"
    )

    check(
        "Phishing intent does NOT receive 'workshop'",
        "workshop" not in phishing[1].query.lower(),
    )

    check(
        "Phishing intent preserved",
        phishing[1].query
        == "what is the phishing reporting process?",
    )

    # --- Single intent regression -----------------------------------

    single_intent_queries = [
        "What are the international travel rules?",
        "Tell me about international travel.",
        "What are the rules and requirements for international travel?",
        "What are the rules and booking requirements for international travel?",
    ]

    for single in single_intent_queries:

        check(
            f"Single intent unchanged: {single}",
            [s.query for s in decomposer.decompose(single)] == [single],
        )

    # --- Existing multi-intent regression ---------------------------

    existing_multi = {
        "What is required for international travel and what happens with late bookings?": [
            "What is required for international travel?",
            "what happens with late bookings?",
        ],
        "What are the travel rules and also what are the reimbursement rules?": [
            "What are the travel rules?",
            "what are the reimbursement rules?",
        ],
        "What approval is needed for travel and does late booking require approval?": [
            "What approval is needed for travel?",
            "does late booking require approval?",
        ],
        "What are the travel rules? What are the reimbursement rules?": [
            "What are the travel rules?",
            "What are the reimbursement rules?",
        ],
    }

    for source, expected in existing_multi.items():

        check(
            f"Existing multi-intent unchanged: {source}",
            [s.query for s in decomposer.decompose(source)] == expected,
        )

    print()
    print("-" * 80)

    if failures:

        print(f"{len(failures)} ASSERTION(S) FAILED")

        for failure in failures:
            print(f"  - {failure}")

        raise SystemExit(1)

    print("ALL ASSERTIONS PASSED")