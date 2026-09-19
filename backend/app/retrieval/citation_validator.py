import re


class CitationValidator:

    # Matches a complete citation block:
    #
    # [DOC_TRAVEL_POLICY Â§2. International Travel]
    #
    # It intentionally captures the whole content inside [ ... ].
    CITATION_BLOCK_PATTERN = re.compile(
        r"\[([A-Za-z0-9_]+)\s+Â§([^\]]+)\]"
    )

    # ---------------------------------------------------------------
    # SINGLE-INTENT PATH
    # ---------------------------------------------------------------

    def validate(
        self,
        answer: str,
        evidence: list[dict],
    ) -> dict:

        valid_citations = set()

        for result in evidence:
            chunk = result["chunk"]

            citation = (
                f"[{chunk['doc_id']} Â§{chunk['section']}]"
            )

            valid_citations.add(citation)

        found_citations = set(
            self.CITATION_BLOCK_PATTERN.findall(answer)
        )

        normalized_found = {
            f"[{doc_id} Â§{section}]"
            for doc_id, section in found_citations
        }

        invalid = normalized_found - valid_citations

        return {
            "valid": len(invalid) == 0,
            "invalid_citations": sorted(invalid),
            "citations_found": sorted(normalized_found),
            "valid_citations": sorted(
                normalized_found & valid_citations
            ),
        }

    # ---------------------------------------------------------------
    # CITATION NORMALIZATION
    # ---------------------------------------------------------------

    def _expand_citation_block(
        self,
        doc_id: str,
        section_text: str,
    ) -> list[str]:
        """
        Convert both single-section and combined-section citations
        into canonical individual citations.

        Example:

        [DOC_TRAVEL_POLICY Â§2. International Travel]

        becomes:

        [DOC_TRAVEL_POLICY Â§2. International Travel]

        And:

        [DOC_REIMBURSEMENT_POLICY Â§1. Eligible Expenses,
         Â§3. Submission Deadline,
         Â§2. International Expenses]

        becomes:

        [DOC_REIMBURSEMENT_POLICY Â§1. Eligible Expenses]
        [DOC_REIMBURSEMENT_POLICY Â§3. Submission Deadline]
        [DOC_REIMBURSEMENT_POLICY Â§2. International Expenses]
        """

        # The first section starts immediately after the first Â§.
        # Additional sections in a combined citation start with
        # ", Â§".
        parts = re.split(r"\s*,\s*Â§", section_text)

        citations = []

        for part in parts:
            section = part.strip()

            if not section:
                continue

            citations.append(
                f"[{doc_id} Â§{section}]"
            )

        return citations

    def _parse_citations(self, answer: str) -> set[str]:
        """
        Parse citation blocks from model output and normalize
        combined citations into individual canonical citations.
        """

        normalized = set()

        for doc_id, section_text in self.CITATION_BLOCK_PATTERN.findall(
            answer
        ):
            expanded = self._expand_citation_block(
                doc_id,
                section_text,
            )

            normalized.update(expanded)

        return normalized

    # ---------------------------------------------------------------
    # MULTI-INTENT PATH
    # ---------------------------------------------------------------

    def _citations_for_evidence(
        self,
        evidence: list[dict],
    ) -> list[str]:
        """
        Build canonical citations directly from vetted evidence.
        """

        seen = []

        for result in evidence:
            chunk = result["chunk"]

            citation = (
                f"[{chunk['doc_id']} Â§{chunk['section']}]"
            )

            if citation not in seen:
                seen.append(citation)

        return seen

    def validate_multi_intent(
        self,
        answer: str,
        intents: list[dict],
    ) -> dict:
        """
        Validate citations for a multi-intent answer.

        Each intent owns its own evidence.

        Rules:
        1. Only supported intents contribute allowed citations.
        2. A citation from another document/intent is invalid.
        3. Combined citations such as:
             [DOC_X Â§1. A, Â§2. B, Â§3. C]
           are expanded into individual citations.
        4. If the model cites valid evidence, those citations are used.
        5. If a supported intent has evidence but the model omitted
           citations, citations are deterministically recovered from
           that intent's vetted evidence.
        6. Unsupported intents never receive recovered citations.
        """

        # -----------------------------------------------------------
        # Build allowed citations separately for every supported
        # intent.
        # -----------------------------------------------------------

        per_intent_allowed = {
            intent.get("intent_id"): self._citations_for_evidence(
                intent.get("evidence", []) or []
            )
            for intent in intents
            if intent.get("supported")
        }

        valid_citations_all = set()

        for citations in per_intent_allowed.values():
            valid_citations_all.update(citations)

        # -----------------------------------------------------------
        # Parse the model's citations.
        #
        # IMPORTANT:
        # _parse_citations() expands combined citation blocks.
        # -----------------------------------------------------------

        normalized_found = self._parse_citations(answer)

        # -----------------------------------------------------------
        # Hallucination check.
        #
        # Any citation not backed by evidence selected/gated for
        # one of the supported intents makes validation fail.
        # -----------------------------------------------------------

        invalid = normalized_found - valid_citations_all

        if invalid:
            return {
                "valid": False,
                "invalid_citations": sorted(invalid),
                "citations_found": sorted(normalized_found),
                "valid_citations": [],
            }

        # -----------------------------------------------------------
        # Build final citations intent-by-intent.
        #
        # If the model cited an intent's evidence, keep those.
        #
        # If it cited none, recover citations from the vetted evidence
        # for that intent.
        # -----------------------------------------------------------

        final_citations: list[str] = []

        for intent_id, allowed in per_intent_allowed.items():

            if not allowed:
                continue

            cited_by_model = [
                citation
                for citation in allowed
                if citation in normalized_found
            ]

            if cited_by_model:
                recovered = cited_by_model
            else:
                recovered = allowed

            for citation in recovered:
                if citation not in final_citations:
                    final_citations.append(citation)

        return {
            "valid": True,
            "invalid_citations": [],
            "citations_found": sorted(normalized_found),
            "valid_citations": sorted(final_citations),
        }