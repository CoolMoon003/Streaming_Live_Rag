import re


class CitationValidator:

    CITATION_PATTERN = re.compile(
        r"\[([A-Za-z0-9_]+)\s+§([^\]]+)\]"
    )

    def validate(
        self,
        answer: str,
        evidence: list[dict],
    ) -> dict:

        valid_citations = set()

        for result in evidence:

            chunk = result["chunk"]

            citation = (
                f"[{chunk['doc_id']} §{chunk['section']}]"
            )

            valid_citations.add(citation)

        found_citations = set(
            self.CITATION_PATTERN.findall(answer)
        )

        normalized_found = {
            f"[{doc_id} §{section}]"
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
    # MULTI-INTENT PATH
    # ---------------------------------------------------------------
    #
    # Small local models reliably follow the single-intent citation
    # instruction but frequently drop inline [DOC_ID §Section] markers
    # once the prompt is reshaped into multiple "INTENT n:" blocks. When
    # that happens, `validate()` alone would report a technically-valid
    # answer (no *invalid* citation was found) with an empty
    # `valid_citations` list, because it can only ever return citations
    # that literally appear in the generated text.
    #
    # `validate_multi_intent` keeps that same hallucination check (any
    # citation string in the answer that isn't backed by evidence still
    # fails validation, exactly as before) but additionally derives the
    # citation list directly from the evidence that was actually
    # selected/gated for each supported intent whenever the model failed
    # to cite that intent's evidence itself. This is deterministic and
    # grounded: it can only ever emit citations that came from evidence
    # already vetted by EvidenceSelector/EvidenceGate for that specific
    # intent, so unsupported intents and unrelated documents can never
    # receive a citation. No additional LLM call is made.

    def _citations_for_evidence(self, evidence: list[dict]) -> list[str]:
        seen = []
        for result in evidence:
            chunk = result["chunk"]
            citation = f"[{chunk['doc_id']} §{chunk['section']}]"
            if citation not in seen:
                seen.append(citation)
        return seen

    def validate_multi_intent(
        self,
        answer: str,
        intents: list[dict],
    ) -> dict:
        # Only supported intents may contribute allowed citations; an
        # unsupported intent's "evidence" list is already empty by the
        # time it reaches this validator (see orchestrator), but the
        # supported flag is checked again here defensively.
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

        found_citations = set(self.CITATION_PATTERN.findall(answer))
        normalized_found = {
            f"[{doc_id} §{section}]" for doc_id, section in found_citations
        }

        invalid = normalized_found - valid_citations_all

        if invalid:
            return {
                "valid": False,
                "invalid_citations": sorted(invalid),
                "citations_found": sorted(normalized_found),
                "valid_citations": [],
            }

        final_citations: list[str] = []

        for intent_id, allowed in per_intent_allowed.items():
            if not allowed:
                continue

            cited_by_model = [c for c in allowed if c in normalized_found]

            # Deterministic recovery: the model was given exactly this
            # evidence for this intent and produced no verifiable
            # hallucination (checked above), but also cited none of it
            # directly. Fall back to the evidence itself rather than
            # leaving a supported, evidenced intent with no citation.
            recovered = cited_by_model if cited_by_model else allowed

            for citation in recovered:
                if citation not in final_citations:
                    final_citations.append(citation)

        return {
            "valid": True,
            "invalid_citations": [],
            "citations_found": sorted(normalized_found),
            "valid_citations": sorted(final_citations),
        }