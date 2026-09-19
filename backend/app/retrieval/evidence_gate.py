from dataclasses import dataclass, field
import re


@dataclass
class EvidenceDecision:
    sufficient: bool
    confidence: float
    reason: str
    supported_chunk_ids: list[str]
    missing_constraints: list[str] = field(default_factory=list)


class EvidenceGate:
    """
    Determines whether retrieved evidence is strong and specific
    enough to support an answer for the given query.

    Distinguishes topical relevance from claim-level sufficiency:
    - Verifies reranker score margin and topical confidence
    - Detects whether the query requests specific constraints (e.g. numeric amounts,
      limits, deadlines) that are missing from the retrieved evidence.
    """

    NUMERIC_QUERY_PATTERNS = [
        re.compile(r"\b(maximum|max|minimum|min|amount|how much|how many|cost|price|rate|fee|budget|allowance)\b", re.IGNORECASE),
        re.compile(r"\b(deadline|time limit|how long|how many days|duration)\b", re.IGNORECASE),
        re.compile(r"(\$|usd|dollars?|cents?|percent|%)", re.IGNORECASE),
    ]

    NUMERIC_VALUE_PATTERNS = [
        re.compile(r"\b\d+(\.\d+)?\b"),
        re.compile(r"\$[\d,]+"),
        re.compile(r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|fifty|hundred|thousand)\b", re.IGNORECASE),
        re.compile(r"\b(per diem|flat rate|capped at)\b", re.IGNORECASE),
    ]

    def __init__(
        self,
        min_relevance_score: float = 0.5,
        minimum_results: int = 1,
        intent_min_relevance_score: float = 0.0,
    ):
        self.min_relevance_score = min_relevance_score
        self.minimum_results = minimum_results

        # Floor used when gating one intent of a multi-intent query. See the
        # note on EvidenceSelector.intent_min_score: the 0.5 floor is a
        # whole-query threshold, and applying it per subquery rejects evidence
        # that is genuinely the best the corpus has for that intent. The
        # "completely unrelated" rule (best_score < 0.0) still applies.
        self.intent_min_relevance_score = intent_min_relevance_score

    def check(
        self,
        results: list[dict],
        query: str = "",
        min_relevance_score: float | None = None,
    ) -> EvidenceDecision:
        """
        min_relevance_score overrides the configured floor for this call only;
        when it is None (the default, and every existing caller) behaviour is
        unchanged.
        """
        effective_min_relevance = (
            self.min_relevance_score
            if min_relevance_score is None
            else min_relevance_score
        )

        if not results:
            return EvidenceDecision(
                sufficient=False,
                confidence=0.0,
                reason="no_evidence",
                supported_chunk_ids=[],
                missing_constraints=[],
            )

        # -------------------------------------------------------------
        # 1. Filter results with reranker scores
        # -------------------------------------------------------------
        valid_results = [
            result
            for result in results
            if result.get("chunk") and result.get("reranker_score") is not None
        ]

        if not valid_results:
            return EvidenceDecision(
                sufficient=False,
                confidence=0.0,
                reason="no_evidence",
                supported_chunk_ids=[],
                missing_constraints=[],
            )

        best_score = max(r["reranker_score"] for r in valid_results)

        # Evidence the multi-intent evidence selector already vetted
        # through its retrieval-stage-agreement fallback (see
        # EvidenceSelector._recover_by_retrieval_agreement) is exempt from
        # the CrossEncoder-only checks below: it already required
        # meaningful BM25/Dense/RRF agreement plus a CrossEncoder score
        # that was not a catastrophic outlier. That flag is only ever set
        # by select_per_intent, so single-intent callers of check() (and
        # any call not going through the multi-intent path) are completely
        # unaffected.
        has_recovered_evidence = any(
            r.get("recovered_via_retrieval_agreement") for r in valid_results
        )

        # Completely unrelated queries produce deeply negative scores (e.g. < 0.0)
        if best_score < 0.0 and not has_recovered_evidence:
            return EvidenceDecision(
                sufficient=False,
                confidence=0.1,
                reason="insufficient_relevance",
                supported_chunk_ids=[],
                missing_constraints=[],
            )

        # -------------------------------------------------------------
        # 2. Check topical relevance threshold
        # -------------------------------------------------------------
        if has_recovered_evidence:
            # Already vetted by the selector's retrieval-agreement
            # fallback; the whole-query/per-intent CrossEncoder floor does
            # not apply to these candidates a second time here.
            top_results = valid_results
        else:
            top_results = [
                r for r in valid_results
                if r["reranker_score"] >= effective_min_relevance
            ]

        if len(top_results) < self.minimum_results:
            return EvidenceDecision(
                sufficient=False,
                confidence=0.25,
                reason="insufficient_relevance",
                supported_chunk_ids=[],
                missing_constraints=[],
            )

        # -------------------------------------------------------------
        # 3. Query constraint sufficiency check (e.g., numeric amounts)
        # -------------------------------------------------------------
        missing_constraints = []
        if query:
            normalized_query = query.strip().lower()

            asks_for_numeric = any(
                pattern.search(normalized_query)
                for pattern in self.NUMERIC_QUERY_PATTERNS
            )

            if asks_for_numeric:
                # Check whether candidate texts contain numeric values
                candidate_texts = " ".join(r["chunk"]["text"] for r in top_results)
                has_numeric_evidence = any(
                    val_pattern.search(candidate_texts)
                    for val_pattern in self.NUMERIC_VALUE_PATTERNS
                )

                if not has_numeric_evidence:
                    missing_constraints.append("numeric_amount")
                    return EvidenceDecision(
                        sufficient=False,
                        confidence=0.35,
                        reason="missing_numeric_fact",
                        supported_chunk_ids=[r["chunk"]["chunk_id"] for r in top_results],
                        missing_constraints=missing_constraints,
                    )

        # -------------------------------------------------------------
        # 4. Sufficient evidence
        # -------------------------------------------------------------
        supported_ids = [
            r["chunk"]["chunk_id"]
            for r in top_results
        ]

        confidence = min(
            0.99,
            max(0.5, (best_score + 2.0) / 10.0),
        )

        return EvidenceDecision(
            sufficient=True,
            confidence=confidence,
            reason=(
                "sufficient_evidence_retrieval_agreement"
                if has_recovered_evidence
                else "sufficient_evidence"
            ),
            supported_chunk_ids=supported_ids,
            missing_constraints=[],
        )

    def check_per_intent(
        self,
        per_intent: list[dict],
    ) -> list[dict]:
        """
        Run the existing gate once per intent.

        A single global check over merged multi-intent evidence cannot express
        "intent 1 supported, intent 2 unsupported" - a strong first intent
        makes the whole query look sufficient. Checking each intent against
        its own subquery and its own selected evidence produces a per-intent
        support map instead.

        check() is called unchanged, so every existing rule (relevance floor,
        minimum results, missing numeric fact) still applies, just scoped to
        one intent.

        Input:  [{"intent_id", "query", "selected"}]  (EvidenceSelector.select_per_intent)
        Output: [{"intent_id", "query", "evidence", "decision", "supported"}]
        """
        support_map: list[dict] = []

        for entry in per_intent:
            evidence = entry.get("selected", [])

            decision = self.check(
                results=evidence,
                query=entry.get("query", ""),
                min_relevance_score=self.intent_min_relevance_score,
            )

            support_map.append(
                {
                    "intent_id": entry.get("intent_id"),
                    "query": entry.get("query", ""),
                    "evidence": evidence,
                    "decision": decision,
                    "supported": decision.sufficient,
                }
            )

        return support_map