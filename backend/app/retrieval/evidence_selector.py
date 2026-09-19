from typing import Any


class EvidenceSelector:
    """
    Selects high-precision supporting evidence chunks from reranked results.

    Prevents unrelated or loosely related chunks from contaminating the
    generation context.
    Pipeline:
        retrieval -> reranking -> evidence selection -> sufficiency check -> generator
    """

    # ------------------------------------------------------------------
    # Multi-intent-only retrieval-agreement fallback (see
    # _recover_by_retrieval_agreement). These two constants define
    # "meaningful agreement" from the earlier retrieval stages (BM25 +
    # Dense, fused by RRF) and are only ever consulted from
    # select_per_intent, never from select().
    # ------------------------------------------------------------------

    # A candidate must fuse to within this fraction of the strongest
    # RRF-fused candidate in its own intent's pool to count as
    # retrieval-agreed. RRF scores for a genuinely tied/near-tied match
    # sit within a percent or two of each other (see the real bug: the
    # correct chunk's RRF score was ~0.3% below the top chunk's), while an
    # unrelated candidate that only entered the pool through one weak
    # stage sits far below this band.
    RETRIEVAL_AGREEMENT_RATIO = 0.95

    # Absolute floor on top of the relative ratio above. RRF's own formula
    # (see fusion.reciprocal_rank_fusion, default k=60) means a candidate
    # found by only one of BM25/Dense near the top of that single list
    # scores at most ~1/(60+1) =~ 0.0164, while a candidate both stages
    # agree is near the top scores close to double that (~0.03+, matching
    # the real bug's on-topic cluster). Without this floor, a single
    # weakly-retrieved candidate would trivially satisfy the ratio above by
    # being compared only to itself. 0.02 sits between the two regimes, so
    # it requires genuine two-stage participation, not mere presence.
    RETRIEVAL_AGREEMENT_MIN_SIGNAL = 0.02

    # Even a retrieval-agreed candidate is only recovered if the
    # CrossEncoder did not single it out as catastrophically worse than
    # its own intent's best CrossEncoder score. This stops the fallback
    # from ever accepting a candidate the CrossEncoder considers
    # unrelated (large negative outlier), as opposed to one it merely
    # under-ranked.
    RETRIEVAL_AGREEMENT_CE_MARGIN = 6.0

    def __init__(
        self,
        min_score: float = 0.5,
        score_margin: float = 4.0,
        max_chunks: int = 3,
        intent_min_score: float = 0.0,
    ):
        self.min_score = min_score
        self.score_margin = score_margin
        self.max_chunks = max_chunks

        # Floor used when selecting for a single intent of a multi-intent
        # query. min_score=0.5 was tuned against whole queries, where the top
        # chunk is the best match in the corpus for the entire utterance. A
        # subquery's own best chunk can be its correct evidence and still
        # score modestly, so the per-intent floor uses the same boundary the
        # gate already treats as "topically unrelated" (below 0.0) instead.
        self.intent_min_score = intent_min_score

    def select(
        self,
        results: list[dict[str, Any]],
        query: str = "",
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Filter reranked results so only chunks with sufficient relevance and
        proximity to the top score are sent to the generator.

        min_score overrides the configured floor for this call only; when it
        is None (the default, and every existing caller) behaviour is
        unchanged.
        """
        effective_min_score = (
            self.min_score if min_score is None else min_score
        )
        if not results:
            return []

        valid_results = [
            r for r in results
            if r.get("chunk") and r.get("reranker_score") is not None
        ]

        if not valid_results:
            return []

        # Results should already be sorted descending by reranker_score
        sorted_results = sorted(
            valid_results,
            key=lambda x: x["reranker_score"],
            reverse=True,
        )

        best_score = sorted_results[0]["reranker_score"]

        if best_score < effective_min_score:
            return []

        selected = []
        score_cutoff = max(effective_min_score, best_score - self.score_margin)

        for result in sorted_results:
            score = result["reranker_score"]
            if score >= score_cutoff:
                # Ensure all required chunk metadata fields are present
                chunk = result["chunk"]
                if all(k in chunk for k in ("chunk_id", "doc_id", "section", "text", "source")):
                    selected.append(result)
            else:
                break

            if len(selected) >= self.max_chunks:
                break

        return selected

    @staticmethod
    def _retrieval_signal(result: dict[str, Any]) -> float | None:
        """
        Pull the pre-CrossEncoder retrieval-stage score off a reranked
        result. The reranker copies the fused-stage dict (see
        StreamingRetriever/CrossEncoderReranker) before adding
        reranker_score, so "rrf_score" (falling back to "score") is still
        the RRF-fused BM25+Dense agreement score, untouched by reranking.
        """
        value = result.get("rrf_score", result.get("score"))
        return float(value) if isinstance(value, (int, float)) else None

    def _recover_by_retrieval_agreement(
        self,
        reranked_pool: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Conservative multi-intent-only fallback.

        Triggered only when the normal per-intent select() found nothing
        (every candidate's CrossEncoder score fell below the per-intent
        floor). Recovers candidates the CrossEncoder under-ranked but that
        the earlier BM25/Dense/RRF retrieval stages agree on: a candidate
        is recovered only if
          (a) its RRF-fused score is within RETRIEVAL_AGREEMENT_RATIO of
              the strongest RRF-fused score in this intent's own
              candidate pool ("meaningful agreement", not just presence),
              and
          (b) its CrossEncoder score is not a catastrophic outlier versus
              this intent's own best CrossEncoder score.

        A candidate that satisfies neither (e.g. it only entered the pool
        through one weak stage, and the CrossEncoder considers it far
        worse than the rest) is never recovered - this is what keeps
        unrelated candidates out even though their CrossEncoder score is
        also negative.
        """
        candidates = [
            r
            for r in reranked_pool
            if r.get("chunk")
            and r.get("reranker_score") is not None
            and self._retrieval_signal(r) is not None
        ]

        # "Agreement" is a relative comparison between candidates. A single
        # candidate trivially "agrees" with itself, so with fewer than two
        # scored candidates there is nothing for the retrieval stages to
        # agree on - recovering it would accept a candidate merely because
        # it exists, which is exactly what this fallback must not do.
        if len(candidates) < 2:
            return []

        top_retrieval_signal = max(self._retrieval_signal(r) for r in candidates)

        # RRF scores are always positive (sum of 1/(k+rank) terms); a
        # non-positive top score means there is no real retrieval-stage
        # agreement to recover from. The absolute floor additionally
        # requires that top score to reflect genuine multi-stage
        # participation rather than a single weak stage (see
        # RETRIEVAL_AGREEMENT_MIN_SIGNAL above).
        if top_retrieval_signal < self.RETRIEVAL_AGREEMENT_MIN_SIGNAL:
            return []

        top_ce_score = max(r["reranker_score"] for r in candidates)
        retrieval_cutoff = self.RETRIEVAL_AGREEMENT_RATIO * top_retrieval_signal
        ce_cutoff = top_ce_score - self.RETRIEVAL_AGREEMENT_CE_MARGIN

        agreed = [
            r
            for r in candidates
            if self._retrieval_signal(r) >= retrieval_cutoff
            and r["reranker_score"] >= ce_cutoff
        ]

        # Strongest retrieval-stage agreement first, so a max_chunks cap
        # keeps the candidates the earlier stages agree on most, not an
        # arbitrary slice.
        agreed.sort(key=self._retrieval_signal, reverse=True)

        recovered = []

        for result in agreed:
            chunk = result["chunk"]
            if not all(
                k in chunk for k in ("chunk_id", "doc_id", "section", "text", "source")
            ):
                continue

            enriched = dict(result)
            enriched["recovered_via_retrieval_agreement"] = True
            recovered.append(enriched)

            if len(recovered) >= self.max_chunks:
                break

        return recovered

    @staticmethod
    def intent_reranked_results(intent_result: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Pull the reranked results out of one MultiIntentRetriever subquery entry.

        Accepts either the retriever's nested shape
            {"results": {"reranked_results": [...]}}
        or an already-flattened
            {"results": [...]}
        """
        retrieval = intent_result.get("results")

        if isinstance(retrieval, dict):
            return retrieval.get("reranked_results", []) or []

        if isinstance(retrieval, list):
            return retrieval

        return []

    def select_per_intent(
        self,
        intent_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Run the existing single-query selection independently for every intent.

        Each intent is scored against its own subquery, so cross-encoder scores
        from different intents are not comparable. Selecting globally lets the
        highest-scoring intent consume the whole evidence budget. Giving every
        intent its own call to select() keeps the existing thresholds and
        max_chunks logic per intent instead of per query.

        Returns one entry per intent:
            {"intent_id": int, "query": str, "selected": list[dict]}

        Evidence appearing under more than one intent is kept only for the
        first intent that selected it, so the merged prompt never repeats a
        chunk. select() itself is untouched, so the single-intent path is
        unchanged.
        """
        per_intent: list[dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()

        for position, intent_result in enumerate(intent_results, start=1):
            intent_id = intent_result.get("intent_id", position)
            subquery = intent_result.get("query", "")

            reranked = self.intent_reranked_results(intent_result)

            # Existing behaviour, applied to this intent's own results only,
            # with the per-intent relevance floor.
            selected = self.select(
                results=reranked,
                query=subquery,
                min_score=self.intent_min_score,
            )

            # CrossEncoder gave every candidate a score below the
            # per-intent floor. Before concluding the intent is
            # unsupported, check whether the earlier retrieval stages
            # (BM25/Dense/RRF) agree strongly on a candidate the
            # CrossEncoder merely under-ranked. Multi-intent-only: the
            # single-intent select() path above is never affected.
            if not selected and reranked:
                selected = self._recover_by_retrieval_agreement(reranked)

            deduped: list[dict[str, Any]] = []

            for result in selected:
                chunk_id = result["chunk"].get("chunk_id")

                if not chunk_id or chunk_id in seen_chunk_ids:
                    continue

                seen_chunk_ids.add(chunk_id)

                # Preserve (or restore) intent provenance on the copy.
                enriched = dict(result)
                enriched["intent_id"] = intent_id
                enriched["subquery"] = subquery

                deduped.append(enriched)

            per_intent.append(
                {
                    "intent_id": intent_id,
                    "query": subquery,
                    "selected": deduped,
                }
            )

        return per_intent

    @staticmethod
    def flatten_per_intent(
        per_intent: list[dict[str, Any]],
        intent_ids: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Flatten per-intent selections into a single evidence list.

        Used for citation validation, which needs every chunk that was shown
        to the generator. When intent_ids is given, only those intents are
        included.
        """
        flattened: list[dict[str, Any]] = []

        for entry in per_intent:
            if intent_ids is not None and entry["intent_id"] not in intent_ids:
                continue

            flattened.extend(entry["selected"])

        return flattened