import asyncio
from typing import Any, AsyncGenerator

from backend.app.models.session import (
    SessionState,
    AnswerState,
    build_delta_query,
    delta_needs_retrieval,
    merge_evidence,
)
from backend.app.controller.retrieval_controller import (
    RetrievalController,
    RetrievalAction,
)
from backend.app.query.multi_intent import (
    MultiIntentDetector,
    MultiIntentDecomposer,
)
from backend.app.query.refinement import (
    QueryRefinementAnalyzer,
    RefinementType,
)
from backend.app.query.refinement_retriever import RefinementRetriever
from backend.app.retrieval.async_streaming_retriever import AsyncStreamingRetriever
from backend.app.retrieval.multi_intent_retriever import MultiIntentRetriever
from backend.app.retrieval.evidence_selector import EvidenceSelector
from backend.app.retrieval.evidence_gate import EvidenceDecision, EvidenceGate
from backend.app.retrieval.citation_validator import CitationValidator
from backend.app.llm.ollama_client import OllamaClient
from backend.app.llm.grounded_generator import GroundedAnswerGenerator


class StreamingRagOrchestrator:
    """
    Event-driven Streaming Live RAG Orchestrator for Samsung Theme 04.

    Coordinates:
    - Incremental partial transcript processing (controller -> early retrieval)
    - Multi-intent query detection & parallel subquery retrieval
    - Utterance commit boundary -> final evidence gating & streaming grounded answer
    - State-preserving refinement (NEW, ADDITIVE, REPLACEMENT, PRESENTATION)
    - Strict corpus grounding & citation validation
    - Stale generation race protection
    """

    def __init__(
        self,
        chunks_path: str = "data/processed/chunks.jsonl",
        async_retriever: AsyncStreamingRetriever | None = None,
        multi_intent_retriever: MultiIntentRetriever | None = None,
        llm_client: OllamaClient | None = None,
        model: str = "llama3.2:3b",
    ):
        self.chunks_path = chunks_path

        # Initialize or reuse shared retrieval components
        if async_retriever is not None:
            self.async_retriever = async_retriever
        else:
            self.async_retriever = AsyncStreamingRetriever(chunks_path)

        if multi_intent_retriever is not None:
            self.multi_intent_retriever = multi_intent_retriever
        else:
            self.multi_intent_retriever = MultiIntentRetriever(
                retriever=self.async_retriever
            )

        # LLM client & generator (defaulting to llama3.2:3b, low latency, think=False)
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            self.llm_client = OllamaClient(
                model=model,
                think=False,
            )

        self.generator = GroundedAnswerGenerator(self.llm_client)

        # Routing, decomposition, and refinement
        self.controller = RetrievalController()
        self.multi_intent_detector = MultiIntentDetector()
        self.multi_intent_decomposer = MultiIntentDecomposer()
        self.refinement_analyzer = QueryRefinementAnalyzer()
        self.refinement_retriever = RefinementRetriever(
            retriever=self.async_retriever.retriever
        )

        # Gating and citation verification
        self.evidence_selector = EvidenceSelector()
        self.evidence_gate = EvidenceGate()
        self.citation_validator = CitationValidator()

    # =========================================================================
    # PARTIAL TRANSCRIPT FLOW
    # =========================================================================

    async def process_partial(
        self,
        session: SessionState,
        transcript_text: str,
    ) -> dict[str, Any]:
        """
        Process a partial transcript chunk arriving before utterance completion.

        Decides whether to WAIT, SUPPRESS, or initiate EARLY RETRIEVAL.
        Does NOT invoke the LLM generator.
        """
        text = transcript_text.strip()

        decision = self.controller.decide(
            transcript=text,
            previous_transcript=session.current_query or session.previous_query,
            is_final=False,
        )

        # -------------------------------------------------------------
        # 1. WAIT: Transcript is incomplete or unstable
        # -------------------------------------------------------------
        if decision.action == RetrievalAction.WAIT:
            return {
                "event": "controller_decision",
                "action": "WAIT",
                "reason": decision.reason,
                "confidence": decision.confidence,
                "query_version": session.query_version,
                "generation_id": session.active_generation_id,
            }

        # -------------------------------------------------------------
        # 2. SUPPRESS: Presentation-only follow-up
        # -------------------------------------------------------------
        if decision.action == RetrievalAction.SUPPRESS:
            return {
                "event": "controller_decision",
                "action": "SUPPRESS",
                "reason": decision.reason,
                "confidence": decision.confidence,
                "query_version": session.query_version,
                "generation_id": session.active_generation_id,
                "latest_answer": session.latest_answer,
                "latest_citations": session.latest_citations,
                "answer_version": session.answer_version,
            }

        # -------------------------------------------------------------
        # 3. RETRIEVE: Early retrieval while user is still speaking
        # -------------------------------------------------------------
        generation_id = session.start_new_query(text)

        # Detect multi-intent vs single-intent
        multi_intent_dec = self.multi_intent_detector.detect(text)

        if multi_intent_dec.is_multi_intent:
            subqueries = self.multi_intent_decomposer.decompose(text)
            subquery_strings = [sq.query for sq in subqueries]

            retrieval_res = await self.multi_intent_retriever.retrieve(
                subqueries=subquery_strings,
                generation_id=generation_id,
            )
            raw_results = retrieval_res["results"]
            is_multi = True
        else:
            retrieval_res = await self.async_retriever.retrieve(
                query=text,
                generation_id=generation_id,
            )
            raw_results = retrieval_res["results"]["reranked_results"]
            is_multi = False

        # Apply stale generation guard: only store if generation_id is still active
        accepted = session.accept_results(
            generation_id=generation_id,
            query=text,
            results=raw_results,
        )

        if not accepted:
            return {
                "event": "retrieval_stale",
                "action": "RETRIEVE",
                "generation_id": generation_id,
                "active_generation_id": session.active_generation_id,
                "reason": "discarded_as_stale",
            }

        return {
            "event": "retrieval_update",
            "action": "RETRIEVE",
            "query_version": session.query_version,
            "generation_id": generation_id,
            "is_multi_intent": is_multi,
            "chunks_retrieved": len(raw_results),
            "confidence": decision.confidence,
            "reason": decision.reason,
        }

    # =========================================================================
    # COMMIT FLOW (UTTERANCE BOUNDARY)
    # =========================================================================

    async def process_commit(
        self,
        session: SessionState,
        transcript_text: str,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Process the committed, stable utterance transcript.

        Reuses early retrieval when valid, selects evidence, gates sufficiency,
        and streams the grounded answer token-by-token.
        """
        text = transcript_text.strip()

        # -------------------------------------------------------------
        # 1. Analyze Refinement Intent
        # -------------------------------------------------------------
        previous = session.previous_query if session.previous_query else session.current_query
        refinement_dec = self.refinement_analyzer.analyze(
            previous_query=previous,
            new_query=text,
        )

        refinement_type = refinement_dec.refinement_type

        # -------------------------------------------------------------
        # 2. Handle PRESENTATION Refinement (No retrieval needed)
        # -------------------------------------------------------------
        if refinement_type == RefinementType.PRESENTATION:
            yield {
                "event": "answer_started",
                "action": "SUPPRESS",
                "refinement_type": "PRESENTATION",
                "answer_version": session.answer_version,
            }

            yield {
                "event": "answer_token",
                "token": session.latest_answer,
            }

            yield {
                "event": "answer_completed",
                "action": "SUPPRESS",
                "refinement_type": "PRESENTATION",
                "answer": session.latest_answer,
                "citations": session.latest_citations,
                "answer_version": session.answer_version,
                "evidence_sufficient": True,
            }
            return

        # -------------------------------------------------------------
        # 3. Retrieve or Reuse Evidence
        # -------------------------------------------------------------
        active_query = text
        generation_id = session.active_generation_id
        evidence = []

        # Per-intent retrieval structure from MultiIntentRetriever, kept
        # alongside the merged evidence so evidence selection and gating can
        # work per intent. Stays None for every single-intent path.
        intent_results: list[dict[str, Any]] | None = None

        if refinement_type == RefinementType.ADDITIVE:
            # start_new_query() overwrites session.previous_query with the
            # refinement text itself, so capture the real previous query first.
            additive_previous = previous
            generation_id = session.start_new_query(text)
            refinement = self.refinement_retriever.retrieve_refinement(
                previous_query=additive_previous,
                refinement_query=text,
                existing_results=session.latest_results,
            )
            active_query = refinement["contextual_query"]
            evidence = refinement["merged_results"]
            session.accept_results(
                generation_id=generation_id,
                query=active_query,
                results=evidence,
            )

        elif refinement_type == RefinementType.REPLACEMENT:
            query_info = self.refinement_retriever.query_builder.build(
                previous_query=session.previous_query or session.current_query,
                refinement_query=text,
                refinement_type="REPLACEMENT",
            )
            clean_query = query_info["query"]
            active_query = clean_query
            generation_id = session.start_new_query(clean_query)

            # Check multi-intent on replacement
            multi_dec = self.multi_intent_detector.detect(clean_query)
            if multi_dec.is_multi_intent:
                subqueries = self.multi_intent_decomposer.decompose(clean_query)
                ret_res = await self.multi_intent_retriever.retrieve(
                    subqueries=[sq.query for sq in subqueries],
                    generation_id=generation_id,
                )
                evidence = ret_res["results"]
                intent_results = ret_res.get("subqueries") or None
            else:
                ret_res = await self.async_retriever.retrieve(
                    query=clean_query,
                    generation_id=generation_id,
                )
                evidence = ret_res["results"]["reranked_results"]

            session.accept_results(
                generation_id=generation_id,
                query=clean_query,
                results=evidence,
            )

        else:
            # NEW Query: reuse the accepted early retrieval when the commit
            # repeats it exactly, or extends it (transcript kept growing).
            reuse = session.find_reusable_retrieval(text)

            if reuse.mode == "exact":
                generation_id = reuse.retrieval.generation_id
                evidence = reuse.retrieval.results

            elif reuse.mode == "extension":
                early_evidence = reuse.retrieval.results
                generation_id = session.start_new_query(text)

                delta_query = build_delta_query(reuse.added_tokens)

                if delta_query and delta_needs_retrieval(
                    delta_query, early_evidence
                ):
                    delta_res = await self.async_retriever.retrieve(
                        query=delta_query,
                        generation_id=generation_id,
                    )
                    evidence = merge_evidence(
                        early_evidence,
                        delta_res["results"]["reranked_results"],
                    )
                else:
                    evidence = early_evidence

                session.accept_results(
                    generation_id=generation_id,
                    query=text,
                    results=evidence,
                )

            else:
                generation_id = session.start_new_query(text)
                multi_dec = self.multi_intent_detector.detect(text)

                if multi_dec.is_multi_intent:
                    subqueries = self.multi_intent_decomposer.decompose(text)
                    ret_res = await self.multi_intent_retriever.retrieve(
                        subqueries=[sq.query for sq in subqueries],
                        generation_id=generation_id,
                    )
                    evidence = ret_res["results"]
                    intent_results = ret_res.get("subqueries") or None
                else:
                    ret_res = await self.async_retriever.retrieve(
                        query=text,
                        generation_id=generation_id,
                    )
                    evidence = ret_res["results"]["reranked_results"]

                session.accept_results(
                    generation_id=generation_id,
                    query=text,
                    results=evidence,
                )

        # -------------------------------------------------------------
        # 4. Evidence Selection Stage
        # 5. Evidence Sufficiency Gate
        # -------------------------------------------------------------
        is_multi_intent = bool(intent_results) and len(intent_results) > 1
        multi_intent_payload: list[dict[str, Any]] = []

        if is_multi_intent:
            # Select and gate per intent. Scores from different subqueries are
            # not comparable, so a single global selection lets one intent take
            # the whole evidence budget and a single global gate cannot say
            # "intent 1 supported, intent 2 unsupported".
            per_intent = self.evidence_selector.select_per_intent(intent_results)
            support_map = self.evidence_gate.check_per_intent(per_intent)

            supported_entries = [
                entry for entry in support_map if entry["supported"]
            ]
            supported_intent_ids = {
                entry["intent_id"] for entry in supported_entries
            }

            candidate_evidence = self.evidence_selector.flatten_per_intent(
                per_intent,
                intent_ids=supported_intent_ids,
            )

            # Pre-selection best score per intent, kept for diagnostics: it is
            # the only way to tell "retrieval found nothing" apart from
            # "evidence was filtered out" when an intent comes back
            # unsupported.
            top_scores = {}
            for intent_result in intent_results:
                reranked = self.evidence_selector.intent_reranked_results(
                    intent_result
                )
                scores = [
                    item["reranker_score"]
                    for item in reranked
                    if item.get("reranker_score") is not None
                ]
                top_scores[intent_result.get("intent_id")] = (
                    max(scores) if scores else None
                )

            multi_intent_payload = [
                {
                    "intent_id": entry["intent_id"],
                    "query": entry["query"],
                    "evidence": entry["evidence"] if entry["supported"] else [],
                    "supported": entry["supported"],
                    "reason": entry["decision"].reason,
                    "selected_chunk_ids": [
                        result["chunk"]["chunk_id"] for result in entry["evidence"]
                    ],
                    "retrieved": len(
                        self.evidence_selector.intent_reranked_results(
                            intent_results[index]
                        )
                    ),
                    "top_score": top_scores.get(entry["intent_id"]),
                }
                for index, entry in enumerate(support_map)
            ]

            if supported_entries:
                gate_decision = EvidenceDecision(
                    sufficient=True,
                    confidence=max(
                        entry["decision"].confidence for entry in supported_entries
                    ),
                    reason="sufficient_evidence",
                    supported_chunk_ids=[
                        chunk_id
                        for entry in supported_entries
                        for chunk_id in entry["decision"].supported_chunk_ids
                    ],
                    missing_constraints=[],
                )
            else:
                # No intent has usable evidence: fall through to the existing
                # refusal path with the first intent's reason.
                gate_decision = support_map[0]["decision"]
        else:
            selected_evidence = self.evidence_selector.select(
                results=evidence,
                query=active_query,
            )
            candidate_evidence = selected_evidence if selected_evidence else evidence

            gate_decision = self.evidence_gate.check(
                results=candidate_evidence,
                query=active_query,
            )

        if not gate_decision.sufficient:
            if gate_decision.reason == "missing_numeric_fact":
                refusal_msg = (
                    "The provided corpus does not contain enough evidence to determine "
                    "the requested amount or numeric figure."
                )
            else:
                refusal_msg = (
                    "The provided corpus does not contain enough evidence to answer this."
                )

            answer_state = session.save_answer(
                query=active_query,
                answer=refusal_msg,
                citations=[],
            )

            yield {
                "event": "uncertainty_emitted",
                "action": "ANSWER",
                "refinement_type": refinement_type.value,
                "reason": gate_decision.reason,
                "answer": refusal_msg,
                "citations": [],
                "answer_version": answer_state.answer_version,
                "query_version": session.query_version,
                "generation_id": generation_id,
                "evidence_sufficient": False,
                "is_multi_intent": is_multi_intent,
                "intents": [
                    {
                        "intent_id": intent["intent_id"],
                        "query": intent["query"],
                        "supported": intent["supported"],
                        "reason": intent["reason"],
                        "retrieved": intent["retrieved"],
                        "top_score": intent["top_score"],
                    }
                    for intent in multi_intent_payload
                ],
            }
            return

        # -------------------------------------------------------------
        # 6. Grounded Answer Streaming
        # -------------------------------------------------------------
        yield {
            "event": "answer_started",
            "action": "ANSWER",
            "refinement_type": refinement_type.value,
            "query": active_query,
            "generation_id": generation_id,
            "query_version": session.query_version,
            "supported_chunks": gate_decision.supported_chunk_ids,
            "is_multi_intent": is_multi_intent,
            "intents": [
                {
                    "intent_id": intent["intent_id"],
                    "query": intent["query"],
                    "supported": intent["supported"],
                    "reason": intent["reason"],
                    "retrieved": intent["retrieved"],
                    "top_score": intent["top_score"],
                    "chunk_ids": intent["selected_chunk_ids"],
                }
                for intent in multi_intent_payload
            ],
        }

        full_answer_parts = []

        if is_multi_intent:
            token_stream = self.generator.generate_multi_intent_stream(
                query=active_query,
                intents=multi_intent_payload,
            )
        else:
            token_stream = self.generator.generate_stream(
                query=active_query,
                evidence=candidate_evidence,
                previous_answer=session.latest_answer,
                refinement_type=refinement_type.value,
            )

        for token in token_stream:
            full_answer_parts.append(token)
            yield {
                "event": "answer_token",
                "token": token,
            }

        full_answer = "".join(full_answer_parts).strip()

        # -------------------------------------------------------------
        # 7. Citation Validation & State Preservation
        # -------------------------------------------------------------
        if is_multi_intent:
            # The per-intent generator prompt is far harder for a small
            # local model to follow verbatim than the single-intent one,
            # so citation recovery is grounded in each intent's own
            # selected/gated evidence rather than requiring the model to
            # echo the exact bracket marker back in its prose.
            citation_check = self.citation_validator.validate_multi_intent(
                answer=full_answer,
                intents=multi_intent_payload,
            )
        else:
            citation_check = self.citation_validator.validate(
                answer=full_answer,
                evidence=candidate_evidence,
            )

        if not citation_check["valid"]:
            final_answer = (
                "The generated answer could not be verified against the provided corpus."
            )
            valid_citations = []
        else:
            final_answer = full_answer
            valid_citations = citation_check["valid_citations"]

        answer_state = session.save_answer(
            query=active_query,
            answer=final_answer,
            citations=valid_citations,
        )

        metrics = self.llm_client.get_last_metrics()

        yield {
            "event": "answer_completed",
            "action": "ANSWER",
            "refinement_type": refinement_type.value,
            "answer": final_answer,
            "citations": valid_citations,
            "answer_version": answer_state.answer_version,
            "query_version": session.query_version,
            "generation_id": generation_id,
            "evidence_sufficient": True,
            "metrics": metrics,
            "citation_valid": citation_check["valid"],
            "is_multi_intent": is_multi_intent,
            "intents": [
                {
                    "intent_id": intent["intent_id"],
                    "query": intent["query"],
                    "supported": intent["supported"],
                    "reason": intent["reason"],
                    "retrieved": intent["retrieved"],
                    "top_score": intent["top_score"],
                    "chunk_ids": intent["selected_chunk_ids"],
                }
                for intent in multi_intent_payload
            ],
        }