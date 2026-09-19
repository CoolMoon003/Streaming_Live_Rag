from backend.app.models.session import SessionState
from backend.app.query.refinement import (
    QueryRefinementAnalyzer,
    RefinementType,
)
from backend.app.query.refinement_retriever import RefinementRetriever
from backend.app.retrieval.evidence_gate import EvidenceGate
from backend.app.retrieval.evidence_selector import EvidenceSelector
from backend.app.retrieval.citation_validator import CitationValidator
from backend.app.llm.grounded_generator import GroundedAnswerGenerator


class SessionAnswerManager:

    def __init__(
        self,
        retriever,
        generator: GroundedAnswerGenerator,
    ):
        self.retriever = retriever
        self.generator = generator

        self.analyzer = QueryRefinementAnalyzer()

        self.refinement_retriever = RefinementRetriever(
            retriever=retriever
        )

        self.evidence_selector = EvidenceSelector()
        self.evidence_gate = EvidenceGate()
        self.citation_validator = CitationValidator()

    def process(
        self,
        session: SessionState,
        query: str,
    ) -> dict:

        decision = self.analyzer.analyze(
            previous_query=session.current_query,
            new_query=query,
        )

        # ==============================================================
        # FIRST QUERY
        # ==============================================================

        if not session.current_query:

            generation_id = session.start_new_query(query)

            retrieval = self.retriever.retrieve(query)

            evidence = retrieval["reranked_results"]

            return self._generate_answer(
                session=session,
                query=query,
                evidence=evidence,
                generation_id=generation_id,
                refinement_type=RefinementType.NEW,
            )

        # ==============================================================
        # PRESENTATION-ONLY REQUEST
        # ==============================================================

        if decision.refinement_type == RefinementType.PRESENTATION:

            return {
                "action": "SUPPRESS",
                "refinement_type": "PRESENTATION",
                "answer": session.latest_answer,
                "citations": session.latest_citations,
                "answer_version": session.answer_version,
            }

        # ==============================================================
        # ADDITIVE REFINEMENT
        # ==============================================================

        if decision.refinement_type == RefinementType.ADDITIVE:

            generation_id = session.start_new_query(query)

            refinement = (
                self.refinement_retriever.retrieve_refinement(
                    previous_query=session.previous_query,
                    refinement_query=query,
                    existing_results=session.latest_results,
                )
            )

            evidence = refinement["merged_results"]

            return self._generate_answer(
                session=session,
                query=refinement["contextual_query"],
                evidence=evidence,
                generation_id=generation_id,
                refinement_type=RefinementType.ADDITIVE,
            )

        # ==============================================================
        # REPLACEMENT / CORRECTION
        # ==============================================================

        if decision.refinement_type == RefinementType.REPLACEMENT:

            # First clean the correction.
            query_info = (
                self.refinement_retriever.query_builder.build(
                    previous_query=session.current_query,
                    refinement_query=query,
                    refinement_type="REPLACEMENT",
                )
            )

            clean_query = query_info["query"]

            # Store the cleaned query as the new active query.
            generation_id = session.start_new_query(
                clean_query
            )

            retrieval = self.retriever.retrieve(
                clean_query
            )

            evidence = retrieval["reranked_results"]

            return self._generate_answer(
                session=session,
                query=clean_query,
                evidence=evidence,
                generation_id=generation_id,
                refinement_type=RefinementType.REPLACEMENT,
            )

        # ==============================================================
        # NEW INDEPENDENT QUERY
        # ==============================================================

        generation_id = session.start_new_query(query)

        retrieval = self.retriever.retrieve(query)

        evidence = retrieval["reranked_results"]

        return self._generate_answer(
            session=session,
            query=query,
            evidence=evidence,
            generation_id=generation_id,
            refinement_type=RefinementType.NEW,
        )

    # ==================================================================
    # ANSWER GENERATION
    # ==================================================================

    def _generate_answer(
        self,
        session: SessionState,
        query: str,
        evidence: list[dict],
        generation_id: int,
        refinement_type: RefinementType,
    ) -> dict:

        # --------------------------------------------------------------
        # Make sure retrieval belongs to the active generation.
        # --------------------------------------------------------------

        accepted = session.accept_results(
            generation_id=generation_id,
            query=query,
            results=evidence,
        )

        if not accepted:

            return {
                "action": "STALE",
                "refinement_type": refinement_type.value,
                "answer": session.latest_answer,
                "citations": session.latest_citations,
                "answer_version": session.answer_version,
            }

        # --------------------------------------------------------------
        # Evidence selection stage:
        # Filter out loosely related/irrelevant chunks before LLM
        # --------------------------------------------------------------
        selected_evidence = self.evidence_selector.select(
            results=evidence,
            query=query,
        )

        # --------------------------------------------------------------
        # Evidence sufficiency check
        # --------------------------------------------------------------
        gate_decision = self.evidence_gate.check(
            results=selected_evidence if selected_evidence else evidence,
            query=query,
        )

        if not gate_decision.sufficient:

            if gate_decision.reason == "missing_numeric_fact":
                answer = (
                    "The provided corpus does not contain enough "
                    "evidence to determine the requested amount or numeric figure."
                )
            else:
                answer = (
                    "The provided corpus does not contain enough "
                    "evidence to answer this."
                )

            answer_state = session.save_answer(
                query=query,
                answer=answer,
                citations=[],
            )

            return {
                "action": "ANSWER",
                "refinement_type": refinement_type.value,
                "answer": answer,
                "citations": [],
                "answer_version": answer_state.answer_version,
                "evidence_sufficient": False,
                "evidence_reason": gate_decision.reason,
            }

        # --------------------------------------------------------------
        # Pass only the selected supporting evidence to the generator
        # --------------------------------------------------------------
        supported_evidence = selected_evidence if selected_evidence else evidence

        answer = self.generator.generate(
            query=query,
            evidence=supported_evidence,
            previous_answer=session.latest_answer,
            refinement_type=refinement_type.value,
        )

        # --------------------------------------------------------------
        # Validate citations generated by the LLM.
        # --------------------------------------------------------------

        citation_check = self.citation_validator.validate(
            answer=answer,
            evidence=supported_evidence,
        )

        if not citation_check["valid"]:

            answer = (
                "The generated answer could not be verified "
                "against the provided corpus."
            )

            citations = []

        else:

            citations = citation_check["valid_citations"]

        # --------------------------------------------------------------
        # Save answer version
        # --------------------------------------------------------------

        answer_state = session.save_answer(
            query=query,
            answer=answer,
            citations=citations,
        )

        return {
            "action": "ANSWER",
            "refinement_type": refinement_type.value,
            "answer": answer,
            "citations": citations,
            "answer_version": answer_state.answer_version,
            "evidence_sufficient": True,
            "evidence_reason": gate_decision.reason,
        }