from typing import Generator
from backend.app.llm.ollama_client import OllamaClient


class GroundedAnswerGenerator:
    """
    Corpus-grounded answer generator with strict citation enforcement.

    Supports:
    - Synchronous and streaming generation
    - Exact citation format [DOC_ID §Section]
    - Strict adherence to supplied evidence (no hallucinations)
    - Additive, replacement, and presentation modes
    """

    INSUFFICIENT_EVIDENCE_MSG = (
        "The provided corpus does not contain enough evidence to answer this."
    )

    def __init__(self, llm: OllamaClient):
        self.llm = llm

    def _build_prompt(
        self,
        query: str,
        evidence: list[dict],
        previous_answer: str = "",
        refinement_type: str = "NEW",
    ) -> tuple[str, list[str]]:
        evidence_text = []
        allowed_citations = []

        for index, result in enumerate(evidence, start=1):
            chunk = result["chunk"]
            citation = f"[{chunk['doc_id']} §{chunk['section']}]"
            allowed_citations.append(citation)

            evidence_text.append(
                f"Evidence {index} {citation}:\n{chunk['text']}"
            )

        context = "\n\n".join(evidence_text)
        citations_list_str = ", ".join(allowed_citations)

        previous_context = ""
        if previous_answer:
            previous_context = f"\nPREVIOUS ANSWER:\n{previous_answer}\n"

        refinement_instruction = ""
        if refinement_type == "ADDITIVE":
            refinement_instruction = """
This is an ADDITIVE refinement.
Preserve the factual claims from the PREVIOUS ANSWER that remain supported.
Incorporate the new facts from the provided EVIDENCE.
Produce a single COMPLETE UPDATED ANSWER (do not answer only the new refinement).
"""
        elif refinement_type == "REPLACEMENT":
            refinement_instruction = """
This is a REPLACEMENT / CORRECTION.
Answer the corrected question using the provided EVIDENCE.
Do NOT preserve claims from the PREVIOUS ANSWER that relate to the replaced query.
"""

        prompt = f"""You are a strict corpus-grounded assistant.

Answer the user's question using ONLY the factual evidence provided below.

Strict Grounding Rules:
1. Do not use outside knowledge or extrapolate beyond the provided text.
2. Every factual claim MUST end with an exact citation from the ALLOWED CITATIONS list.
3. Citation format: [DOC_ID §Section]
4. Allowed Citations: {citations_list_str}
5. NEVER invent document IDs, sections, or numbers not explicitly in the evidence.
6. If the evidence is insufficient to answer the question, say exactly:
   "{self.INSUFFICIENT_EVIDENCE_MSG}"
7. Keep the answer concise, accurate, and direct.
{refinement_instruction}{previous_context}
USER QUESTION:
{query}

EVIDENCE:
{context}

ANSWER:
"""
        return prompt, allowed_citations

    def generate(
        self,
        query: str,
        evidence: list[dict],
        previous_answer: str = "",
        refinement_type: str = "NEW",
    ) -> str:
        if not evidence:
            return self.INSUFFICIENT_EVIDENCE_MSG

        prompt, _ = self._build_prompt(
            query=query,
            evidence=evidence,
            previous_answer=previous_answer,
            refinement_type=refinement_type,
        )

        raw_answer = self.llm.generate(prompt).strip()
        return raw_answer

    def generate_stream(
        self,
        query: str,
        evidence: list[dict],
        previous_answer: str = "",
        refinement_type: str = "NEW",
    ) -> Generator[str, None, None]:
        if not evidence:
            yield self.INSUFFICIENT_EVIDENCE_MSG
            return

        prompt, _ = self._build_prompt(
            query=query,
            evidence=evidence,
            previous_answer=previous_answer,
            refinement_type=refinement_type,
        )

        for token in self.llm.generate_stream(prompt):
            yield token

    # =====================================================================
    # MULTI-INTENT PATH
    # =====================================================================
    #
    # Used only when the query decomposed into more than one intent. The
    # single-intent prompt above is left untouched.

    INTENT_INSUFFICIENT_MSG = (
        "The provided policy corpus does not contain enough evidence "
        "to answer this part."
    )

    def _build_multi_intent_prompt(
        self,
        query: str,
        intents: list[dict],
    ) -> tuple[str, list[str]]:
        """
        Build a prompt that shows each intent with its own evidence block.

        intents: [{"intent_id", "query", "evidence", "supported"}]

        Unsupported intents carry no evidence and are given the exact
        insufficiency sentence to copy, so the wording is deterministic and
        the small local model is never asked to invent a refusal.
        """
        intent_blocks = []
        allowed_citations: list[str] = []
        skeleton_lines = []

        for intent in intents:
            intent_id = intent.get("intent_id")
            subquery = intent.get("query", "")
            evidence = intent.get("evidence", []) or []
            supported = bool(intent.get("supported")) and bool(evidence)

            header = f"INTENT {intent_id}: {subquery}"

            if not supported:
                skeleton_lines.append(
                    f"Intent {intent_id}: {self.INTENT_INSUFFICIENT_MSG}"
                )
                intent_blocks.append(
                    f"{header}\n"
                    f"EVIDENCE FOR INTENT {intent_id}: NONE\n"
                    f"Required text for this intent (write it exactly, "
                    f"add no facts and no citation):\n"
                    f"{self.INTENT_INSUFFICIENT_MSG}"
                )
                continue

            evidence_lines = []
            intent_citations: list[str] = []

            for index, result in enumerate(evidence, start=1):
                chunk = result["chunk"]
                citation = f"[{chunk['doc_id']} §{chunk['section']}]"

                if citation not in allowed_citations:
                    allowed_citations.append(citation)

                if citation not in intent_citations:
                    intent_citations.append(citation)

                evidence_lines.append(
                    f"Evidence {intent_id}.{index} {citation}:\n{chunk['text']}"
                )

            skeleton_lines.append(
                f"Intent {intent_id}: <1-3 short sentences answering this "
                f"intent only, each ending with {intent_citations[0]}>"
            )

            intent_blocks.append(
                f"{header}\n"
                f"EVIDENCE FOR INTENT {intent_id}:\n"
                + "\n\n".join(evidence_lines)
                + f"\nCITATIONS FOR INTENT {intent_id} "
                f"(copy exactly, use only these): "
                + " ".join(intent_citations)
            )

        context = "\n\n".join(intent_blocks)

        intent_labels = "\n".join(
            f"Intent {intent.get('intent_id')}: {intent.get('query', '')}"
            for intent in intents
        )
        output_format = "\n".join(skeleton_lines)
        intent_count = len(intents)

        prompt = f"""You are a strict corpus-grounded assistant.

The user asked a question containing MULTIPLE separate intents.
You must answer EVERY intent listed below. Answering only one is a failure.

DETECTED INTENTS ({intent_count}):
{intent_labels}

OUTPUT FORMAT - exactly {intent_count} sections, in this order, each on its own line(s):
{output_format}

RULES:
1. Write one section per intent, in the order above, each starting with "Intent <N>:". Never skip an intent and never merge two intents into one section.
2. Answer each intent using ONLY the evidence under that same intent. Never use another intent's evidence.
3. Keep each section to 1-3 short sentences or bullets. State only the facts needed; do not copy the evidence text.
4. Every factual sentence must end with a citation in the format [DOC_ID §Section], copied exactly from that intent's CITATIONS line. Every intent that has evidence must contain at least one citation.
5. For an intent marked NO evidence, write exactly its required text, with no citation and nothing else.
6. Never use outside knowledge. Never invent document IDs, sections, numbers or citations.
7. Write nothing before the first section or after the last section: no introduction, conclusion, summary, or comment about other intents.
Before answering, silently check that you wrote {intent_count} sections, that each supported intent has a citation, and that every citation belongs to that intent's own evidence. Never write this check in your answer.

USER QUESTION:
{query}

EVIDENCE BY INTENT:
{context}

ANSWER:
"""
        return prompt, allowed_citations

    def generate_multi_intent(
        self,
        query: str,
        intents: list[dict],
    ) -> str:
        if not intents:
            return self.INSUFFICIENT_EVIDENCE_MSG

        prompt, _ = self._build_multi_intent_prompt(
            query=query,
            intents=intents,
        )

        return self.llm.generate(prompt).strip()

    def generate_multi_intent_stream(
        self,
        query: str,
        intents: list[dict],
    ) -> Generator[str, None, None]:
        if not intents:
            yield self.INSUFFICIENT_EVIDENCE_MSG
            return

        prompt, _ = self._build_multi_intent_prompt(
            query=query,
            intents=intents,
        )

        for token in self.llm.generate_stream(prompt):
            yield token