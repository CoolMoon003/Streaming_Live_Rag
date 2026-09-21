import re
import unicodedata


class CitationValidator:

    # Matches a complete citation block:
    #
    # [DOC_TRAVEL_POLICY §2. International Travel]
    #
    # It intentionally captures the whole content inside [ ... ].
    CITATION_BLOCK_PATTERN = re.compile(
        r"\[([A-Za-z0-9_]+)\s+§([^\]]+)\]"
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
                f"[{chunk['doc_id']} §{chunk['section']}]"
            )

            valid_citations.add(citation)

        found_citations = set(
            self.CITATION_BLOCK_PATTERN.findall(answer)
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

        [DOC_TRAVEL_POLICY §2. International Travel]

        becomes:

        [DOC_TRAVEL_POLICY §2. International Travel]

        And:

        [DOC_REIMBURSEMENT_POLICY §1. Eligible Expenses,
         §3. Submission Deadline,
         §2. International Expenses]

        becomes:

        [DOC_REIMBURSEMENT_POLICY §1. Eligible Expenses]
        [DOC_REIMBURSEMENT_POLICY §3. Submission Deadline]
        [DOC_REIMBURSEMENT_POLICY §2. International Expenses]
        """

        # The first section starts immediately after the first §.
        # Additional sections in a combined citation start with
        # ", §".
        parts = re.split(r"\s*,\s*§", section_text)

        citations = []

        for part in parts:
            section = part.strip()

            if not section:
                continue

            citations.append(
                f"[{doc_id} §{section}]"
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
                f"[{chunk['doc_id']} §{chunk['section']}]"
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
             [DOC_X §1. A, §2. B, §3. C]
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

    # ---------------------------------------------------------------
    # ATTRIBUTION HELPER — repair uncited, evidence-backed sentences
    # ---------------------------------------------------------------

    # "Intent 2:", "**Intent 2:**", "- Intent 2." ... (same shapes the
    # groundedness metric recognises). Everything after the label on the
    # same line is answer text and IS processed.
    _ATTR_INTENT_LABEL = re.compile(
        r"^\s*(?:[-*\u2022]\s*)?\**\s*intent\s+(\d+)\s*\**\s*[:.)-]\s*\**\s*",
        re.IGNORECASE,
    )
    _ATTR_BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+")
    # Sentence boundary = . ! ? + whitespace, never inside a [...] block
    # (citation sections look like "2. International Travel").
    # The capture group keeps the separators so the line can be rebuilt
    # byte-for-byte.
    _ATTR_SENTENCE_SPLIT = re.compile(r"((?<=[.!?])\s+(?![^\[\]]*\]))")
    _ATTR_LEADING_CITATIONS = re.compile(r"^\s*((?:\[[^\]]*\]\s*)+)")
    # body | terminal punctuation | closing markdown/quotes
    _ATTR_TRAILING = re.compile(
        r"^(.*?)([.!?]*)([*_`\"'\u2019\u201d)]*)$", re.DOTALL
    )
    # A sentence shorter than this is too generic to be attributed safely.
    _ATTR_MIN_TOKENS = 3

    @staticmethod
    def _attr_tokens(text: str) -> list[str]:
        """Normalise text to comparable word tokens.

        Case, whitespace, punctuation, markdown emphasis, apostrophes and
        curly quotes are ignored; the words themselves (and numbers) are
        not, so a paraphrase or a changed figure never matches.
        """
        text = unicodedata.normalize("NFKC", text or "").lower()
        for ch in ("'", "\u2019", "\u2018", "`"):
            text = text.replace(ch, "")
        return re.findall(r"[^\W_]+", text)

    def _attr_pairs(self, entry: dict) -> list[tuple[str, str]]:
        """(canonical citation, padded token string) for one intent's own
        vetted evidence. Unsupported intents yield nothing."""
        pairs: list[tuple[str, str]] = []

        if not entry.get("supported", True):
            return pairs

        for result in entry.get("evidence") or []:
            chunk = result.get("chunk", {}) or {}
            doc_id = chunk.get("doc_id", "")
            section = chunk.get("section", "")
            text = chunk.get("text", "")

            if not (doc_id and section and text):
                continue

            tokens = self._attr_tokens(text)

            if not tokens:
                continue

            pairs.append(
                (
                    f"[{doc_id} §{section}]",
                    " " + " ".join(tokens) + " ",
                )
            )

        return pairs

    def _attr_find_citation(
        self,
        sentence: str,
        pairs: list[tuple[str, str]],
    ) -> str | None:
        """Citation of the first evidence chunk that contains this sentence
        (after normalisation), else None. Never guesses."""
        tokens = self._attr_tokens(sentence)

        if len(tokens) < self._ATTR_MIN_TOKENS:
            return None

        needle = " " + " ".join(tokens) + " "

        for citation, haystack in pairs:
            if needle in haystack:
                return citation

        return None

    def _attribute_body(
        self,
        body: str,
        pairs: list[tuple[str, str]],
    ) -> str:
        """Sentence-level attribution for the text of ONE line.

        A citation that follows a sentence's full stop ("... made. [X]")
        belongs to that sentence, exactly as the groundedness metric reads
        it, so a trailing citation group only covers the LAST sentence
        before it.
        """
        parts = self._ATTR_SENTENCE_SPLIT.split(body)
        fragments = parts[0::2]
        separators = parts[1::2]

        units: list[dict] = []

        for fragment in fragments:
            lead_text = ""
            text = fragment
            lead = self._ATTR_LEADING_CITATIONS.match(fragment)

            if (
                lead
                and units
                and self.CITATION_BLOCK_PATTERN.search(lead.group(1))
            ):
                units[-1]["cited"] = True
                lead_text = fragment[: lead.end()]
                text = fragment[lead.end():]

            units.append(
                {
                    "lead": lead_text,
                    "text": text,
                    "cited": bool(
                        self.CITATION_BLOCK_PATTERN.search(text)
                    ),
                }
            )

        rebuilt: list[str] = []

        for index, unit in enumerate(units):
            text = unit["text"]

            if not unit["cited"] and text.strip():
                core = text.rstrip()
                tail = text[len(core):]
                citation = self._attr_find_citation(core, pairs)

                if citation:
                    body_part, punct, closers = self._ATTR_TRAILING.match(
                        core
                    ).groups()
                    text = (
                        f"{body_part.rstrip()} {citation}"
                        f"{punct}{closers}{tail}"
                    )

            rebuilt.append(unit["lead"] + text)

            if index < len(separators):
                rebuilt.append(separators[index])

        return "".join(rebuilt)

    def attribute_sentences(
        self,
        answer: str,
        intent_results: list[dict],
    ) -> str:
        """Attach a missing citation to each factual sentence that is
        verbatim-supported (modulo whitespace / case / punctuation /
        markdown) by evidence belonging to the SAME intent.

        Guarantees
        ----------
        * Sentence level: every sentence of every "Intent N:" section is
          considered on its own, including the text that follows the label
          on the header line and sentences that share a line with other
          cited sentences.
        * Only that intent's own vetted evidence is consulted; there is no
          cross-intent attribution. Unsupported intents never receive a
          citation.
        * Citation IDs are copied from evidence chunks, never invented;
          existing citations are preserved untouched.
        * A sentence is attributed only if its normalised word sequence is
          contained in a single evidence chunk of its intent (and has at
          least ``_ATTR_MIN_TOKENS`` words). Paraphrases, extra facts and
          changed numbers stay uncited.
        * Deterministic (first matching chunk in evidence order) and
          idempotent.
        * Text before the first "Intent N:" label is left untouched.

        The citation is inserted before the sentence's terminal
        punctuation: ``sentence [DOC §Section].``
        """
        intent_pairs: dict[str, list[tuple[str, str]]] = {}

        for position, entry in enumerate(intent_results or [], start=1):
            key = entry.get("intent_id")
            key = str(position if key is None else key)
            intent_pairs.setdefault(key, self._attr_pairs(entry))

        out_lines: list[str] = []
        pairs: list[tuple[str, str]] = []
        in_intent = False

        for line in (answer or "").splitlines(keepends=True):
            stripped = line.rstrip("\r\n")
            eol = line[len(stripped):]

            label = self._ATTR_INTENT_LABEL.match(stripped)

            if label:
                in_intent = True
                pairs = intent_pairs.get(label.group(1), [])
                prefix = label.group(0)
                body = stripped[label.end():]
            elif in_intent:
                bullet = self._ATTR_BULLET.match(stripped)
                prefix = bullet.group(0) if bullet else ""
                body = stripped[len(prefix):]
            else:
                out_lines.append(line)
                continue

            if not pairs or not body.strip():
                out_lines.append(line)
                continue

            out_lines.append(
                prefix + self._attribute_body(body, pairs) + eol
            )

        return "".join(out_lines)

    def enforce_unsupported_intents(
        self,
        answer: str,
        intent_results: list[dict],
        insufficiency_msg: str,
    ) -> str:
        """Force every unsupported intent's section to the deterministic
        abstention sentence.

        An intent is unsupported when the evidence gate rejected it or it
        has no evidence. Whatever the model wrote under such an intent's
        "Intent N:" label (an uncited fabricated claim, a citation borrowed
        from another intent, extra prose) is replaced by
        ``insufficiency_msg``; the label itself and blank lines are kept.
        Supported intents, unlabelled text and sections the model did not
        write are left untouched. Deterministic and idempotent.
        """
        unsupported: set[str] = set()

        for position, entry in enumerate(intent_results or [], start=1):
            key = entry.get("intent_id")
            key = str(position if key is None else key)

            if not (entry.get("supported") and entry.get("evidence")):
                unsupported.add(key)

        if not unsupported:
            return answer

        out_lines: list[str] = []
        skipping = False

        for line in (answer or "").splitlines(keepends=True):
            stripped = line.rstrip("\r\n")
            eol = line[len(stripped):]
            label = self._ATTR_INTENT_LABEL.match(stripped)

            if label:
                skipping = label.group(1) in unsupported

                if skipping:
                    out_lines.append(
                        f"{label.group(0)}{insufficiency_msg}{eol}"
                    )
                else:
                    out_lines.append(line)

                continue

            if skipping and stripped.strip():
                continue

            out_lines.append(line)

        return "".join(out_lines)