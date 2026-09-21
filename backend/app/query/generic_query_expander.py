"""Deterministic query expansion for GENERIC single-topic questions.

Used only as a one-shot fallback AFTER the normal retrieval + evidence gate
found nothing usable (see StreamingRagOrchestrator). Queries such as
"What is the travel policy?" name a whole document instead of a fact, so the
cross-encoder has nothing specific to match and scores every chunk low.

The expansion adds terms that are ALREADY in the corpus (the matched
document's own anchor chunk). It never calls an LLM or a model, never
invents terms, and returns None unless the query is purely generic:
every non-generic word must map onto ONE corpus document. A query with any
extra specific word ("...for astronauts", "...weather policy") is left
alone, so the fallback cannot turn an unsupported question into a supported
one. The refined query is used for retrieval only; the evidence gate and the
answer generator still see the user's original question.
"""

from __future__ import annotations

import re
from collections import Counter

from backend.app.retrieval.chunk_loader import load_chunks


class GenericQueryExpander:
    # Question / filler words plus the word "policy" itself. Whatever is left
    # after removing these is the query's topic.
    GENERIC_WORDS = frozenset(
        """
        a an the is are was were be what whats which who tell me us about
        explain describe give show summary summarize summarise overview
        outline details detail info information general please can could
        would you i we our my your of for on in to and or do does how
        policy policies rule rules guideline guidelines
        """.split()
    )

    # Words never worth adding to a refined query.
    STOPWORDS = frozenset(
        """
        a an the and or of to in on at by for with from as is are was were be
        been it its this that these those must may should can will shall not
        no any all each other than then when where which who whom
        employee employees employee's their there they them
        require requires required requirement requirements prior before
        after normally usually generally
        """.split()
    )

    # Doc-id words that describe the file type, not the topic.
    DOC_ID_NOISE = frozenset({"doc", "policy", "policies"})

    MAX_TOPIC_TOKENS = 3
    EXTRA_TERMS = 4

    _TOKEN = re.compile(r"[a-z0-9]+")

    def __init__(self, chunks_path: str):
        self.chunks_path = chunks_path
        self._docs: dict[str, list[dict]] | None = None

    # ------------------------------------------------------------------
    def _load(self) -> dict[str, list[dict]]:
        if self._docs is None:
            docs: dict[str, list[dict]] = {}
            try:
                for chunk in load_chunks(self.chunks_path):
                    docs.setdefault(chunk.get("doc_id") or "", []).append(chunk)
            except Exception:
                docs = {}
            self._docs = docs
        return self._docs

    @classmethod
    def _tokens(cls, text: str) -> list[str]:
        return cls._TOKEN.findall((text or "").lower().replace("'s", ""))

    @staticmethod
    def _same_word(a: str, b: str) -> bool:
        """travel ~ travelling ~ travels (shared 5-letter stem, min length 5)."""
        if a == b:
            return True
        return min(len(a), len(b)) >= 5 and a[:5] == b[:5]

    # ------------------------------------------------------------------
    def expand(self, query: str) -> str | None:
        """Return a refined retrieval query, or None when not applicable."""
        docs = self._load()
        if not docs:
            return None

        topic = [t for t in self._tokens(query) if t not in self.GENERIC_WORDS]
        if not topic or len(topic) > self.MAX_TOPIC_TOKENS:
            return None

        # Every topic word must appear in the SAME document's id, and exactly
        # one document may match; otherwise the query is not a plain
        # "tell me about <document>" request.
        matches = []
        for doc_id in docs:
            doc_words = [
                w for w in self._tokens(doc_id) if w not in self.DOC_ID_NOISE
            ]
            if doc_words and all(
                any(self._same_word(t, w) for w in doc_words) for t in topic
            ):
                matches.append(doc_id)

        if len(matches) != 1:
            return None

        chunks = docs[matches[0]]

        # Anchor chunk: the one that talks about the topic the most
        # (ties -> earliest chunk).
        def topic_hits(chunk: dict) -> int:
            words = self._tokens(f"{chunk.get('section', '')} {chunk.get('text', '')}")
            return sum(1 for w in words if any(self._same_word(w, t) for t in topic))

        anchor = max(chunks, key=lambda c: (topic_hits(c), -chunks.index(c)))

        used = set(topic) | {"policy"}
        terms = topic + ["policy"]

        # Anchor section title (drop the "2." numbering), then the anchor
        # chunk's most frequent content words. All taken from corpus text.
        title_words = [
            w for w in self._tokens(anchor.get("section", ""))
            if not w.isdigit() and w not in self.STOPWORDS
        ]
        body_words = [
            w for w in self._tokens(anchor.get("text", ""))
            if len(w) > 3 and w not in self.STOPWORDS
        ]
        freq = Counter(body_words)
        first_seen = {}
        for i, w in enumerate(body_words):
            first_seen.setdefault(w, i)
        ranked_body = sorted(freq, key=lambda w: (-freq[w], first_seen[w]))

        extra = 0
        for w in title_words + ranked_body:
            if w in used or any(self._same_word(w, u) for u in used):
                continue
            terms.append(w)
            used.add(w)
            if w not in title_words:
                extra += 1
                if extra >= self.EXTRA_TERMS:
                    break

        refined = " ".join(terms)
        return refined if refined.lower() != query.strip().lower() else None