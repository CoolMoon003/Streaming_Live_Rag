"""
Phase 5 - deterministic metrics helpers.

Kept intentionally small and dependency-free: retrieval-quality metrics
(Recall@k, MRR) plus a simple wall-clock latency tracker. No new third-party
dependencies are introduced.
Groundedness and live per-turn accounting (tokens, latency, cost) are added at
the bottom of this file; they are also stdlib-only.
"""

import re
import time
from contextlib import contextmanager
from statistics import median
from typing import Any, Iterable


def recall_at_k(
    retrieved_chunk_ids: list[str],
    expected_chunk_ids: list[str],
    k: int,
) -> float:
    """
    1.0 if any expected chunk id appears in the top-k retrieved ids, else 0.0.

    Returns 0.0 (not an error) when expected_chunk_ids is empty, since an
    eval row with no labeled ground truth cannot contribute a hit.
    """
    if not expected_chunk_ids:
        return 0.0

    top_k = set(retrieved_chunk_ids[: max(0, k)])
    expected = set(expected_chunk_ids)

    return 1.0 if top_k & expected else 0.0


def reciprocal_rank(
    retrieved_chunk_ids: list[str],
    expected_chunk_ids: list[str],
) -> float:
    """1 / rank of the first retrieved chunk that is in expected_chunk_ids, else 0.0."""
    expected = set(expected_chunk_ids)

    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in expected:
            return 1.0 / rank

    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class LatencyTracker:
    """
    Accumulates wall-clock timings for named pipeline stages across many
    measured calls, and reports the mean per stage in milliseconds.

    This is intentionally a plain dict-backed accumulator, not an
    observability system: Phase 5 only asks for a simple structured
    latency summary.
    """

    def __init__(self):
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    @contextmanager
    def measure(self, stage: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.record(stage, elapsed_ms)

    def record(self, stage: str, elapsed_ms: float) -> None:
        self._totals[stage] = self._totals.get(stage, 0.0) + elapsed_ms
        self._counts[stage] = self._counts.get(stage, 0) + 1

    def mean_ms(self, stage: str) -> float | None:
        count = self._counts.get(stage, 0)
        if count == 0:
            return None
        return self._totals[stage] / count

    def summary(self) -> dict[str, Any]:
        return {
            stage: round(self.mean_ms(stage), 2)
            for stage in self._totals
        }


# ===========================================================================
# GROUNDEDNESS  --  citation / evidence groundedness (NOT semantic truth)
# ===========================================================================
#
# DEFINITION
#     groundedness = supported claim units / total claim units
#     (In comments below, <sect> stands for the section sign, U+00A7.)
#
# The pipeline does not expose explicit "claims", so this uses the closest
# defensible unit: a CLAIM UNIT is one sentence (or bullet / line fragment)
# of the final answer text.
#
# A claim unit is SUPPORTED when it carries at least one inline citation
# "[DOC_ID <sect>Section]" AND every inline citation on it points to a chunk that
# was allowed for that unit:
#   * single-intent answer : the chunks the evidence gate marked supported
#   * multi-intent answer  : the chunks selected+gated for THAT intent's
#                            "Intent N:" section (a citation borrowed from
#                            another intent is unsupported)
# Combined citations "[DOC <sect>1. A, <sect>2. B]" are expanded into individual ones.
#
# NOT CLAIM UNITS (excluded from numerator and denominator):
#   * empty / markup-only fragments and "Intent N:" labels
#   * abstentions ("... does not contain enough evidence ...", or the
#     "could not be verified" replacement message) - counted separately
#   * uncited lead-ins that end with ":"
#
# WHAT THIS DOES *NOT* MEASURE
#   It checks that each statement is attributed to evidence the system
#   actually retrieved and vetted. It does NOT check that the sentence is
#   semantically entailed by the cited chunk, so it is a deterministic
#   proxy for groundedness, not a factual-correctness score. No LLM judge,
#   no extra model, no API call.
#
# The parser below is deliberately independent of CitationValidator so the
# metric does not depend on the component it is auditing.

_SECTION_SIGN = "\u00a7"
_CITATION_RE = re.compile(r"\[([A-Za-z0-9_]+)\s+\u00a7([^\]]+)\]")
_INTENT_LABEL_RE = re.compile(
    r"^\s*(?:[-*\u2022]\s*)?\**\s*intent\s+(\d+)\s*\**\s*[:.)-]\s*\**\s*",
    re.IGNORECASE,
)
_BULLET_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+")
# Sentence boundary = ., ! or ? followed by whitespace, but never a boundary
# inside [...] (citation sections look like "2. International Travel").
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?![^\[\]]*\])")
_LEADING_CITATIONS_RE = re.compile(r"^\s*((?:\[[^\]]*\]\s*)+)")
_ABSTENTION_MARKERS = (
    "does not contain enough evidence",
    "could not be verified against",
)


def canonical_citation(doc_id: str, section: str) -> str:
    """Canonical citation text: [DOC_ID <sect>Section] with whitespace collapsed."""
    return f"[{doc_id} {_SECTION_SIGN}{' '.join(str(section).split())}]"


def parse_citations(text: str) -> list[str]:
    """Ordered canonical citations found in text (combined blocks expanded)."""
    found: list[str] = []
    for doc_id, section_text in _CITATION_RE.findall(text or ""):
        for part in re.split(r"\s*,\s*\u00a7", section_text):
            if part.strip():
                found.append(canonical_citation(doc_id, part))
    return found


def _split_sections(answer: str) -> list[tuple[str | None, list[str]]]:
    """Split an answer into (intent_id | None, lines). None = no label."""
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in (answer or "").splitlines():
        label = _INTENT_LABEL_RE.match(line)
        if label:
            sections.append((label.group(1), [line[label.end():]]))
        else:
            sections[-1][1].append(line)
    return sections


def _units_from_lines(lines: list[str]) -> list[dict]:
    """Sentence-level units, each with the citations attached to it."""
    units: list[dict] = []
    for raw in lines:
        line = _BULLET_RE.sub("", raw).strip()
        if not line:
            continue
        line_units: list[dict] = []
        for fragment in _SENTENCE_SPLIT_RE.split(line):
            fragment = fragment.strip()
            if not fragment:
                continue
            lead = _LEADING_CITATIONS_RE.match(fragment)
            if lead and line_units:
                # A citation placed after the previous sentence's full stop
                # belongs to that sentence.
                line_units[-1]["citations"].extend(parse_citations(lead.group(1)))
                fragment = fragment[lead.end():].strip()
                if not fragment:
                    continue
            line_units.append(
                {"text": fragment, "citations": parse_citations(fragment)}
            )
        units.extend(line_units)
    return units


def _is_claim(unit: dict) -> bool:
    """Return True when this sentence unit should be counted as a factual claim.

    Exclusion rules (applied in order):
      1. No alphanumeric content → not a claim  (e.g. markup-only lines).
      2. Uncited colon lead-in → not a claim    (e.g. "Here are the rules:").
      3. Uncited question → not a claim          (e.g. echoed intent headings
         like "What is the reimbursement limit?" that multi-intent generation
         emits as section headers before answering).
         A question that does carry a citation is still counted as a claim,
         because the author chose to cite it.
      4. Anything else → is a claim.
    """
    bare = _CITATION_RE.sub("", unit["text"])
    if not re.search(r"[A-Za-z0-9]", bare):
        return False
    if not unit["citations"] and unit["text"].rstrip().endswith(":"):
        return False
    # Rule 3: uncited question — intent echo heading, not a factual claim.
    if not unit["citations"] and unit["text"].rstrip().endswith("?"):
        return False
    return True


def _is_abstention(unit: dict) -> bool:
    low = unit["text"].lower()
    return any(marker in low for marker in _ABSTENTION_MARKERS)


def _allowed_set(citations: Iterable[str] | None) -> set[str]:
    return set(parse_citations(" ".join(citations or [])))


def answer_groundedness(
    answer: str,
    allowed_citations: Iterable[str] | None = None,
    intents: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Citation/evidence groundedness of one final answer (see block above).

    Single-intent: pass allowed_citations (canonical "[DOC <sect>Section]" strings
    of the gate-supported evidence).

    Multi-intent: pass intents = [{"intent_id", "supported": bool,
    "allowed_citations": [...]}]. Each "Intent N:" section is scored against
    that intent's own allowed citations; unlabelled text is scored against the
    union of the supported intents' citations. An unsupported intent has no
    allowed citations, so any claim written for it counts as unsupported.

    "groundedness" is None when the answer has no claim units (for example a
    pure abstention): there is nothing to score, and 0.0 would be misleading.
    """
    per_intent_allowed: dict[str, set[str]] = {}
    union: set[str] = set(_allowed_set(allowed_citations))
    for intent in intents or []:
        allowed = (
            _allowed_set(intent.get("allowed_citations"))
            if intent.get("supported")
            else set()
        )
        per_intent_allowed[str(intent.get("intent_id"))] = allowed
        union |= allowed

    counts: dict[str | None, list[int]] = {}
    abstentions = 0
    invalid: set[str] = set()
    unsupported_examples: list[str] = []

    for intent_key, lines in _split_sections(answer):
        allowed = union if intent_key is None else per_intent_allowed.get(intent_key, set())
        bucket = counts.setdefault(intent_key, [0, 0])  # [total, supported]

        for unit in _units_from_lines(lines):
            if _is_abstention(unit):
                abstentions += 1
                continue
            if not _is_claim(unit):
                continue

            bucket[0] += 1
            bad = [c for c in unit["citations"] if c not in allowed]
            invalid.update(bad)

            if unit["citations"] and not bad:
                bucket[1] += 1
            elif len(unsupported_examples) < 3:
                unsupported_examples.append(unit["text"][:160])

    total = sum(b[0] for b in counts.values())
    supported = sum(b[1] for b in counts.values())

    result: dict[str, Any] = {
        "total_claims": total,
        "supported_claims": supported,
        "unsupported_claims": total - supported,
        "groundedness": (supported / total) if total else None,
        "abstention_units": abstentions,
        "invalid_citations": sorted(invalid),
        "unsupported_examples": unsupported_examples,
    }

    if intents is not None:
        per_intent = []
        for intent in intents:
            key = str(intent.get("intent_id"))
            t, s = counts.get(key, [0, 0])
            per_intent.append(
                {
                    "intent_id": intent.get("intent_id"),
                    "intent_supported": bool(intent.get("supported")),
                    "total_claims": t,
                    "supported_claims": s,
                    "groundedness": (s / t) if t else None,
                }
            )
        result["per_intent"] = per_intent

    return result


def aggregate_groundedness(results: list[dict]) -> dict[str, Any]:
    """
    Corpus-level groundedness over answer_groundedness() results.

    groundedness_micro = sum(supported claims) / sum(total claims)  (headline)
    groundedness_macro = mean of per-turn groundedness over turns with claims
    Turns without claim units (abstentions) are counted, not scored.
    """
    scored = [r for r in results if r["total_claims"]]
    total = sum(r["total_claims"] for r in scored)
    supported = sum(r["supported_claims"] for r in scored)
    return {
        "turns": len(results),
        "turns_scored": len(scored),
        "turns_without_claims": len(results) - len(scored),
        "total_claims": total,
        "supported_claims": supported,
        "groundedness_micro": (supported / total) if total else None,
        "groundedness_macro": mean([r["groundedness"] for r in scored]) if scored else None,
        "invalid_citation_count": sum(len(r["invalid_citations"]) for r in results),
    }


# ===========================================================================
# LIVE TURN ACCOUNTING  --  latency, tokens, cost per turn (local Ollama)
# ===========================================================================
#
# These are LIVE metrics: they need a real generation, so they are never
# reported by the offline retrieval evaluation. A "turn record" is a dict:
#   generated            bool   an answer_token was streamed
#   pre_generation_ms    commit -> answer_started (retrieval/select/gate)
#   ttft_ms              commit -> first streamed answer_token (pipeline TTFT)
#   llm_ttft_ms          Ollama request -> first token (LLM-only TTFT)
#   llm_generation_ms    Ollama request -> generation done
#   server_processing_ms commit -> final event (whole server-side turn)
#   retrieval_calls      retrieve() calls counted by the harness
#   prompt_tokens / completion_tokens / total_tokens   as reported by Ollama,
#                        None when Ollama did not report them (never estimated)

LOCAL_COST_BASIS = (
    "Local Ollama inference: no per-token cloud/API billing, so cloud/API "
    "cost per turn is 0. Local compute (electricity, GPU/CPU time) is not "
    "free and is NOT monetised here."
)


def summarize_values(values: list[float | int | None]) -> dict[str, Any] | None:
    """n / mean / median / max over the non-None values; None if there are none."""
    present = [v for v in values if v is not None]
    if not present:
        return None
    return {
        "n": len(present),
        "mean": mean(present),
        "median": median(present),
        "max": max(present),
    }


def cost_per_turn_local(turns: list[dict], model: str) -> dict[str, Any]:
    """Cost-per-turn accounting for the local-Ollama demo."""
    with_tokens = [t for t in turns if t.get("total_tokens") is not None]
    if with_tokens:
        tokens: dict[str, Any] = {
            "status": "available",
            "turns_with_token_counts": len(with_tokens),
            "mean_prompt_tokens": mean([t["prompt_tokens"] for t in with_tokens]),
            "mean_completion_tokens": mean([t["completion_tokens"] for t in with_tokens]),
            "mean_total_tokens": mean([t["total_tokens"] for t in with_tokens]),
        }
    else:
        tokens = {"status": "unavailable"}

    return {
        "model": model,
        "cloud_api_cost_usd_per_turn": 0.0,
        "cloud_api_cost_inr_per_turn": 0.0,
        "cost_basis": LOCAL_COST_BASIS,
        "token_usage": tokens,
        "mean_retrieval_calls_per_turn": summarize_values(
            [t.get("retrieval_calls") for t in turns]
        ),
    }


def summarize_live_turns(turns: list[dict], model: str) -> dict[str, Any]:
    """Latency and cost summary over live turn records."""
    generated = [t for t in turns if t.get("generated")]
    return {
        "turns": len(turns),
        "turns_with_generation": len(generated),
        "latency_ms": {
            "ttft_pipeline": summarize_values([t.get("ttft_ms") for t in generated]),
            "ttft_llm_only": summarize_values([t.get("llm_ttft_ms") for t in generated]),
            "pre_generation": summarize_values([t.get("pre_generation_ms") for t in turns]),
            "llm_generation": summarize_values([t.get("llm_generation_ms") for t in generated]),
            "server_processing": summarize_values([t.get("server_processing_ms") for t in turns]),
        },
        "cost_per_turn": cost_per_turn_local(turns, model),
    }