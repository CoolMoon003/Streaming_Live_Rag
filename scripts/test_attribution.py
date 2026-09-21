"""Offline tests for CitationValidator.attribute_sentences() and
enforce_unsupported_intents().

Uses the REAL Phase 6 corpus chunks and the REAL groundedness metric; no
retrieval models, no Ollama. Includes a replay of the exact P6-Q017 /
P6-Q018 answers observed from llama3.2:3b.

    python -m scripts.test_attribution
"""
import sys
from pathlib import Path

from backend.app.retrieval.chunk_loader import load_chunks
from backend.app.retrieval.citation_validator import CitationValidator
from evaluation.metrics import answer_groundedness, canonical_citation

CHUNKS = {
    c["chunk_id"]: c
    for c in load_chunks(
        Path(__file__).resolve().parent.parent
        / "data" / "processed" / "phase6_chunks.jsonl"
    )
}
MSG = "The provided policy corpus does not contain enough evidence to answer this part."
V = CitationValidator()
FAILS: list[str] = []


def check(name, condition, detail=""):
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    if not condition:
        FAILS.append(name)
        if detail:
            print(f"         {detail}")


def intent(intent_id, *chunk_ids, supported=True):
    return {
        "intent_id": intent_id,
        "supported": supported,
        "evidence": [{"chunk": CHUNKS[i]} for i in chunk_ids] if supported else [],
    }


def score(answer, intents):
    return answer_groundedness(
        answer,
        intents=[
            {
                "intent_id": str(i["intent_id"]),
                "supported": i["supported"],
                "allowed_citations": [
                    canonical_citation(r["chunk"]["doc_id"], r["chunk"]["section"])
                    for r in i["evidence"]
                ],
            }
            for i in intents
        ],
    )


WS = "DOC_WORKSHOP_EVENT_POLICY"
Q17 = [intent(1, f"{WS}_C001", f"{WS}_C006"), intent(2, f"{WS}_C003")]
Q17_ANSWER = (
    "Intent 1: Internal workshops require approval from the responsible "
    "department before venue or service commitments are made. Approved "
    "workshop expenses may include venue rental, catering, equipment, "
    "printing, and other directly related business costs. "
    f"[{WS} §1. Workshop Approval] [{WS} §6. Event Expenses]\n\n"
    "Intent 2: Catering arrangements should reflect the expected attendance "
    "and event duration. Approved catering may include meals, refreshments, "
    f"and dietary alternatives for participants. [{WS} §3. Catering]"
)


def test_q017_replay():
    print("P6-Q017 replay (observed llama3.2:3b output)")
    before = score(Q17_ANSWER, Q17)
    check("before repair: 2/4 (reproduces the bug)",
          (before["supported_claims"], before["total_claims"]) == (2, 4), before)

    fixed = V.attribute_sentences(Q17_ANSWER, Q17)
    after = score(fixed, Q17)
    check("after repair: 4/4, groundedness 1.0",
          (after["supported_claims"], after["total_claims"]) == (4, 4)
          and after["groundedness"] == 1.0, after)
    check("first sentence of intent 1 cites §1 Workshop Approval",
          f"made [{WS} §1. Workshop Approval]." in fixed)
    check("first sentence of intent 2 cites §3 Catering",
          f"event duration [{WS} §3. Catering]." in fixed)
    check("no citation outside the intents' own evidence",
          V.validate_multi_intent(fixed, Q17)["valid"])
    check("idempotent", V.attribute_sentences(fixed, Q17) == fixed)
    check("existing citations preserved",
          f"[{WS} §6. Event Expenses]" in fixed
          and fixed.count(f"[{WS} §3. Catering]") == 2)


def test_q018_abstention():
    print("P6-Q018 abstention preserved")
    it = "DOC_IT_ASSET_POLICY"
    intents = [intent(1, f"{it}_C004"), intent(2, supported=False)]
    answer = (
        "Intent 1: Lost or stolen company equipment must be reported promptly "
        f"to the IT service desk and the employee's manager [{it} §4. Loss or Theft].\n\n"
        f"Intent 2: {MSG}"
    )
    check("attribution leaves it unchanged", V.attribute_sentences(answer, intents) == answer)
    check("enforcement leaves it unchanged",
          V.enforce_unsupported_intents(answer, intents, MSG) == answer)
    g = score(answer, intents)
    check("groundedness 1.0", g["groundedness"] == 1.0, g)


def test_safety():
    print("attribution safety")
    paraphrase = (
        "Intent 1: Workshops need departmental sign-off first.\n"
        "Intent 2: Meals and refreshments are permitted."
    )
    check("paraphrases are NOT cited",
          V.attribute_sentences(paraphrase, Q17) == paraphrase)

    changed = ("Intent 2: Approved catering may include meals, refreshments, "
               "and unlimited alcohol for participants.")
    check("changed facts are NOT cited",
          V.attribute_sentences(changed, Q17) == changed)

    borrowed = (
        "Intent 1: Catering arrangements should reflect the expected attendance "
        "and event duration.\nIntent 2: Internal workshops require approval from "
        "the responsible department before venue or service commitments are made."
    )
    check("no cross-intent borrowing",
          V.attribute_sentences(borrowed, Q17) == borrowed)

    it = "DOC_IT_ASSET_POLICY"
    unsupported = [intent(1, f"{it}_C004"), intent(2, supported=False)]
    text = ("Intent 2: Lost or stolen company equipment must be reported promptly "
            "to the IT service desk and the employee's manager.")
    check("unsupported intent never receives a citation",
          V.attribute_sentences(text, unsupported) == text)

    preamble = "Sure! Internal workshops require approval from the responsible department.\n" + Q17_ANSWER
    check("text before the first Intent label untouched",
          V.attribute_sentences(preamble, Q17).startswith("Sure! Internal workshops require approval from the responsible department.\n"))

    fake = "Intent 1: Approved catering may include meals. [DOC_FAKE §9. Nope]"
    check("never invents or repairs fabricated IDs",
          V.attribute_sentences(fake, Q17) == fake
          and not V.validate_multi_intent(fake, Q17)["valid"])


def test_formats():
    print("format robustness")
    messy = (
        "**Intent 1:** - Internal   workshops require approval from the "
        "responsible department before venue or service commitments are made\n"
        "* approved WORKSHOP expenses may include venue rental, catering, "
        "equipment, printing, and other directly related business costs.\r\n"
        "**Intent 2:**\r\n"
        "1. **Catering arrangements should reflect the expected attendance and event duration.**\r\n"
        f"2. Approved catering may include meals. [{WS} §3. Catering]\r\n"
    )
    out = V.attribute_sentences(messy, Q17)
    check("bold labels / bullets / numbering / CRLF / case / spacing",
          out.count(f"[{WS} §1. Workshop Approval]") == 1
          and out.count(f"[{WS} §6. Event Expenses]") == 1
          and out.count(f"[{WS} §3. Catering]") == 2
          and "\r\n" in out, out)
    check("idempotent", V.attribute_sentences(out, Q17) == out)

    trailing = (
        "Intent 1: Internal workshops require approval from the responsible "
        "department before venue or service commitments are made. "
        f"[{WS} §1. Workshop Approval]"
    )
    check("citation after the full stop is respected",
          V.attribute_sentences(trailing, Q17) == trailing)


def test_enforce_unsupported():
    print("unsupported-intent enforcement")
    it = "DOC_IT_ASSET_POLICY"
    intents = [intent(1, f"{it}_C004"), intent(2, supported=False)]
    halluc = (
        f"Intent 1: Report it promptly. [{it} §4. Loss or Theft]\n\n"
        "Intent 2: Employees may claim unlimited amounts with no documentation required.\n"
        f"More made-up prose. [{it} §4. Loss or Theft]"
    )
    out = V.enforce_unsupported_intents(halluc, intents, MSG)
    check("hallucinated claim replaced by the abstention sentence",
          out.rstrip().endswith(f"Intent 2: {MSG}") and "unlimited" not in out and "made-up" not in out, out)
    check("supported intent untouched",
          out.startswith(f"Intent 1: Report it promptly. [{it} §4. Loss or Theft]\n\n"))
    check("idempotent", V.enforce_unsupported_intents(out, intents, MSG) == out)
    all_ok = [intent(1, f"{it}_C004")]
    check("no unsupported intents -> unchanged",
          V.enforce_unsupported_intents(halluc, all_ok, MSG) == halluc)


def main() -> int:
    for fn in (test_q017_replay, test_q018_abstention, test_safety,
               test_formats, test_enforce_unsupported):
        fn()
    print("-" * 72)
    print("ALL PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())