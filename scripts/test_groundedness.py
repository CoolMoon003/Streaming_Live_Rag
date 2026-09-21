"""Deterministic tests for the groundedness metric, Ollama token capture and
cost-per-turn accounting. No Ollama, no models, no network.

These tests check the SCORER and the ACCOUNTING with synthetic answers. They
are not a measurement of the live system; the live numbers come from
    .venv\\Scripts\\python -m scripts.evaluate_phase5 --phase6 --live

Run:
    .venv\\Scripts\\python -m scripts.test_groundedness
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import (
    aggregate_groundedness,
    answer_groundedness,
    canonical_citation as C,
    cost_per_turn_local,
    parse_citations,
    summarize_live_turns,
)

TRAVEL = C("DOC_A", "2. International Travel")
CLAIMS = C("DOC_B", "1. Eligible Expenses")
DEADLINE = C("DOC_B", "3. Submission Deadline")
FAKE = C("DOC_X", "9. Invented Section")

_results = []


def check(name, condition, detail=""):
    _results.append(bool(condition))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  {detail}" if not condition else ""))


def test_single_intent():
    print("single-intent")

    # Citation placed AFTER the full stop, and section titles containing ". "
    full = f"International travel needs prior approval. {TRAVEL} Airfare may be claimed. {TRAVEL}"
    r = answer_groundedness(full, [TRAVEL])
    check("fully cited -> 1.0 (2/2)", r["groundedness"] == 1.0 and r["total_claims"] == 2, r)

    inline = f"International travel needs prior approval {TRAVEL}."
    r = answer_groundedness(inline, [TRAVEL])
    check("citation before full stop -> supported", r["groundedness"] == 1.0, r)

    half = f"International travel needs prior approval. {TRAVEL} Hotels are always free."
    r = answer_groundedness(half, [TRAVEL])
    check("one uncited sentence -> 0.5", r["groundedness"] == 0.5 and r["unsupported_claims"] == 1, r)

    fake = f"Approval is needed. {FAKE}"
    r = answer_groundedness(fake, [TRAVEL])
    check("fabricated citation -> unsupported + reported",
          r["groundedness"] == 0.0 and r["invalid_citations"] == [FAKE], r)

    block = "Claims are limited. [DOC_B \u00a71. Eligible Expenses, \u00a73. Submission Deadline]"
    check("combined citation expands", parse_citations(block) == [CLAIMS, DEADLINE], parse_citations(block))
    r = answer_groundedness(block, [CLAIMS, DEADLINE])
    check("combined citation, both allowed -> 1.0", r["groundedness"] == 1.0, r)
    r = answer_groundedness(block, [CLAIMS])
    check("combined citation, one not allowed -> unsupported", r["groundedness"] == 0.0, r)

    r = answer_groundedness("The provided corpus does not contain enough evidence to answer this.", [TRAVEL])
    check("pure abstention -> None, not 0.0",
          r["groundedness"] is None and r["total_claims"] == 0 and r["abstention_units"] == 1, r)

    lead = f"Here are the rules:\n- Approval is required. {TRAVEL}\n- Receipts are needed. {TRAVEL}"
    r = answer_groundedness(lead, [TRAVEL])
    check("uncited lead-in ignored, bullets scored", r["total_claims"] == 2 and r["groundedness"] == 1.0, r)

    r = answer_groundedness(f"Approval is required. {TRAVEL}", [])
    check("no allowed evidence -> unsupported", r["groundedness"] == 0.0, r)


def test_multi_intent():
    print("multi-intent")
    intents = [
        {"intent_id": 1, "supported": True, "allowed_citations": [TRAVEL]},
        {"intent_id": 2, "supported": True, "allowed_citations": [CLAIMS]},
    ]

    ok = f"Intent 1: Approval is required. {TRAVEL}\nIntent 2: Receipts are needed. {CLAIMS}"
    r = answer_groundedness(ok, intents=intents)
    check("each intent cites its own evidence -> 1.0",
          r["groundedness"] == 1.0 and [p["groundedness"] for p in r["per_intent"]] == [1.0, 1.0], r)

    md = f"**Intent 1:** Approval is required. {TRAVEL}\n**Intent 2:** Receipts are needed. {CLAIMS}"
    r = answer_groundedness(md, intents=intents)
    check("markdown-bold intent labels parsed", r["groundedness"] == 1.0 and r["total_claims"] == 2, r)

    swapped = f"Intent 1: Approval is required. {TRAVEL}\nIntent 2: Receipts are needed. {TRAVEL}"
    r = answer_groundedness(swapped, intents=intents)
    check("citation borrowed from another intent -> unsupported",
          r["groundedness"] == 0.5 and r["per_intent"][1]["groundedness"] == 0.0, r)

    partial = [
        {"intent_id": 1, "supported": True, "allowed_citations": [TRAVEL]},
        {"intent_id": 2, "supported": False, "allowed_citations": []},
    ]
    abst = (f"Intent 1: Approval is required. {TRAVEL}\n"
            "Intent 2: The provided policy corpus does not contain enough evidence to answer this part.")
    r = answer_groundedness(abst, intents=partial)
    check("unsupported intent abstains -> not penalised",
          r["groundedness"] == 1.0 and r["abstention_units"] == 1 and r["per_intent"][1]["groundedness"] is None, r)

    invented = f"Intent 1: Approval is required. {TRAVEL}\nIntent 2: The limit is 500 dollars. {CLAIMS}"
    r = answer_groundedness(invented, intents=partial)
    check("claim written for an unsupported intent -> unsupported",
          r["groundedness"] == 0.5 and r["per_intent"][1]["supported_claims"] == 0, r)


def test_is_claim_question_rule():
    """Regression tests for _is_claim() question-exclusion (PATCH A)."""
    print("_is_claim question rule")

    # 1. Uncited question -> NOT a claim (intent echo heading pattern)
    r = answer_groundedness("What is the reimbursement limit?", [TRAVEL])
    check("uncited question -> not a claim (total_claims == 0)",
          r["total_claims"] == 0 and r["groundedness"] is None, r)

    # 2. Cited question -> IS a claim (author chose to cite it)
    cited_q = f"What is the reimbursement limit? {TRAVEL}"
    r = answer_groundedness(cited_q, [TRAVEL])
    check("cited question -> is a claim, supported",
          r["total_claims"] == 1 and r["groundedness"] == 1.0, r)

    # 3. Uncited factual sentence -> still a claim
    r = answer_groundedness("International travel requires prior approval.", [TRAVEL])
    check("uncited factual sentence -> is a claim, unsupported",
          r["total_claims"] == 1 and r["groundedness"] == 0.0, r)

    # 4. Uncited colon lead-in -> NOT a claim (existing rule unchanged)
    r = answer_groundedness("Here are the rules:", [TRAVEL])
    check("uncited colon lead-in -> not a claim (unchanged rule)",
          r["total_claims"] == 0 and r["groundedness"] is None, r)

    # 5. Pure abstention -> NOT a claim (unchanged rule)
    r = answer_groundedness(
        "The provided corpus does not contain enough evidence to answer this.", [TRAVEL]
    )
    check("abstention -> not a claim (unchanged rule)",
          r["total_claims"] == 0 and r["abstention_units"] == 1, r)


def test_aggregate():
    print("aggregate")
    a = {"total_claims": 4, "supported_claims": 4, "groundedness": 1.0, "invalid_citations": []}
    b = {"total_claims": 1, "supported_claims": 0, "groundedness": 0.0, "invalid_citations": [FAKE]}
    c = {"total_claims": 0, "supported_claims": 0, "groundedness": None, "invalid_citations": []}
    g = aggregate_groundedness([a, b, c])
    check("micro = 4/5, macro = 0.5, abstention counted not scored",
          g["groundedness_micro"] == 0.8 and g["groundedness_macro"] == 0.5
          and g["turns"] == 3 and g["turns_scored"] == 2 and g["turns_without_claims"] == 1
          and g["invalid_citation_count"] == 1, g)
    g = aggregate_groundedness([c])
    check("no claims anywhere -> None", g["groundedness_micro"] is None and g["groundedness_macro"] is None, g)


class _FakeResponse:
    def __init__(self, payload=None, lines=None):
        self._payload, self._lines = payload, lines or []

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload

    def iter_lines(self):
        return iter(self._lines)


def test_ollama_usage_capture():
    print("ollama token capture")
    from backend.app.llm import ollama_client

    real_post = ollama_client.requests.post
    try:
        client = ollama_client.OllamaClient()

        ollama_client.requests.post = lambda *a, **k: _FakeResponse(
            payload={"response": "ok", "prompt_eval_count": 120, "eval_count": 30})
        client.generate("p")
        m = client.get_last_metrics()
        check("generate(): counts captured",
              (m["prompt_tokens"], m["completion_tokens"], m["total_tokens"]) == (120, 30, 150), m)

        lines = [json.dumps(x).encode() for x in (
            {"response": "Hel", "done": False},
            {"response": "lo", "done": False},
            {"response": "", "done": True, "prompt_eval_count": 200, "eval_count": 12},
        )]
        ollama_client.requests.post = lambda *a, **k: _FakeResponse(lines=lines)
        text = "".join(client.generate_stream("p"))
        m = client.get_last_metrics()
        check("generate_stream(): counts from final chunk",
              text == "Hello" and (m["prompt_tokens"], m["completion_tokens"], m["total_tokens"]) == (200, 12, 212), m)
        check("existing ttft_ms / latency_ms keys still present", "ttft_ms" in m and "latency_ms" in m, m)

        ollama_client.requests.post = lambda *a, **k: _FakeResponse(payload={"response": "ok"})
        client.generate("p")
        m = client.get_last_metrics()
        check("counts absent -> None (stale values reset, never estimated)",
              m["prompt_tokens"] is None and m["completion_tokens"] is None and m["total_tokens"] is None, m)
    finally:
        ollama_client.requests.post = real_post


def test_cost_and_live_summary():
    print("cost per turn / live summary")
    turns = [
        {"generated": True, "ttft_ms": 900.0, "llm_ttft_ms": 700.0, "pre_generation_ms": 200.0,
         "llm_generation_ms": 2000.0, "server_processing_ms": 2300.0, "retrieval_calls": 1,
         "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        {"generated": True, "ttft_ms": 1100.0, "llm_ttft_ms": 800.0, "pre_generation_ms": 300.0,
         "llm_generation_ms": 3000.0, "server_processing_ms": 3400.0, "retrieval_calls": 2,
         "prompt_tokens": 200, "completion_tokens": 40, "total_tokens": 240},
        {"generated": False, "pre_generation_ms": 250.0, "server_processing_ms": 260.0,
         "retrieval_calls": 1, "prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
    ]
    s = summarize_live_turns(turns, "llama3.2:3b")
    cost, lat = s["cost_per_turn"], s["latency_ms"]
    check("cloud/API cost is 0 USD and 0 INR",
          cost["cloud_api_cost_usd_per_turn"] == 0.0 and cost["cloud_api_cost_inr_per_turn"] == 0.0, cost)
    check("mean tokens over turns that reported them",
          cost["token_usage"]["mean_total_tokens"] == 180 and cost["token_usage"]["turns_with_token_counts"] == 2, cost)
    check("TTFT stats only over generated turns", lat["ttft_pipeline"]["n"] == 2 and lat["ttft_pipeline"]["mean"] == 1000.0, lat)
    check("retrieval calls per turn mean = 4/3",
          abs(cost["mean_retrieval_calls_per_turn"]["mean"] - 4 / 3) < 1e-9, cost)

    none = cost_per_turn_local([{"generated": True, "total_tokens": None}], "m")
    check("no token counts -> reported unavailable, still not invented",
          none["token_usage"] == {"status": "unavailable"}, none)


def main() -> int:
    print("=" * 72)
    print("GROUNDEDNESS / TOKEN / COST-PER-TURN TESTS")
    print("=" * 72)
    test_single_intent()
    test_multi_intent()
    test_is_claim_question_rule()
    test_aggregate()
    test_ollama_usage_capture()
    test_cost_and_live_summary()
    passed = sum(_results)
    print("-" * 72)
    print(f"{passed}/{len(_results)} checks passed")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())