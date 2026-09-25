"""Phase 5 / Phase 6 evaluation runner (offline retrieval by default; --live adds Ollama metrics).

Usage:
    .venv\\Scripts\\python -m scripts.evaluate_phase5
    .venv\\Scripts\\python -m scripts.evaluate_phase5 --phase6 --offline-only
    .venv\\Scripts\\python -m scripts.evaluate_phase5 --phase6 --live --json phase6_report.json

OFFLINE section: Recall@k, MRR, multi-intent coverage / pass rate (no LLM).
LIVE section (--live, needs Ollama): TTFT, generation latency, server latency,
groundedness, token usage, cost per turn.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.retrieval.multi_intent_retriever import MultiIntentRetriever
from backend.app.retrieval.streaming_retriever import StreamingRetriever
from evaluation import eval_dataset
from evaluation.metrics import (
    aggregate_groundedness,
    answer_groundedness,
    canonical_citation,
    mean,
    recall_at_k,
    reciprocal_rank,
    summarize_live_turns,
)

PHASE5_CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
PHASE6_CHUNKS = ROOT / "data" / "processed" / "phase6_chunks.jsonl"
PHASE6_QUERIES = ROOT / "data" / "processed" / "phase6_eval_queries.json"

ID_KEYS = ("expected_chunks", "relevant_chunk_ids", "expected_chunk_ids",
           "expected_chunk_id", "gold_chunk_ids", "gold_chunks", "relevant_chunks",
           "relevant_ids", "expected_ids", "chunk_ids")
QUERY_KEYS = ("query", "question", "text")
SUB_KEYS = ("subqueries", "sub_queries", "intents")
# Keys tried (in order) when a retriever returns a dict of several result lists.
RANKED_KEYS = ("final_results", "final_chunks", "final", "reranked_results",
               "rrf_results", "results", "chunks", "merged")


def _first(rec, keys, default=None):
    for k in keys:
        if isinstance(rec, dict) and rec.get(k) not in (None, "", []):
            return rec[k]
    return default


def _dedupe(seq):
    return list(dict.fromkeys(seq))


def _ids(results):
    """Ordered chunk ids from a retriever result: a list of chunk dicts/objects,
    {"chunk": {...}, "score": ...} wrappers, or a dict of result lists."""
    if isinstance(results, dict):
        cid = results.get("chunk_id")
        if isinstance(cid, str):
            return [cid]
        if isinstance(results.get("chunk"), dict):
            return _ids(results["chunk"])
        if isinstance(results.get("id"), str):
            return [results["id"]]
        for key in RANKED_KEYS:
            if key in results:
                ids = _ids(results[key])
                if ids:
                    return ids
        out = []
        for v in results.values():
            if isinstance(v, (dict, list, tuple)):
                out.extend(_ids(v))
        return _dedupe(out)
    if isinstance(results, (list, tuple)):
        out = []
        for r in results:
            out.extend(_ids(r))
        return _dedupe(out)
    cid = getattr(results, "chunk_id", None)
    return [cid] if isinstance(cid, str) else []


def _known_ids(path):
    """All chunk ids in the corpus file."""
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        cid = _first(d, ("chunk_id", "id"))
        if isinstance(cid, str):
            out.add(cid)
    if not out:
        raise RuntimeError(f"No chunk ids found in {path}")
    return out


def _collect_ids(obj, known):
    """Walk ANY nested structure and return the ordered, de-duplicated corpus
    chunk ids found in it. Never hashes a dict."""
    out = []

    def walk(o, depth):
        if o is None or depth > 12:
            return
        if isinstance(o, str):
            if o in known:
                out.append(o)
        elif isinstance(o, dict):
            for k, v in o.items():
                if isinstance(k, str):
                    walk(k, depth + 1)
                walk(v, depth + 1)
        elif isinstance(o, (list, tuple, set, frozenset)):
            for v in o:
                walk(v, depth + 1)
        elif hasattr(o, "model_dump"):
            walk(o.model_dump(), depth + 1)
        elif hasattr(o, "__dict__"):
            walk(vars(o), depth + 1)

    walk(obj, 0)
    return _dedupe(out)


def _call_metric(fn, retrieved, relevant, *extra):
    try:
        return fn(retrieved, relevant, *extra)
    except TypeError:
        return fn(retrieved, set(relevant), *extra)


def _as_dict(x):
    if isinstance(x, dict):
        return x
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return dataclasses.asdict(x)
    if hasattr(x, "model_dump"):
        return x.model_dump()
    if hasattr(x, "__dict__"):
        return dict(vars(x))
    return None


def _as_records(obj):
    """Turn a dataset object (list, dict of lists, dataclass list...) into record dicts."""
    if isinstance(obj, (list, tuple)):
        recs = [_as_dict(i) for i in obj]
        return [r for r in recs if r and _first(r, QUERY_KEYS)]
    if isinstance(obj, dict):
        if _first(obj, QUERY_KEYS):
            return [obj]
        out = []
        for v in obj.values():
            if isinstance(v, (list, tuple, dict)):
                out.extend(_as_records(v))
        return out
    return []


def _phase5_records():
    """Find the existing Phase 5 cases in evaluation/eval_dataset.py without
    assuming a variable name: use the largest public data object holding
    query-bearing records; only call loader functions if no data object matches."""
    names = [n for n in dir(eval_dataset) if not n.startswith("_")]
    data, funcs = [], []
    for n in names:
        obj = getattr(eval_dataset, n)
        if isinstance(obj, (types.ModuleType, type)):
            continue
        if callable(obj):
            funcs.append((n, obj))
            continue
        recs = _as_records(obj)
        if recs:
            data.append((n, recs))
    if not data:
        for n, fn in funcs:
            try:
                recs = _as_records(fn())
            except Exception:
                continue
            if recs:
                data.append((n, recs))
    if not data:
        raise RuntimeError(
            "No Phase 5 cases found in evaluation/eval_dataset.py; public names: "
            + ", ".join(names)
        )
    name, recs = max(data, key=lambda t: len(t[1]))
    print(f"Phase 5 dataset: evaluation.eval_dataset.{name} ({len(recs)} cases)")
    return recs


def load_records(phase6: bool):
    if phase6:
        return json.loads(PHASE6_QUERIES.read_text(encoding="utf-8"))
    return _phase5_records()


def _labels(rec):
    val = _first(rec, ID_KEYS, [])
    return [val] if isinstance(val, str) else list(val)


def is_multi(rec):
    kind = str(_first(rec, ("intent_type", "type", "category", "intent"), "")).lower()
    subs = _first(rec, SUB_KEYS)
    return "multi" in kind or (isinstance(subs, list) and len(subs) > 1)


def expected_groups(rec):
    """One list of acceptable chunk ids per intent."""
    flat = _labels(rec)
    intents = _first(rec, SUB_KEYS, [])
    # Flat Phase 6 schema: intents (strings) map positionally to expected_chunks.
    if (isinstance(intents, list) and intents and all(isinstance(i, str) for i in intents)
            and isinstance(flat, list) and len(flat) == len(intents)):
        return [[c] for c in flat]
    groups = []
    for s in intents if isinstance(intents, list) else []:
        g = _first(s, ID_KEYS) if isinstance(s, dict) else None
        if g:
            groups.append(list(g))
    return groups or [list(flat)]


def subquery_texts(rec):
    subs = _first(rec, SUB_KEYS, [])
    out = []
    for s in subs if isinstance(subs, list) else []:
        out.append(s if isinstance(s, str) else _first(s, QUERY_KEYS, ""))
    return [s for s in out if s] or [_first(rec, QUERY_KEYS, "")]


class _RetrieverAdapter:
    """MultiIntentRetriever awaits retriever.retrieve(query, generation_id=...),
    but StreamingRetriever.retrieve() is sync and takes only (query)."""

    def __init__(self, inner):
        self._inner = inner

    async def retrieve(self, query, *args, **kwargs):
        return await asyncio.to_thread(self._inner.retrieve, query)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def evaluate(phase6: bool, no_multi_intent: bool = False) -> dict:
    """Offline retrieval evaluation.

    Parameters
    ----------
    phase6:
        When True, use the Phase 6 corpus and gold labels.
    no_multi_intent:
        When True, multi-intent rows are handled by a single ``StreamingRetriever``
        call on the complete original query string instead of ``MultiIntentRetriever``.
        Intent-level coverage is still calculated using the same per-intent gold chunk
        IDs, but the pool of retrieved chunks comes from the single-query result.
        All other behaviour (single-intent rows, metrics, gold labels) is unchanged.
        Default False = existing full multi-intent retrieval behaviour.
    """
    chunks_path = PHASE6_CHUNKS if phase6 else PHASE5_CHUNKS
    records = load_records(phase6)
    known = _known_ids(chunks_path)
    retriever = StreamingRetriever(chunks_path=str(chunks_path))
    # MultiIntentRetriever is constructed only when needed (full mode).
    multi = (
        None
        if no_multi_intent
        else MultiIntentRetriever(
            chunks_path=str(chunks_path), retriever=_RetrieverAdapter(retriever)
        )
    )

    r1, r3, r5, rr, lat = [], [], [], [], []
    m_cov, m_pass = [], []
    single_n = multi_n = 0

    for rec in records:
        rid = rec.get("id", rec.get("query_id", "?")) if isinstance(rec, dict) else "?"
        if is_multi(rec):
            multi_n += 1
            groups = [_collect_ids(g, known) for g in expected_groups(rec)]

            if no_multi_intent:
                # --no-multi-intent: issue ONE query (the full original text) through
                # the plain StreamingRetriever, then score each intent's gold chunks
                # against the combined result pool.  Same top-K / ranking parameters
                # as any single-intent query; MultiIntentRetriever is NOT used.
                original_query = _first(rec, QUERY_KEYS, "")
                t0 = time.perf_counter()
                res = retriever.retrieve(original_query)
                lat.append((time.perf_counter() - t0) * 1000)
                pool = set(_collect_ids(res, known))
                hits = [bool(g) and any(c in pool for c in g) for g in groups]
            else:
                # Full multi-intent path: parallel sub-queries via MultiIntentRetriever.
                t0 = time.perf_counter()
                res = asyncio.run(multi.retrieve(subquery_texts(rec), f"eval-{rid}"))
                lat.append((time.perf_counter() - t0) * 1000)
                subs = res.get("subqueries") if isinstance(res, dict) else None
                hits = []
                for i, g in enumerate(groups):
                    # Score each intent against its own subquery results when available.
                    src = subs[i] if isinstance(subs, list) and i < len(subs) else res
                    pool = set(_collect_ids(src, known))
                    hits.append(bool(g) and any(c in pool for c in g))

            m_cov.append(sum(hits) / len(hits) if hits else 0.0)
            m_pass.append(1.0 if hits and all(hits) else 0.0)
            continue

        single_n += 1
        t0 = time.perf_counter()
        res = retriever.retrieve(_first(rec, QUERY_KEYS, ""))
        lat.append((time.perf_counter() - t0) * 1000)
        got = _ids(res)
        rel = _labels(rec)
        r1.append(_call_metric(recall_at_k, got, rel, 1))
        r3.append(_call_metric(recall_at_k, got, rel, 3))
        r5.append(_call_metric(recall_at_k, got, rel, 5))
        rr.append(_call_metric(reciprocal_rank, got, rel))

    return {
        "corpus": chunks_path.name,
        "records": len(records),
        "multi_intent_mode": "no_multi_intent" if no_multi_intent else "full",
        "single_intent": single_n,
        "multi_intent": multi_n,
        "Recall@1": mean(r1),
        "Recall@3": mean(r3),
        "Recall@5": mean(r5),
        "MRR": mean(rr),
        "mean_latency_ms": mean(lat),
        "multi_intent_intent_coverage": mean(m_cov) if m_cov else None,
        "multi_intent_query_pass_rate": mean(m_pass) if m_pass else None,
    }


# ---------------------------------------------------------------------------
# LIVE evaluation (needs Ollama). Opt-in via --live; never run by default.
# ---------------------------------------------------------------------------
# One cold COMMIT per Phase 6 query on a fresh session: retrieval + selection
# + gating + real Ollama generation + citation validation. Early (partial-
# transcript) retrieval is not exercised, so TTFT here is the pipeline's
# cold-commit TTFT, not the reuse-optimised demo path. The harness only reads
# events the orchestrator already emits; it changes no pipeline behaviour.

WARMUP_PROMPT = "Reply with the single word OK."


def _allowed_citations(chunk_ids, citation_of):
    return [citation_of[c] for c in chunk_ids or [] if c in citation_of]


def _score_groundedness(started, completed, citation_of):
    """Score the final answer using only evidence the pipeline already exposes."""
    answer = completed.get("answer", "")
    if completed.get("is_multi_intent"):
        intents = [
            {
                "intent_id": i.get("intent_id"),
                "supported": bool(i.get("supported")),
                "allowed_citations": _allowed_citations(i.get("chunk_ids"), citation_of),
            }
            for i in completed.get("intents", [])
        ]
        return answer_groundedness(answer, intents=intents)
    supported = (started or {}).get("supported_chunks", [])
    return answer_groundedness(answer, _allowed_citations(supported, citation_of))


async def run_live_turns(orchestrator, records, citation_of, session_factory, retrieval_counter):
    """Run each record as one cold commit; return one turn record per query."""
    turns = []
    for rec in records:
        rid = rec.get("id", rec.get("query_id", "?")) if isinstance(rec, dict) else "?"
        text = _first(rec, QUERY_KEYS, "")
        session = session_factory(f"live-eval-{rid}")
        retrieval_counter[0] = 0
        started = completed = refused = None
        ttft_ms = pre_generation_ms = None

        t0 = time.perf_counter()
        async for event in orchestrator.process_commit(session, text):
            elapsed = (time.perf_counter() - t0) * 1000.0
            kind = event.get("event")
            if kind == "answer_started":
                started, pre_generation_ms = event, elapsed
            elif kind == "answer_token" and ttft_ms is None:
                ttft_ms = elapsed
            elif kind == "answer_completed":
                completed = event
            elif kind == "uncertainty_emitted":
                refused = event
        server_ms = (time.perf_counter() - t0) * 1000.0

        llm = (completed or {}).get("metrics") or {}
        generated = ttft_ms is not None
        turn = {
            "id": rid,
            "query": text,
            "multi_intent": bool(is_multi(rec)),
            "outcome": "answered" if completed else ("refused" if refused else "no_answer"),
            "generated": generated,
            "citation_valid": (completed or {}).get("citation_valid"),
            "ttft_ms": ttft_ms,
            # Refusals stop before generation, so all of their time is pre-generation.
            "pre_generation_ms": pre_generation_ms if pre_generation_ms is not None else server_ms,
            "llm_ttft_ms": llm.get("ttft_ms") if generated else None,
            "llm_generation_ms": llm.get("latency_ms") if generated else None,
            "server_processing_ms": server_ms,
            "retrieval_calls": retrieval_counter[0],
            "prompt_tokens": llm.get("prompt_tokens"),
            "completion_tokens": llm.get("completion_tokens"),
            "total_tokens": llm.get("total_tokens"),
            "answer": (completed or refused or {}).get("answer"),
            "groundedness": _score_groundedness(started, completed, citation_of) if completed else None,
        }
        turns.append(turn)
    return turns


def _report_live(turns, model) -> dict:
    scored = [t["groundedness"] for t in turns if t["groundedness"]]
    single = [t["groundedness"] for t in turns if t["groundedness"] and not t["multi_intent"]]
    multi = [t["groundedness"] for t in turns if t["groundedness"] and t["multi_intent"]]
    return {
        "available": True,
        "model": model,
        "turns": len(turns),
        "answered": sum(t["outcome"] == "answered" for t in turns),
        "refused": sum(t["outcome"] == "refused" for t in turns),
        "validator_rejections": sum(t["citation_valid"] is False for t in turns),
        "summary": summarize_live_turns(turns, model),
        "groundedness": aggregate_groundedness(scored),
        "groundedness_single_intent": aggregate_groundedness(single),
        "groundedness_multi_intent": aggregate_groundedness(multi),
        "per_turn": turns,
    }


def evaluate_live(records, chunks_path) -> dict:
    """Run the live pass. Returns {"available": False, "reason": ...} if Ollama is unreachable."""
    from backend.app.models.session import SessionState
    from backend.app.orchestration.streaming_rag_orchestrator import StreamingRagOrchestrator
    from backend.app.retrieval.chunk_loader import load_chunks

    orchestrator = StreamingRagOrchestrator(chunks_path=str(chunks_path))
    llm = orchestrator.llm_client

    try:  # untimed warm-up so model load time does not distort the first turn
        llm.generate(WARMUP_PROMPT)
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    # Count retrieve() invocations at the lowest level (one per query/intent).
    counter = [0]
    inner = orchestrator.async_retriever.retrieve

    async def counted(*args, **kwargs):
        counter[0] += 1
        return await inner(*args, **kwargs)

    orchestrator.async_retriever.retrieve = counted

    # Same loader as the pipeline, so citation strings match what was generated.
    citation_of = {
        c["chunk_id"]: canonical_citation(c["doc_id"], c["section"])
        for c in load_chunks(chunks_path)
    }

    turns = asyncio.run(
        run_live_turns(
            orchestrator, records, citation_of,
            lambda sid: SessionState(session_id=sid), counter,
        )
    )
    return _report_live(turns, llm.model)


def _fmt(value, digits=4):
    if value is None:
        return "unavailable"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _fmt_stats(stats, unit=""):
    if not stats:
        return "unavailable"
    return (f"mean {stats['mean']:.1f}{unit} | median {stats['median']:.1f}{unit} | "
            f"max {stats['max']:.1f}{unit} | n={stats['n']}")


def _print_groundedness(label, g):
    print(f"    {label}: micro {_fmt(g['groundedness_micro'])} | macro {_fmt(g['groundedness_macro'])} | "
          f"claims {g['supported_claims']}/{g['total_claims']} | turns scored {g['turns_scored']} | "
          f"invalid citations {g['invalid_citation_count']}")


def print_live_report(live) -> None:
    print()
    print("LIVE (real Ollama generation; cold commit per query, no early retrieval)")
    if not live.get("available"):
        print(f"  unavailable: {live.get('reason')}")
        return
    s = live["summary"]
    lat, cost = s["latency_ms"], s["cost_per_turn"]
    print(f"  model: {live['model']} (untimed warm-up call excluded)")
    print(f"  turns: {live['turns']} | answered: {live['answered']} | refused: {live['refused']} | "
          f"citation-validator rejections: {live['validator_rejections']}")
    print("  latency:")
    print(f"    TTFT, pipeline (commit -> first token): {_fmt_stats(lat['ttft_pipeline'], ' ms')}")
    print(f"    TTFT, LLM only (Ollama request -> first token): {_fmt_stats(lat['ttft_llm_only'], ' ms')}")
    print(f"    pre-generation (retrieval + select + gate): {_fmt_stats(lat['pre_generation'], ' ms')}")
    print(f"    LLM generation latency: {_fmt_stats(lat['llm_generation'], ' ms')}")
    print(f"    server processing per turn: {_fmt_stats(lat['server_processing'], ' ms')}")
    print("  groundedness (citation/evidence proxy; NOT semantic factual truth):")
    _print_groundedness("all      ", live["groundedness"])
    _print_groundedness("single   ", live["groundedness_single_intent"])
    _print_groundedness("multi    ", live["groundedness_multi_intent"])
    print("  cost per turn:")
    print(f"    cloud/API: ${cost['cloud_api_cost_usd_per_turn']:.2f} / INR {cost['cloud_api_cost_inr_per_turn']:.2f}")
    print(f"    basis: {cost['cost_basis']}")
    tok = cost["token_usage"]
    if tok["status"] == "available":
        print(f"    tokens/turn (Ollama-reported, {tok['turns_with_token_counts']} turns): "
              f"prompt {tok['mean_prompt_tokens']:.1f} | completion {tok['mean_completion_tokens']:.1f} | "
              f"total {tok['mean_total_tokens']:.1f}")
    else:
        print("    tokens/turn: unavailable (Ollama returned no token counts)")
    rc = cost["mean_retrieval_calls_per_turn"]
    print(f"    retrieval calls/turn: {_fmt(rc['mean'], 2) if rc else 'unavailable'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 5/6 retrieval evaluation")
    ap.add_argument("--phase6", action="store_true", help="use the Phase 6 corpus and labels")
    ap.add_argument("--offline-only", action="store_true", help="retrieval only; never calls Ollama")
    ap.add_argument("--live", action="store_true",
                    help="also run the live pass (needs Ollama): TTFT, latency, groundedness, tokens, cost per turn")
    ap.add_argument(
        "--no-multi-intent",
        action="store_true",
        dest="no_multi_intent",
        help=(
            "Ablation: for multi-intent Phase 6 queries, retrieve with plain StreamingRetriever "
            "on the full original query instead of MultiIntentRetriever. "
            "Gold labels and all metrics are unchanged. Single-intent rows are unaffected."
        ),
    )
    ap.add_argument("--json", metavar="PATH", help="also write the full report as JSON")
    args = ap.parse_args()

    if args.live and args.offline_only:
        ap.error("--live and --offline-only are mutually exclusive")

    try:
        report = evaluate(args.phase6, no_multi_intent=args.no_multi_intent)
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"Evaluation failed: {type(exc).__name__}: {exc}")
        return 1

    mode_tag = f"  [multi_intent_mode: {report['multi_intent_mode']}]"
    print(f"Phase {'6' if args.phase6 else '5'} evaluation (offline retrieval only){mode_tag}")
    for k, v in report.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("  note: mean_latency_ms is offline retrieval latency, not TTFT")
    print("  groundedness: not measurable offline (needs generated answers) - see LIVE")

    live = None
    if args.live:
        chunks_path = PHASE6_CHUNKS if args.phase6 else PHASE5_CHUNKS
        try:
            live = evaluate_live(load_records(args.phase6), chunks_path)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            live = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
        print_live_report(live)
    else:
        print()
        print("LIVE: not run (TTFT, generation latency, server latency, cost per turn, tokens, "
              "groundedness). Re-run with --live; requires Ollama.")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"offline": report, "live": live}, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\nJSON report written to {args.json}")

    return 2 if live is not None and not live.get("available") else 0


if __name__ == "__main__":
    raise SystemExit(main())