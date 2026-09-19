"""Phase 5 / Phase 6 retrieval evaluation runner (offline, no Ollama).

Usage:
    .venv\\Scripts\\python -m scripts.evaluate_phase5
    .venv\\Scripts\\python -m scripts.evaluate_phase5 --phase6 --offline-only
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
from evaluation.metrics import mean, recall_at_k, reciprocal_rank

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


def evaluate(phase6: bool) -> dict:
    chunks_path = PHASE6_CHUNKS if phase6 else PHASE5_CHUNKS
    records = load_records(phase6)
    known = _known_ids(chunks_path)
    retriever = StreamingRetriever(chunks_path=str(chunks_path))
    multi = MultiIntentRetriever(
        chunks_path=str(chunks_path), retriever=_RetrieverAdapter(retriever)
    )

    r1, r3, r5, rr, lat = [], [], [], [], []
    m_cov, m_pass = [], []
    single_n = multi_n = 0

    for rec in records:
        rid = rec.get("id", rec.get("query_id", "?")) if isinstance(rec, dict) else "?"
        if is_multi(rec):
            multi_n += 1
            t0 = time.perf_counter()
            res = asyncio.run(multi.retrieve(subquery_texts(rec), f"eval-{rid}"))
            lat.append((time.perf_counter() - t0) * 1000)
            groups = [_collect_ids(g, known) for g in expected_groups(rec)]
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
        "corpus": chunks_path.name, "records": len(records),
        "single_intent": single_n, "multi_intent": multi_n,
        "Recall@1": mean(r1), "Recall@3": mean(r3), "Recall@5": mean(r5),
        "MRR": mean(rr), "mean_latency_ms": mean(lat),
        "multi_intent_intent_coverage": mean(m_cov) if m_cov else None,
        "multi_intent_query_pass_rate": mean(m_pass) if m_pass else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 5/6 retrieval evaluation")
    ap.add_argument("--phase6", action="store_true", help="use the Phase 6 corpus and labels")
    ap.add_argument("--offline-only", action="store_true", help="retrieval only; never calls Ollama")
    args = ap.parse_args()

    try:
        report = evaluate(args.phase6)
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"Evaluation failed: {type(exc).__name__}: {exc}")
        return 1

    print(f"Phase {'6' if args.phase6 else '5'} evaluation (offline retrieval only)")
    for k, v in report.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())