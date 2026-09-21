"""Controlled generation-model comparison on the Phase 6 live evaluation.

EVALUATION ONLY. Nothing in the production pipeline is modified. The one
experimental variable is the Ollama model handed to the existing
StreamingRagOrchestrator (its existing `model=` constructor parameter, the same
object path `evaluate_phase5.evaluate_live()` uses). Every query is run through
the existing `evaluate_phase5.run_live_turns()` (one cold commit per query, the
Phase 6 corpus, the Phase 6 queries in file order), so TTFT, latency, tokens and
groundedness come from the existing telemetry/metrics, never from a competing
measurement. Unavailable values are written as null, never estimated.

Usage (from the project root, inside the project virtualenv):

    .venv\\Scripts\\python -m scripts.benchmark_models
    .venv\\Scripts\\python -m scripts.benchmark_models --report-only     # rebuild the .md from the .json

Outputs (existing reports are never overwritten; pass --force to replace these two):
    benchmark/model_comparison_results.json   (written incrementally, after every query)
    benchmark/model_comparison_report.md
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import inspect
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests

from backend.app.retrieval.chunk_loader import load_chunks
from evaluation.metrics import aggregate_groundedness, canonical_citation
from scripts import evaluate_phase5 as ev

DEFAULT_MODELS = ["llama3.2:3b", "qwen2.5:3b", "gemma3:4b", "qwen3:4b", "qwen3:8b"]
PRODUCTION_MODEL = "llama3.2:3b"
BASE_URL = "http://localhost:11434"
OUT_JSON = ROOT / "benchmark" / "model_comparison_results.json"
OUT_MD = ROOT / "benchmark" / "model_comparison_report.md"

# Files/dirs that must be byte-identical before and after the benchmark.
PROTECTED = ["backend", "demo", "evaluation", "data", "requirements.txt", "scripts/evaluate_phase5.py"]
SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", "telemetry"}


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stats(values: list) -> dict | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return {
        "n": len(present),
        "min": min(present),
        "mean": statistics.fmean(present),
        "median": statistics.median(present),
        "max": max(present),
    }


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def snapshot() -> dict[str, str]:
    """sha256 of every protected file (relative posix path -> digest)."""
    out: dict[str, str] = {}
    for entry in PROTECTED:
        base = ROOT / entry
        files = [base] if base.is_file() else (
            [p for p in base.rglob("*") if p.is_file()] if base.is_dir() else []
        )
        for p in files:
            rel = p.relative_to(ROOT)
            if any(part in SKIP_DIRS for part in rel.parts) or p.suffix == ".pyc":
                continue
            out[rel.as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def snapshot_diff(before: dict, after: dict) -> dict:
    return {
        "added": sorted(set(after) - set(before)),
        "removed": sorted(set(before) - set(after)),
        "modified": sorted(k for k in before if k in after and before[k] != after[k]),
    }


def total_ram_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        pass
    try:  # Windows
        import ctypes

        class MEMSTAT(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MEMSTAT()
        m.dwLength = ctypes.sizeof(MEMSTAT)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return int(m.ullTotalPhys)
    except Exception:
        return None


# --------------------------------------------------------------------------
# Ollama administration (harness only: list / pre-load / unload / inspect)
# --------------------------------------------------------------------------
def ollama_get(base: str, path: str, timeout: float = 15.0) -> dict:
    r = requests.get(base + path, timeout=timeout)
    r.raise_for_status()
    return r.json()


def list_models(base: str) -> dict[str, dict]:
    return {m["name"]: m for m in ollama_get(base, "/api/tags").get("models", [])}


def preload(base: str, model: str, timeout: float) -> None:
    """Untimed model load with a generous timeout, so a cold 8B load is not
    mistaken for a generation failure. The measured runs use the unmodified client."""
    r = requests.post(base + "/api/generate", timeout=timeout,
                      json={"model": model, "prompt": "", "stream": False, "keep_alive": "30m"})
    r.raise_for_status()


def unload(base: str, model: str) -> None:
    try:
        requests.post(base + "/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)
    except Exception:
        pass


def loaded_info(base: str, model: str) -> dict | None:
    try:
        for m in ollama_get(base, "/api/ps").get("models", []):
            if model in (m.get("name"), m.get("model")):
                return {k: m.get(k) for k in ("name", "size", "size_vram", "context_length")}
    except Exception:
        pass
    return None


def model_dependence_scan() -> dict:
    """Which backend files reference Ollama / an LLM client at all."""
    hits: dict[str, list[str]] = {}
    for p in sorted((ROOT / "backend").rglob("*.py")):
        text = "\n".join(  # comments are not code paths
            ln for ln in p.read_text(encoding="utf-8", errors="ignore").lower().splitlines()
            if not ln.lstrip().startswith("#")
        )
        found = [k for k in ("ollamaclient", "llm_client", "/api/generate", "requests.post") if k in text]
        if found:
            hits[p.relative_to(ROOT).as_posix()] = found
    return hits


# --------------------------------------------------------------------------
# running queries through the EXISTING live-evaluation path
# --------------------------------------------------------------------------
class _Observer:
    """Transparent wrapper: forwards every orchestrator event unchanged and
    remembers them so controller/reuse/evidence fields can be recorded."""

    def __init__(self, inner):
        self.inner = inner
        self.events: list[dict] = []
        self.tokens: list[str] = []

    async def process_commit(self, session, text):
        self.events, self.tokens = [], []
        async for event in self.inner.process_commit(session, text):
            if event.get("event") == "answer_token":
                self.tokens.append(event.get("token", ""))
            else:
                self.events.append(event)
            yield event


def make_orchestrator(model: str, chunks_path: Path):
    """Exactly what evaluate_live() builds, with only `model` varied."""
    from backend.app.orchestration.streaming_rag_orchestrator import StreamingRagOrchestrator

    return StreamingRagOrchestrator(chunks_path=str(chunks_path), model=model)


def classify_error(exc: BaseException) -> str:
    infra = (requests.exceptions.RequestException, ConnectionError, TimeoutError, OSError)
    return "infrastructure" if isinstance(exc, infra) else "pipeline_exception"


def event_facts(events: list[dict]) -> dict:
    def last(kind):
        return next((e for e in reversed(events) if e.get("event") == kind), None)

    ctrl, started = last("controller_decision"), last("answer_started")
    refused, completed = last("uncertainty_emitted"), last("answer_completed")
    final = completed or refused or started or {}
    shape = started or refused or {}
    intents = [
        {k: i.get(k) for k in ("intent_id", "query", "supported", "reason", "retrieved", "top_score", "chunk_ids")}
        for i in (shape.get("intents") or []) if isinstance(i, dict)
    ]
    chunk_ids = set((started or {}).get("supported_chunks") or [])
    for i in intents:
        chunk_ids.update(i.get("chunk_ids") or [])
    return {
        "controller_action": ctrl.get("action") if ctrl else None,
        "commit_action": shape.get("action"),
        "refinement_type": final.get("refinement_type"),
        "retrieval_reuse_mode": final.get("retrieval_reuse_mode"),
        "retrieval_reuse_reason": final.get("retrieval_reuse_reason"),
        "retrieval_calls_this_turn": final.get("retrieval_calls_this_turn"),
        "multi_intent_detected": shape.get("is_multi_intent"),
        "refusal_reason": (refused or {}).get("reason"),
        "intents": intents,
        "evidence_chunk_ids": sorted(chunk_ids),
    }


def query_record(model: str, rec: dict, turn: dict | None, obs: _Observer,
                 error: dict | None, elapsed_ms: float) -> dict:
    facts = event_facts(obs.events)
    g = (turn or {}).get("groundedness")
    outcome = (turn or {}).get("outcome") if turn else "error"
    return {
        "model": model,
        "query_id": ev._first(rec, ("id", "query_id"), "?"),
        "query": ev._first(rec, ev.QUERY_KEYS, ""),
        "query_type": "multi_intent" if ev.is_multi(rec) else "single_intent",
        "status": outcome,
        "error": error,
        "elapsed_ms_measured_by_harness": round(elapsed_ms, 1),
        **facts,
        "retrieval_calls": (turn or {}).get("retrieval_calls"),
        "generated": (turn or {}).get("generated", False),
        "ttft_pipeline_ms": (turn or {}).get("ttft_ms"),
        "ttft_llm_only_ms": (turn or {}).get("llm_ttft_ms"),
        "llm_generation_ms": (turn or {}).get("llm_generation_ms"),
        "pre_generation_ms": (turn or {}).get("pre_generation_ms"),
        "server_ms": (turn or {}).get("server_processing_ms"),
        "prompt_tokens": (turn or {}).get("prompt_tokens"),
        "completion_tokens": (turn or {}).get("completion_tokens"),
        "total_tokens": (turn or {}).get("total_tokens"),
        "citation_valid": (turn or {}).get("citation_valid"),
        "groundedness": g["groundedness"] if g else None,
        "supported_claims": g["supported_claims"] if g else None,
        "total_claims": g["total_claims"] if g else None,
        "invalid_citations": g["invalid_citations"] if g else None,
        "invalid_citation_count": len(g["invalid_citations"]) if g else None,
        "abstention_units": g["abstention_units"] if g else None,
        "groundedness_detail": g,
        "refused": outcome == "refused",
        "answer": (turn or {}).get("answer") if turn else None,
        "partial_answer_before_error": "".join(obs.tokens) if error else None,
    }


async def run_one(observer, rec, citation_of, counter):
    from backend.app.models.session import SessionState

    turns = await ev.run_live_turns(observer, [rec], citation_of, lambda sid: SessionState(session_id=sid), counter)
    return turns[0]


def summarize_model(recs: list[dict]) -> dict:
    ok = [r for r in recs if r["error"] is None]
    gen = [r for r in ok if r["generated"]]
    scored = [r["groundedness_detail"] for r in ok if r["groundedness_detail"]]
    single = [r["groundedness_detail"] for r in ok if r["groundedness_detail"] and r["query_type"] == "single_intent"]
    multi = [r["groundedness_detail"] for r in ok if r["groundedness_detail"] and r["query_type"] == "multi_intent"]
    return {
        "queries_attempted": len(recs),
        "queries_completed": len(ok),
        "queries_errored": len(recs) - len(ok),
        "answered": sum(r["status"] == "answered" for r in ok),
        "refused": sum(r["status"] == "refused" for r in ok),
        "no_answer": sum(r["status"] == "no_answer" for r in ok),
        "validator_rejections": sum(r["citation_valid"] is False for r in ok),
        "latency_ms": {
            "ttft_pipeline": stats([r["ttft_pipeline_ms"] for r in gen]),
            "ttft_llm_only": stats([r["ttft_llm_only_ms"] for r in gen]),
            "llm_generation": stats([r["llm_generation_ms"] for r in gen]),
            "pre_generation": stats([r["pre_generation_ms"] for r in ok]),
            "server": stats([r["server_ms"] for r in ok]),
        },
        "tokens": {k: stats([r[f"{k}_tokens"] for r in ok])
                   for k in ("prompt", "completion", "total")},
        "groundedness_all": aggregate_groundedness(scored),
        "groundedness_single_intent": aggregate_groundedness(single),
        "groundedness_multi_intent": aggregate_groundedness(multi),
        "invalid_citation_count": sum(r["invalid_citation_count"] or 0 for r in ok),
        "whole_query_refusals": [r["query_id"] for r in ok if r["status"] == "refused"],
        "queries_with_intent_level_abstention": [r["query_id"] for r in ok if (r["abstention_units"] or 0) > 0],
        "errored_query_ids": [r["query_id"] for r in recs if r["error"] is not None],
    }


async def benchmark_model(model, records, citation_of, orch, args, m: dict, save) -> None:
    counter = [0]
    inner = orch.async_retriever.retrieve

    async def counted(*a, **k):  # same counting hook evaluate_live() installs
        counter[0] += 1
        return await inner(*a, **k)

    orch.async_retriever.retrieve = counted
    obs = _Observer(orch)

    # ---- smoke test through the real pipeline (first Phase 6 query) ----
    t0 = time.perf_counter()
    try:
        turn = await run_one(obs, records[0], citation_of, counter)
        ok = bool(turn.get("generated"))
        m["smoke"] = {"query_id": turn["id"], "passed": ok, "outcome": turn["outcome"],
                      "ms": round((time.perf_counter() - t0) * 1000, 1),
                      "error": None if ok else "pipeline ran but no answer tokens were generated"}
    except Exception as exc:
        m["smoke"] = {"query_id": ev._first(records[0], ("id", "query_id"), "?"), "passed": False,
                      "ms": round((time.perf_counter() - t0) * 1000, 1),
                      "error": {"kind": classify_error(exc), "type": type(exc).__name__, "message": str(exc)[:500]}}
    print(f"  smoke test ({m['smoke']['query_id']}): {'OK' if m['smoke']['passed'] else 'FAILED'} "
          f"{m['smoke']['error'] or ''}", flush=True)
    save()
    if not m["smoke"]["passed"]:
        m["status"] = "smoke_failed"
        return

    # ---- full run, identical order for every model ----
    turns = []
    for n, rec in enumerate(records, 1):
        rid = ev._first(rec, ("id", "query_id"), "?")
        t0 = time.perf_counter()
        turn = error = None
        try:
            turn = await run_one(obs, rec, citation_of, counter)
        except Exception as exc:
            error = {"kind": classify_error(exc), "type": type(exc).__name__, "message": str(exc)[:500]}
        elapsed = (time.perf_counter() - t0) * 1000.0
        r = query_record(model, rec, turn, obs, error, elapsed)
        m["queries"].append(r)
        if turn:
            turns.append(turn)
        g = r["groundedness"]
        print(f"  query {rid} ({n}/{len(records)}) ... "
              + (f"ERROR[{error['kind']}] {error['type']}" if error else
                 f"{r['status']} | groundedness {'n/a' if g is None else f'{g:.2f}'} | "
                 f"ttft {'n/a' if r['ttft_pipeline_ms'] is None else str(round(r['ttft_pipeline_ms'])) + ' ms'}"),
              flush=True)
        save()
    m["summary"] = summarize_model(m["queries"])
    m["existing_report_summary"] = ev._report_live(turns, model) if turns else None
    if m["existing_report_summary"]:
        m["existing_report_summary"].pop("per_turn", None)
    m["status"] = "complete" if not m["summary"]["queries_errored"] else "complete_with_errors"


def run_model(model, idx, total, records, args, data, base) -> None:
    m = data["models"][model]
    save = lambda: write_json(args.output_json, data)  # noqa: E731
    print(f"\n[{idx}/{total}] {model}", flush=True)
    orch = None
    try:
        try:
            preload(base, model, args.load_timeout)
        except Exception as exc:
            m["status"], m["error"] = "preload_failed", {"type": type(exc).__name__, "message": str(exc)[:500]}
            print(f"  preload FAILED: {exc}", flush=True)
            return
        try:
            orch = make_orchestrator(model, ev.PHASE6_CHUNKS)
        except Exception as exc:
            m["status"], m["error"] = "pipeline_init_failed", {"type": type(exc).__name__, "message": str(exc)[:500]}
            print(f"  pipeline init FAILED: {exc}", flush=True)
            return
        client = orch.llm_client
        if args.client_timeout:
            client.timeout = args.client_timeout  # opt-in, recorded; default = production value
        m["client_options"] = {"model": client.model, "temperature": client.temperature,
                               "num_predict": client.num_predict, "think": client.think,
                               "timeout_s": client.timeout}
        try:  # same untimed warm-up evaluate_live() performs
            client.generate(ev.WARMUP_PROMPT)
        except Exception as exc:
            m["status"], m["error"] = "warmup_failed", {"type": type(exc).__name__, "message": str(exc)[:500]}
            print(f"  warm-up FAILED: {exc}", flush=True)
            return
        m["ollama_loaded"] = loaded_info(base, model)
        citation_of = {c["chunk_id"]: canonical_citation(c["doc_id"], c["section"])
                       for c in load_chunks(ev.PHASE6_CHUNKS)}
        asyncio.run(benchmark_model(model, records, citation_of, orch, args, m, save))
    finally:
        if not m.get("status"):
            m["status"] = "not_run"
        del orch
        gc.collect()
        unload(base, model)
        save()


# --------------------------------------------------------------------------
# cross-model checks
# --------------------------------------------------------------------------
def retrieval_consistency(data: dict) -> dict:
    """Retrieval/decomposition are LLM-free, so the evidence and intent split handed to
    generation must be identical across models. Verify rather than assume."""
    by_q: dict[str, dict[str, tuple]] = {}
    for name, m in data["models"].items():
        for r in m.get("queries", []):
            if r["evidence_chunk_ids"] or r["intents"]:
                sig = (tuple(r["evidence_chunk_ids"]),
                       tuple((i["intent_id"], i["query"], i["supported"]) for i in r["intents"]))
                by_q.setdefault(r["query_id"], {})[name] = sig
    mismatches = [{"query_id": q, "models": sorted(v)} for q, v in by_q.items() if len(set(v.values())) > 1]
    return {"queries_compared": len(by_q), "identical_for_all_models": not mismatches, "mismatches": mismatches}


# --------------------------------------------------------------------------
# Markdown report
# --------------------------------------------------------------------------
def _f(v, d=1, unit=""):
    if v is None:
        return "null"
    return f"{v:.{d}f}{unit}" if isinstance(v, float) else f"{v}{unit}"


def _srow(model, s, d=0):
    return (f"| {model} | {s['n']} | {_f(s['mean'], d)} | {_f(s['median'], d)} | {_f(s['min'], d)} | {_f(s['max'], d)} |"
            if s else f"| {model} | 0 | null | null | null | null |")


def _range_sentence(label, pairs, d=1, unit=""):
    pairs = [(m, v) for m, v in pairs if v is not None]
    if len(pairs) < 2:
        return None
    lo, hi = min(pairs, key=lambda p: p[1]), max(pairs, key=lambda p: p[1])
    if lo[1] == hi[1]:
        return f"- {label} was identical across models ({_f(lo[1], d, unit)})."
    return (f"- {label} ranged from {_f(lo[1], d, unit)} ({lo[0]}) to {_f(hi[1], d, unit)} ({hi[0]}) "
            f"across the models measured in this run.")


def write_report(data: dict, path: Path) -> None:
    models = data["models"]
    done = {n: m for n, m in models.items() if m.get("summary")}
    env, cfg = data["environment"], data["config"]
    L: list[str] = ["# Streaming-Live-RAG Model Comparison", "",
                    f"_Generated {data.get('finished_utc') or data['started_utc']} (UTC) by `scripts/benchmark_models.py`. "
                    "All numbers are copied from `benchmark/model_comparison_results.json`; this report ranks nothing "
                    "and declares no winner._", ""]

    L += ["## Experimental setup", "",
          f"- **Environment:** {env.get('platform')} | CPU: {env.get('processor') or 'n/a'} "
          f"({env.get('cpu_count')} logical cores) | RAM: "
          f"{_f(env['ram_gb'], 1, ' GB') if env.get('ram_gb') else 'null'} | Python {env.get('python')}",
          f"- **Ollama:** local instance at `{data['ollama']['base_url']}`, version {data['ollama'].get('version') or 'null'}; "
          "models run **sequentially** (one loaded at a time, unloaded afterwards).",
          f"- **Corpus:** `{cfg['chunks']}` ({cfg.get('n_chunks', 'n/a')} chunks, unchanged).",
          f"- **Phase 6 query set:** `{cfg['queries']}`, {cfg['n_queries']} queries in file order "
          f"({', '.join(cfg['query_ids'])}); unchanged. One cold commit per query, fresh session per query.",
          "- **Retrieval pipeline (identical for every model):** rule-based controller and multi-intent decomposition, "
          "BM25 + FAISS dense retrieval, RRF fusion, CrossEncoder reranking, evidence gate/selector, grounded generation "
          "prompt, citation validator, deterministic sentence attribution.",
          f"- **Ollama options (production configuration, unchanged):** temperature {cfg['ollama_options'].get('temperature')}, "
          f"num_predict {cfg['ollama_options'].get('num_predict')}, think={cfg['ollama_options'].get('think')}, "
          f"client timeout {cfg['ollama_options'].get('timeout_s')} s"
          + (f" (benchmark override: {cfg['client_timeout_override_s']} s)" if cfg.get("client_timeout_override_s") else "")
          + ".",
          f"- **Models tested:** {', '.join(f'`{n}`' for n in models)}.",
          f"- **Default production model:** `{data['production_default_model']['orchestrator_default']}` "
          "(unchanged by this benchmark).",
          "- **Model-dependent stages:** generation only. A scan of `backend/` found Ollama/LLM-client references only in: "
          + ", ".join(f"`{p}`" for p in data["model_dependence_check"]) + ". "
          "Controller, decomposition, retrieval, reranking, evidence selection and validation do not call the LLM.", ""]

    L += ["## Model configuration", "", "| Model | Size (on disk) | Parameters | Quantization | Role | Run status |", "|---|---|---|---|---|---|"]
    for n, m in models.items():
        gb = f"{m['size_bytes'] / 1e9:.2f} GB" if m.get("size_bytes") else "null"
        L.append(f"| {n} | {gb} | {m.get('parameter_size') or 'null'} | {m.get('quantization') or 'null'} | {m['role']} | {m.get('status')} |")
    L.append("")

    if not done:
        L += ["_No model completed a run; see Limitations._", ""]
    else:
        L += ["## Latency comparison", "",
              "Milliseconds, over queries that generated tokens (server latency: all completed queries). "
              "Values are Ollama/pipeline-reported; null = not available.", ""]
        for title, key in (("TTFT, pipeline (commit → first token)", "ttft_pipeline"),
                           ("TTFT, LLM only (Ollama request → first token)", "ttft_llm_only"),
                           ("LLM generation latency", "llm_generation"),
                           ("Pre-generation (retrieval + select + gate)", "pre_generation"),
                           ("Server processing per turn", "server")):
            L += [f"**{title}**", "", "| Model | n | mean | median | min | max |", "|---|---|---|---|---|---|"]
            L += [_srow(n, m["summary"]["latency_ms"][key]) for n, m in done.items()] + [""]

        L += ["## Token usage", "", "Ollama-reported counts (`prompt_eval_count` / `eval_count`); never estimated.", "",
              "| Model | turns with counts | mean prompt | mean completion | mean total |", "|---|---|---|---|---|"]
        for n, m in done.items():
            t = m["summary"]["tokens"]
            L.append(f"| {n} | {t['total']['n'] if t['total'] else 0} | {_f(t['prompt']['mean'] if t['prompt'] else None)} | "
                     f"{_f(t['completion']['mean'] if t['completion'] else None)} | {_f(t['total']['mean'] if t['total'] else None)} |")
        L.append("")

        L += ["## Grounding and citation behavior", "",
              "Groundedness is the project's citation/evidence proxy (`evaluation/metrics.py`), **not** semantic factual truth. "
              "Micro = supported claims / all claims; macro = mean of per-turn scores over turns with claims.", "",
              "| Model | micro | macro | claims supported/total | single-intent micro | multi-intent micro | invalid citations | validator rejections | whole-query refusals | queries with intent-level abstention | errored queries |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for n, m in done.items():
            s, a = m["summary"], m["summary"]["groundedness_all"]
            L.append(f"| {n} | {_f(a['groundedness_micro'], 3)} | {_f(a['groundedness_macro'], 3)} | {a['supported_claims']}/{a['total_claims']} | "
                     f"{_f(s['groundedness_single_intent']['groundedness_micro'], 3)} | {_f(s['groundedness_multi_intent']['groundedness_micro'], 3)} | "
                     f"{s['invalid_citation_count']} | {s['validator_rejections']} | {len(s['whole_query_refusals'])} | "
                     f"{len(s['queries_with_intent_level_abstention'])} | {s['queries_errored']} |")
        L += ["", "Refusal / abstention detail (the Phase 6 file carries no per-query 'should refuse' label, so these are observed behaviors, not scored expectations):", ""]
        for n, m in done.items():
            s = m["summary"]
            L.append(f"- `{n}`: whole-query refusals: {', '.join(s['whole_query_refusals']) or 'none'}; "
                     f"intent-level abstentions in: {', '.join(s['queries_with_intent_level_abstention']) or 'none'}.")
        L.append("")

    L += ["## Query-level results", "",
          "| Query | Model | Type | Status | Groundedness (supported/total) | Invalid cit. | Citation valid | Pipeline TTFT ms | LLM gen ms | Server ms | Error |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    qids = cfg["query_ids"]
    for q in qids:
        for n, m in models.items():
            r = next((x for x in m.get("queries", []) if x["query_id"] == q), None)
            if r is None:
                L.append(f"| {q} | {n} | — | not run | — | — | — | — | — | — | {m.get('status')} |")
                continue
            err = f"{r['error']['kind']}: {r['error']['type']}" if r["error"] else ""
            gs = "null" if r["groundedness"] is None else f"{r['groundedness']:.2f} ({r['supported_claims']}/{r['total_claims']})"
            L.append(f"| {q} | {n} | {r['query_type'].replace('_intent', '')} | {r['status']} | {gs} | {_f(r['invalid_citation_count'])} | "
                     f"{_f(r['citation_valid'])} | {_f(r['ttft_pipeline_ms'], 0)} | {_f(r['llm_generation_ms'], 0)} | {_f(r['server_ms'], 0)} | {err} |")
    L += ["", "Full answers, controller/reuse fields, retrieval calls, intent split and evidence IDs are in the JSON.", ""]

    L += ["## Observed trade-offs", ""]
    if len(done) >= 2:
        g = lambda f: [(n, f(m["summary"])) for n, m in done.items()]  # noqa: E731
        med = lambda k: (lambda s: (s["latency_ms"][k] or {}).get("median"))  # noqa: E731
        mean_t = lambda k: (lambda s: (s["tokens"][k] or {}).get("mean"))  # noqa: E731
        sents = [
            _range_sentence("Median LLM generation latency", g(med("llm_generation")), 0, " ms"),
            _range_sentence("Median pipeline TTFT", g(med("ttft_pipeline")), 0, " ms"),
            _range_sentence("Median server latency per turn", g(med("server")), 0, " ms"),
            _range_sentence("Mean completion tokens per turn", g(mean_t("completion")), 1),
            _range_sentence("Mean total tokens per turn", g(mean_t("total")), 1),
            _range_sentence("Groundedness (micro)", g(lambda s: s["groundedness_all"]["groundedness_micro"]), 3),
            _range_sentence("Groundedness (macro)", g(lambda s: s["groundedness_all"]["groundedness_macro"]), 3),
            _range_sentence("Total invalid citations", g(lambda s: s["invalid_citation_count"]), 0),
        ]
        L += [s for s in sents if s]
    else:
        L.append("- Fewer than two models completed, so no cross-model comparison is reported.")
    L += ["- Each line above describes one metric in this run only. Metrics are not combined, and no overall ordering is implied.", ""]

    rc = data.get("retrieval_consistency") or {}
    fails = {n: m for n, m in models.items() if m.get("status") not in ("complete",)}
    L += ["## Limitations", "",
          "- Single machine, local Ollama inference, one run per model: latency reflects this hardware and any concurrent load (CPU/RAM/GPU contention was not controlled beyond running models one at a time). Run-to-run variance was **not** measured.",
          f"- Small evaluation set ({cfg['n_queries']} queries); a single claim changes a model's micro score noticeably. Differences should not be read as statistically significant.",
          "- Groundedness is a citation/evidence proxy, not a semantic correctness judgement; answers were not human-graded.",
          "- Temperature 0 makes decoding near-deterministic but not guaranteed bit-identical across runs.",
          f"- **Fairness / model-dependent stages:** only generation depends on the Ollama model. Retrieval evidence and intent decomposition identical across models in this run: **{rc.get('identical_for_all_models')}** "
          f"({rc.get('queries_compared', 0)} queries compared" + (f"; mismatches: {rc['mismatches']}" if rc.get("mismatches") else "") + ").",
          "- The production client timeout (default 60 s) was left as-is unless overridden above; a slow model can therefore fail with a timeout, which is recorded as an infrastructure error, not as an incorrect RAG answer.",
          "- `think=False` is sent for every model exactly as production does; how each model family honors it was not altered."]
    if fails:
        L.append("- **Runs that were incomplete or had errors:**")
        for n, m in fails.items():
            L.append(f"  - `{n}`: status `{m.get('status')}`"
                     + (f", error `{m['error']['type']}: {m['error']['message'][:160]}`" if m.get("error") else "")
                     + (f", smoke: {m['smoke']['error']}" if m.get("smoke") and not m["smoke"]["passed"] else "")
                     + (f", errored queries: {', '.join(m['summary']['errored_query_ids'])}" if m.get("summary") and m["summary"]["errored_query_ids"] else ""))
    else:
        L.append("- No model run failed or errored.")
    L.append("")

    L += ["## Reproducibility", "", "```", "ollama serve", "# models must already be installed: " + " ".join(models),
          r".venv\Scripts\python -m scripts.benchmark_models          # full run (add --force to replace existing outputs)",
          r".venv\Scripts\python -m scripts.benchmark_models --report-only --force   # rebuild this report from the JSON",
          "```", "",
          f"Protected-file integrity (sha256 before/after): {json.dumps(data.get('file_integrity', {}).get('changes'))}.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Generation-model comparison on the Phase 6 live evaluation")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS), help="comma-separated Ollama model names")
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--output-json", type=Path, default=OUT_JSON)
    ap.add_argument("--output-md", type=Path, default=OUT_MD)
    ap.add_argument("--force", action="store_true", help="allow replacing the two output files")
    ap.add_argument("--report-only", action="store_true", help="rebuild the Markdown from the existing JSON")
    ap.add_argument("--load-timeout", type=float, default=900.0, help="harness-only pre-load timeout (s)")
    ap.add_argument("--client-timeout", type=float, default=None,
                    help="OPTIONAL override of the Ollama client timeout for this process only (default: production 60 s)")
    ap.add_argument("--smoke-only", action="store_true", help="verify models + smoke test each, skip the full run")
    args = ap.parse_args()

    if args.report_only:
        if args.output_md.exists() and not args.force:
            print(f"{args.output_md} exists; use --force to replace it.")
            return 2
        write_report(json.loads(args.output_json.read_text(encoding="utf-8")), args.output_md)
        print(f"wrote {args.output_md}")
        return 0

    for p in (args.output_json, args.output_md):
        if p.exists() and not args.force:
            print(f"{p} already exists; refusing to overwrite (use --force).")
            return 2

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    base = args.base_url.rstrip("/")

    # production default must be untouched
    from backend.app.llm.ollama_client import OllamaClient
    from backend.app.orchestration.streaming_rag_orchestrator import StreamingRagOrchestrator
    prod = {
        "orchestrator_default": inspect.signature(StreamingRagOrchestrator.__init__).parameters["model"].default,
        "ollama_client_default": inspect.signature(OllamaClient.__init__).parameters["model"].default,
    }
    if set(prod.values()) != {PRODUCTION_MODEL}:
        print(f"Production default is not {PRODUCTION_MODEL}: {prod}. Aborting.")
        return 2

    try:
        installed = list_models(base)
    except Exception as exc:
        print(f"Ollama is not reachable at {base}: {type(exc).__name__}: {exc}")
        return 2
    missing = [m for m in models if m not in installed]
    if missing:
        print(f"Models not installed in Ollama (this script never downloads): {missing}")
        return 2
    print(f"Ollama OK at {base}; all {len(models)} models installed: {', '.join(models)}")

    records = ev.load_records(True)
    ids = [ev._first(r, ("id", "query_id"), "?") for r in records]
    try:
        version = ollama_get(base, "/api/version").get("version")
    except Exception:
        version = None
    ram = total_ram_bytes()
    n_chunks = sum(1 for line in ev.PHASE6_CHUNKS.read_text(encoding="utf-8").splitlines() if line.strip())

    data: dict[str, Any] = {
        "schema_version": 1, "status": "running", "started_utc": utc_now(), "finished_utc": None,
        "environment": {"platform": platform.platform(), "machine": platform.machine(),
                        "processor": platform.processor(), "cpu_count": os.cpu_count(),
                        "ram_gb": round(ram / 1e9, 1) if ram else None, "python": platform.python_version()},
        "ollama": {"base_url": base, "version": version},
        "production_default_model": prod,
        "config": {"chunks": ev.PHASE6_CHUNKS.relative_to(ROOT).as_posix(), "n_chunks": n_chunks,
                   "queries": ev.PHASE6_QUERIES.relative_to(ROOT).as_posix(), "n_queries": len(records),
                   "query_ids": ids, "ollama_options": {}, "client_timeout_override_s": args.client_timeout,
                   "smoke_only": args.smoke_only},
        "model_dependence_check": model_dependence_scan(),
        "models": {},
    }
    for name in models:
        d = installed[name].get("details") or {}
        data["models"][name] = {
            "role": "Production default / demo model" if name == PRODUCTION_MODEL else "Comparison candidate",
            "size_bytes": installed[name].get("size"), "parameter_size": d.get("parameter_size"),
            "quantization": d.get("quantization_level"), "status": None, "smoke": None, "queries": [],
        }

    before = snapshot()
    write_json(args.output_json, data)
    run_records = records[:1] if args.smoke_only else records
    try:
        for i, name in enumerate(models, 1):
            run_model(name, i, len(models), run_records, args, data, base)
            opts = data["models"][name].get("client_options")
            if opts and not data["config"]["ollama_options"]:
                data["config"]["ollama_options"] = {k: v for k, v in opts.items() if k != "model"}
        data["status"] = "complete"
    except KeyboardInterrupt:
        data["status"] = "interrupted"
        print("\ninterrupted - partial results kept", flush=True)
    finally:
        data["finished_utc"] = utc_now()
        opts = [m.get("client_options") for m in data["models"].values() if m.get("client_options")]
        data["config"]["ollama_options_identical_across_models"] = all(
            {k: v for k, v in o.items() if k != "model"} == {k: v for k, v in opts[0].items() if k != "model"} for o in opts
        ) if opts else None
        data["retrieval_consistency"] = retrieval_consistency(data)
        after = snapshot()
        data["file_integrity"] = {"files_hashed": len(before), "changes": snapshot_diff(before, after)}
        write_json(args.output_json, data)
        write_report(data, args.output_md)

    changes = data["file_integrity"]["changes"]
    prod_mod = changes["modified"] + changes["removed"] + changes["added"]
    print("\nMODEL BENCHMARK COMPLETE\n\nModels:")
    for n in models:
        print(f"  * {n}  [{data['models'][n]['status']}]")
    print(f"\nFiles created:\n  * {args.output_json}\n  * {args.output_md}\n  * scripts/benchmark_models.py (if not already present)")
    print("\nFiles modified:\n  none by this script (benchmark outputs only)")
    print("\nProduction files modified (sha256 over backend/, demo/, evaluation/, data/, requirements.txt, evaluate_phase5.py):")
    print("  " + (", ".join(prod_mod) if prod_mod else "none"))
    print(f"\nDefault production model:\n  {prod['orchestrator_default']}")
    bad = {n: m["status"] for n, m in data["models"].items() if m["status"] != "complete"}
    print("\nModel-specific failures:\n  " + (json.dumps(bad) if bad else "none"))
    return 0 if data["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())