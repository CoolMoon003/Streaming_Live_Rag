"""
G6 telemetry tests. Deterministic: no Ollama, no Hugging Face, no network.

Drives the REAL StreamingRagOrchestrator (fixture retriever + stub LLM, exactly
as scripts/evaluate_session_refinement.py does) and the REAL server handlers /
JSONL writer, then checks the telemetry that comes out and the G6 evaluator
that reads it.

Usage:
    python -m scripts.test_g6_telemetry
    python -m scripts.test_g6_telemetry --emit-sample data/telemetry/g6_sample.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from backend.app.models.session import SessionState
from backend.app.orchestration.streaming_rag_orchestrator import StreamingRagOrchestrator
from scripts import evaluate_g6_telemetry as g6
from scripts.evaluate_session_refinement import (
    FIXTURE_CHUNKS,
    FixtureAsyncRetriever,
    Q_TRAVEL,
    Q_UNRELATED,
    StubLLMClient,
)

try:  # the server module needs fastapi; the rest of G6 does not
    import backend.app.api.server as server
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    server = None
    SERVER_IMPORT_ERROR = str(exc)

REPORTED = {
    "ttft_ms": 12.5, "latency_ms": 40.0,
    "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
}
UNREPORTED = {
    "ttft_ms": 12.5, "latency_ms": 40.0,
    "prompt_tokens": None, "completion_tokens": None, "total_tokens": None,
}

_results: list[tuple[str, bool]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail and not ok else ""))


def skip(name: str, why: str) -> None:
    print(f"  [SKIP] {name}: {why}")


class MetricsStubLLM(StubLLMClient):
    """Stub LLM that reports Ollama-shaped metrics (or none)."""

    def __init__(self, metrics: dict):
        super().__init__()
        self._metrics = metrics

    @staticmethod
    def _citations_in(prompt: str) -> list[str]:
        # Scan only the EVIDENCE section: the prompt's own format instruction
        # ("[DOC_ID §Section]") would otherwise be cited back as a fake source.
        return StubLLMClient._citations_in(prompt.split("EVIDENCE:", 1)[-1])

    def get_last_metrics(self) -> dict:
        return dict(self._metrics)


def make_orch(metrics: dict = REPORTED):
    retriever = FixtureAsyncRetriever()
    orch = StreamingRagOrchestrator(
        chunks_path="unused-fixture-path",
        async_retriever=retriever,
        llm_client=MetricsStubLLM(metrics),
    )
    return orch, retriever


class FakeWS:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))


async def drain(orch, session, text) -> list[dict]:
    return [e async for e in orch.process_commit(session, text)]


async def run_server_steps(steps, path: Path, metrics: dict = REPORTED, session_id: str = "g6-test"):
    """Run (kind, text) steps through the real server handlers; telemetry -> path."""
    server.TELEMETRY_PATH = path
    orch, _ = make_orch(metrics)
    session = SessionState(session_id=session_id)
    trail = server.TelemetryTrail(session_id)
    ws = FakeWS()
    for kind, text in steps:
        handler = server.handle_partial if kind == "partial" else server.handle_commit
        await handler(ws, orch, session, {}, text, trail)
    trail.emit("session_ended", source="server", reason="disconnect")
    return ws.sent


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


SCENARIO = [
    ("partial", "What are the"),                       # controller WAIT
    ("partial", Q_TRAVEL),                             # early retrieval
    ("commit", Q_TRAVEL),                              # exact reuse -> LLM
    ("partial", "Repeat that in two bullets"),         # controller SUPPRESS
    ("commit", "Repeat that in two bullets"),          # presentation, no retrieval
    ("commit", Q_UNRELATED),                           # new retrieval -> refusal
]


# =============================================================================
# TESTS
# =============================================================================

def test_session_defaults() -> None:
    print("\nsession telemetry defaults")
    a, b = SessionState("a"), SessionState(session_id="b")
    check("token_metrics defaults to empty dict", a.token_metrics == {})
    a.token_metrics["x"] = 1
    check("token_metrics is not shared between sessions", b.token_metrics == {})
    check("existing constructor / fields unchanged",
          a.session_id == "a" and a.query_version == 0 and a.latest_answer == "")


async def test_trace_envelope() -> None:
    print("\ntrace envelope")
    orch, _ = make_orch()
    session = SessionState(session_id="env")

    partial = await orch.process_partial(session, Q_TRAVEL)
    check("partial event carries envelope",
          partial["session_id"] == "env" and partial["stage"] == "partial"
          and isinstance(partial["query_version"], int)
          and isinstance(partial["generation_id"], int)
          and isinstance(partial["answer_version"], int)
          and partial["retrieval_calls_this_turn"] == 1
          and partial["retrieval_reuse_mode"] == "early_retrieval")
    check("timestamp is ISO-8601", datetime.fromisoformat(partial["timestamp"]) is not None)

    events = await drain(orch, session, Q_TRAVEL)
    kinds = [e["event"] for e in events]
    check("commit emitted started/token/completed",
          kinds[0] == "answer_started" and "answer_token" in kinds and kinds[-1] == "answer_completed",
          str(kinds))
    check("every commit event has session_id/timestamp/stage/lineage",
          all(e["session_id"] == "env" and e["stage"] == "commit" and e["timestamp"]
              and isinstance(e["query_version"], int) and isinstance(e["generation_id"], int)
              and isinstance(e["answer_version"], int) for e in events))
    check("event names and payloads preserved",
          events[-1]["action"] == "ANSWER" and events[-1]["answer"] and "metrics" in events[-1])

    original = {"event": "x", "generation_id": 99, "query_version": 7}
    snapshot = copy.deepcopy(original)
    wrapped = orch._trace_envelope(session, original, stage="commit")
    check("envelope never overwrites fields the event already has",
          wrapped["generation_id"] == 99 and wrapped["query_version"] == 7)
    check("envelope does not mutate its input", original == snapshot)

    blob = json.dumps(events, default=str)
    check("no evidence text leaks through IDs-only envelope fields",
          all("supported_chunks" in e or e["event"] != "answer_started" for e in events)
          and FIXTURE_CHUNKS[0]["chunk"]["text"] not in json.dumps(
              [{k: v for k, v in e.items() if k not in ("answer", "token")} for e in events]))


async def test_token_metrics() -> None:
    print("\ntoken metrics propagation")
    orch, _ = make_orch(REPORTED)
    session = SessionState(session_id="tok")
    await orch.process_partial(session, Q_TRAVEL)
    done = (await drain(orch, session, Q_TRAVEL))[-1]
    check("answer is grounded and citation-valid (harness sanity)",
          done["citation_valid"] is True and len(done["citations"]) >= 1, str(done["citations"]))
    check("existing metrics object preserved", done["metrics"] == REPORTED)
    check("token_metrics carries Ollama-reported counts and timings",
          done["token_metrics"] == {
              "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
              "llm_ttft_ms": 12.5, "llm_generation_ms": 40.0})
    check("reason is 'reported'", done["token_metrics_reason"] == "reported")
    check("session.token_metrics mirrors the turn", session.token_metrics == done["token_metrics"])

    orch, _ = make_orch(UNREPORTED)
    session = SessionState(session_id="tok2")
    await orch.process_partial(session, Q_TRAVEL)
    done = (await drain(orch, session, Q_TRAVEL))[-1]
    check("unreported tokens stay None (never estimated)",
          done["token_metrics"]["prompt_tokens"] is None
          and done["token_metrics"]["completion_tokens"] is None
          and done["token_metrics"]["total_tokens"] is None)
    check("reason says Ollama did not report",
          done["token_metrics_reason"] == "ollama_did_not_report_tokens")

    orch, _ = make_orch(REPORTED)
    session = SessionState(session_id="tok3")
    refusal = (await drain(orch, session, Q_UNRELATED))[-1]
    check("refusal: token_metrics null + explicit no-LLM reason",
          refusal["event"] == "uncertainty_emitted" and refusal["token_metrics"] is None
          and refusal["token_metrics_reason"] == "no_llm_call_evidence_insufficient")
    check("refusal clears session.token_metrics", session.token_metrics == {})

    await orch.process_partial(session, Q_TRAVEL)
    await drain(orch, session, Q_TRAVEL)
    pres = (await drain(orch, session, "Repeat that in two bullets"))[-1]
    check("presentation: token_metrics null + explicit no-LLM reason",
          pres["event"] == "answer_completed" and pres["token_metrics"] is None
          and pres["token_metrics_reason"] == "no_llm_call_presentation_reuse")


async def test_reuse_telemetry() -> None:
    print("\nreuse decision telemetry")
    cases = [
        ("exact", [("partial", Q_TRAVEL)], Q_TRAVEL, ("exact", 0)),
        ("extension (no delta)", [("partial", Q_TRAVEL)], Q_TRAVEL + " for the whole team", ("extension", 0)),
        ("extension + delta", [("partial", Q_TRAVEL)], Q_TRAVEL + " and workshop catering arrangements",
         ("extension_delta", 1)),
        ("new", [], Q_TRAVEL, ("new", 1)),
        ("additive", [("partial", Q_TRAVEL), ("commit", Q_TRAVEL)],
         "Also what about reimbursement claims for these trips", ("additive", 1)),
        ("replacement", [("partial", Q_TRAVEL), ("commit", Q_TRAVEL)],
         "No, I meant reimbursement claims for business trips", ("replacement", 1)),
        ("presentation", [("partial", Q_TRAVEL), ("commit", Q_TRAVEL)],
         "Repeat that in two bullets", ("presentation_none", 0)),
    ]
    for label, setup, final_text, (mode, calls) in cases:
        orch, retriever = make_orch()
        session = SessionState(session_id="reuse")
        for kind, text in setup:
            if kind == "partial":
                await orch.process_partial(session, text)
            else:
                await drain(orch, session, text)
        retriever.call_log.clear()
        retriever.retriever.call_log.clear()
        events = await drain(orch, session, final_text)
        tagged = [e for e in events if e["event"] in ("answer_started", "answer_completed", "uncertainty_emitted")]
        check(f"{label}: mode={mode} calls={calls}",
              bool(tagged) and all(e["retrieval_reuse_mode"] == mode
                                   and e["retrieval_calls_this_turn"] == calls for e in tagged),
              str([(e["retrieval_reuse_mode"], e["retrieval_calls_this_turn"]) for e in tagged]))
        check(f"{label}: reported calls match real retriever calls",
              len(retriever.retriever.call_log) == calls,
              f"real={len(retriever.retriever.call_log)}")


def test_jsonl_writer() -> None:
    print("\nJSONL telemetry writer")
    if server is None:
        skip("writer tests", f"server module unavailable ({SERVER_IMPORT_ERROR})")
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nested" / "dir" / "live_events.jsonl"
        server.TELEMETRY_PATH = path

        class Odd:
            def __repr__(self) -> str:
                return "<odd>"

        ok1 = server._write_telemetry({"session_id": "w", "event": "a", "text": "caf\u00e9 \u2713",
                                       "obj": Odd(), "nan": float("nan"), "s": {1, 2}})
        ok2 = server._write_telemetry({"session_id": "w", "event": "b"})
        check("writer reports success and creates the directory", ok1 and ok2 and path.exists())
        raw = path.read_bytes()
        lines = raw.decode("utf-8").splitlines()
        check("one event per line, append mode", len(lines) == 2)
        check("UTF-8 kept as UTF-8 (not \\u escapes)", "caf\u00e9".encode("utf-8") in raw)
        rec = json.loads(lines[0])
        check("non-serialisable values coerced to plain JSON",
              rec["obj"] == "<odd>" and rec["nan"] is None and sorted(rec["s"]) == [1, 2])
        check("server timestamp added when absent", datetime.fromisoformat(rec["timestamp"]) is not None)
        check("caller session_id kept", rec["session_id"] == "w")

        blocker = Path(tmp) / "blocker"
        blocker.write_text("i am a file", encoding="utf-8")
        server.TELEMETRY_PATH = blocker / "cannot" / "live.jsonl"
        try:
            result = server._write_telemetry({"event": "x"})
            check("write failure returns False and never raises", result is False)
        except Exception as exc:  # pragma: no cover
            check("write failure never raises", False, repr(exc))


async def test_session_stream() -> None:
    print("\nmultiple events in one session (real server handlers)")
    if server is None:
        skip("server session tests", f"server module unavailable ({SERVER_IMPORT_ERROR})")
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "live_events.jsonl"
        sent = await run_server_steps(SCENARIO, path)
        recs = read_jsonl(path)
        names = [r["event"] for r in recs]

        check("all records share the session_id", {r["session_id"] for r in recs} == {"g6-test"})
        check("seq strictly increasing from 1", [r["seq"] for r in recs] == list(range(1, len(recs) + 1)))
        check("no duplicate (session_id, seq)", len({(r["session_id"], r["seq"]) for r in recs}) == len(recs))
        check("no per-token lines", "answer_token" not in names)
        for needed in ("session_started", "controller_decision", "retrieval_update", "commit_start",
                       "reuse_decision", "retrieval", "answer_started", "first_token",
                       "answer_completed", "uncertainty_emitted", "request_finished", "session_ended"):
            check(f"captures {needed}", needed in names)

        text = path.read_text(encoding="utf-8")
        answer = next(m for m in sent if m.get("event") == "answer_completed" and m.get("token_metrics"))["answer"]
        check("no raw transcript stored",
              Q_TRAVEL not in text and Q_UNRELATED not in text and "Repeat that in two bullets" not in text)
        check("no evidence text stored",
              all(e["chunk"]["text"] not in text for e in FIXTURE_CHUNKS))
        check("no answer text stored", answer not in text)
        check("WebSocket payloads unchanged in kind",
              {"orchestrator_event", "request_finished", "reuse_decision"} <= {m["type"] for m in sent})

        report = g6.analyze(*_load(path))
        check("evaluator: strict G6 passes on real telemetry", report["strict_g6"]["passed"],
              str(report["strict_g6"]["failures"]))
        check("evaluator: generated answer citation-valid with citations recorded",
              report["turns"]["unverified_generated_answers"] == 0
              and next(r for r in recs if r["event"] == "answer_completed" and r.get("token_metrics"))["citations"])
        t, c = report["turns"], report["controller_actions"]
        check("evaluator: turn counts", t["commit_turns"] == 3 and t["complete_turns"] == 3
              and t["generated_answers"] == 1 and t["uncertainty_refusals"] == 1
              and t["presentation_turns"] == 1, str(t))
        check("evaluator: controller counts WAIT/SUPPRESS/RETRIEVE",
              (c["WAIT"], c["SUPPRESS"], c["RETRIEVE"]) == (1, 1, 1), str(c))
        r = report["retrieval"]
        check("evaluator: retrieval/reuse accounting",
              r["total_retrieval_calls"] == 2 and r["early_retrieval_calls"] == 1
              and r["commit_retrieval_calls"] == 1 and r["exact_reuse_count"] == 1
              and r["new_retrieval_count"] == 1 and r["presentation_no_retrieval_count"] == 1, str(r))
        check("evaluator: token summary from Ollama-reported counts",
              report["tokens"]["total_tokens"]["n"] == 1
              and report["tokens"]["total_tokens"]["mean"] == 120
              and report["latency_ms"]["ttft_llm_only_ms"]["median"] == 12.5)


async def test_reset_lifecycle() -> None:
    print("\nreset / session lifecycle")
    if server is None:
        skip("reset test", f"server module unavailable ({SERVER_IMPORT_ERROR})")
        return
    with tempfile.TemporaryDirectory() as tmp:
        server.TELEMETRY_PATH = Path(tmp) / "live_events.jsonl"
        old = server.TelemetryTrail("old-session")
        new = server.TelemetryTrail("new-session", previous_session_id="old-session")
        recs = read_jsonl(server.TELEMETRY_PATH)
        check("session_started then session_reset (with previous id)",
              [r["event"] for r in recs] == ["session_started", "session_reset"]
              and recs[1]["previous_session_id"] == "old-session" and new.session_id == "new-session"
              and old.session_id == "old-session")


# ----- evaluator behaviour on synthetic / mutated telemetry -------------------

def _load(path: Path):
    return g6.load_records(path)


def _good_turn_records(sid: str = "s") -> list[dict]:
    """A hand-built, fully valid answered commit turn plus a partial."""
    base = 100

    def rec(seq, event, **kw):
        return {"seq": seq, "timestamp": f"2026-01-01T00:00:{seq:02d}.000+00:00",
                "session_id": sid, "event": event, **kw}

    tm = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
          "llm_ttft_ms": 12.5, "llm_generation_ms": 40.0}
    lineage = dict(query_version=1, generation_id=1, answer_version=1)
    return [
        rec(1, "controller_decision", action="WAIT", query_version=0, generation_id=0,
            retrieval_calls_this_turn=0, server_ms=0.5),
        rec(2, "request_finished", kind="partial", server_ms=0.5),
        rec(3, "commit_start", turn_id=f"{sid}:c1", query_chars=10),
        rec(4, "retrieval", turn_id=f"{sid}:c1", retrieval_reuse_mode="new", retrieval_calls=1),
        rec(5, "answer_started", turn_id=f"{sid}:c1", retrieval_reuse_mode="new",
            retrieval_calls_this_turn=1, **lineage),
        rec(6, "first_token", turn_id=f"{sid}:c1", ttft_ms=5.0),
        rec(7, "answer_completed", turn_id=f"{sid}:c1", citations=["[D §S]"], citation_valid=True,
            retrieval_reuse_mode="new", token_metrics=dict(tm), token_metrics_reason="reported", **lineage),
        rec(8, "request_finished", kind="commit", turn_id=f"{sid}:c1", server_ms=50.0,
            outcome="answered", llm_called=True, tokens_streamed=3, ttft_ms=5.0, pre_generation_ms=2.0,
            retrieval_calls=1, retrieval_reuse_mode="new", token_metrics_reason="reported",
            llm_ttft_ms=12.5, llm_generation_ms=40.0, prompt_tokens=100, completion_tokens=20,
            total_tokens=120),
    ]


def _write(path: Path, recs: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")


def _strict(recs: list[dict], tmp: str, name: str = "t.jsonl") -> dict:
    p = Path(tmp) / name
    _write(p, recs)
    return g6.analyze(*_load(p))


def test_malformed_handling() -> None:
    print("\nmalformed telemetry handling")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "bad.jsonl"
        good = _good_turn_records()
        lines = [json.dumps(r) for r in good]
        raw = ("\n".join(lines[:2]) + "\n{not json\n\n[1,2,3]\n\"str\"\n" + "\n".join(lines[2:]) + "\n").encode()
        raw += b'\xff\xfe{"bad":"utf8"}\n'
        p.write_bytes(raw)
        records, malformed = g6.load_records(p)
        check("valid records still parsed", len(records) == len(good))
        check("malformed/non-object/bad-utf8 lines reported with line numbers",
              [m["line"] for m in malformed] == [3, 5, 6, 13], str([m["line"] for m in malformed]))
        report = g6.analyze(records, malformed)
        check("analysis survives malformed input", report["events"]["malformed_lines"] == 4)
        check("strict fails on malformed lines", not report["strict_g6"]["passed"])
        check("empty file: strict fails, non-strict ok",
              not g6.analyze([], [])["strict_g6"]["passed"])


def test_strict_behaviour() -> None:
    print("\nstrict evaluator behaviour")
    with tempfile.TemporaryDirectory() as tmp:
        clean = _strict(_good_turn_records(), tmp)
        check("valid telemetry passes strict", clean["strict_g6"]["passed"], str(clean["strict_g6"]["failures"]))
        check("valid turn counted complete", clean["turns"]["complete_turns"] == 1
              and clean["turns"]["incomplete_turns"] == 0)

        def mutate(fn) -> dict:
            recs = copy.deepcopy(_good_turn_records())
            fn(recs)
            return _strict(recs, tmp, "m.jsonl")

        def fails(name: str, fn, expect: str | None = None) -> None:
            rep = mutate(fn)
            failures = " ".join(rep["strict_g6"]["failures"])
            check(name, not rep["strict_g6"]["passed"] and (expect is None or expect in failures),
                  failures or "passed unexpectedly")

        fails("missing session_id fails", lambda r: r[0].pop("session_id"), "required_base_fields_present")
        fails("missing timestamp fails", lambda r: r[3].pop("timestamp"), "required_base_fields_present")
        fails("unparseable timestamp fails", lambda r: r[3].__setitem__("timestamp", "yesterday"))
        fails("negative timing fails", lambda r: r[7].__setitem__("server_ms", -3.0), "no_negative_timings")
        fails("negative nested timing fails",
              lambda r: r[6]["token_metrics"].__setitem__("llm_generation_ms", -1.0))
        fails("finished turn missing answer_completed fails", lambda r: r.pop(6),
              "finished_turn_lifecycle_complete")
        fails("finished turn missing retrieval fails", lambda r: r.pop(3), "finished_turn_lifecycle_complete")
        fails("finished turn missing commit_start fails", lambda r: r.pop(2),
              "finished_turn_lifecycle_complete")
        fails("out-of-order events fail", lambda r: r.insert(5, r.pop(6)), "ordering")
        fails("duplicate event write fails", lambda r: r.append(copy.deepcopy(r[7])), "no_duplicate_events")
        fails("seq going backwards fails", lambda r: r[4].__setitem__("seq", 1))
        fails("timestamp going backwards fails",
              lambda r: r[5].__setitem__("timestamp", "2025-12-31T00:00:00.000+00:00"), "ordering")
        fails("generated turn missing token_metrics fails", lambda r: r[6].pop("token_metrics"),
              "required_event_fields_present")
        fails("generated turn missing llm latency fails",
              lambda r: r[6]["token_metrics"].__setitem__("llm_ttft_ms", None), "generated_turns")
        fails("'reported' tokens that are null fail",
              lambda r: r[6]["token_metrics"].__setitem__("prompt_tokens", None), "generated_turns")
        fails("token totals that do not add up fail",
              lambda r: r[6]["token_metrics"].__setitem__("total_tokens", 999), "generated_turns")
        fails("turn summary disagreeing with answer_completed fails",
              lambda r: r[7].__setitem__("prompt_tokens", 1), "finished_turn_lifecycle_complete")
        fails("retrieval call mismatch fails", lambda r: r[3].__setitem__("retrieval_calls", 0),
              "finished_turn_lifecycle_complete")

        # --- unreported by Ollama is NOT a telemetry defect ----------------
        def ollama_silent(r):
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                r[6]["token_metrics"][k] = None
                r[7][k] = None
            r[6]["token_metrics_reason"] = "ollama_did_not_report_tokens"
            r[7]["token_metrics_reason"] = "ollama_did_not_report_tokens"

        rep = mutate(ollama_silent)
        check("Ollama-unreported tokens pass strict", rep["strict_g6"]["passed"],
              str(rep["strict_g6"]["failures"]))
        check("...and are listed as unavailable, not as missing fields",
              len(rep["unavailable_metrics"]) == 3 and not rep["missing_required_fields"]
              and rep["tokens"]["turns_tokens_not_reported_by_ollama"] == 1
              and rep["tokens"]["total_tokens"]["n"] == 0)

        def lying_reason(r):
            r[6]["token_metrics_reason"] = "ollama_did_not_report_tokens"
            r[7]["token_metrics_reason"] = "ollama_did_not_report_tokens"

        fails("'unreported' claimed while counts are present fails", lying_reason, "generated_turns")

        # --- unfinished turns are reported, not strict failures ------------
        rep = mutate(lambda r: r.pop(7))
        check("unfinished (in-flight/aborted) turn reported as incomplete",
              rep["turns"]["unfinished_turns"] == 1 and rep["turns"]["complete_turns"] == 0)
        check("...and does not fail strict on its own", rep["strict_g6"]["passed"],
              str(rep["strict_g6"]["failures"]))

        rep = mutate(lambda r: (r.pop(7), r.append({"seq": 9, "timestamp": "2026-01-01T00:00:09.000+00:00",
                                                     "session_id": "s", "event": "request_error",
                                                     "turn_id": "s:c1", "kind": "commit"})))
        check("errored turn counted separately", rep["turns"]["error_turns"] == 1)


def test_refusal_and_presentation_strictness() -> None:
    print("\nno-LLM turns")
    with tempfile.TemporaryDirectory() as tmp:
        def refusal(mode_reason="no_llm_call_evidence_insufficient", tm=None):
            def rec(seq, event, **kw):
                return {"seq": seq, "timestamp": f"2026-01-01T00:00:{seq:02d}.000+00:00",
                        "session_id": "r", "event": event, **kw}
            return [
                rec(1, "commit_start", turn_id="r:c1", query_chars=5),
                rec(2, "retrieval", turn_id="r:c1", retrieval_reuse_mode="new", retrieval_calls=1),
                rec(3, "uncertainty_emitted", turn_id="r:c1", query_version=1, generation_id=1,
                    answer_version=1, reason="no_evidence", retrieval_reuse_mode="new",
                    token_metrics=tm, token_metrics_reason=mode_reason),
                rec(4, "request_finished", kind="commit", turn_id="r:c1", server_ms=3.0,
                    outcome="refused", llm_called=False, tokens_streamed=0, retrieval_calls=1,
                    retrieval_reuse_mode="new", pre_generation_ms=3.0),
            ]

        rep = _strict(refusal(), tmp)
        check("valid refusal passes; null token_metrics + reason is correct",
              rep["strict_g6"]["passed"] and rep["turns"]["uncertainty_refusals"] == 1,
              str(rep["strict_g6"]["failures"]))
        rep = _strict(refusal(tm={"prompt_tokens": 1}), tmp)
        check("refusal that reports tokens fails", not rep["strict_g6"]["passed"])
        rep = _strict(refusal(mode_reason=None), tmp)
        check("refusal without a no-LLM reason fails", not rep["strict_g6"]["passed"])


# =============================================================================
# MAIN
# =============================================================================

async def run_all() -> int:
    print("=" * 78)
    print("G6 TELEMETRY TESTS")
    print("=" * 78)
    test_session_defaults()
    await test_trace_envelope()
    await test_token_metrics()
    await test_reuse_telemetry()
    test_jsonl_writer()
    await test_session_stream()
    await test_reset_lifecycle()
    test_malformed_handling()
    test_strict_behaviour()
    test_refusal_and_presentation_strictness()

    failed = [n for n, ok in _results if not ok]
    print("\n" + "=" * 78)
    print(f"{len(_results) - len(failed)}/{len(_results)} checks passed")
    if failed:
        print("FAILED:")
        for n in failed:
            print(f"  - {n}")
        return 1
    print("G6 TELEMETRY TESTS: ALL PASSED")
    return 0


async def emit_sample(path: Path) -> None:
    if server is None:
        raise SystemExit(f"server module unavailable: {SERVER_IMPORT_ERROR}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    await run_server_steps(SCENARIO, path)
    print(f"wrote {path} ({len(read_jsonl(path))} records)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-sample", metavar="PATH",
                    help="write a deterministic-scenario telemetry file and exit")
    args = ap.parse_args()
    if args.emit_sample:
        asyncio.run(emit_sample(Path(args.emit_sample)))
        sys.exit(0)
    sys.exit(asyncio.run(run_all()))