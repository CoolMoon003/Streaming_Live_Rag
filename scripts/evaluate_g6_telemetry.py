"""
G6 - Telemetry / observability evaluator.

Reads the JSON Lines telemetry the live server writes
(data/telemetry/live_events.jsonl by default) and reports what can actually be
DERIVED from it: turns, controller actions, retrieval/reuse accounting,
latency and token summaries, and whether the telemetry itself is complete and
internally consistent.

It is deterministic and dependency-free (stdlib + evaluation.metrics): no
Hugging Face, no Ollama, no network. It never estimates anything - a metric
that is not in the telemetry is reported as unavailable, and "the LLM server
did not report token counts" is kept strictly apart from "the telemetry field
is missing or broken".

Usage:
    python -m scripts.evaluate_g6_telemetry
    python -m scripts.evaluate_g6_telemetry --input data/telemetry/live_events.jsonl
    python -m scripts.evaluate_g6_telemetry --json g6_telemetry_report.json
    python -m scripts.evaluate_g6_telemetry --strict-g6

Exit codes: 0 ok, 1 strict-G6 failure, 2 input file unreadable.

Record schema (written by backend/app/api/server.py, TelemetryTrail):
    every record   seq, timestamp, session_id, event
    per commit     commit_start -> [reuse_decision] -> retrieval ->
                   (uncertainty_emitted | answer_started -> [first_token] ->
                   answer_completed) -> request_finished
    per partial    controller_decision | retrieval_update | retrieval_stale
                   -> request_finished
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.metrics import summarize_values

DEFAULT_INPUT = Path("data") / "telemetry" / "live_events.jsonl"

REQUIRED_BASE = ("seq", "timestamp", "session_id", "event")

# Fields that must be present AND non-null for each event type.
EVENT_REQUIRED: dict[str, tuple[str, ...]] = {
    "commit_start": ("turn_id", "query_chars"),
    "reuse_decision": ("turn_id", "mode"),
    "retrieval": ("turn_id", "retrieval_reuse_mode", "retrieval_calls"),
    "first_token": ("turn_id", "ttft_ms"),
    "answer_started": (
        "turn_id", "query_version", "generation_id", "answer_version",
        "retrieval_reuse_mode", "retrieval_calls_this_turn",
    ),
    "uncertainty_emitted": (
        "turn_id", "query_version", "generation_id", "answer_version",
        "reason", "retrieval_reuse_mode", "token_metrics_reason",
    ),
    "answer_completed": (
        "turn_id", "query_version", "generation_id", "answer_version",
        "citations", "retrieval_reuse_mode", "token_metrics_reason",
    ),
    "controller_decision": (
        "action", "query_version", "generation_id", "retrieval_calls_this_turn",
    ),
    "retrieval_update": (
        "action", "query_version", "generation_id", "retrieval_calls_this_turn",
    ),
    "retrieval_stale": ("action", "generation_id", "retrieval_calls_this_turn"),
    "request_finished": ("kind", "server_ms"),
}

# Fields that must be present but MAY legitimately be null.
EVENT_PRESENT: dict[str, tuple[str, ...]] = {
    "uncertainty_emitted": ("token_metrics",),
    "answer_completed": ("token_metrics",),
}

COMMIT_FINISH_REQUIRED = ("turn_id", "outcome", "retrieval_calls", "tokens_streamed")
TOKEN_METRIC_KEYS = (
    "prompt_tokens", "completion_tokens", "total_tokens",
    "llm_ttft_ms", "llm_generation_ms",
)
PARTIAL_EVENTS = ("controller_decision", "retrieval_update", "retrieval_stale")
TURN_TERMINALS = ("answer_completed", "uncertainty_emitted")

REUSE_MODES = (
    "exact", "extension", "extension_delta", "new",
    "additive", "replacement", "presentation_none",
)

TIMESTAMP_TOLERANCE_MS = 1000.0  # wall-clock adjustments; seq is authoritative
TIMING_TOLERANCE_MS = 1.0        # rounding slack for cross-field timing checks


# =============================================================================
# LOADING
# =============================================================================

def load_records(path: Path) -> tuple[list[tuple[int, dict]], list[dict]]:
    """Return ([(line_no, record)], [malformed]). Blank lines are ignored."""
    records: list[tuple[int, dict]] = []
    malformed: list[dict] = []

    with path.open("rb") as f:
        for line_no, raw in enumerate(f, start=1):
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                malformed.append({"line": line_no, "error": f"{type(exc).__name__}: {exc}"})
                continue
            if not isinstance(obj, dict):
                malformed.append({"line": line_no, "error": "not a JSON object"})
                continue
            records.append((line_no, obj))

    return records, malformed


# =============================================================================
# HELPERS
# =============================================================================

def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _stats(values: list[Any]) -> dict[str, Any]:
    summary = summarize_values([v for v in values if _is_num(v)])
    return summary if summary else {"n": 0, "mean": None, "median": None, "max": None}


def _walk_ms(rec: dict) -> list[tuple[str, Any]]:
    """Every *_ms field, top level and inside nested dicts."""
    found: list[tuple[str, Any]] = []
    for key, value in rec.items():
        if isinstance(value, dict):
            found.extend((f"{key}.{k}", v) for k, v in _walk_ms(value))
        elif key.endswith("_ms") or key == "latency_ms":
            found.append((key, value))
    return found


class _Findings:
    def __init__(self) -> None:
        self.violations: list[dict] = []
        self.missing_required: list[dict] = []
        self.unavailable: list[dict] = []

    def violation(self, kind: str, detail: str, rec: dict | None = None,
                  line: int | None = None, turn_id: str | None = None) -> None:
        self.violations.append({
            "kind": kind,
            "detail": detail,
            "line": line,
            "session_id": (rec or {}).get("session_id"),
            "turn_id": turn_id or (rec or {}).get("turn_id"),
        })

    def missing(self, rec: dict, line: int, field: str, why: str = "missing_or_null") -> None:
        self.missing_required.append({
            "line": line, "event": rec.get("event"),
            "session_id": rec.get("session_id"), "field": field, "problem": why,
        })


# =============================================================================
# PER-RECORD CHECKS
# =============================================================================

def _check_record(line: int, rec: dict, fx: _Findings) -> None:
    for name in REQUIRED_BASE:
        if rec.get(name) in (None, ""):
            fx.missing(rec, line, name)

    if "seq" in rec and rec["seq"] not in (None, "") and not _is_int(rec["seq"]):
        fx.missing(rec, line, "seq", "not_an_integer")
    if rec.get("timestamp") not in (None, "") and _parse_ts(rec["timestamp"]) is None:
        fx.missing(rec, line, "timestamp", "not_iso8601")

    event = rec.get("event")
    for name in EVENT_REQUIRED.get(event, ()):
        if rec.get(name) is None:
            fx.missing(rec, line, name)
    for name in EVENT_PRESENT.get(event, ()):
        if name not in rec:
            fx.missing(rec, line, name, "key_absent")

    if event == "request_finished" and rec.get("kind") == "commit":
        for name in COMMIT_FINISH_REQUIRED:
            if rec.get(name) is None:
                fx.missing(rec, line, name)

    for name, value in _walk_ms(rec):
        if _is_num(value) and value < 0:
            fx.violation("negative_timing", f"{name}={value}", rec, line)


# =============================================================================
# PER-SESSION ORDERING / DUPLICATES
# =============================================================================

def _check_sessions(recs: list[tuple[int, dict]], fx: _Findings) -> dict[str, list[tuple[int, dict]]]:
    sessions: "OrderedDict[str, list[tuple[int, dict]]]" = OrderedDict()
    for line, rec in recs:
        sid = rec.get("session_id")
        if sid:
            sessions.setdefault(sid, []).append((line, rec))

    for sid, items in sessions.items():
        seen_seq: set[int] = set()
        prev_seq: int | None = None
        prev_ts: datetime | None = None

        for line, rec in items:
            seq = rec.get("seq")
            if _is_int(seq):
                if seq in seen_seq:
                    fx.violation("duplicate_event", f"seq {seq} written more than once", rec, line)
                elif prev_seq is not None and seq < prev_seq:
                    fx.violation("ordering", f"seq {seq} follows {prev_seq}", rec, line)
                seen_seq.add(seq)
                prev_seq = seq if prev_seq is None else max(prev_seq, seq)

            ts = _parse_ts(rec.get("timestamp"))
            if ts is not None:
                if prev_ts is not None:
                    back_ms = (prev_ts - ts).total_seconds() * 1000.0
                    if back_ms > TIMESTAMP_TOLERANCE_MS:
                        fx.violation(
                            "ordering",
                            f"timestamp goes backwards by {back_ms:.0f} ms", rec, line,
                        )
                prev_ts = ts if prev_ts is None else max(prev_ts, ts)

    return sessions


# =============================================================================
# PER-TURN LIFECYCLE
# =============================================================================

def _first(events: list[tuple[int, dict]], name: str) -> tuple[int, dict] | None:
    return next(((i, r) for i, (_, r) in enumerate(events) if r.get("event") == name), None)


def _check_token_metrics(turn_id: str, completed: dict, line: int, fx: _Findings) -> str:
    """Validate answer_completed.token_metrics for an LLM turn.

    Returns "reported" | "unavailable" | "broken".
    """
    tm = completed.get("token_metrics")
    reason = completed.get("token_metrics_reason")
    if not isinstance(tm, dict):
        fx.violation("missing_metric", "generated turn has no token_metrics object",
                     completed, line, turn_id)
        return "broken"

    absent = [k for k in TOKEN_METRIC_KEYS if k not in tm]
    if absent:
        for k in absent:
            fx.missing(completed, line, f"token_metrics.{k}", "key_absent")
        return "broken"

    # Latency figures are always measured by the LLM client -> a null is broken.
    for k in ("llm_ttft_ms", "llm_generation_ms"):
        if not _is_num(tm[k]):
            fx.violation("missing_metric", f"token_metrics.{k} is null/non-numeric",
                         completed, line, turn_id)
            return "broken"

    counts = [tm["prompt_tokens"], tm["completion_tokens"], tm["total_tokens"]]
    all_int = all(_is_int(c) for c in counts)

    if reason == "reported":
        if not all_int:
            fx.violation("token_metrics_inconsistent",
                         "reason 'reported' but a token count is null", completed, line, turn_id)
            return "broken"
        if tm["total_tokens"] != tm["prompt_tokens"] + tm["completion_tokens"]:
            fx.violation("token_metrics_inconsistent",
                         "total_tokens != prompt_tokens + completion_tokens",
                         completed, line, turn_id)
            return "broken"
        if any(c < 0 for c in counts):
            fx.violation("token_metrics_inconsistent", "negative token count",
                         completed, line, turn_id)
            return "broken"
        return "reported"

    if reason == "ollama_did_not_report_tokens":
        if all_int:
            fx.violation("token_metrics_inconsistent",
                         "reason says tokens unreported but all counts are present",
                         completed, line, turn_id)
            return "broken"
        return "unavailable"

    fx.violation("token_metrics_inconsistent", f"unknown token_metrics_reason {reason!r}",
                 completed, line, turn_id)
    return "broken"


def _analyze_turn(turn_id: str, events: list[tuple[int, dict]], fx: _Findings) -> dict:
    names = [r.get("event") for _, r in events]
    lines = {line for line, _ in events}
    turn: dict[str, Any] = {
        "turn_id": turn_id, "session_id": events[0][1].get("session_id"),
        "status": "incomplete", "outcome": None, "events": names,
    }

    finished = [(l, r) for l, r in events
                if r.get("event") == "request_finished" and r.get("kind") == "commit"]
    errored = any(n == "request_error" for n in names)

    if not finished:
        turn["status"] = "error" if errored else "incomplete"
        turn["outcome"] = "error" if errored else None
        return turn

    fin_line, fin = finished[-1]
    outcome = fin.get("outcome")
    turn["outcome"] = outcome

    def problem(detail: str, rec: dict | None = None, line: int | None = None) -> None:
        fx.violation("lifecycle", detail, rec or fin, line or fin_line, turn_id)

    if len(finished) > 1:
        problem("request_finished written more than once")
    if names[0] != "commit_start":
        problem(f"first event is {names[0]!r}, expected commit_start")
    if names[-1] != "request_finished":
        problem(f"last event is {names[-1]!r}, expected request_finished")
    if names.count("commit_start") != 1:
        problem("commit_start missing or duplicated")

    retrieval = _first(events, "retrieval")
    started = _first(events, "answer_started")
    completed = _first(events, "answer_completed")
    refused = _first(events, "uncertainty_emitted")
    first_token = _first(events, "first_token")

    if retrieval is None:
        problem("no retrieval record")
    for name in ("retrieval", "answer_started", "uncertainty_emitted", "answer_completed",
                 "first_token", "commit_start"):
        if names.count(name) > 1:
            problem(f"{name} written more than once")

    # Event ordering: retrieval < answer_started/uncertainty < first_token < completed
    def before(a: tuple | None, b: tuple | None, label: str) -> None:
        if a is not None and b is not None and a[0] >= b[0]:
            fx.violation("ordering", f"{label} out of order", fin, fin_line, turn_id)

    before(retrieval, started, "retrieval/answer_started")
    before(retrieval, refused, "retrieval/uncertainty_emitted")
    before(started, first_token, "answer_started/first_token")
    before(first_token, completed, "first_token/answer_completed")
    before(started, completed, "answer_started/answer_completed")

    # Retrieval accounting must agree between the retrieval record and the turn summary.
    if retrieval is not None:
        r_calls = retrieval[1].get("retrieval_calls")
        if _is_int(r_calls) and _is_int(fin.get("retrieval_calls")) and r_calls != fin["retrieval_calls"]:
            problem(f"retrieval_calls {r_calls} != request_finished.retrieval_calls {fin['retrieval_calls']}")
        if retrieval[1].get("retrieval_reuse_mode") != fin.get("retrieval_reuse_mode"):
            problem("retrieval_reuse_mode differs between retrieval and request_finished")

    llm_called = bool(fin.get("llm_called"))

    if outcome == "refused":
        if refused is None:
            problem("outcome refused without uncertainty_emitted")
        else:
            ref = refused[1]
            if ref.get("token_metrics") is not None or not str(ref.get("token_metrics_reason", "")).startswith("no_llm_call"):
                problem("refusal must have token_metrics null with a no_llm_call reason", ref)
        if started or completed:
            problem("refused turn has answer_started/answer_completed")
        if llm_called:
            problem("refused turn claims llm_called")

    elif outcome in ("answered", "presentation"):
        if started is None:
            problem("no answer_started")
        if completed is None:
            problem("no answer_completed")
        if refused is not None:
            problem("answered turn also has uncertainty_emitted")

        if completed is not None:
            c_line = events[completed[0]][0]
            comp = completed[1]
            if outcome == "answered":
                if not llm_called:
                    problem("answered turn without llm_called")
                state = _check_token_metrics(turn_id, comp, c_line, fx)
                if not isinstance(comp.get("citation_valid"), bool):
                    fx.missing(comp, c_line, "citation_valid")
                turn["token_state"] = state
                tm = comp.get("token_metrics") or {}
                if state != "broken":
                    for k in TOKEN_METRIC_KEYS:
                        if fin.get(k) != tm.get(k):
                            problem(f"request_finished.{k} != answer_completed.token_metrics.{k}")
                    # Pipeline timings are measured by the server for every generated turn.
                    for k in ("ttft_ms", "pre_generation_ms"):
                        if not _is_num(fin.get(k)):
                            fx.violation("missing_metric", f"request_finished.{k} is null",
                                         fin, fin_line, turn_id)
                    if first_token is None:
                        problem("generated turn has no first_token record")
                    if state == "unavailable":
                        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                            if tm.get(k) is None:
                                fx.unavailable.append({
                                    "turn_id": turn_id, "metric": k,
                                    "reason": "ollama_did_not_report_tokens",
                                })
            else:  # presentation: no LLM involved
                if llm_called:
                    problem("presentation turn claims llm_called")
                if comp.get("token_metrics") is not None or not str(comp.get("token_metrics_reason", "")).startswith("no_llm_call"):
                    problem("presentation turn must have token_metrics null with a no_llm_call reason", comp, c_line)
    else:
        problem(f"request_finished outcome {outcome!r} is not a completed outcome")

    # Cross-field timing sanity (only meaningful when both sides are numbers).
    def num(k: str) -> float | None:
        return fin.get(k) if _is_num(fin.get(k)) else None

    server, ttft, pre = num("server_ms"), num("ttft_ms"), num("pre_generation_ms")
    if server is not None and ttft is not None and ttft > server + TIMING_TOLERANCE_MS:
        fx.violation("timing_inconsistent", f"ttft_ms {ttft} > server_ms {server}", fin, fin_line, turn_id)
    if ttft is not None and pre is not None and pre > ttft + TIMING_TOLERANCE_MS:
        fx.violation("timing_inconsistent", f"pre_generation_ms {pre} > ttft_ms {ttft}", fin, fin_line, turn_id)
    lt, lg = num("llm_ttft_ms"), num("llm_generation_ms")
    if lt is not None and lg is not None and lt > lg + TIMING_TOLERANCE_MS:
        fx.violation("timing_inconsistent", f"llm_ttft_ms {lt} > llm_generation_ms {lg}", fin, fin_line, turn_id)

    # A finished turn is only "complete" if nothing anywhere in it is wrong.
    broken = (
        any(v["turn_id"] == turn_id for v in fx.violations)
        or any(m["line"] in lines for m in fx.missing_required)
    )
    turn["status"] = "broken" if broken else "complete"
    turn["summary"] = fin
    return turn


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze(records: list[tuple[int, dict]], malformed: list[dict] | None = None) -> dict:
    malformed = malformed or []
    fx = _Findings()

    for line, rec in records:
        _check_record(line, rec, fx)

    sessions = _check_sessions(records, fx)

    # ---- group commit turns
    by_turn: "OrderedDict[str, list[tuple[int, dict]]]" = OrderedDict()
    for line, rec in records:
        tid = rec.get("turn_id")
        if tid:
            by_turn.setdefault(tid, []).append((line, rec))
    turns = [_analyze_turn(tid, ev, fx) for tid, ev in by_turn.items()]

    complete = [t for t in turns if t["status"] == "complete"]
    broken = [t for t in turns if t["status"] == "broken"]
    incomplete = [t for t in turns if t["status"] == "incomplete"]
    errored = [t for t in turns if t["status"] == "error"]
    finished = complete + broken  # turns that claim to be completed

    # ---- controller / early retrieval
    partial = [r for _, r in records if r.get("event") in PARTIAL_EVENTS]
    controller = Counter(str(r.get("action")) for r in partial if r.get("action"))
    controller_actions = {a: controller.get(a, 0) for a in ("WAIT", "SUPPRESS", "RETRIEVE")}
    for extra, n in controller.items():
        controller_actions.setdefault(extra, n)

    early_calls = sum(
        r.get("retrieval_calls_this_turn", 0) for r in partial
        if r.get("event") in ("retrieval_update", "retrieval_stale")
        and _is_int(r.get("retrieval_calls_this_turn"))
    )

    # ---- reuse / commit retrieval (one `retrieval` record per commit turn)
    retrieval_records = [r for _, r in records if r.get("event") == "retrieval"]
    reuse = Counter(str(r.get("retrieval_reuse_mode")) for r in retrieval_records
                    if r.get("retrieval_reuse_mode"))
    reuse_counts = {m: reuse.get(m, 0) for m in REUSE_MODES}
    for extra, n in reuse.items():
        reuse_counts.setdefault(extra, n)
    predicted = Counter(str(r.get("mode")) for _, r in records
                        if r.get("event") == "reuse_decision" and r.get("mode"))
    commit_calls = sum(r.get("retrieval_calls", 0) for r in retrieval_records
                       if _is_int(r.get("retrieval_calls")))

    retrieval_summary = {
        "total_retrieval_calls": early_calls + commit_calls,
        "early_retrieval_calls": early_calls,
        "commit_retrieval_calls": commit_calls,
        "exact_reuse_count": reuse_counts["exact"],
        "extension_reuse_count": reuse_counts["extension"],
        "extension_plus_delta_retrieval_count": reuse_counts["extension_delta"],
        "new_retrieval_count": reuse_counts["new"],
        "additive_retrieval_count": reuse_counts["additive"],
        "replacement_retrieval_count": reuse_counts["replacement"],
        "presentation_no_retrieval_count": reuse_counts["presentation_none"],
        "early_retrieval_stale_discarded": sum(
            1 for r in partial if r.get("event") == "retrieval_stale"),
    }

    # ---- answers / refusals
    answered = [t for t in finished if t["outcome"] == "answered"]
    refused_events = [r for _, r in records if r.get("event") == "uncertainty_emitted"]
    unverified = [t for t in answered
                  if t["summary"].get("citation_valid") is False]

    # ---- latency & tokens
    fins = [t["summary"] for t in finished]
    gen = [t["summary"] for t in answered]
    latency = {
        "ttft_pipeline_ms": _stats([s.get("ttft_ms") for s in gen]),
        "ttft_llm_only_ms": _stats([s.get("llm_ttft_ms") for s in gen]),
        "llm_generation_ms": _stats([s.get("llm_generation_ms") for s in gen]),
        "server_processing_ms": _stats([s.get("server_ms") for s in fins]),
        "pre_generation_ms": _stats([s.get("pre_generation_ms") for s in fins]),
    }
    tokens = {
        "prompt_tokens": _stats([s.get("prompt_tokens") for s in gen]),
        "completion_tokens": _stats([s.get("completion_tokens") for s in gen]),
        "total_tokens": _stats([s.get("total_tokens") for s in gen]),
        "turns_with_reported_tokens": sum(1 for t in answered if t.get("token_state") == "reported"),
        "turns_tokens_not_reported_by_ollama": sum(
            1 for t in answered if t.get("token_state") == "unavailable"),
    }

    # ---- checks
    kinds = Counter(v["kind"] for v in fx.violations)
    checks = [
        ("no_malformed_lines", not malformed, f"{len(malformed)} malformed line(s)"),
        ("required_base_fields_present",
         not any(m["field"] in REQUIRED_BASE for m in fx.missing_required),
         "every record has seq/timestamp/session_id/event"),
        ("required_event_fields_present",
         not any(m["field"] not in REQUIRED_BASE for m in fx.missing_required),
         f"{sum(1 for m in fx.missing_required if m['field'] not in REQUIRED_BASE)} missing event field(s)"),
        ("no_duplicate_events", kinds.get("duplicate_event", 0) == 0,
         f"{kinds.get('duplicate_event', 0)} duplicate (session_id, seq)"),
        ("event_ordering_consistent", kinds.get("ordering", 0) == 0,
         f"{kinds.get('ordering', 0)} ordering violation(s)"),
        ("no_negative_timings", kinds.get("negative_timing", 0) == 0,
         f"{kinds.get('negative_timing', 0)} negative timing(s)"),
        ("timings_internally_consistent", kinds.get("timing_inconsistent", 0) == 0,
         f"{kinds.get('timing_inconsistent', 0)} inconsistent timing(s)"),
        ("finished_turn_lifecycle_complete",
         not broken and kinds.get("lifecycle", 0) == 0,
         f"{len(broken)} finished turn(s) with a broken lifecycle"),
        ("generated_turns_have_required_metrics",
         kinds.get("missing_metric", 0) == 0 and kinds.get("token_metrics_inconsistent", 0) == 0,
         f"{kinds.get('missing_metric', 0) + kinds.get('token_metrics_inconsistent', 0)} metric problem(s)"),
    ]

    strict_failures: list[str] = []
    if not records and not malformed:
        strict_failures.append("no telemetry records found")
    for name, ok, detail in checks:
        if not ok:
            strict_failures.append(f"{name}: {detail}")

    report = {
        "events": {
            "records": len(records),
            "malformed_lines": len(malformed),
            "unique_sessions": len(sessions),
            "by_event": dict(sorted(Counter(r.get("event") for _, r in records).items(),
                                    key=lambda kv: str(kv[0]))),
        },
        "turns": {
            "commit_turns": len(turns),
            "complete_turns": len(complete),
            "incomplete_turns": len(incomplete) + len(broken),
            "unfinished_turns": len(incomplete),
            "broken_lifecycle_turns": len(broken),
            "error_turns": len(errored),
            "generated_answers": len(answered),
            "uncertainty_refusals": len(refused_events),
            "presentation_turns": sum(1 for t in finished if t["outcome"] == "presentation"),
            "unverified_generated_answers": len(unverified),
        },
        "controller_actions": controller_actions,
        "reuse_decisions": reuse_counts,
        "reuse_decisions_predicted_by_server": dict(predicted),
        "retrieval": retrieval_summary,
        "latency_ms": latency,
        "tokens": tokens,
        "completeness_checks": [
            {"check": n, "passed": bool(ok), "detail": d} for n, ok, d in checks
        ],
        "missing_required_fields": fx.missing_required,
        "violations": fx.violations,
        # Reported by the LLM server as absent - NOT a telemetry defect.
        "unavailable_metrics": fx.unavailable,
        "malformed_lines": malformed,
        "turn_details": [
            {k: v for k, v in t.items() if k != "summary"} for t in turns
        ],
        "strict_g6": {"passed": not strict_failures, "failures": strict_failures},
    }
    return report


# =============================================================================
# OUTPUT
# =============================================================================

def _fmt(stats: dict, unit: str = "") -> str:
    if not stats or not stats.get("n"):
        return "n=0"
    return (f"n={stats['n']}  mean={stats['mean']:.1f}{unit}  "
            f"median={stats['median']:.1f}{unit}  max={stats['max']:.1f}{unit}")


def print_report(report: dict, source: str) -> None:
    ev, tn = report["events"], report["turns"]
    line = "=" * 78
    print(line)
    print(f"G6 TELEMETRY REPORT  ({source})")
    print(line)
    print(f"  events: {ev['records']} valid, {ev['malformed_lines']} malformed | "
          f"unique sessions: {ev['unique_sessions']}")
    print(f"  commit turns: {tn['commit_turns']}  complete: {tn['complete_turns']}  "
          f"incomplete: {tn['incomplete_turns']} (unfinished {tn['unfinished_turns']}, "
          f"broken {tn['broken_lifecycle_turns']})  errored: {tn['error_turns']}")
    print(f"  generated answers: {tn['generated_answers']}  "
          f"uncertainty/refusals: {tn['uncertainty_refusals']}  "
          f"presentation: {tn['presentation_turns']}  "
          f"unverified answers: {tn['unverified_generated_answers']}")
    print(f"  controller actions: {report['controller_actions']}")
    print(f"  reuse decisions (actual): {report['reuse_decisions']}")
    r = report["retrieval"]
    print(f"  retrieval calls: total={r['total_retrieval_calls']} "
          f"(early={r['early_retrieval_calls']}, commit={r['commit_retrieval_calls']})  "
          f"exact={r['exact_reuse_count']} extension={r['extension_reuse_count']} "
          f"extension+delta={r['extension_plus_delta_retrieval_count']} "
          f"new={r['new_retrieval_count']}")
    lat = report["latency_ms"]
    print("  latency:")
    print(f"    TTFT, pipeline (commit -> first token): {_fmt(lat['ttft_pipeline_ms'], ' ms')}")
    print(f"    TTFT, LLM only:                         {_fmt(lat['ttft_llm_only_ms'], ' ms')}")
    print(f"    LLM generation:                         {_fmt(lat['llm_generation_ms'], ' ms')}")
    print(f"    server processing per turn:             {_fmt(lat['server_processing_ms'], ' ms')}")
    tk = report["tokens"]
    print("  tokens (as reported by Ollama; never estimated):")
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        print(f"    {k:<18} {_fmt(tk[k])}")
    print(f"    turns with reported tokens: {tk['turns_with_reported_tokens']}  "
          f"not reported by Ollama: {tk['turns_tokens_not_reported_by_ollama']}")
    print("  completeness checks:")
    for c in report["completeness_checks"]:
        print(f"    [{'PASS' if c['passed'] else 'FAIL'}] {c['check']}: {c['detail']}")
    if report["unavailable_metrics"]:
        print(f"  metrics unavailable because Ollama did not report them: "
              f"{len(report['unavailable_metrics'])} (not a telemetry defect)")
    if report["missing_required_fields"]:
        print(f"  MISSING/BROKEN telemetry fields: {len(report['missing_required_fields'])}")
        for m in report["missing_required_fields"][:10]:
            print(f"    line {m['line']}: {m['event']}.{m['field']} ({m['problem']})")
    for v in report["violations"][:10]:
        print(f"    violation [{v['kind']}] line {v['line']}: {v['detail']}")
    for m in report["malformed_lines"][:10]:
        print(f"    malformed line {m['line']}: {m['error']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="G6 telemetry evaluator")
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="telemetry JSONL (default: %(default)s)")
    parser.add_argument("--json", metavar="PATH", help="also write the report as JSON")
    parser.add_argument("--strict-g6", action="store_true",
                        help="exit non-zero if telemetry is malformed, incomplete or inconsistent")
    args = parser.parse_args(argv)

    path = Path(args.input)
    try:
        records, malformed = load_records(path)
    except OSError as exc:
        print(f"Cannot read telemetry file {path}: {exc}", file=sys.stderr)
        return 2

    report = analyze(records, malformed)
    report["input"] = str(path)
    print_report(report, str(path))

    if args.json:
        out = Path(args.json)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nWrote {out}")

    strict = report["strict_g6"]
    if args.strict_g6:
        print("\nSTRICT G6 GATE: " + ("PASS" if strict["passed"] else "FAIL"))
        for failure in strict["failures"]:
            print(f"  - {failure}")
        print("=" * 78)
        return 0 if strict["passed"] else 1

    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())