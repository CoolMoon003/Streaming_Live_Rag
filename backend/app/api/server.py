"""Local demo adapter for Streaming-Live-RAG (Phase 7).

Thin FastAPI + WebSocket layer over the EXISTING StreamingRagOrchestrator.
It does not change retrieval, gating, generation or session logic: it only
forwards partial/commit text to the orchestrator and relays its events.

Start (from the project root):
    .venv\\Scripts\\python -m uvicorn backend.app.api.server:app --port 8000
Open:
    http://localhost:8000
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

from backend.app.models.session import SessionState, normalize_query_tokens

ROOT = Path(__file__).resolve().parents[3]
CHUNKS_PATH = ROOT / "data" / "processed" / "phase6_chunks.jsonl"
DEMO_PAGE = ROOT / "demo" / "index.html"
EARLY_EVIDENCE_LIMIT = 5

# G6 telemetry: one append-only JSON Lines file, created on first write.
# Read at call time so tests / deployments can redirect it.
TELEMETRY_DIR = ROOT / "data" / "telemetry"
TELEMETRY_PATH = TELEMETRY_DIR / "live_events.jsonl"

logger = logging.getLogger("streaming_live_rag.telemetry")
_telemetry_lock = threading.Lock()


# --------------------------------------------------------------------------
# Helpers (pure; no production logic)
# --------------------------------------------------------------------------
def load_chunk_lookup(path: Path) -> dict[str, dict[str, Any]]:
    """chunk_id -> {chunk_id, doc_id, section, text, source}, read once at startup."""
    lookup: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cid = d.get("chunk_id")
            if not cid:
                continue
            lookup[cid] = {
                "chunk_id": cid,
                "doc_id": d.get("doc_id") or d.get("document_id") or "",
                "section": d.get("section", ""),
                "text": d.get("text", ""),
                "source": d.get("source", ""),
            }
    return lookup


def chunks_for(ids: list[str], lookup: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(lookup[i]) for i in dict.fromkeys(ids) if i in lookup]


def early_chunk_ids(session: SessionState) -> list[str]:
    """Top chunk ids of the retrieval the session just accepted (read-only)."""
    ids: list[str] = []
    for item in session.latest_results or []:
        chunk = item.get("chunk") if isinstance(item, dict) else None
        cid = (chunk or {}).get("chunk_id") or (item.get("chunk_id") if isinstance(item, dict) else None)
        if cid:
            ids.append(cid)
    return list(dict.fromkeys(ids))[:EARLY_EVIDENCE_LIMIT]


def answer_evidence_ids(event: dict[str, Any]) -> tuple[list[str], dict[str, list[int]]]:
    """Chunk ids named by answer_started, and which intents each belongs to."""
    ids: list[str] = list(event.get("supported_chunks") or [])
    owners: dict[str, list[int]] = {}
    for intent in event.get("intents") or []:
        for cid in intent.get("chunk_ids") or []:
            ids.append(cid)
            owners.setdefault(cid, []).append(intent.get("intent_id"))
    return ids, owners


def describe_reuse(orchestrator: Any, session: SessionState, text: str) -> dict[str, Any] | None:
    """Predict, BEFORE the commit runs, which retrieval path process_commit will take.

    Uses only public, side-effect-free calls that process_commit itself uses
    (refinement analyzer, session.find_reusable_retrieval, delta helpers).
    Returns None if anything is unavailable, in which case the UI omits it.
    """
    try:
        from backend.app.models.session import build_delta_query, delta_needs_retrieval

        previous = session.previous_query if session.previous_query else session.current_query
        dec = orchestrator.refinement_analyzer.analyze(previous_query=previous, new_query=text)
        rtype = dec.refinement_type.value
        out: dict[str, Any] = {"type": "reuse_decision", "refinement_type": rtype,
                               "added_tokens": [], "reason": ""}
        if rtype == "PRESENTATION":
            out.update(mode="none", label="No retrieval (presentation only)")
        elif rtype == "ADDITIVE":
            out.update(mode="additive", label="Additive refinement retrieval")
        elif rtype == "REPLACEMENT":
            out.update(mode="new", label="New retrieval (replacement)")
        else:
            reuse = session.find_reusable_retrieval(text)
            out["reason"] = reuse.reason
            if reuse.mode == "exact":
                out.update(mode="exact", label="Exact reuse")
            elif reuse.mode == "extension":
                delta_q = build_delta_query(reuse.added_tokens)
                needs = bool(delta_q and delta_needs_retrieval(delta_q, reuse.retrieval.results))
                out.update(
                    mode="extension_delta" if needs else "extension",
                    label="Extension reuse + delta retrieval" if needs else "Extension reuse (no delta retrieval)",
                    added_tokens=list(reuse.added_tokens),
                )
            else:
                out.update(mode="new", label="New retrieval")
        return out
    except Exception:
        return None


async def send(ws: WebSocket, msg: dict[str, Any]) -> None:
    await ws.send_text(json.dumps(msg, default=str))


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 1)


# --------------------------------------------------------------------------
# G6 telemetry (observational only: never changes what is sent to the client)
# --------------------------------------------------------------------------
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json_safe(value: Any) -> Any:
    """Plain JSON types only: no NaN/Inf, no arbitrary Python objects."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    return str(value)


def _write_telemetry(event: dict[str, Any]) -> bool:
    """Append one event to the telemetry JSONL file (UTF-8, one line per event).

    Creates the directory on demand. Never raises: telemetry failure must not
    break the WebSocket. Returns True when the line was written.
    """
    try:
        record = _json_safe(dict(event))
        record.setdefault("timestamp", _utc_now())
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"

        path = Path(TELEMETRY_PATH)
        with _telemetry_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(line)
        return True
    except Exception as exc:  # telemetry must never break a live session
        logger.warning("telemetry write failed: %s: %s", type(exc).__name__, exc)
        return False


# Orchestrator event fields that are safe to persist (IDs, counts, scalars).
# Anything else - transcript text, answer text, evidence chunks, tokens - is
# deliberately NOT copied.
_EVENT_FIELDS = (
    "action", "reason", "confidence", "refinement_type",
    "query_version", "generation_id", "active_generation_id", "answer_version",
    "is_multi_intent", "chunks_retrieved", "evidence_sufficient",
    "citation_valid", "citations", "latest_citations", "supported_chunks",
    "retrieval_reuse_mode", "retrieval_reuse_reason",
    "retrieval_calls_this_turn", "token_metrics", "token_metrics_reason",
    "metrics",
)
_INTENT_FIELDS = ("intent_id", "supported", "reason", "retrieved", "top_score", "chunk_ids")

# Voice-layer telemetry (Phase 8, additive only). These never reach the
# orchestrator; they are purely descriptive fields the browser reports about
# its own microphone/TTS/session behaviour, logged next to the existing G6
# events so a turn's input_mode is visible, and as a separate "voice_event"
# record for anything that isn't tied to a specific turn.
_VOICE_EVENT_FIELDS = (
    "input_mode", "transcription_ms", "raw_transcript_length",
    "final_transcript_length", "answer_spoken", "duplicate_detected",
    "similarity_score", "mode",
)


def voice_event_fields(msg: dict[str, Any]) -> dict[str, Any]:
    """Allowlisted, telemetry-safe fields out of a client `voice_event` message."""
    out = {k: msg[k] for k in _VOICE_EVENT_FIELDS if k in msg}
    name = msg.get("event")
    if isinstance(name, str) and name:
        out["voice_event_name"] = name
    return out


def telemetry_fields(event: dict[str, Any]) -> dict[str, Any]:
    """Reduce an orchestrator event to its telemetry-safe fields."""
    out: dict[str, Any] = {}
    for key in _EVENT_FIELDS:
        if key in event:
            out[key] = event[key]  # explicit None (e.g. token_metrics) is kept

    if event.get("timestamp"):
        out["event_timestamp"] = event["timestamp"]
    if isinstance(event.get("answer"), str):
        out["answer_chars"] = len(event["answer"])
    if isinstance(event.get("latest_answer"), str):
        out["latest_answer_chars"] = len(event["latest_answer"])
    if isinstance(event.get("query"), str):
        out["query_chars"] = len(event["query"])
    if isinstance(event.get("intents"), list):
        out["intents"] = [
            {k: i.get(k) for k in _INTENT_FIELDS if k in i}
            for i in event["intents"]
            if isinstance(i, dict)
        ]
    return out


class TelemetryTrail:
    """Per-WebSocket-session recorder. Owns the seq counter and turn ids."""

    def __init__(self, session_id: str, **start_fields: Any):
        self.session_id = session_id
        self._seq = 0
        self._commit_turns = 0
        self.emit(
            "session_reset" if "previous_session_id" in start_fields else "session_started",
            source="server",
            **start_fields,
        )

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        self._seq += 1
        record = {
            "seq": self._seq,
            "timestamp": _utc_now(),
            "session_id": self.session_id,
            "event": event,
            **fields,
        }
        _write_telemetry(record)
        return record

    def log_orchestrator_event(self, event: dict[str, Any], **extra: Any) -> None:
        fields = telemetry_fields(event)
        self.emit(str(event.get("event")), source="orchestrator", **fields, **extra)

    # ---- commit turn lifecycle -------------------------------------------
    def begin_commit(self, text: str, input_mode: str = "text") -> tuple[str, dict[str, Any]]:
        self._commit_turns += 1
        turn_id = f"{self.session_id}:c{self._commit_turns}"
        self.emit(
            "commit_start", source="server", turn_id=turn_id,
            query_chars=len(text),
            query_token_count=len(normalize_query_tokens(text)),
            input_mode=input_mode,
        )
        state: dict[str, Any] = {
            "turn_id": turn_id, "started": None, "started_ms": None,
            "refused": None, "completed": None,
            "first_token_ms": None, "tokens_streamed": 0,
            "input_mode": input_mode,
        }
        return turn_id, state

    def observe_commit_event(self, state: dict[str, Any], event: dict[str, Any], ms: float) -> None:
        kind = event.get("event")
        turn_id = state["turn_id"]

        if kind == "answer_token":
            # Never one telemetry line per token: only count + first-token time.
            state["tokens_streamed"] += 1
            if state["first_token_ms"] is None:
                state["first_token_ms"] = ms
                self.emit("first_token", source="server", turn_id=turn_id, ttft_ms=ms)
            return

        if kind in ("answer_started", "uncertainty_emitted"):
            self.emit(
                "retrieval", source="orchestrator", turn_id=turn_id,
                retrieval_reuse_mode=event.get("retrieval_reuse_mode"),
                retrieval_reuse_reason=event.get("retrieval_reuse_reason"),
                retrieval_calls=event.get("retrieval_calls_this_turn"),
                refinement_type=event.get("refinement_type"),
                query_version=event.get("query_version"),
                generation_id=event.get("generation_id"),
                is_multi_intent=event.get("is_multi_intent"),
            )

        self.log_orchestrator_event(event, turn_id=turn_id, server_ms=ms)

        if kind == "answer_started":
            state["started"], state["started_ms"] = event, ms
        elif kind == "uncertainty_emitted":
            state["refused"] = event
        elif kind == "answer_completed":
            state["completed"] = event

    def finish_commit(self, state: dict[str, Any], server_ms: float) -> None:
        completed = state["completed"]
        refused = state["refused"]
        token_metrics = (completed or {}).get("token_metrics")
        llm_called = isinstance(token_metrics, dict)
        token_metrics = token_metrics if llm_called else {}
        source = completed or refused or state["started"] or {}

        if refused:
            outcome = "refused"
        elif completed and llm_called:
            outcome = "answered"
        elif completed:
            outcome = "presentation"
        else:
            outcome = "no_answer"

        self.emit(
            "request_finished", source="server", kind="commit",
            turn_id=state["turn_id"], server_ms=server_ms, outcome=outcome,
            llm_called=llm_called, input_mode=state.get("input_mode", "text"),
            tokens_streamed=state["tokens_streamed"],
            ttft_ms=state["first_token_ms"],
            pre_generation_ms=(
                state["started_ms"] if state["started_ms"] is not None else server_ms
            ),
            llm_ttft_ms=token_metrics.get("llm_ttft_ms"),
            llm_generation_ms=token_metrics.get("llm_generation_ms"),
            prompt_tokens=token_metrics.get("prompt_tokens"),
            completion_tokens=token_metrics.get("completion_tokens"),
            total_tokens=token_metrics.get("total_tokens"),
            token_metrics_reason=source.get("token_metrics_reason"),
            retrieval_calls=source.get("retrieval_calls_this_turn"),
            retrieval_reuse_mode=source.get("retrieval_reuse_mode"),
            refinement_type=source.get("refinement_type"),
            citation_valid=(completed or {}).get("citation_valid"),
            citation_count=len((completed or refused or {}).get("citations") or []),
        )


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    os.chdir(ROOT)  # existing components use project-relative index paths
    from backend.app.orchestration.streaming_rag_orchestrator import StreamingRagOrchestrator

    app.state.chunk_lookup = load_chunk_lookup(CHUNKS_PATH)
    app.state.orchestrator = StreamingRagOrchestrator(chunks_path=str(CHUNKS_PATH))
    yield


app = FastAPI(title="Streaming-Live-RAG demo", lifespan=lifespan)


@app.get("/")
async def index():
    if DEMO_PAGE.exists():
        return FileResponse(DEMO_PAGE)
    return HTMLResponse("demo/index.html not found", status_code=404)


def _trail_for(session: SessionState) -> TelemetryTrail:
    """The session's trail; created once so seq numbers never restart."""
    trail = getattr(session, "_g6_trail", None)
    if trail is None:
        trail = TelemetryTrail(session.session_id)
        session._g6_trail = trail
    return trail


def make_send(ws: WebSocket, session: SessionState, active: dict[str, Any]):
    """Bind a send function to one turn's session.

    Every message is tagged with the session it belongs to. If a `reset`
    has since made a different session active, the message is dropped
    instead of reaching the client: this is what stops a retrieval/answer
    that was already running when RESET was pressed from repopulating the
    UI after the fact (it is still allowed to finish server-side; its
    output is just discarded).
    """
    async def _send(msg: dict[str, Any]) -> None:
        msg = dict(msg)
        msg.setdefault("session_id", session.session_id)
        if active.get("session_id") != session.session_id:
            return
        await send(ws, msg)
    return _send


async def handle_partial(ws, orch, session, lookup, text, trail=None, input_mode="text",
                          send_fn=None):
    trail = trail or _trail_for(session)
    send_fn = send_fn or (lambda msg: send(ws, msg))
    t0 = time.perf_counter()
    try:
        event = await orch.process_partial(session, text)
    except BaseException as exc:
        trail.emit("request_error", source="server", kind="partial",
                   error_type=type(exc).__name__, server_ms=_ms(t0))
        raise
    ms = _ms(t0)
    trail.log_orchestrator_event(event, server_ms=ms)
    await send_fn({"type": "orchestrator_event", "stage": "partial", **event, "server_ms": ms})
    if event.get("event") == "retrieval_update":
        await send_fn({"type": "evidence_chunks", "stage": "early_retrieval",
                       "chunks": chunks_for(early_chunk_ids(session), lookup)})
    trail.emit("request_finished", source="server", kind="partial", server_ms=ms,
               outcome=event.get("event"), action=event.get("action"),
               retrieval_calls=event.get("retrieval_calls_this_turn"),
               input_mode=input_mode)
    await send_fn({"type": "request_finished", "kind": "partial", "server_ms": ms})


async def handle_commit(ws, orch, session, lookup, text, trail=None, input_mode="text",
                         send_fn=None):
    trail = trail or _trail_for(session)
    send_fn = send_fn or (lambda msg: send(ws, msg))
    turn_id, state = trail.begin_commit(text, input_mode=input_mode)
    reuse = describe_reuse(orch, session, text)
    if reuse:
        trail.emit("reuse_decision", source="server_prediction", turn_id=turn_id,
                   mode=reuse.get("mode"), refinement_type=reuse.get("refinement_type"),
                   reason=reuse.get("reason"),
                   added_token_count=len(reuse.get("added_tokens") or []))
        await send_fn(reuse)
    t0 = time.perf_counter()
    try:
        async for event in orch.process_commit(session, text):
            ms = _ms(t0)
            trail.observe_commit_event(state, event, ms)
            await send_fn({"type": "orchestrator_event", "stage": "commit", **event,
                           "server_ms": ms})
            if event.get("event") == "answer_started" and event.get("action") == "ANSWER":
                ids, owners = answer_evidence_ids(event)
                chunks = chunks_for(ids, lookup)
                for c in chunks:
                    c["intent_ids"] = owners.get(c["chunk_id"], [])
                await send_fn({"type": "evidence_chunks", "stage": "answer", "chunks": chunks})
            elif event.get("event") == "uncertainty_emitted":
                await send_fn({"type": "evidence_chunks", "stage": "none", "chunks": []})
    except BaseException as exc:
        trail.emit("request_error", source="server", kind="commit", turn_id=turn_id,
                   error_type=type(exc).__name__, server_ms=_ms(t0))
        raise
    total_ms = _ms(t0)
    trail.finish_commit(state, total_ms)
    await send_fn({"type": "request_finished", "kind": "commit", "server_ms": total_ms})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    orch = app.state.orchestrator
    lookup = app.state.chunk_lookup
    session = SessionState(session_id=f"demo-{uuid.uuid4().hex[:8]}")
    trail = TelemetryTrail(session.session_id)

    # `active` names the one session whose events are still allowed to reach
    # the client; `session_lock` serializes partial/commit turns *within*
    # one session so a burst of partials/commits can never run concurrently
    # against it (retrieval-spam guard). A `reset` swaps both to a fresh
    # session id and a fresh lock, so it is never blocked behind whatever
    # the previous session's lock is currently holding.
    active: dict[str, Any] = {"session_id": session.session_id}
    session_lock = asyncio.Lock()
    pending_tasks: set[asyncio.Task] = set()

    async def run_turn(handler, session_obj, trail_obj, text, input_mode, lock):
        send_fn = make_send(ws, session_obj, active)
        async with lock:
            # Superseded by a reset while queued behind an earlier turn on
            # this same (pre-reset) session: never start it.
            if active.get("session_id") != session_obj.session_id:
                return
            try:
                await handler(ws, orch, session_obj, lookup, text, trail_obj,
                              input_mode=input_mode, send_fn=send_fn)
            except asyncio.CancelledError:
                raise
            except WebSocketDisconnect:
                pass
            except Exception as exc:  # keep the socket alive for the demo
                try:
                    await send_fn({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
                except Exception:
                    pass

    try:
        while True:
            # A plain `await ws.receive_text()` with the handler awaited
            # inline (the previous shape) blocks this loop for the entire
            # duration of a commit's retrieval + streamed generation, so a
            # `reset` sent mid-stream could not be received until the
            # commit finished. Dispatching partial/commit as background
            # tasks keeps this loop free to receive the next message -
            # including `reset` - immediately.
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
                kind = msg.get("type")
                text = str(msg.get("text", "")).strip()
                input_mode = str(msg.get("input_mode") or "text")
                if kind == "reset":
                    previous_id = session.session_id
                    session = SessionState(session_id=f"demo-{uuid.uuid4().hex[:8]}")
                    trail = TelemetryTrail(session.session_id, previous_session_id=previous_id)
                    session_lock = asyncio.Lock()
                    active["session_id"] = session.session_id
                    await send(ws, {"type": "reset_done", "session_id": session.session_id})
                elif kind in ("partial", "commit"):
                    if not text:
                        await send(ws, {"type": "error", "message": "Empty transcript."})
                        continue
                    handler = handle_partial if kind == "partial" else handle_commit
                    task = asyncio.create_task(
                        run_turn(handler, session, trail, text, input_mode, session_lock)
                    )
                    pending_tasks.add(task)
                    task.add_done_callback(pending_tasks.discard)
                elif kind == "voice_event":
                    # Client-reported voice/session telemetry only (mic, TTS,
                    # duplicate detection). Never touches the orchestrator or
                    # session state - just an additional G6-adjacent record.
                    trail.emit("voice_event", source="client", **voice_event_fields(msg))
                else:
                    await send(ws, {"type": "error", "message": f"Unknown message type: {kind!r}"})
            except WebSocketDisconnect:
                raise
            except Exception as exc:  # keep the socket alive for the demo
                await send(ws, {"type": "error", "message": f"{type(exc).__name__}: {exc}"})
    except WebSocketDisconnect:
        pass
    finally:
        for t in pending_tasks:
            t.cancel()
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        trail.emit("session_ended", source="server", reason="disconnect")