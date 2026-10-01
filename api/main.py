"""
FastAPI wrapper for the Data Pipeline Intelligence Platform.

Endpoints:
  POST /incident            — start triage
  POST /approve              — submit HITL decision
  GET  /status/{thread_id}   — get triage status summary
  GET  /plan/{thread_id}     — get full triage + remediation plan for human review
  GET  /health               — load balancer health check

Authentication: X-API-Key header.
All endpoints propagate X-Request-ID through logging.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import time
import uuid
from pathlib import Path
from threading import Lock, Thread
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Sys path so imports work from any cwd ─────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.graph import graph, make_initial_state  # noqa: E402

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pipeline-api")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Pipeline Intelligence API", version="1.0.0")
START_TIME = time.time()
_incidents_processed = 0

# Simple in-memory store (swap for Redis/Postgres in production)
_threads: dict[str, dict] = {}
_threads_lock = Lock()

_API_KEY = os.getenv("API_KEY", "")
if not _API_KEY:
    raise RuntimeError(
        "API_KEY environment variable must be set (no insecure default is allowed)."
    )
if _API_KEY == "dev-key-change-me":
    logger.warning(
        "API_KEY is set to the well-known placeholder value — this is unsafe for production."
    )

# How long to retain completed/errored thread records before pruning (memory-leak guard).
_THREAD_TTL_SECONDS = int(os.getenv("THREAD_TTL_SECONDS", str(24 * 3600)))


# ── Auth ──────────────────────────────────────────────────────────────────────
def _require_auth(x_api_key: Optional[str]) -> None:
    # Constant-time comparison to avoid timing side-channels.
    if not x_api_key or not secrets.compare_digest(x_api_key, _API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


def _prune_stale_threads() -> None:
    """Drop finished thread records older than the TTL to bound memory growth."""
    cutoff = time.time() - _THREAD_TTL_SECONDS
    with _threads_lock:
        stale = [
            tid for tid, rec in _threads.items()
            if rec.get("status") in ("completed", "error") and rec.get("created_at", time.time()) < cutoff
        ]
        for tid in stale:
            del _threads[tid]


# ── Graph helpers ─────────────────────────────────────────────────────────────
def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _run_to_hitl(thread_id: str, initial_state: dict, request_id: str) -> None:
    logger.info("thread=%s request_id=%s event=triage_start", thread_id, request_id)
    try:
        for _ in graph.stream(initial_state, _config(thread_id)):
            pass
        with _threads_lock:
            _threads[thread_id]["status"] = "awaiting_approval"
        logger.info("thread=%s request_id=%s event=hitl_gate_reached", thread_id, request_id)
    except Exception as exc:
        logger.exception("thread=%s request_id=%s event=triage_error", thread_id, request_id)
        with _threads_lock:
            _threads[thread_id]["status"] = "error"
            _threads[thread_id]["error"] = str(exc)


def _resume(thread_id: str, request_id: str) -> None:
    global _incidents_processed
    logger.info("thread=%s request_id=%s event=resume", thread_id, request_id)
    try:
        for _ in graph.stream(None, _config(thread_id)):
            pass
        state = graph.get_state(_config(thread_id)).values
        with _threads_lock:
            _threads[thread_id]["status"] = "completed"
            _threads[thread_id]["severity"] = state.get("triage", {}).get("severity")
            _threads[thread_id]["steps_executed"] = len(state.get("execution_log", []))
            _incidents_processed += 1
        logger.info("thread=%s request_id=%s event=completed", thread_id, request_id)
    except Exception as exc:
        logger.exception("thread=%s request_id=%s event=resume_error", thread_id, request_id)
        with _threads_lock:
            _threads[thread_id]["status"] = "error"
            _threads[thread_id]["error"] = str(exc)


# ── Request / response schemas ────────────────────────────────────────────────
class IncidentRequest(BaseModel):
    raw_log: str = Field(..., min_length=1, max_length=20_000)
    pipeline_id: str = Field(..., min_length=1, max_length=200)


class ApprovalRequest(BaseModel):
    thread_id: str = Field(..., min_length=1, max_length=200)
    decision: str   # "approved" | "rejected"
    feedback: str = Field("", max_length=5_000)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/incident")
async def create_incident(
    req: IncidentRequest,
    x_api_key: Optional[str] = Header(None),
    x_request_id: Optional[str] = Header(None),
):
    """Start a new pipeline incident triage run."""
    _require_auth(x_api_key)
    request_id = x_request_id or str(uuid.uuid4())

    _prune_stale_threads()

    thread_id = str(uuid.uuid4())
    initial_state = make_initial_state(req.raw_log)

    with _threads_lock:
        _threads[thread_id] = {
            "thread_id": thread_id,
            "pipeline_id": req.pipeline_id,
            "status": "running",
            "created_at": time.time(),
            "request_id": request_id,
        }

    # Run async — returns immediately, graph runs in background to HITL gate
    t = Thread(target=_run_to_hitl, args=(thread_id, initial_state, request_id), daemon=True)
    t.start()

    logger.info(
        "request_id=%s event=incident_created thread_id=%s pipeline=%s",
        request_id, thread_id, req.pipeline_id,
    )
    return JSONResponse(
        status_code=202,
        content={
            "thread_id": thread_id,
            "status": "running",
            "message": "Triage started. Poll /status/{thread_id} for progress.",
            "request_id": request_id,
        },
        headers={"X-Request-ID": request_id},
    )


@app.post("/approve")
async def submit_approval(
    req: ApprovalRequest,
    x_api_key: Optional[str] = Header(None),
    x_request_id: Optional[str] = Header(None),
):
    """Submit human approval or rejection for a pending HITL gate."""
    _require_auth(x_api_key)
    request_id = x_request_id or str(uuid.uuid4())

    if req.decision not in ("approved", "rejected"):
        raise HTTPException(status_code=400, detail="decision must be 'approved' or 'rejected'")

    with _threads_lock:
        if req.thread_id not in _threads:
            raise HTTPException(status_code=404, detail=f"Unknown thread_id: {req.thread_id}")

        status = _threads[req.thread_id].get("status")
        if status != "awaiting_approval":
            raise HTTPException(
                status_code=409,
                detail=f"Thread is not awaiting approval (current status: {status})",
            )
        _threads[req.thread_id]["status"] = "resuming"

    cfg = _config(req.thread_id)
    state = graph.get_state(cfg).values
    attempts = state.get("approval_attempts", 0)

    graph.update_state(
        cfg,
        {
            "hitl_decision": req.decision,
            "human_feedback": req.feedback,
            "approval_attempts": attempts + 1,
        },
    )

    t = Thread(target=_resume, args=(req.thread_id, request_id), daemon=True)
    t.start()

    logger.info(
        "request_id=%s event=approval_submitted thread_id=%s decision=%s",
        request_id, req.thread_id, req.decision,
    )
    return JSONResponse(
        content={
            "thread_id": req.thread_id,
            "decision": req.decision,
            "status": "resuming",
            "request_id": request_id,
        },
        headers={"X-Request-ID": request_id},
    )


@app.get("/status/{thread_id}")
async def get_status(
    thread_id: str,
    x_api_key: Optional[str] = Header(None),
    x_request_id: Optional[str] = Header(None),
):
    """Get current triage status for a thread."""
    _require_auth(x_api_key)
    request_id = x_request_id or str(uuid.uuid4())

    with _threads_lock:
        if thread_id not in _threads:
            raise HTTPException(status_code=404, detail=f"Unknown thread_id: {thread_id}")
        record = dict(_threads[thread_id])

    live: dict = {}
    try:
        state = graph.get_state(_config(thread_id)).values
        live = {
            "severity": state.get("triage", {}).get("severity"),
            "root_cause_snippet": (state.get("root_cause") or "")[:200],
            "plan_steps": len(state.get("plan", {}).get("steps", [])),
            "steps_executed": len(state.get("execution_log", [])),
            "approval_attempts": state.get("approval_attempts", 0),
            "audit_entries": len(state.get("audit_trail", [])),
        }
    except Exception:
        logger.exception("thread=%s request_id=%s event=status_live_state_error", thread_id, request_id)

    return JSONResponse(
        content={
            "thread_id": thread_id,
            "pipeline_id": record.get("pipeline_id"),
            "status": record.get("status"),
            **live,
            "request_id": request_id,
        },
        headers={"X-Request-ID": request_id},
    )


@app.get("/plan/{thread_id}")
async def get_plan(
    thread_id: str,
    x_api_key: Optional[str] = Header(None),
    x_request_id: Optional[str] = Header(None),
):
    """Full triage + remediation plan for human review (what /approve actually approves)."""
    _require_auth(x_api_key)
    request_id = x_request_id or str(uuid.uuid4())

    with _threads_lock:
        if thread_id not in _threads:
            raise HTTPException(status_code=404, detail=f"Unknown thread_id: {thread_id}")

    try:
        state = graph.get_state(_config(thread_id)).values
    except Exception:
        logger.exception("thread=%s request_id=%s event=plan_fetch_error", thread_id, request_id)
        raise HTTPException(status_code=500, detail="Could not load graph state for thread.")

    return JSONResponse(
        content={
            "thread_id": thread_id,
            "triage": state.get("triage", {}),
            "root_cause": state.get("root_cause", ""),
            "plan": state.get("plan", {}),
            "execution_log": state.get("execution_log", []),
            "audit_trail": state.get("audit_trail", []),
            "approval_attempts": state.get("approval_attempts", 0),
            "request_id": request_id,
        },
        headers={"X-Request-ID": request_id},
    )


@app.get("/health")
async def health():
    """Load-balancer health check — fast, no external dependencies."""
    return {
        "status": "ok",
        "uptime_s": round(time.time() - START_TIME, 1),
        "incidents_processed": _incidents_processed,
        "model": os.getenv("OPENAI_MODEL") or os.getenv("deployment_name", "unknown"),
    }


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    """Never leak stack traces / internals to clients; log full detail server-side."""
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    logger.exception("request_id=%s event=unhandled_exception path=%s", request_id, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error.", "request_id": request_id},
        headers={"X-Request-ID": request_id},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
