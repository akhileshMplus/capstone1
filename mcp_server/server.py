"""
MCP Server for the Data Pipeline Intelligence Platform.

Tools:
  trigger_triage      — start a triage run, return thread_id immediately
  submit_approval     — approve / reject a pending HITL gate
  get_incident_status — status summary for a thread_id

Resource:
  incident_history    — paginated read-only view of processed incidents

Authentication: X-API-Key header (set MCP_API_KEY env var).
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from typing import Literal

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

# ── Import the compiled graph ─────────────────────────────────────────────────
# Adjust sys.path so this module can be run standalone from the repo root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.graph import graph, make_initial_state  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pipeline-mcp")

mcp = FastMCP("pipeline-intelligence")

# ── Simple in-memory incident store ──────────────────────────────────────────
# In production this would be a durable DB.
_incidents: dict[str, dict] = {}
_incidents_lock = Lock()
_INCIDENT_TTL_SECONDS = int(os.getenv("INCIDENT_TTL_SECONDS", str(24 * 3600)))


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _prune_stale_incidents() -> None:
    """Drop finished incident records older than the TTL to bound memory growth."""
    cutoff = time.time()
    with _incidents_lock:
        stale = []
        for tid, rec in _incidents.items():
            if rec.get("status") not in ("completed", "error"):
                continue
            created = rec.get("_created_ts", cutoff)
            if cutoff - created > _INCIDENT_TTL_SECONDS:
                stale.append(tid)
        for tid in stale:
            del _incidents[tid]


def _stream_to_hitl(thread_id: str, initial_state: dict) -> None:
    """Run the graph in a background thread until the HITL interrupt."""
    try:
        for _ in graph.stream(initial_state, _config(thread_id)):
            pass
        with _incidents_lock:
            _incidents[thread_id]["status"] = "awaiting_approval"
    except Exception as exc:
        logger.exception("thread=%s event=triage_error", thread_id)
        with _incidents_lock:
            _incidents[thread_id]["status"] = "error"
            _incidents[thread_id]["error"] = str(exc)


def _stream_resume(thread_id: str) -> None:
    """Resume the graph after a HITL decision."""
    try:
        for _ in graph.stream(None, _config(thread_id)):
            pass
        state = graph.get_state(_config(thread_id)).values
        with _incidents_lock:
            _incidents[thread_id]["status"] = "completed"
            _incidents[thread_id]["severity"] = state.get("triage", {}).get("severity")
            _incidents[thread_id]["steps_executed"] = len(state.get("execution_log", []))
            _incidents[thread_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        logger.exception("thread=%s event=resume_error", thread_id)
        with _incidents_lock:
            _incidents[thread_id]["status"] = "error"
            _incidents[thread_id]["error"] = str(exc)


# ── Auth helper ───────────────────────────────────────────────────────────────
_API_KEY = os.getenv("MCP_API_KEY", "")
if not _API_KEY:
    raise RuntimeError(
        "MCP_API_KEY environment variable must be set (no insecure default is allowed)."
    )
if _API_KEY == "dev-key-change-me":
    logger.warning(
        "MCP_API_KEY is set to the well-known placeholder value — this is unsafe for production."
    )


def _check_auth(api_key: str) -> None:
    # Constant-time comparison to avoid timing side-channels.
    if not api_key or not secrets.compare_digest(api_key, _API_KEY):
        raise PermissionError("Invalid API key.")


# ── Input schemas ─────────────────────────────────────────────────────────────
class TriageRequest(BaseModel):
    raw_log: str
    pipeline_id: str
    api_key: str = ""


class ApprovalRequest(BaseModel):
    thread_id: str
    decision: Literal["approved", "rejected"]
    feedback: str = ""
    api_key: str = ""


# ── Tools ─────────────────────────────────────────────────────────────────────
@mcp.tool()
def trigger_triage(request: TriageRequest) -> dict:
    """
    Start a new pipeline incident triage run.
    Returns thread_id immediately — triage runs asynchronously until the
    HITL gate, at which point it pauses awaiting a submit_approval call.
    """
    _check_auth(request.api_key)

    _prune_stale_incidents()

    thread_id = str(uuid.uuid4())
    initial_state = make_initial_state(request.raw_log)

    with _incidents_lock:
        _incidents[thread_id] = {
            "thread_id": thread_id,
            "pipeline_id": request.pipeline_id,
            "status": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "_created_ts": time.time(),
            "severity": None,
            "steps_executed": 0,
        }

    # Run to HITL in a background thread so the tool returns immediately
    t = Thread(target=_stream_to_hitl, args=(thread_id, initial_state), daemon=True)
    t.start()

    return {"thread_id": thread_id, "status": "running", "message": "Triage started."}


@mcp.tool()
def submit_approval(request: ApprovalRequest) -> dict:
    """
    Submit a human approval or rejection for a pending HITL gate.
    On rejection, include feedback so the agent can replan.
    """
    _check_auth(request.api_key)

    with _incidents_lock:
        if request.thread_id not in _incidents:
            return {"error": f"Unknown thread_id: {request.thread_id}"}

        current_status = _incidents[request.thread_id].get("status")
        if current_status != "awaiting_approval":
            return {
                "error": f"Thread is not awaiting approval (current status: {current_status})"
            }
        _incidents[request.thread_id]["status"] = "resuming"

    # Inject the decision into the graph state
    cfg = _config(request.thread_id)
    state = graph.get_state(cfg).values
    attempts = state.get("approval_attempts", 0)

    graph.update_state(
        cfg,
        {
            "hitl_decision": request.decision,
            "human_feedback": request.feedback,
            "approval_attempts": attempts + 1,
        },
    )

    # Resume graph in background
    t = Thread(target=_stream_resume, args=(request.thread_id,), daemon=True)
    t.start()

    return {
        "thread_id": request.thread_id,
        "decision": request.decision,
        "message": "Decision recorded. Graph resuming.",
    }


@mcp.tool()
def get_incident_status(thread_id: str, api_key: str = "") -> dict:
    """
    Get current triage status, severity, and execution log for a thread.
    Returns a structured subset — not the full internal state.
    Requires a valid api_key (same as trigger_triage / submit_approval).
    """
    _check_auth(api_key)

    with _incidents_lock:
        if thread_id not in _incidents:
            return {"error": f"Unknown thread_id: {thread_id}"}
        record = dict(_incidents[thread_id])

    # Also pull live state from the graph if available
    live: dict = {}
    try:
        state = graph.get_state(_config(thread_id)).values
        live = {
            "severity": state.get("triage", {}).get("severity"),
            "root_cause": state.get("root_cause", "")[:200],
            "plan_steps": len(state.get("plan", {}).get("steps", [])),
            "steps_executed": len(state.get("execution_log", [])),
            "approval_attempts": state.get("approval_attempts", 0),
            "audit_entries": len(state.get("audit_trail", [])),
        }
    except Exception:
        logger.exception("thread=%s event=status_live_state_error", thread_id)

    return {
        "thread_id": thread_id,
        "pipeline_id": record.get("pipeline_id"),
        "status": record.get("status"),
        "created_at": record.get("created_at"),
        "completed_at": record.get("completed_at"),
        **live,
    }


# ── Resource ──────────────────────────────────────────────────────────────────
@mcp.resource("incidents://history/{api_key}")
def incident_history(api_key: str) -> str:
    """
    Streamable resource: full list of processed incidents with outcomes.
    Requires the shared MCP API key as part of the URI (mirrors tool auth).
    Paginate by filtering on your end — returns all records (bounded by process lifetime).
    """
    _check_auth(api_key)
    with _incidents_lock:
        records = list(_incidents.values())
    return json.dumps(
        {
            "incidents": records,
            "total": len(records),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        indent=2,
        default=str,
    )


if __name__ == "__main__":
    port = int(os.getenv("MCP_PORT", "8001"))
    print(f"Starting MCP server on port {port}")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
