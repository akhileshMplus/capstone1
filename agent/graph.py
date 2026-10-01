"""
LangGraph orchestration for the Data Pipeline Intelligence Platform.

Graph:
    ingest → retrieve → diagnose → plan → [hitl_gate] → execute
                                              ↑
                                    rejected (bounded retries) ──┘

Interrupt is placed *before* hitl_gate so the graph pauses there awaiting
a human decision (injected via graph.update_state()).
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.tools import execute_sql, parse_log, validate_fix
from agent.prompts import (
    SEVERITY_TMPL,
    ROOT_CAUSE_TMPL,
    PLAN_TMPL,
    TriageResult,
    RemediationPlan,
)
from rag.retriever import hybrid_retrieve, has_strong_precedent

logger = logging.getLogger("pipeline-agent")

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "data" / "pipeline_registry.json"

MAX_HITL_RETRIES = 3   # bounded retry loop — escalate after this many rejections


# ── LLM helper ────────────────────────────────────────────────────────────────
_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        endpoint = os.getenv("OPENAI_BASE_URL") or os.getenv("endpoint")
        api_key = os.getenv("OPENAI_API_KEY", "sk-placeholder")
        # Explicit timeout + retries so a stalled upstream never hangs a graph node forever.
        _client = OpenAI(base_url=endpoint, api_key=api_key, timeout=30.0, max_retries=2)
    return _client


def _llm(prompt: str, max_tokens: int = 1024) -> str:
    model = os.getenv("OPENAI_MODEL") or os.getenv("deployment_name", "gpt-4o-mini")
    client = _get_client()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error("event=llm_call_failed error=%s", e)
        return f"LLM_ERROR: {e}"


_JINJA_ENV = None


def _jinja_render(template_str: str, **kwargs) -> str:
    """Render a Jinja2 template string; adds a tojson filter that handles non-JSON types (e.g. datetime)."""
    global _JINJA_ENV
    if _JINJA_ENV is None:
        from jinja2 import Environment
        _JINJA_ENV = Environment()
        _JINJA_ENV.filters["tojson"] = lambda v, indent=None: json.dumps(v, indent=indent, default=str)
    return _JINJA_ENV.from_string(template_str).render(**kwargs)


def _parse_json_response(raw: str) -> dict:
    """Strip markdown fences and parse the first JSON object found."""
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON object found in response: {raw[:200]}")


def _audit(trail: list, event: str, data: dict | None = None) -> list:
    """Append an audit event and return the updated trail."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
    }
    if data:
        entry.update(data)
    return trail + [entry]


# ── State schema ──────────────────────────────────────────────────────────────
class PipelineState(TypedDict):
    raw_log:            str
    parsed_incident:    dict
    pipeline_meta:      dict
    similar_incidents:  list
    sql_result:         dict
    root_cause:         str
    triage:             dict
    plan:               dict
    hitl_decision:      str   # "approved" | "rejected" | "pending"
    human_feedback:     str
    approval_attempts:  int
    execution_log:      list
    audit_trail:        list


# ── Pipeline registry lookup ──────────────────────────────────────────────────
def _lookup_pipeline(name: str) -> dict:
    try:
        registry = json.loads(REGISTRY_PATH.read_text())
        for p in registry:
            if p["name"] == name:
                return p
    except Exception:
        logger.exception("event=pipeline_registry_lookup_failed pipeline=%s", name)
    return {"name": name, "criticality": "unknown", "sla_hours": 8,
            "downstream": [], "owner": "unknown"}


# ── NODE 1: Ingest ────────────────────────────────────────────────────────────
def node_ingest(state: PipelineState) -> dict:
    parsed = parse_log(state["raw_log"])
    pipeline_name = parsed.get("pipeline") or "unknown"
    meta = _lookup_pipeline(pipeline_name)

    trail = _audit(
        state.get("audit_trail", []),
        "INGEST",
        {
            "pipeline": pipeline_name,
            "error_type": parsed.get("error_type"),
            "affected_rows": parsed.get("affected_rows"),
            "timestamp": parsed.get("timestamp"),
        },
    )
    return {
        "parsed_incident": parsed,
        "pipeline_meta": meta,
        "audit_trail": trail,
    }


# ── NODE 2: Retrieve ──────────────────────────────────────────────────────────
def node_retrieve(state: PipelineState) -> dict:
    parsed = state["parsed_incident"]
    pipeline_name = parsed.get("pipeline", "unknown")
    error_type = parsed.get("error_type", "")
    hint = parsed.get("llm_root_cause_hint", "")

    query = f"{pipeline_name} {error_type} {hint}"
    results = hybrid_retrieve(query, top_k=3)

    strong = has_strong_precedent(results)
    trail = _audit(
        state["audit_trail"],
        "RETRIEVE",
        {
            "query": query[:100],
            "results": [
                {"id": r["incident"]["id"], "rrf_score": r["rrf_score"]}
                for r in results
            ],
            "strong_precedent": strong,
        },
    )
    return {
        "similar_incidents": results,
        "audit_trail": trail,
    }


# ── NODE 3: Diagnose ──────────────────────────────────────────────────────────
def node_diagnose(state: PipelineState) -> dict:
    parsed = state["parsed_incident"]
    meta = state["pipeline_meta"]
    similar = state["similar_incidents"]
    pipeline_name = parsed.get("pipeline", "unknown")

    # Diagnostic SQL — check row counts for this pipeline's tables
    sql_q = (
        "SELECT table_name, measured_at, row_count, expected_min "
        "FROM row_counts ORDER BY measured_at DESC LIMIT 5"
    )
    sql_result = execute_sql(sql_q)

    # Root cause — few-shot grounded
    rc_prompt = _jinja_render(
        ROOT_CAUSE_TMPL,
        pipeline_name=pipeline_name,
        parsed_incident=parsed,
        sql_result=sql_result,
        similar_incidents=similar,
    )
    root_cause = _llm(rc_prompt, max_tokens=768)

    # Severity — CoT
    downstream = meta.get("downstream", [])
    sev_prompt = _jinja_render(
        SEVERITY_TMPL,
        pipeline_name=pipeline_name,
        criticality=meta.get("criticality", "unknown"),
        sla_hours=meta.get("sla_hours", 8),
        error_type=parsed.get("error_type", ""),
        parsed_incident=parsed,
        downstream=downstream,
        sql_result=sql_result,
        similar_incidents=similar,
    )
    sev_raw = _llm(sev_prompt, max_tokens=2048)

    # Parse severity — fall back to P2 on failure
    triage_dict: dict = {}
    try:
        raw_dict = _parse_json_response(sev_raw)
        triage = TriageResult(**raw_dict)
        triage_dict = triage.model_dump()
    except Exception:
        triage_dict = {
            "pipeline": pipeline_name,
            "severity": "P2",
            "error_type": parsed.get("error_type", "unknown"),
            "sla_at_risk": True,
            "downstream_count": len(downstream),
            "reasoning": f"[Parse fallback] Raw LLM: {sev_raw[:300]}",
        }

    trail = _audit(
        state["audit_trail"],
        "DIAGNOSE",
        {
            "severity": triage_dict.get("severity"),
            "root_cause_snippet": root_cause[:150],
        },
    )
    return {
        "sql_result": sql_result,
        "root_cause": root_cause,
        "triage": triage_dict,
        "audit_trail": trail,
    }


# ── NODE 4: Plan ──────────────────────────────────────────────────────────────
def node_plan(state: PipelineState) -> dict:
    parsed = state["parsed_incident"]
    meta = state["pipeline_meta"]
    triage = state["triage"]
    similar = state["similar_incidents"]
    sql_result = state.get("sql_result", {})
    human_feedback = state.get("human_feedback", "")
    pipeline_name = parsed.get("pipeline", "unknown")

    # Generate a stable incident ID from thread context
    incident_id = f"INC-NEW-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    plan_prompt = _jinja_render(
        PLAN_TMPL,
        incident_id=incident_id,
        pipeline_name=pipeline_name,
        severity=triage.get("severity", "P2"),
        root_cause=state.get("root_cause", ""),
        parsed_incident=parsed,
        similar_incidents=similar,
        sql_result=sql_result,
        human_feedback=human_feedback,
    )
    plan_raw = _llm(plan_prompt, max_tokens=3072)

    plan_dict: dict = {}
    try:
        raw_dict = _parse_json_response(plan_raw)
        # Ensure requires_approval is always True
        raw_dict["requires_approval"] = True
        plan = RemediationPlan(**raw_dict)
        plan_dict = plan.model_dump()
    except Exception as exc:
        # Fallback plan — keep something useful rather than crashing
        plan_dict = {
            "incident_id": incident_id,
            "root_cause": state.get("root_cause", "Unknown"),
            "similar_incidents": [r["incident"]["id"] for r in similar],
            "steps": [
                {
                    "step_number": 1,
                    "action": "Manual investigation required — automated plan generation failed.",
                    "sql_or_cmd": None,
                    "risk_level": "low",
                    "reversible": True,
                }
            ],
            "estimated_mins": 60,
            "requires_approval": True,
            "_parse_error": str(exc),
            "_raw": plan_raw[:500],
        }

    # Validate every step that has sql_or_cmd
    for step in plan_dict.get("steps", []):
        if step.get("sql_or_cmd"):
            validation = validate_fix(step["sql_or_cmd"])
            step["_validation"] = validation
            # If hard-rejected (dangerous, proceed_to_hitl=False), mark the step
            if not validation.get("proceed_to_hitl", True):
                step["_blocked"] = True
                step["_block_reason"] = validation["reason"]

    trail = _audit(
        state["audit_trail"],
        "PLAN",
        {
            "incident_id": plan_dict.get("incident_id"),
            "steps": len(plan_dict.get("steps", [])),
            "estimated_mins": plan_dict.get("estimated_mins"),
            "revision_from_feedback": bool(human_feedback),
        },
    )
    return {
        "plan": plan_dict,
        "audit_trail": trail,
    }


# ── NODE 5: HITL Gate ─────────────────────────────────────────────────────────
def node_hitl(state: PipelineState) -> dict:
    """
    This node prints the approval request and pauses.
    The graph is compiled with interrupt_before=["hitl_gate"] so execution
    stops here; the decision arrives via graph.update_state().
    """
    plan = state.get("plan", {})
    triage = state.get("triage", {})
    attempts = state.get("approval_attempts", 0)

    step_lines = []
    for step in plan.get("steps", []):
        blocked = " BLOCKED" if step.get("_blocked") else ""
        line = f"  [{step['step_number']}] ({step['risk_level'].upper()}) {step['action']}{blocked}"
        if step.get("sql_or_cmd"):
            line += f"\n       SQL: {step['sql_or_cmd'][:100]}"
        if v := step.get("_validation"):
            line += f"\n       Validation: {v.get('rating', '?')} — {v.get('reason', '')[:80]}"
        step_lines.append(line)

    logger.info(
        "event=hitl_approval_required pipeline=%s severity=%s sla_at_risk=%s "
        "attempt=%s/%s estimated_mins=%s\nRoot cause: %s\nReasoning: %s\nSteps (%d):\n%s",
        triage.get("pipeline", "unknown"),
        triage.get("severity", "?"),
        triage.get("sla_at_risk"),
        attempts + 1, MAX_HITL_RETRIES,
        plan.get("estimated_mins", "?"),
        state.get("root_cause", "")[:200],
        triage.get("reasoning", "")[:300],
        len(plan.get("steps", [])),
        "\n".join(step_lines),
    )

    trail = _audit(
        state["audit_trail"],
        "HITL_GATE",
        {
            "attempt": attempts + 1,
            "severity": triage.get("severity"),
            "steps_count": len(plan.get("steps", [])),
        },
    )
    return {"audit_trail": trail}


# ── NODE 6: Execute ───────────────────────────────────────────────────────────
def node_execute(state: PipelineState) -> dict:
    """
    Execute approved remediation steps.
    Low/medium risk steps run against the mock DB.
    High-risk steps are logged but not auto-executed (require manual run).
    Hard-blocked steps are skipped.
    """
    plan = state.get("plan", {})
    execution_log: list = list(state.get("execution_log", []))
    trail = list(state.get("audit_trail", []))

    for step in plan.get("steps", []):
        step_num = step["step_number"]
        sql = step.get("sql_or_cmd")
        risk = step.get("risk_level", "high")
        blocked = step.get("_blocked", False)

        if blocked:
            execution_log.append({
                "step": step_num,
                "action": step["action"],
                "status": "skipped_blocked",
                "reason": step.get("_block_reason", "Hard-rejected by validate_fix"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            continue

        if risk == "high":
            # High-risk steps are proposed but not auto-executed
            execution_log.append({
                "step": step_num,
                "action": step["action"],
                "sql_or_cmd": sql,
                "status": "proposed_manual_execution_required",
                "reason": "High-risk step: must be executed manually by engineer.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            continue

        if sql:
            result = execute_sql(sql)
            if result.get("requires_approval"):
                # Write statement slipped through — log it, don't execute
                execution_log.append({
                    "step": step_num,
                    "action": step["action"],
                    "sql_or_cmd": sql,
                    "status": "blocked_write_statement",
                    "result": result,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            else:
                execution_log.append({
                    "step": step_num,
                    "action": step["action"],
                    "sql_or_cmd": sql,
                    "status": "executed" if "error" not in result else "error",
                    "result": result,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
        else:
            execution_log.append({
                "step": step_num,
                "action": step["action"],
                "status": "no_sql_noted",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    trail = _audit(trail, "EXECUTE", {"steps_executed": len(execution_log)})
    logger.info(
        "event=execution_complete steps_processed=%d\n%s",
        len(execution_log),
        "\n".join(
            f"  Step {e['step']}: {e['status']} — {e['action'][:60]}" for e in execution_log
        ),
    )

    return {
        "execution_log": execution_log,
        "audit_trail": trail,
    }


# ── Routing ───────────────────────────────────────────────────────────────────
def route_after_hitl(state: PipelineState) -> str:
    decision = state.get("hitl_decision", "pending")
    attempts = state.get("approval_attempts", 0)

    if decision == "approved":
        return "execute"

    if decision == "rejected":
        if attempts < MAX_HITL_RETRIES:
            return "plan"   # Re-plan with human feedback
        else:
            # Retries exhausted — escalate
            logger.warning(
                "event=hitl_escalation retries=%d message=%s",
                MAX_HITL_RETRIES,
                "Incident escalated to senior engineer for manual resolution.",
            )
            return END

    # "pending" or anything else — shouldn't happen, but be safe
    return END


# ── Build graph ───────────────────────────────────────────────────────────────
def build_graph():
    builder = StateGraph(PipelineState)

    builder.add_node("ingest",    node_ingest)
    builder.add_node("retrieve",  node_retrieve)
    builder.add_node("diagnose",  node_diagnose)
    builder.add_node("plan",      node_plan)
    builder.add_node("hitl_gate", node_hitl)
    builder.add_node("execute",   node_execute)

    builder.set_entry_point("ingest")
    builder.add_edge("ingest",    "retrieve")
    builder.add_edge("retrieve",  "diagnose")
    builder.add_edge("diagnose",  "plan")
    builder.add_edge("plan",      "hitl_gate")
    builder.add_conditional_edges(
        "hitl_gate",
        route_after_hitl,
        {"execute": "execute", "plan": "plan", END: END},
    )
    builder.add_edge("execute", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory, interrupt_before=["hitl_gate"])


# Compiled graph singleton — import this in server / API
graph = build_graph()


# ── Initial state factory ─────────────────────────────────────────────────────
def make_initial_state(raw_log: str) -> PipelineState:
    return PipelineState(
        raw_log=raw_log,
        parsed_incident={},
        pipeline_meta={},
        similar_incidents=[],
        sql_result={},
        root_cause="",
        triage={},
        plan={},
        hitl_decision="pending",
        human_feedback="",
        approval_attempts=0,
        execution_log=[],
        audit_trail=[],
    )
