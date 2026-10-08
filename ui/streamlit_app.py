"""
Streamlit dashboard for validating the Data Pipeline Intelligence Platform end-to-end.

This is a thin client over the FastAPI backend (api/main.py) — it does not talk to
the LangGraph graph directly, so it exercises the exact same contract a real
caller would use.

IMPORTANT for cloud deployment: this app is only the frontend. Streamlit Community
Cloud runs just this script — it does NOT also run the separate FastAPI backend.
The backend must be deployed separately (e.g. Render/Railway/Fly.io/a VM) and its
public URL configured below (via Secrets or the sidebar). See README.md's
"Deploying to the cloud" section.

Run locally:
    pip install -r requirements-dev.txt
    streamlit run ui/streamlit_app.py
"""
from __future__ import annotations

import os
import time

import requests
import streamlit as st


def _default(key: str, fallback: str) -> str:
    """Prefer st.secrets (cloud deployments), then env vars, then a hardcoded fallback."""
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass  # no secrets.toml configured — fine for local runs
    return os.getenv(key, fallback)


SAMPLE_LOG = """2025-06-01 02:14 UTC | pipeline: daily_sales_aggregation
ERROR: NullPointerException in transform_step
Rows loaded: 0 (expected: ~1,200,000)
Upstream: raw_sales CDC job last successful run: 2025-05-31 01:00 UTC
SLA window: data must be available by 06:00 UTC
"""

st.set_page_config(page_title="Pipeline Intelligence — Triage Console", layout="wide")

# ── Session state ──────────────────────────────────────────────────────────────
if "threads" not in st.session_state:
    st.session_state.threads = []  # list of {"thread_id": ..., "pipeline_id": ...}

# ── Sidebar: connection settings ──────────────────────────────────────────────
with st.sidebar:
    st.header("Connection")
    base_url = st.text_input(
        "API base URL", value=_default("API_BASE_URL", "http://localhost:8000")
    ).rstrip("/")
    api_key = st.text_input("X-API-Key", value=_default("API_KEY", ""), type="password")
    st.caption("Must match the API_KEY the FastAPI server was started with.")
    st.caption("Pre-fill both via `.streamlit/secrets.toml` (API_BASE_URL / API_KEY) when deployed.")

    if st.button("Check /health"):
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            st.json(r.json())
        except requests.RequestException as exc:
            st.error(f"Could not reach API: {exc}")

HEADERS = {"X-API-Key": api_key, "Content-Type": "application/json"}


def _api(method: str, path: str, **kwargs):
    """Call the backend and surface HTTP/network errors in the UI instead of crashing."""
    try:
        resp = requests.request(method, f"{base_url}{path}", headers=HEADERS, timeout=30, **kwargs)
        if resp.status_code >= 400:
            st.error(f"{method} {path} → {resp.status_code}: {resp.text}")
            return None
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"Could not reach API at {base_url}: {exc}")
        return None


st.title("🔎 Pipeline Intelligence — Triage Console")

tab_new, tab_review, tab_history = st.tabs(["New Incident", "Review & Approve", "History"])

# ── Tab 1: submit a new incident ──────────────────────────────────────────────
with tab_new:
    st.subheader("Submit a pipeline failure log")
    pipeline_id = st.text_input("pipeline_id", value="daily_sales_aggregation")
    raw_log = st.text_area("raw_log", value=SAMPLE_LOG, height=180)

    if st.button("Start triage", type="primary"):
        if not api_key:
            st.warning("Enter your API key in the sidebar first.")
        else:
            result = _api("POST", "/incident", json={"raw_log": raw_log, "pipeline_id": pipeline_id})
            if result:
                st.success(f"Triage started — thread_id = {result['thread_id']}")
                st.session_state.threads.append(
                    {"thread_id": result["thread_id"], "pipeline_id": pipeline_id}
                )

# ── Tab 2: review the plan and approve / reject ───────────────────────────────
with tab_review:
    st.subheader("Review pending approvals")
    known_ids = [t["thread_id"] for t in st.session_state.threads]
    thread_id = st.selectbox("thread_id", options=known_ids) if known_ids else st.text_input("thread_id")

    if thread_id:
        col_a, col_b = st.columns([1, 1])
        with col_a:
            refresh = st.button("Refresh status")
        with col_b:
            auto = st.checkbox("Auto-refresh every 3s")

        status = _api("GET", f"/status/{thread_id}") if (refresh or auto or True) else None

        if status:
            st.metric("Status", status.get("status", "unknown"))
            cols = st.columns(4)
            cols[0].metric("Severity", status.get("severity") or "—")
            cols[1].metric("Plan steps", status.get("plan_steps", 0))
            cols[2].metric("Steps executed", status.get("steps_executed", 0))
            cols[3].metric("Audit entries", status.get("audit_entries", 0))

            if status.get("status") == "awaiting_approval":
                plan_data = _api("GET", f"/plan/{thread_id}")
                if plan_data:
                    triage = plan_data.get("triage", {})
                    plan = plan_data.get("plan", {})

                    st.markdown(f"**Root cause:** {plan_data.get('root_cause', '')}")
                    st.markdown(f"**Reasoning:** {triage.get('reasoning', '')}")
                    st.markdown(
                        f"**SLA at risk:** {triage.get('sla_at_risk')} &nbsp;|&nbsp; "
                        f"**Downstream count:** {triage.get('downstream_count')} &nbsp;|&nbsp; "
                        f"**Estimated resolution:** {plan.get('estimated_mins', '?')} min"
                    )

                    steps = plan.get("steps", [])
                    st.markdown(f"**Proposed steps ({len(steps)}):**")
                    for step in steps:
                        blocked = step.get("_blocked")
                        label = f"[{step['step_number']}] ({step['risk_level'].upper()}) {step['action']}"
                        if blocked:
                            label += "  🚫 BLOCKED"
                        with st.expander(label):
                            if step.get("sql_or_cmd"):
                                st.code(step["sql_or_cmd"], language="sql")
                            st.write("Reversible:", step.get("reversible"))
                            if v := step.get("_validation"):
                                st.write(f"Validation: **{v.get('rating')}** — {v.get('reason')}")
                            if blocked:
                                st.error(step.get("_block_reason", "Hard-rejected by validate_fix."))

                    st.divider()
                    feedback = st.text_area("Feedback (required for rejection)")
                    c1, c2 = st.columns(2)
                    if c1.button("✅ Approve", type="primary"):
                        r = _api(
                            "POST", "/approve",
                            json={"thread_id": thread_id, "decision": "approved", "feedback": feedback},
                        )
                        if r:
                            st.success("Approved — resuming graph execution.")
                    if c2.button("❌ Reject"):
                        r = _api(
                            "POST", "/approve",
                            json={"thread_id": thread_id, "decision": "rejected", "feedback": feedback},
                        )
                        if r:
                            st.warning("Rejected — the planner will revise using your feedback.")

            elif status.get("status") == "completed":
                st.success("Triage completed.")
                plan_data = _api("GET", f"/plan/{thread_id}")
                if plan_data:
                    st.json(plan_data.get("execution_log", []))

        if auto:
            time.sleep(3)
            st.rerun()

# ── Tab 3: session history ────────────────────────────────────────────────────
with tab_history:
    st.subheader("Incidents started this session")
    if not st.session_state.threads:
        st.info("No incidents submitted yet.")
    for t in st.session_state.threads:
        status = _api("GET", f"/status/{t['thread_id']}")
        if status:
            st.write(
                f"`{t['thread_id']}` — pipeline=`{t['pipeline_id']}` — "
                f"status=**{status.get('status')}** — severity={status.get('severity') or '—'}"
            )
