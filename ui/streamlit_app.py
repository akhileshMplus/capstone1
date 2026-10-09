"""
Data Pipeline Intelligence — Triage Console
=======================================
Self-contained Streamlit app: calls LangGraph directly, no FastAPI backend required.
Runs on Streamlit Community Cloud as-is.

Local:
    cd capstone1/
    streamlit run ui/streamlit_app.py

Cloud (streamlit.app):
    Set secrets in the Streamlit Cloud dashboard:
        OPENAI_API_KEY = "sk-..."
        OPENAI_MODEL   = "gpt-4o-mini"   # optional
        OPENAI_BASE_URL = "https://..."  # optional — Azure endpoint
"""
from __future__ import annotations

import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

# ── Path: make top-level packages (agent/, rag/) importable ──────────────────
APP_DIR = Path(__file__).resolve().parent          # capstone1/ui/
ROOT    = APP_DIR.parent                            # capstone1/
sys.path.insert(0, str(ROOT))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pipeline Intelligence — Triage Console",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.badge-p1{background:#ef4444;color:#fff;padding:3px 10px;border-radius:12px;font-weight:700;font-size:.85rem}
.badge-p2{background:#f97316;color:#fff;padding:3px 10px;border-radius:12px;font-weight:700;font-size:.85rem}
.badge-p3{background:#eab308;color:#fff;padding:3px 10px;border-radius:12px;font-weight:700;font-size:.85rem}
.risk-low{background:#22c55e;color:#fff;padding:2px 8px;border-radius:8px;font-size:.75rem;font-weight:600}
.risk-medium{background:#f97316;color:#fff;padding:2px 8px;border-radius:8px;font-size:.75rem;font-weight:600}
.risk-high{background:#ef4444;color:#fff;padding:2px 8px;border-radius:8px;font-size:.75rem;font-weight:600}
.audit-entry{border-left:3px solid #3b82f6;padding:6px 12px;margin:4px 0;background:#1e293b;border-radius:0 6px 6px 0;font-size:.8rem}
.step-card{border:1px solid #334155;border-radius:8px;padding:1rem;margin:.4rem 0;background:#0f172a}
</style>
""", unsafe_allow_html=True)

# ── Sample logs ───────────────────────────────────────────────────────────────
SAMPLES = {
    "CDC Failure — Sales Pipeline (P1)": """\
2025-06-01 02:14 UTC | pipeline: daily_sales_aggregation
ERROR: NullPointerException in transform_step
Rows loaded: 0 (expected: ~1,200,000)
Upstream: raw_sales CDC job last successful run: 2025-05-31 01:00 UTC
SLA window: data must be available by 06:00 UTC
Downstream: finance_dashboard, inventory_reorder_job
""",
    "Duplicate Rows — Fraud Feed (P1)": """\
2025-06-03 14:55 UTC | pipeline: fraud_detection_feed
ERROR: DataQualityException — duplicate rows detected
Duplicate count: 14,203 records in transactions table
Scheduler retried pipeline twice at 14:30 and 14:55 UTC
SLA window: 1 hour (CRITICAL pipeline)
""",
    "Schema Drift — Customer 360 (P2)": """\
2025-06-05 09:22 UTC | pipeline: customer_360_snapshot
ERROR: Schema mismatch — column 'loyalty_tier' not found in staging.customers
Source: CRM sync job completed at 09:15 UTC
Affected rows: unknown (job aborted before transform)
Recent change: source team deployed CRM v2.4.1 on 2025-06-04
""",
    "SLA Breach — Inventory Job (P2)": """\
2025-06-07 11:45 UTC | pipeline: inventory_reorder_job
WARN: SLA breach — job running for 7h 12m (SLA: 6h)
Stage: full table scan on inventory table (180M rows)
No index on product_id column detected in EXPLAIN output
Downstream: wms_system (warehouse management — shipments delayed)
""",
}

# ── Secret / env helper ───────────────────────────────────────────────────────
def _secret(key: str, fallback: str = "") -> str:
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, fallback)


# ── Sidebar: config ───────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    api_key = st.text_input(
        "OpenAI API Key",
        value=_secret("OPENAI_API_KEY"),
        type="password",
        help="Set OPENAI_API_KEY in Streamlit Secrets (cloud) or as env var (local).",
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    model = st.text_input(
        "Model",
        value=_secret("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI model or Azure deployment name.",
    )
    if model:
        os.environ["OPENAI_MODEL"] = model

    base_url = st.text_input(
        "Base URL (Azure only)",
        value=_secret("OPENAI_BASE_URL"),
        help="Leave blank for standard OpenAI.",
    )
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["endpoint"] = base_url

    st.divider()
    st.caption("**RAG:** BM25 + ChromaDB · RRF k=60")
    st.caption("**Max HITL retries:** 3")
    st.caption("**Indexes:** built in-memory at startup")
    st.divider()
    if st.button("🔄 Reset Session", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ── Guard ─────────────────────────────────────────────────────────────────────
if not os.getenv("OPENAI_API_KEY"):
    st.warning("⚠️  Set your **OpenAI API Key** in the sidebar to begin.", icon="🔑")
    st.info(
        "**Cloud deployment:** add `OPENAI_API_KEY` to your app's Secrets in the "
        "Streamlit Cloud dashboard → App settings → Secrets.",
        icon="☁️",
    )
    st.stop()


# ── Cache: RAG + graph ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔧 Building RAG indexes and compiling graph…")
def _load_platform():
    """
    Build in-memory RAG indexes and compile the LangGraph graph.
    Cached for the process lifetime — survives Streamlit reruns.
    Uses chromadb.EphemeralClient() so no filesystem persistence is needed
    (Streamlit Cloud has ephemeral storage that resets between deployments).
    """
    import json
    import pickle
    import re as _re

    import chromadb
    from chromadb.utils import embedding_functions
    from rank_bm25 import BM25Okapi

    # Load incidents from data/
    data_path = ROOT / "data" / "incidents.json"
    incidents = json.loads(data_path.read_text())

    def _tok(text):
        return _re.findall(r"\w+", text.lower())

    # BM25
    corpus = [
        _tok(f"{i['error']} {i['root_cause']} {i['pipeline']} {' '.join(i['tags'])}")
        for i in incidents
    ]
    bm25 = BM25Okapi(corpus)
    incident_ids = [i["id"] for i in incidents]

    # ChromaDB — in-memory (EphemeralClient works on any host, no filesystem needed)
    chroma = chromadb.EphemeralClient()
    ef = embedding_functions.DefaultEmbeddingFunction()
    collection = chroma.get_or_create_collection("incidents", embedding_function=ef)
    docs      = [f"{i['error']}. {i['root_cause']}. Tags: {', '.join(i['tags'])}" for i in incidents]
    metadatas = [{"id": i["id"], "severity": i["severity"],
                  "resolution": i["resolution"], "fix_sql": i["fix_sql"],
                  "resolved_mins": i["resolved_mins"]} for i in incidents]
    collection.add(documents=docs, metadatas=metadatas, ids=incident_ids)

    incidents_by_id = {i["id"]: i for i in incidents}

    # Patch the retriever module so hybrid_retrieve uses these in-memory objects
    import rag.retriever as _ret
    _ret._bm25_data      = {"bm25": bm25, "incident_ids": incident_ids}
    _ret._collection     = collection
    _ret._incidents_by_id = incidents_by_id

    # Build the graph
    from agent.graph import build_graph, make_initial_state as _make
    g = build_graph()
    return g, _make


graph, make_initial_state = _load_platform()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _cfg(tid):
    return {"configurable": {"thread_id": tid}}

def _sev_badge(sev):
    cls = {"P1":"badge-p1","P2":"badge-p2","P3":"badge-p3"}.get(sev,"badge-p3")
    return f'<span class="{cls}">{sev}</span>' if sev else ""

def _risk_chip(risk):
    cls = {"low":"risk-low","medium":"risk-medium","high":"risk-high"}.get(risk,"risk-medium")
    return f'<span class="{cls}">{risk.upper()}</span>'


# ── Session state defaults ────────────────────────────────────────────────────
def _init():
    defaults = {
        "phase": "idle",         # idle | running | awaiting_approval | complete | escalated
        "thread_id": None,
        "gs": {},                # latest graph state snapshot
        "rejection_count": 0,
        "rejection_log": [],
        "history": [],           # list of completed thread summaries
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ── Title + tabs ──────────────────────────────────────────────────────────────
st.title("🔎 Pipeline Intelligence — Triage Console")

tab_new, tab_review, tab_history = st.tabs(["New Incident", "Review & Approve", "History"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — New Incident
# ════════════════════════════════════════════════════════════════════════════
with tab_new:
    if st.session_state.phase not in ("idle",):
        sev = st.session_state.gs.get("triage", {}).get("severity")
        phase_labels = {
            "awaiting_approval": f"⏸ Triage complete — awaiting your approval in **Review & Approve**.",
            "complete":          f"✅ Incident resolved — see **History** for the audit trail.",
            "escalated":         f"🚨 Incident escalated after 3 rejections.",
        }
        msg = phase_labels.get(st.session_state.phase, f"Status: {st.session_state.phase}")
        st.info(msg)
        if st.button("Start a New Incident", type="primary"):
            # Archive current if complete
            if st.session_state.phase == "complete":
                gs = st.session_state.gs
                st.session_state.history.append({
                    "thread_id": st.session_state.thread_id,
                    "severity": gs.get("triage", {}).get("severity"),
                    "pipeline": gs.get("triage", {}).get("pipeline"),
                    "status": "completed",
                    "ts": datetime.now(timezone.utc).isoformat()[:19],
                })
            st.session_state.phase = "idle"
            st.session_state.thread_id = None
            st.session_state.gs = {}
            st.session_state.rejection_count = 0
            st.session_state.rejection_log = []
            st.rerun()
    else:
        st.subheader("Submit a pipeline failure log")

        col_sel, _ = st.columns([1, 2])
        with col_sel:
            preset = st.selectbox("Load sample", ["(paste your own)"] + list(SAMPLES.keys()))
        default_log = SAMPLES.get(preset, "")

        pipeline_id = st.text_input("pipeline_id", value="daily_sales_aggregation")
        raw_log = st.text_area("raw_log", value=default_log, height=200,
                               placeholder="Paste your pipeline failure log here…")

        if st.button("🚀 Start triage", type="primary", disabled=not raw_log.strip()):
            thread_id = str(uuid.uuid4())
            st.session_state.thread_id = thread_id
            st.session_state.rejection_count = 0
            st.session_state.rejection_log = []
            st.session_state.phase = "running"

            cfg = _cfg(thread_id)
            initial = make_initial_state(raw_log)

            with st.spinner("Running pipeline: ingest → retrieve → diagnose → plan…"):
                try:
                    for _ in graph.stream(initial, cfg):
                        pass
                    st.session_state.gs = dict(graph.get_state(cfg).values)
                    st.session_state.phase = "awaiting_approval"
                except Exception as exc:
                    st.error(f"Graph error: {exc}")
                    st.session_state.phase = "idle"
                    st.stop()

            st.success("Triage complete — go to **Review & Approve** tab.")
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Review & Approve
# ════════════════════════════════════════════════════════════════════════════
with tab_review:
    phase = st.session_state.phase

    if phase == "idle":
        st.info("Submit an incident in **New Incident** first.")

    elif phase in ("awaiting_approval", "replanning"):
        gs = st.session_state.gs
        triage  = gs.get("triage", {})
        plan    = gs.get("plan", {})
        similar = gs.get("similar_incidents", [])
        parsed  = gs.get("parsed_incident", {})
        meta    = gs.get("pipeline_meta", {})

        # ── Progress indicator ────────────────────────────────────────────
        steps = ["Ingest","Retrieve","Diagnose","Plan","⏸ HITL Gate","Execute"]
        cols = st.columns(len(steps))
        for i, (c, s) in enumerate(zip(cols, steps)):
            (c.success if i < 4 else c.warning if i == 4 else c.info)(s, icon="✅" if i<4 else "⏸" if i==4 else "⬜")

        st.divider()

        # ── Triage summary ────────────────────────────────────────────────
        st.subheader("🎯 Triage")
        c1, c2, c3 = st.columns(3)
        sev = triage.get("severity", "?")
        c1.markdown(f"**Severity**<br>{_sev_badge(sev)}", unsafe_allow_html=True)
        c2.metric("SLA at Risk", "YES" if triage.get("sla_at_risk") else "NO")
        c3.metric("Downstream Pipelines", triage.get("downstream_count", "?"))

        with st.expander("CoT Reasoning", expanded=False):
            st.write(triage.get("reasoning") or "_Not available_")

        # ── Root cause ────────────────────────────────────────────────────
        st.subheader("🔬 Root Cause")
        st.info(gs.get("root_cause") or "_Not yet determined_")

        # ── SQL diagnostics ───────────────────────────────────────────────
        sql_r = gs.get("sql_result", {})
        if sql_r.get("rows"):
            with st.expander("📊 SQL Diagnostic", expanded=False):
                import pandas as pd
                try:
                    st.dataframe(pd.DataFrame(sql_r["rows"], columns=sql_r["columns"]))
                except Exception:
                    st.json(sql_r)

        # ── Similar incidents ─────────────────────────────────────────────
        if similar:
            with st.expander(f"📚 Similar Past Incidents ({len(similar)})", expanded=False):
                for r in similar:
                    inc = r["incident"]
                    strong = "✅" if r["rrf_score"] >= 0.3 else "⚠️"
                    st.markdown(
                        f"{strong} **{inc['id']}** `{r['rrf_score']:.4f}` "
                        f"{_sev_badge(inc['severity'])} _{inc['error'][:70]}_",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Resolution: {inc['resolution'][:120]}")
                    st.divider()

        # ── Remediation plan ──────────────────────────────────────────────
        st.divider()
        rej = st.session_state.rejection_count
        if rej:
            st.warning(f"♻️ Revised plan (after {rej} rejection{'s' if rej>1 else ''})")
        st.subheader("🛠️ Remediation Plan")

        if plan:
            mc = st.columns(3)
            mc[0].metric("Est. Resolution", f"{plan.get('estimated_mins','?')} min")
            mc[1].metric("Steps", len(plan.get("steps", [])))
            mc[2].metric("Similar Refs", ", ".join(plan.get("similar_incidents", [])) or "—")

            for step in plan.get("steps", []):
                blocked = step.get("_blocked", False)
                val = step.get("_validation", {})
                col_a, col_b = st.columns([4, 1])
                with st.container():
                    hdr = f"Step {step['step_number']} {'🚫 BLOCKED' if blocked else ''} — {step['action'][:70]}"
                    with st.expander(hdr, expanded=True):
                        if step.get("sql_or_cmd"):
                            st.code(step["sql_or_cmd"], language="sql")
                        st.markdown(
                            f"{_risk_chip(step.get('risk_level','medium'))} &nbsp; "
                            f"{'↩️ Reversible' if step.get('reversible') else '🔒 Irreversible'}",
                            unsafe_allow_html=True,
                        )
                        if val:
                            icons = {"safe":"✅","risky":"⚠️","dangerous":"🚫"}
                            r = val.get("rating","?")
                            st.caption(f"{icons.get(r,'?')} Validation: **{r}** — {val.get('reason','')[:100]}")
                        if blocked:
                            st.error(step.get("_block_reason","Hard-rejected by validate_fix."))

        # ── HITL Gate ─────────────────────────────────────────────────────
        st.divider()
        from agent.graph import MAX_HITL_RETRIES
        retries_left = MAX_HITL_RETRIES - st.session_state.rejection_count

        st.markdown(
            f"### ⏸ Human Approval Required "
            f"<span style='font-size:.8rem;color:#94a3b8'>— attempt {rej+1}/{MAX_HITL_RETRIES}</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"Rejections remaining before escalation: **{retries_left}**")

        feedback = st.text_area(
            "Feedback / rejection reason",
            placeholder="e.g. The DELETE is too aggressive — scope it with a date filter.",
            height=90,
            key=f"feedback_{rej}",
        )

        a_col, r_col, _ = st.columns([1, 1, 3])
        approve = a_col.button("✅ Approve", type="primary", use_container_width=True)
        reject  = r_col.button("❌ Reject",  use_container_width=True)

        if approve:
            cfg = _cfg(st.session_state.thread_id)
            attempts = graph.get_state(cfg).values.get("approval_attempts", 0)
            graph.update_state(cfg, {
                "hitl_decision": "approved",
                "human_feedback": feedback,
                "approval_attempts": attempts + 1,
            })
            with st.spinner("Executing approved steps…"):
                try:
                    for _ in graph.stream(None, cfg):
                        pass
                    st.session_state.gs = dict(graph.get_state(cfg).values)
                    st.session_state.rejection_log.append(
                        {"attempt": rej + 1, "decision": "approved", "feedback": feedback}
                    )
                    st.session_state.phase = "complete"
                except Exception as exc:
                    st.error(f"Execution error: {exc}")
                    st.stop()
            st.rerun()

        if reject:
            if not feedback.strip():
                st.error("Please provide feedback before rejecting.")
            else:
                cfg = _cfg(st.session_state.thread_id)
                attempts = graph.get_state(cfg).values.get("approval_attempts", 0)
                graph.update_state(cfg, {
                    "hitl_decision": "rejected",
                    "human_feedback": feedback,
                    "approval_attempts": attempts + 1,
                })
                st.session_state.rejection_log.append(
                    {"attempt": rej + 1, "decision": "rejected", "feedback": feedback}
                )

                if retries_left <= 1:
                    # Exhaust — route to END
                    with st.spinner("Escalating…"):
                        for _ in graph.stream(None, cfg):
                            pass
                    st.session_state.gs = dict(graph.get_state(cfg).values)
                    st.session_state.phase = "escalated"
                else:
                    # Replan
                    with st.spinner("Replanning with your feedback…"):
                        try:
                            for _ in graph.stream(None, cfg):
                                pass
                        except Exception as exc:
                            st.error(f"Replan error: {exc}")
                            st.stop()
                    st.session_state.gs = dict(graph.get_state(cfg).values)
                    st.session_state.rejection_count += 1
                    st.session_state.phase = "awaiting_approval"
                st.rerun()

    elif phase == "complete":
        gs = st.session_state.gs
        triage   = gs.get("triage", {})
        exec_log = gs.get("execution_log", [])
        audit    = gs.get("audit_trail", [])

        # All-green progress
        steps = ["Ingest","Retrieve","Diagnose","Plan","HITL Gate","Execute"]
        cols = st.columns(len(steps))
        for c, s in zip(cols, steps):
            c.success(s, icon="✅")

        st.divider()
        st.success("✅ Incident triage and remediation complete!", icon="🎉")

        m1,m2,m3,m4 = st.columns(4)
        m1.markdown(f"**Severity**<br>{_sev_badge(triage.get('severity'))}", unsafe_allow_html=True)
        m2.metric("Steps Executed", len(exec_log))
        m3.metric("Audit Entries", len(audit))
        m4.metric("Rejections", st.session_state.rejection_count)

        st.divider()
        left, right = st.columns([3,2])

        with left:
            st.subheader("🚀 Execution Log")
            icons = {"executed":"✅","proposed_manual_execution_required":"📋",
                     "no_sql_noted":"📝","blocked_write_statement":"🚫",
                     "skipped_blocked":"⛔","error":"❌"}
            for entry in exec_log:
                ico = icons.get(entry.get("status",""),"❓")
                with st.expander(f"{ico} Step {entry['step']}: {entry['action'][:60]}"):
                    st.markdown(f"**Status:** `{entry.get('status','').replace('_',' ').title()}`")
                    if entry.get("sql_or_cmd"):
                        st.code(entry["sql_or_cmd"], language="sql")
                    if entry.get("reason"):
                        st.caption(entry["reason"])
                    if entry.get("result", {}).get("rows"):
                        import pandas as pd
                        r = entry["result"]
                        st.dataframe(pd.DataFrame(r["rows"], columns=r["columns"]))

            st.subheader("🔬 Root Cause")
            st.info(gs.get("root_cause","")[:300] or "_Not available_")

            # HITL decision history
            if st.session_state.rejection_log:
                st.subheader("🔄 HITL Decision History")
                for h in st.session_state.rejection_log:
                    icon = "✅" if h["decision"] == "approved" else "❌"
                    st.markdown(
                        f"{icon} **Attempt {h['attempt']}:** {h['decision'].upper()} — "
                        + (f"_{h['feedback'][:120]}_" if h.get("feedback") else "_no feedback_")
                    )

        with right:
            st.subheader("📜 Audit Trail")
            for entry in audit:
                ts = entry.get("timestamp","")[:19].replace("T"," ")
                extra = {k:v for k,v in entry.items() if k not in ("event","timestamp")}
                st.markdown(
                    f'<div class="audit-entry"><strong>{entry.get("event","?")}</strong> '
                    f'<span style="color:#64748b">{ts}</span><br/>'
                    + (f'<span style="color:#94a3b8">{json.dumps(extra, default=str)[:160]}</span>' if extra else "")
                    + "</div>",
                    unsafe_allow_html=True,
                )

    elif phase == "escalated":
        st.error("🚨 Incident escalated — 3 rejections received. Requires senior engineer review.", icon="🚨")
        for h in st.session_state.rejection_log:
            st.markdown(f"❌ **Attempt {h['attempt']}:** _{h.get('feedback','')[:200]}_")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — History (session)
# ════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.subheader("Incidents this session")
    history = st.session_state.history

    # Add current if complete
    current_summary = None
    if st.session_state.phase == "complete":
        gs = st.session_state.gs
        current_summary = {
            "thread_id": st.session_state.thread_id,
            "severity": gs.get("triage", {}).get("severity"),
            "pipeline": gs.get("triage", {}).get("pipeline"),
            "status": "completed",
            "ts": "current",
        }

    all_records = history + ([current_summary] if current_summary else [])

    if not all_records:
        st.info("No incidents resolved yet this session.")
    else:
        for rec in reversed(all_records):
            c1, c2, c3 = st.columns([3,1,1])
            c1.markdown(f"`{(rec['thread_id'] or '')[:18]}…`  **{rec.get('pipeline','?')}**")
            c2.markdown(_sev_badge(rec.get("severity")), unsafe_allow_html=True)
            c3.caption(rec.get("ts",""))