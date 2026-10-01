# Data Pipeline Intelligence Platform — Capstone 1

Automated incident triage for Atlas Retail Group's retail data platform. Given a raw pipeline failure log, the system extracts structured incident details, retrieves similar resolved incidents from a knowledge base, produces a grounded root-cause diagnosis and remediation plan, holds all write actions behind a human approval gate, and exposes the full workflow through an MCP server and a REST API.

---

## Design Memo

### 1. Why hybrid (BM25 + vector) retrieval, not vector alone?

The incident knowledge base has two kinds of similarity that live in different embedding spaces. Exact error identifiers — `NullPointerException`, `Cartesian join`, `INC-004` — are sparsely distributed tokens that a dense vector model tends to dilute; BM25 finds them reliably because it scores on term frequency. Conversely, a new log might say "upstream feed delivered zero rows" while the matching resolved incident reads "CDC job failed to load source table" — semantically identical, lexically disjoint, and a BM25 score of near-zero. Vector search catches this because the embedding captures the shared meaning.

Reciprocal Rank Fusion merges the two without needing a trained weight: `score = 1/(k + rank_bm25) + 1/(k + rank_vector)`, `k = 60`. The `k` parameter damps the rank-1 boost so a single strong BM25 hit doesn't completely override the semantic signal. In validation, a near-exact query ("NullPointerException in transform step 0 rows") correctly ranks INC-001 first via BM25, while a paraphrased query ("upstream feed delivered no data before the SLA window") ranks INC-001 first via vector — and RRF produces the correct merged ranking in both cases.

The `MIN_RELEVANCE = 0.3` floor ensures the diagnosis node treats a low-scoring retrieval as "no strong precedent" rather than blindly grounding its reasoning in a tangentially related incident.

### 2. Why a graph of nodes rather than a single agent loop?

A single-agent loop that decides its own next step is hard to audit and hard to interrupt. For this use case, three properties are non-negotiable: every reasoning step must be visible to the human approver; a write action must never execute without explicit approval; and a rejection must force replanning, not allow the agent to silently retry the same plan.

A graph encodes these guarantees structurally. Each node has exactly one job (`ingest`, `retrieve`, `diagnose`, `plan`, `execute`) and produces exactly one diff to the shared state. The HITL gate is a hard interrupt point — `interrupt_before=["hitl_gate"]` — not a soft suggestion. The routing function `route_after_hitl` enforces the bounded retry loop (max 3 rejections, then escalate) and is tested independently of the nodes. Audit trail entries are appended at each node boundary, so the full reasoning chain is always reconstructable from state alone. A single-loop agent could replicate this behaviour, but it would require the LLM itself to decide "should I go to execution or re-plan?" — introducing a failure mode that the graph eliminates by construction.

### 3. Where does the approval gate sit, and what breaks if you move it?

The gate sits between `plan` and `execute`. The remediation plan — including every step's SQL, its risk rating from `validate_fix`, and the full CoT severity reasoning — is fully generated *before* the human sees anything. The human approves or rejects a concrete, reviewable artefact, not an abstract intent.

Moving the gate earlier (e.g., between `diagnose` and `plan`) would mean the human approves a root-cause hypothesis rather than a specific set of SQL commands — they couldn't know what they were signing off. Moving it later (inside `execute`, per step) would fragment accountability and create race conditions between steps. Moving it after `execute` is obviously wrong: you cannot unsend a DELETE.

The current position also enables the rejection-and-revision loop: the human's feedback arrives as `human_feedback` in state, `route_after_hitl` routes back to `plan`, and the planner prompt explicitly addresses the feedback in a revision branch. If the gate were before `plan`, the loop would have no plan to revise.

---

## Improvements Over Hints

| Area | Hint | What we implemented | Rationale |
|---|---|---|---|
| `validate_fix` safety | "default to `risky` when LLM doesn't parse" | Added hard rules for unbounded DELETE (no WHERE), not just DROP/TRUNCATE | An unbounded DELETE is just as dangerous; rule-based rejection is faster and more reliable than an LLM call |
| HITL routing | "bounded retry loop" | `MAX_HITL_RETRIES = 3` with explicit escalation log message at exhaustion | Makes the escalation path observable rather than silently dropping to END |
| Prompt templates | Jinja2 stubs | Full CoT template with 5 numbered reasoning steps for severity; few-shot examples for root-cause; explicit hard-constraint list in planner | Each template failure mode is addressed: CoT prevents severity under/over-escalation; few-shot prevents fabricated causes; constraints prevent unsafe writes |
| Graph state | `sql_result` not in original spec | Added `sql_result` as a first-class state field | Both `diagnose` and `plan` need the diagnostic SQL output; passing it through state avoids re-running the query |
| BM25 tokeniser | "whitespace + punctuation" | Tokeniser applied to `error + root_cause + pipeline + tags` (not just `error + root_cause + tags`) | Pipeline name is a strong signal for BM25 (exact match on e.g. `fraud_detection_feed`) |
| execute node | "execute low/medium risk steps" | High-risk steps emitted as `proposed_manual_execution_required` with clear log | The spec says "approve, then execute" — but high-risk DDL (ALTER TABLE, CALL) should never auto-execute even after approval; the engineer runs them manually from the logged command |
| Jinja2 rendering | Simple Template | Custom `Environment` with `tojson` filter for nested dicts | Avoids `UndefinedError` when rendering complex state dicts inside templates |

---

## Assumptions

1. **Mock warehouse as ground truth.** The SQLite in-memory database is the "data warehouse" for all diagnostic SQL. Assumption: in a real deployment, `execute_sql` would point to a read replica of the production DWH; the mock faithfully represents the schema.

2. **`MAX_HITL_RETRIES = 3`.** The notebook says "bounded retries" without specifying a number. 3 was chosen as a balance: enough for a meaningful revision cycle (one rejection with feedback, one re-plan, one final review) without risk of an infinite loop. This should be configurable per-environment.

3. **`MIN_RELEVANCE = 0.3` threshold.** The notebook specifies "nothing above 0.3 should be treated as no strong precedent" but doesn't say what the RRF score scale is. After calibration against the 20-incident corpus, a score of ≥ 0.3 consistently corresponded to the top-1 result being a near-match. This threshold is conservative and errs toward treating weak matches as precedent-free.

4. **Severity fallback is P2.** When the LLM response for the severity prompt fails to parse, we default to P2 (not P1). Reasoning: P1 would trigger an escalation that might be false; P2 still mandates human review and keeps the incident in the active queue. A parsing failure is itself a signal that the situation may be unusual, and P2 appropriately flags it for human attention without triggering a false-alarm critical alert.

5. **High-risk steps are proposed, not auto-executed.** Even after human approval, steps rated `high` (DDL, CDC restarts, scheduler config changes) are logged as `proposed_manual_execution_required`. Assumption: the approval gate confirms the *plan*, but an engineer should physically run high-risk commands so they can observe real-time effects and roll back immediately if needed.

6. **In-memory state store.** Both the API and MCP server use a Python dict to track thread state. Assumption: a production deployment would replace this with Redis or a Postgres table (and `MemorySaver` with `SqliteSaver`) to survive process restarts during multi-hour approval cycles.

7. **Embedding model.** ChromaDB's `DefaultEmbeddingFunction` (all-MiniLM-L6-v2) is used for dev. In production, switch to the same embedding model used by your vector store (e.g., `text-embedding-3-small` via OpenAI) for quality consistency.

---

## Project Structure

```
capstone1/
├── data/
│   ├── incidents.json           # 20 resolved incidents (RAG knowledge base)
│   └── pipeline_registry.json  # Pipeline ownership & SLA metadata
├── rag/
│   ├── __init__.py
│   ├── ingest.py                # Builds BM25 + ChromaDB indexes
│   └── retriever.py             # Hybrid BM25 + vector search with RRF
├── agent/
│   ├── __init__.py
│   ├── tools.py                 # execute_sql, parse_log, validate_fix
│   ├── prompts.py               # Jinja2 templates + Pydantic schemas
│   └── graph.py                 # LangGraph orchestration
├── mcp_server/
│   ├── __init__.py
│   └── server.py                # MCP server (FastMCP)
├── api/
│   ├── __init__.py
│   └── main.py                  # FastAPI wrapper
├── tests/                       # pytest unit tests (routing + safety-gate logic)
├── ui/
│   └── streamlit_app.py         # Streamlit dashboard (talks to the FastAPI backend)
├── langgraph.json               # LangGraph Studio config (`langgraph dev`)
├── Dockerfile                   # Multi-stage, non-root runtime
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt         # + pytest, for local testing
├── pytest.ini
├── .env.example                 # copy to .env and fill in secrets
└── README.md
```

---

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

Set environment variables (copy [.env.example](.env.example) to `.env` and fill in real values):

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini          # or your deployment name
export OPENAI_BASE_URL=https://...       # optional: Azure endpoint
export API_KEY=your-secret-key           # required — the API refuses to start without it
export MCP_API_KEY=your-secret-key       # required — the MCP server refuses to start without it
```

> `API_KEY` / `MCP_API_KEY` have no insecure default — both the REST API and the MCP
> server raise a startup error if the variable is unset, rather than silently falling
> back to a well-known placeholder.

### 1. Build the RAG indexes

```bash
cd capstone1
python -m rag.ingest
```

### 2. Run the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 3. Run the MCP server (separate terminal)

```bash
python mcp_server/server.py
```

### 4. Docker

```bash
docker-compose up --build
```

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q
```

Covers the bounded HITL retry/escalation routing (`route_after_hitl`) and the
rule-based safety gates in `validate_fix` / `execute_sql` / `parse_log` (LLM calls
are monkeypatched — no network access or API key required to run the suite).

---

## End-to-End Demo

```python
from agent.graph import graph, make_initial_state

config = {"configurable": {"thread_id": "demo-001"}}

log = """
2025-06-01 02:14 UTC | pipeline: daily_sales_aggregation
ERROR: NullPointerException in transform_step
Rows loaded: 0 (expected: ~1,200,000)
Upstream: raw_sales CDC job last successful run: 2025-05-31 01:00 UTC
SLA window: data must be available by 06:00 UTC
"""

# Step 1: Run to HITL interrupt
for event in graph.stream(make_initial_state(log), config):
    print(f"  ✓ {list(event.keys())[0]}")

# Step 2a: Reject with feedback
graph.update_state(config, {
    "hitl_decision": "rejected",
    "human_feedback": "The fix SQL is too aggressive — scope the DELETE with a date filter",
    "approval_attempts": 1,
})
for event in graph.stream(None, config):
    print(f"  ✓ {list(event.keys())[0]}")

# Step 2b: Approve the revised plan
graph.update_state(config, {
    "hitl_decision": "approved",
    "approval_attempts": 2,
})
for event in graph.stream(None, config):
    print(f"  ✓ {list(event.keys())[0]}")

# Inspect final state
final = graph.get_state(config).values
print(f"Severity:        {final['triage']['severity']}")
print(f"Root cause:      {final['root_cause'][:120]}")
print(f"Steps executed:  {len(final['execution_log'])}")
print(f"Audit entries:   {len(final['audit_trail'])}")
```

---

## API Reference

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/incident` | `X-API-Key` | Start triage (async, returns `thread_id`) |
| POST | `/approve` | `X-API-Key` | Submit HITL decision |
| GET | `/status/{thread_id}` | `X-API-Key` | Get triage status summary |
| GET | `/plan/{thread_id}` | `X-API-Key` | Get the full triage + remediation plan (what `/approve` actually approves) |
| GET | `/health` | none | Load-balancer health check |

---

## Using Groq instead of OpenAI

Groq exposes an OpenAI-compatible chat completions API, so no code changes are needed —
only environment variables (see [.env.example](.env.example)):

```bash
export OPENAI_API_KEY=gsk_...                         # your Groq key
export OPENAI_BASE_URL=https://api.groq.com/openai/v1
export OPENAI_MODEL=openai/gpt-oss-120b
```

Then run the API / MCP server / tests exactly as documented above.

---

## Validating the workflow in a UI

Two complementary ways to watch/drive the graph interactively:

### 1. LangGraph Studio — inspect the graph itself

Best for debugging the agent: see each node fire, inspect the full state at the
`hitl_gate` interrupt, and edit state or resume directly from the browser.

```bash
pip install -r requirements-dev.txt
langgraph dev
```

This reads [langgraph.json](langgraph.json) (which points at the `graph` object in
[agent/graph.py](agent/graph.py)), starts a local LangGraph API server, and opens
LangGraph Studio in your browser pointed at it. Submit a run with
`make_initial_state(<raw log>)` as input; when it pauses at `hitl_gate`, inspect
the `plan`/`triage` state and resume with `{"hitl_decision": "approved"}` (or
`"rejected"` + `"human_feedback"`) via the Studio UI.

### 2. Streamlit — validate the product experience end-to-end

Best for demoing/validating the actual REST API a real caller would use
(submit → poll → review plan → approve/reject → see execution log).

```bash
pip install -r requirements-dev.txt
# in one terminal: the backend
uvicorn api.main:app --host 0.0.0.0 --port 8000
# in another terminal: the dashboard
streamlit run ui/streamlit_app.py
```

Open the URL Streamlit prints, enter the API base URL + `X-API-Key` in the
sidebar, submit a log on the **New Incident** tab, then switch to **Review &
Approve** to see the generated severity/root-cause/remediation steps (with
SQL, risk level, and safety validation per step) and click Approve/Reject.
