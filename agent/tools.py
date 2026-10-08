"""
Diagnostic tools for the pipeline intelligence agent.

Three tools:
  execute_sql   — read-only warehouse queries (SQLite mock in dev)
  parse_log     — structured extraction from raw log text
  validate_fix  — rule-based + LLM safety screen for proposed SQL fixes
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from datetime import datetime
from threading import Lock
from typing import Any

from openai import OpenAI

logger = logging.getLogger("pipeline-agent.tools")

# ── LLM client ────────────────────────────────────────────────────────────────
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        endpoint = os.getenv("OPENAI_BASE_URL") or os.getenv("endpoint")
        api_key = os.getenv("OPENAI_API_KEY", "sk-placeholder")
        # Explicit timeout + retries so a stalled upstream never hangs a graph node forever.
        _client = OpenAI(base_url=endpoint, api_key=api_key, timeout=30.0, max_retries=2)
    return _client


def _llm(prompt: str, max_tokens: int = 512) -> str:
    """Single-turn LLM call. Returns the text response."""
    client = _get_client()
    deployment = os.getenv("OPENAI_MODEL") or os.getenv("deployment_name", "gpt-4o-mini")
    try:
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error("event=llm_call_failed error=%s", e)
        return f"LLM_ERROR: {e}"


# ── Mock data warehouse ────────────────────────────────────────────────────────
def _build_mock_db() -> sqlite3.Connection:
    # check_same_thread=False: this connection is built once at import time but
    # queried from the per-request background worker threads spawned by the API/MCP server.
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.executescript("""
        CREATE TABLE pipelines (
            id TEXT PRIMARY KEY,
            name TEXT,
            owner TEXT,
            sla_hours REAL,
            criticality TEXT
        );
        INSERT INTO pipelines VALUES
            ('PL-001','daily_sales_aggregation','data-eng@co.com',6,'high'),
            ('PL-003','fraud_detection_feed','risk-eng@co.com',1,'critical'),
            ('PL-005','payment_reconciliation','data-eng@co.com',2,'critical');

        CREATE TABLE incidents_log (
            id TEXT PRIMARY KEY,
            pipeline TEXT,
            error TEXT,
            status TEXT,
            created_at TEXT
        );
        INSERT INTO incidents_log VALUES
            ('INC-001','daily_sales_aggregation','NullPointerException','resolved','2025-01-10 02:14'),
            ('INC-004','fraud_detection_feed','Duplicate rows: 14203','resolved','2025-02-03 14:55');

        CREATE TABLE row_counts (
            table_name TEXT,
            measured_at TEXT,
            row_count INTEGER,
            expected_min INTEGER
        );
        INSERT INTO row_counts VALUES
            ('raw_sales','2025-06-01 01:00',0,500000),
            ('raw_sales','2025-05-31 01:00',1234567,500000),
            ('transactions','2025-06-01 14:00',2480312,2000000);
    """)
    return conn


_DB: sqlite3.Connection = _build_mock_db()
_DB_LOCK = Lock()  # serialize access since the connection is shared across worker threads

# Patterns that signal a write operation — block all of these at the tool level.
_WRITE_KEYWORDS = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|REPLACE|MERGE|CALL|EXEC)\b",
    re.IGNORECASE,
)


# ── TOOL 1: execute_sql ───────────────────────────────────────────────────────
def execute_sql(query: str) -> dict[str, Any]:
    """
    Run a read-only SQL query against the mock data warehouse.

    Returns:
        {"columns": [...], "rows": [[...], ...], "row_count": N}
        {"error": "...", "requires_approval": True}  — for non-SELECT or errors
    """
    stripped = query.strip()

    # Hard block on write statements — they must go through the HITL gate
    if _WRITE_KEYWORDS.match(stripped):
        return {
            "error": (
                f"Write statement blocked: '{stripped[:60]}...'. "
                "This must be routed through the human approval gate."
            ),
            "requires_approval": True,
        }

    try:
        with _DB_LOCK:
            cursor = _DB.execute(stripped)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
        return {
            "columns": columns,
            "rows": [list(r) for r in rows],
            "row_count": len(rows),
        }
    except Exception as exc:
        return {"error": str(exc)}


# ── TOOL 2: parse_log ────────────────────────────────────────────────────────
def parse_log(raw_log: str) -> dict[str, Any]:
    """
    Extract structured fields from a raw pipeline log string.

    Regex handles: timestamps, pipeline name, row counts, error type.
    LLM fallback summarises the root cause from free-form text.

    Returns:
        {
            "pipeline": str | None,
            "error_type": str | None,
            "timestamp": str | None,
            "affected_rows": int | None,
            "expected_rows": int | None,
            "stack_trace_summary": str | None,
            "llm_root_cause_hint": str,
        }
    """
    result: dict[str, Any] = {
        "pipeline": None,
        "error_type": None,
        "timestamp": None,
        "affected_rows": None,
        "expected_rows": None,
        "stack_trace_summary": None,
        "llm_root_cause_hint": "",
    }

    # Timestamp — ISO-ish patterns: "2025-06-01 02:14 UTC"
    ts_match = re.search(
        r"(\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}(?::\d{2})?(?:\s*UTC)?)", raw_log
    )
    if ts_match:
        result["timestamp"] = ts_match.group(1).strip()

    # Pipeline name — "pipeline: <name>" or "pipeline=<name>"
    pipe_match = re.search(r"pipeline[:\s=]+([a-zA-Z0-9_\-]+)", raw_log, re.IGNORECASE)
    if pipe_match:
        result["pipeline"] = pipe_match.group(1).strip()

    # Error type — first ERROR:/WARN:/EXCEPTION line
    err_match = re.search(
        r"(?:ERROR|Exception|WARN)[:\s]+(.+?)(?:\n|$)", raw_log, re.IGNORECASE
    )
    if err_match:
        result["error_type"] = err_match.group(1).strip()[:200]

    # Affected rows — "Rows loaded: 0" / "0 rows loaded"
    row_match = re.search(
        r"(?:rows?\s+loaded|loaded)[:\s]+(\d[\d,]*)", raw_log, re.IGNORECASE
    )
    if row_match:
        result["affected_rows"] = int(row_match.group(1).replace(",", ""))
    else:
        row_match2 = re.search(r"(\d[\d,]*)\s+rows?\s+loaded", raw_log, re.IGNORECASE)
        if row_match2:
            result["affected_rows"] = int(row_match2.group(1).replace(",", ""))

    # Expected rows — "expected: ~1,200,000" / "Expected: ~1.2M"
    exp_match = re.search(
        r"expected[:\s~]+~?([\d,\.]+)([MK]?)", raw_log, re.IGNORECASE
    )
    if exp_match:
        val = float(exp_match.group(1).replace(",", ""))
        suffix = exp_match.group(2).upper()
        if suffix == "M":
            val *= 1_000_000
        elif suffix == "K":
            val *= 1_000
        result["expected_rows"] = int(val)

    # Stack trace summary — grab the first stack trace line if present
    stack_match = re.search(r"(at\s+[\w\.\$]+\(.*?\))", raw_log)
    if stack_match:
        result["stack_trace_summary"] = stack_match.group(1).strip()

    # LLM root-cause hint — one sentence
    llm_prompt = (
        "You are a data engineering expert. Given this raw pipeline log, "
        "provide a ONE-SENTENCE root-cause hypothesis based only on the evidence present.\n\n"
        f"LOG:\n{raw_log}\n\n"
        "Root cause hypothesis (one sentence, no preamble):"
    )
    result["llm_root_cause_hint"] = _llm(llm_prompt, max_tokens=512)

    return result


# ── TOOL 3: validate_fix ─────────────────────────────────────────────────────
# Patterns for dangerous unguarded statements
_UNGUARDED_DROP = re.compile(r"DROP\s+(TABLE|DATABASE|SCHEMA)\b", re.IGNORECASE)
_UNGUARDED_TRUNCATE = re.compile(r"TRUNCATE\b", re.IGNORECASE)
_HAS_WHERE = re.compile(r"\bWHERE\b", re.IGNORECASE)
_UNBOUNDED_DELETE = re.compile(r"DELETE\s+FROM\b", re.IGNORECASE)


def validate_fix(fix_sql: str) -> dict[str, Any]:
    """
    Validate a proposed SQL fix before it reaches the human approver.

    Rule-based checks run first; LLM safety rating is secondary.
    When LLM response doesn't parse cleanly, we default to 'risky', never 'safe'.

    Returns:
        {
            "safe": bool,
            "rating": "safe" | "risky" | "dangerous",
            "reason": str,
            "proceed_to_hitl": bool,
        }
    """
    stripped = fix_sql.strip()

    # ── Hard rule: unguarded DROP / TRUNCATE — immediately dangerous ──────────
    if _UNGUARDED_DROP.search(stripped):
        return {
            "safe": False,
            "rating": "dangerous",
            "reason": "Unguarded DROP detected. This statement would permanently destroy a table or schema without a WHERE clause. Must not proceed.",
            "proceed_to_hitl": False,
        }

    if _UNGUARDED_TRUNCATE.search(stripped):
        return {
            "safe": False,
            "rating": "dangerous",
            "reason": "TRUNCATE detected. This removes all rows without a WHERE clause and cannot be rolled back in most warehouses. Must not proceed.",
            "proceed_to_hitl": False,
        }

    # ── Warn on unbounded DELETE (no WHERE) ──────────────────────────────────
    if _UNBOUNDED_DELETE.search(stripped) and not _HAS_WHERE.search(stripped):
        return {
            "safe": False,
            "rating": "dangerous",
            "reason": "Unbounded DELETE (no WHERE clause) would remove all rows from the target table. Must not proceed.",
            "proceed_to_hitl": False,
        }

    # ── LLM safety rating ─────────────────────────────────────────────────────
    llm_prompt = (
        "You are a senior database administrator reviewing a SQL fix for a production data pipeline.\n"
        "Rate the following SQL statement on its safety for execution in production.\n\n"
        "Respond with a JSON object ONLY — no markdown, no explanation outside the JSON:\n"
        '{"rating": "safe"|"risky"|"dangerous", "reason": "<one sentence>"}\n\n'
        "Criteria:\n"
        "  safe     — read-only or a narrowly scoped write with a clear WHERE clause and obvious rollback path\n"
        "  risky    — modifies data but scoped; should be reviewed and have a rollback plan\n"
        "  dangerous — unguarded or irreversible; affects unbounded rows or drops structures\n\n"
        f"SQL:\n{stripped}\n\n"
        "JSON response:"
    )
    raw_response = _llm(llm_prompt, max_tokens=1024)

    rating = "risky"   # safe default — never default to "safe"
    reason = "LLM safety rating could not be parsed; defaulting to risky."

    try:
        # Strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?", "", raw_response).strip().rstrip("`")
        # Extract first JSON object
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            raw_rating = parsed.get("rating", "risky").lower()
            if raw_rating in ("safe", "risky", "dangerous"):
                rating = raw_rating
            reason = parsed.get("reason", reason)
    except Exception:
        pass  # keep defaults

    safe = rating == "safe"
    # Both "risky" and "dangerous" go to HITL if dangerous wasn't caught above,
    # but dangerous that slipped through here still gets proceed_to_hitl=True
    # so the human can explicitly reject it.
    return {
        "safe": safe,
        "rating": rating,
        "reason": reason,
        "proceed_to_hitl": True,  # all non-hard-rejected fixes go to human review
    }
