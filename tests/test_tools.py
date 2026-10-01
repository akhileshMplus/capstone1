"""
Unit tests for agent/tools.py — validate_fix safety rules and execute_sql guardrails.

These tests avoid real network calls: any code path that would call the LLM
either short-circuits before doing so (hard rules) or has `_llm` monkeypatched.
"""
from __future__ import annotations

import agent.tools as tools


# ── validate_fix: hard rules (must never reach the LLM) ──────────────────────
def test_validate_fix_rejects_unguarded_drop_table():
    result = tools.validate_fix("DROP TABLE transactions;")
    assert result["rating"] == "dangerous"
    assert result["proceed_to_hitl"] is False


def test_validate_fix_rejects_truncate():
    result = tools.validate_fix("TRUNCATE transactions;")
    assert result["rating"] == "dangerous"
    assert result["proceed_to_hitl"] is False


def test_validate_fix_rejects_unbounded_delete():
    result = tools.validate_fix("DELETE FROM transactions;")
    assert result["rating"] == "dangerous"
    assert result["proceed_to_hitl"] is False


def test_validate_fix_allows_bounded_delete_to_llm_review(monkeypatch):
    monkeypatch.setattr(tools, "_llm", lambda prompt, max_tokens=200: '{"rating": "safe", "reason": "scoped"}')
    result = tools.validate_fix("DELETE FROM transactions WHERE id = 1;")
    assert result["rating"] == "safe"
    assert result["proceed_to_hitl"] is True


def test_validate_fix_defaults_to_risky_when_llm_response_unparseable(monkeypatch):
    monkeypatch.setattr(tools, "_llm", lambda prompt, max_tokens=200: "not json at all")
    result = tools.validate_fix("UPDATE transactions SET status = 'ok' WHERE id = 1;")
    assert result["rating"] == "risky"
    assert result["safe"] is False
    assert result["proceed_to_hitl"] is True


# ── execute_sql: write-statement guardrail ────────────────────────────────────
def test_execute_sql_blocks_write_statements():
    result = tools.execute_sql("DELETE FROM pipelines WHERE id = 'PL-001';")
    assert result["requires_approval"] is True
    assert "error" in result


def test_execute_sql_runs_read_only_query():
    result = tools.execute_sql("SELECT name FROM pipelines WHERE id = 'PL-001';")
    assert "error" not in result
    assert result["rows"] == [["daily_sales_aggregation"]]


# ── parse_log: structured extraction (LLM hint mocked out) ───────────────────
def test_parse_log_extracts_structured_fields(monkeypatch):
    monkeypatch.setattr(tools, "_llm", lambda prompt, max_tokens=128: "CDC job failure.")
    log = (
        "2025-06-01 02:14 UTC | pipeline: daily_sales_aggregation\n"
        "ERROR: NullPointerException in transform_step\n"
        "Rows loaded: 0 (expected: ~1,200,000)\n"
    )
    parsed = tools.parse_log(log)
    assert parsed["pipeline"] == "daily_sales_aggregation"
    assert parsed["error_type"].startswith("NullPointerException")
    assert parsed["affected_rows"] == 0
    assert parsed["expected_rows"] == 1_200_000
    assert parsed["llm_root_cause_hint"] == "CDC job failure."
