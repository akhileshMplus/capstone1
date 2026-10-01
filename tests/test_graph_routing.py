"""
Unit tests for the bounded HITL retry/escalation routing logic in agent/graph.py.

route_after_hitl is a pure function of state — no LLM or I/O — so it's tested
directly without mocking.
"""
from __future__ import annotations

from langgraph.graph import END

from agent.graph import MAX_HITL_RETRIES, route_after_hitl


def test_route_after_hitl_approved_goes_to_execute():
    assert route_after_hitl({"hitl_decision": "approved", "approval_attempts": 0}) == "execute"


def test_route_after_hitl_rejected_below_retry_limit_replans():
    state = {"hitl_decision": "rejected", "approval_attempts": MAX_HITL_RETRIES - 1}
    assert route_after_hitl(state) == "plan"


def test_route_after_hitl_rejected_at_retry_limit_escalates_to_end():
    state = {"hitl_decision": "rejected", "approval_attempts": MAX_HITL_RETRIES}
    assert route_after_hitl(state) == END


def test_route_after_hitl_pending_ends():
    assert route_after_hitl({"hitl_decision": "pending", "approval_attempts": 0}) == END


def test_route_after_hitl_missing_decision_defaults_to_end():
    assert route_after_hitl({}) == END
