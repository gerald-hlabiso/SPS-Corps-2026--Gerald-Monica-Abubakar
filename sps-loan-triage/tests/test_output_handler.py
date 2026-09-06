"""
tests/test_output_handler.py
Tests for Mode 3 output assembly, validation, and logging.
"""
import pytest
from datetime import datetime, timezone
from tools.output_handler import assemble_final_output


def _make_state(overrides=None):
    """Return a minimal valid AgentState dict for testing."""
    state = {
        "application_input": {"credit_score": 680},
        "validated_input": {
            "credit_score": 680,
            "monthly_income": 5000.0,
            "debt_to_income_ratio": 0.35,
            "recent_delinquencies": 0,
            "loan_amount_requested": 15000.0,
        },
        "risk_score": 30.5,
        "risk_tier": "Low",
        "borderline_flag": False,
        "triage_recommendation": "escalate_to_underwriting",
        "decision_explanation": None,
        "policy_references": [],
        "llm_status": "skipped",
        "fallback_used": False,
        "pending_review": False,
        "review_reason": None,
        "human_decision": None,
        "error_flag": False,
        "error_stage": None,
        "error_message": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "final_output": None,
    }
    if overrides:
        state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# Required fields in output
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = [
    "risk_score", "risk_tier", "borderline_flag", "triage_recommendation",
    "decision_explanation", "policy_references", "llm_status", "fallback_used",
    "pending_review", "error_flag", "error_stage", "error_message", "timestamp",
]


def test_all_required_fields_present():
    output = assemble_final_output(_make_state())
    for field in REQUIRED_FIELDS:
        assert field in output, f"Missing field: {field}"


def test_output_is_dict():
    output = assemble_final_output(_make_state())
    assert isinstance(output, dict)


def test_policy_references_is_list():
    output = assemble_final_output(_make_state())
    assert isinstance(output["policy_references"], list)


# ---------------------------------------------------------------------------
# Path A (non-borderline, no LLM) output structure
# ---------------------------------------------------------------------------

def test_path_a_output_has_no_explanation():
    output = assemble_final_output(_make_state())
    assert output["decision_explanation"] is None
    assert output["llm_status"] == "skipped"
    assert output["fallback_used"] is False


def test_path_a_error_flag_is_false():
    output = assemble_final_output(_make_state())
    assert output["error_flag"] is False


# ---------------------------------------------------------------------------
# Fallback output structure
# ---------------------------------------------------------------------------

def test_fallback_output_marks_fallback_used():
    state = _make_state({
        "fallback_used": True,
        "decision_explanation": None,
        "policy_references": [],
        "llm_status": "failed_after_retries",
        "error_flag": True,
        "error_stage": "mode_2",
        "error_message": "Ollama timeout",
    })
    output = assemble_final_output(state)
    assert output["fallback_used"] is True
    assert output["llm_status"] == "failed_after_retries"
    assert output["decision_explanation"] is None


# ---------------------------------------------------------------------------
# Error output structure (Mode 3 failure)
# ---------------------------------------------------------------------------

def test_error_output_has_error_flag():
    # Trigger a Mode 3 failure by passing invalid state
    state = _make_state({"risk_score": "not_a_float"})  # invalid type
    output = assemble_final_output(state)
    assert output["error_flag"] is True


# ---------------------------------------------------------------------------
# HITL fields
# ---------------------------------------------------------------------------

def test_pending_review_false_by_default():
    output = assemble_final_output(_make_state())
    assert output["pending_review"] is False
    assert output["review_reason"] is None
    assert output["human_decision"] is None


def test_pending_review_true_when_set():
    state = _make_state({
        "pending_review": True,
        "review_reason": "LLM fallback triggered",
    })
    output = assemble_final_output(state)
    assert output["pending_review"] is True
    assert output["review_reason"] == "LLM fallback triggered"


def test_human_decision_included_in_output():
    state = _make_state({
        "pending_review": False,
        "human_decision": "approve",
    })
    output = assemble_final_output(state)
    assert output["human_decision"] == "approve"


# ---------------------------------------------------------------------------
# Timestamp
# ---------------------------------------------------------------------------

def test_timestamp_is_present_and_nonempty():
    output = assemble_final_output(_make_state())
    assert output["timestamp"]
    assert len(output["timestamp"]) > 10
