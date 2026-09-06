"""
tests/test_pipeline.py
Integration tests for the full LangGraph pipeline.
All tests use _no_llm_mode=True to avoid Ollama dependency.
"""
import pytest
from orchestrator import run_pipeline, route_after_mode_1
from tools.scoring import run_scoring_engine
from state import initial_state
from config_loader import get_config


# ---------------------------------------------------------------------------
# Path A — non-borderline, no LLM
# ---------------------------------------------------------------------------

def _non_borderline_input():
    """A clearly high-risk input that sits well outside the borderline zone."""
    cfg = get_config()
    return {
        "credit_score": 300,
        "monthly_income": 1500.0,
        "debt_to_income_ratio": 0.95,
        "recent_delinquencies": 5,
        "loan_amount_requested": 50000.0,
        "_no_llm_mode": True,
    }


def _non_borderline_low_risk_input():
    """A clearly low-risk input that sits well outside the borderline zone."""
    return {
        "credit_score": 850,
        "monthly_income": 10000.0,
        "debt_to_income_ratio": 0.05,
        "recent_delinquencies": 0,
        "loan_amount_requested": 5000.0,
        "_no_llm_mode": True,
    }


def test_path_a_runs_end_to_end():
    result = run_pipeline(_non_borderline_input())
    assert result is not None
    assert isinstance(result, dict)


def test_path_a_returns_required_fields():
    result = run_pipeline(_non_borderline_low_risk_input())
    for field in ("risk_score", "risk_tier", "borderline_flag",
                  "triage_recommendation", "llm_status", "error_flag"):
        assert field in result, f"Missing field: {field}"


def test_path_a_llm_status_is_skipped():
    result = run_pipeline(_non_borderline_input())
    assert result["llm_status"] == "skipped"


def test_path_a_no_error():
    result = run_pipeline(_non_borderline_low_risk_input())
    assert result["error_flag"] is False


def test_path_a_risk_score_is_float():
    result = run_pipeline(_non_borderline_low_risk_input())
    assert isinstance(result["risk_score"], float)


def test_path_a_risk_tier_is_valid():
    result = run_pipeline(_non_borderline_input())
    assert result["risk_tier"] in ("Low", "Moderate", "High")


def test_path_a_recommendation_valid():
    result = run_pipeline(_non_borderline_input())
    assert result["triage_recommendation"] in (
        "escalate_to_underwriting", "recommend_decline"
    )


# ---------------------------------------------------------------------------
# Path B routing — verify borderline cases are correctly flagged
# ---------------------------------------------------------------------------

def _borderline_input():
    """Construct an input that lands in the borderline zone."""
    cfg = get_config()
    # Score ≈ escalation_threshold via moderate credit score and DTI
    return {
        "credit_score": 620,
        "monthly_income": 4500.0,
        "debt_to_income_ratio": 0.41,
        "recent_delinquencies": 1,
        "loan_amount_requested": 15000.0,
    }


def test_borderline_input_is_detected_as_borderline():
    inp = _borderline_input()
    from tools.validator import validate_input
    _, validated, _ = validate_input(inp)
    result = run_scoring_engine(validated)
    assert result["borderline_flag"] is True, (
        f"Expected borderline, got score={result['risk_score']}"
    )


def test_route_after_mode_1_returns_policy_retrieval_for_borderline():
    """route_after_mode_1 must return 'policy_retrieval' when borderline and LLM enabled."""
    state = initial_state(_borderline_input())
    # Simulate mode_1 output
    state = {
        **state,
        "borderline_flag": True,
        "error_flag": False,
        "application_input": _borderline_input(),  # no _no_llm_mode
    }
    route = route_after_mode_1(state)
    assert route == "policy_retrieval"


def test_route_after_mode_1_returns_review_checkpoint_for_no_llm():
    """When _no_llm_mode is set, even borderline cases skip to review_checkpoint."""
    inp = {**_borderline_input(), "_no_llm_mode": True}
    state = initial_state(inp)
    state = {**state, "borderline_flag": True, "error_flag": False}
    route = route_after_mode_1(state)
    assert route == "review_checkpoint"


def test_route_after_mode_1_returns_end_error_on_error():
    state = initial_state(_borderline_input())
    state = {**state, "error_flag": True}
    route = route_after_mode_1(state)
    assert route == "end_error"


# ---------------------------------------------------------------------------
# Mode 1 failure produces correct error JSON
# ---------------------------------------------------------------------------

def test_invalid_input_produces_error_output():
    result = run_pipeline({
        "credit_score": "bad_value",
        "monthly_income": 5000.0,
        "debt_to_income_ratio": 0.35,
        "recent_delinquencies": 0,
        "loan_amount_requested": 15000.0,
    })
    assert result["error_flag"] is True
    assert result.get("error_stage") == "mode_1"
    assert result.get("error_message")


def test_missing_field_produces_error_output():
    result = run_pipeline({
        "credit_score": 680,
        "monthly_income": 5000.0,
        # missing debt_to_income_ratio
        "recent_delinquencies": 0,
        "loan_amount_requested": 15000.0,
    })
    assert result["error_flag"] is True


def test_empty_input_produces_error_output():
    result = run_pipeline({})
    assert result["error_flag"] is True
    assert result.get("error_stage") == "mode_1"
