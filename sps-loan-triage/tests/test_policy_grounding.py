"""Tests for deterministic policy applicability and LLM grounding guards."""

import pytest

from agent.reasoning_agent import _normalize_policy_references, _validate_grounding
from orchestrator import policy_retrieval_node
from schemas import ReasoningAgentOutput
from state import initial_state
from tools.policy_retrieval import (
    format_policy_context,
    required_policy_action,
    retrieve_policy_clauses,
)


def _ids(clauses):
    return {clause.split(":", 1)[0] for clause in clauses}


def _case(credit_score=600, dti=0.45, delinquencies=1):
    return {
        "credit_score": credit_score,
        "monthly_income": 4500.0,
        "debt_to_income_ratio": dti,
        "recent_delinquencies": delinquencies,
        "loan_amount_requested": 15000.0,
    }


def test_exact_regression_case_excludes_false_threshold_policies():
    clauses, status = retrieve_policy_clauses(
        risk_tier="Moderate",
        borderline_flag=True,
        validated_input=_case(),
        triage_recommendation="recommend_decline",
    )
    ids = _ids(clauses)

    assert status == "found"
    assert "POL-002" in ids
    assert "POL-003" not in ids
    assert "POL-004" not in ids
    assert "POL-008" in ids


def test_mandatory_policy_action_overrides_decline():
    clauses, _ = retrieve_policy_clauses(
        risk_tier="Moderate",
        borderline_flag=True,
        validated_input=_case(),
        triage_recommendation="recommend_decline",
    )
    assert required_policy_action(clauses) == "escalate_to_underwriting"


def test_below_threshold_case_does_not_receive_dti_policy():
    clauses, _ = retrieve_policy_clauses(
        risk_tier="Moderate",
        borderline_flag=True,
        validated_input=_case(credit_score=620, dti=0.41),
        triage_recommendation="escalate_to_underwriting",
    )
    ids = _ids(clauses)
    assert "POL-002" not in ids
    assert "POL-003" not in ids
    assert "POL-004" not in ids
    assert "POL-008" not in ids


def _reasoning_state():
    state = initial_state(_case())
    clauses, _ = retrieve_policy_clauses(
        risk_tier="Moderate",
        borderline_flag=True,
        validated_input=_case(),
        triage_recommendation="recommend_decline",
    )
    return {
        **state,
        "validated_input": _case(),
        "risk_score": 38.95,
        "risk_tier": "Moderate",
        "borderline_flag": True,
        "triage_recommendation": "escalate_to_underwriting",
        "policy_context": format_policy_context(clauses),
    }, clauses


def test_grounding_guard_rejects_false_credit_threshold():
    state, clauses = _reasoning_state()
    output = ReasoningAgentOutput(
        decision_explanation="The credit score is below 580 and is subprime.",
        policy_references=[clauses[0]],
    )
    with pytest.raises(ValueError, match="credit score"):
        _validate_grounding(output, state)


def test_grounding_guard_rejects_policy_not_in_context():
    state, _ = _reasoning_state()
    output = ReasoningAgentOutput(
        decision_explanation="The 45% DTI requires escalation.",
        policy_references=["POL-999: Invented policy."],
    )
    with pytest.raises(ValueError, match="not supplied"):
        _validate_grounding(output, state)


def test_grounded_explanation_passes():
    state, clauses = _reasoning_state()
    dti_clause = next(c for c in clauses if c.startswith("POL-002:"))
    output = ReasoningAgentOutput(
        decision_explanation=(
            "The applicant's 45.0% debt-to-income ratio exceeds 43%, "
            "so the case requires escalation to underwriting."
        ),
        policy_references=[dti_clause],
    )
    _validate_grounding(output, state)


def test_policy_node_resolves_conflict_and_refreshes_context():
    state = initial_state(_case())
    state = {
        **state,
        "validated_input": _case(),
        "risk_score": 38.95,
        "risk_tier": "Moderate",
        "borderline_flag": True,
        "triage_recommendation": "recommend_decline",
    }

    result = policy_retrieval_node(state)

    assert result["triage_recommendation"] == "escalate_to_underwriting"
    assert "POL-002:" in result["policy_context"]
    assert "POL-003:" not in result["policy_context"]
    assert "POL-004:" not in result["policy_context"]
    assert "POL-006:" not in result["policy_context"]


def test_policy_ids_are_expanded_to_exact_clauses():
    state, clauses = _reasoning_state()
    output = ReasoningAgentOutput(
        decision_explanation="The applicant's 45.0% DTI requires escalation.",
        policy_references=["POL-002", "POL-005"],
    )

    normalized = _normalize_policy_references(output, state)

    assert normalized.policy_references
    assert all(ref.startswith("POL-") and ": " in ref for ref in normalized.policy_references)
    assert next(c for c in clauses if c.startswith("POL-002:")) in normalized.policy_references


def test_grounding_guard_rejects_ambiguous_credit_threshold_claim():
    state, clauses = _reasoning_state()
    output = ReasoningAgentOutput(
        decision_explanation="The applicant's credit score of 600 is below the threshold.",
        policy_references=[next(c for c in clauses if c.startswith("POL-002:"))],
    )
    with pytest.raises(ValueError, match="credit score"):
        _validate_grounding(output, state)
