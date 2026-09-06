"""
tests/test_scoring.py
Tests for the Mode 1 Deterministic Scoring Engine.
"""
import pytest
from tools.scoring import (
    compute_risk_score,
    assign_risk_tier,
    determine_base_recommendation,
    detect_borderline,
    run_scoring_engine,
)
from config_loader import get_config

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_input():
    return {
        "credit_score": 680,
        "monthly_income": 5000.0,
        "debt_to_income_ratio": 0.35,
        "recent_delinquencies": 0,
        "loan_amount_requested": 15000.0,
    }


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_identical_inputs_produce_identical_outputs(base_input):
    r1 = run_scoring_engine(base_input)
    r2 = run_scoring_engine(base_input)
    assert r1["risk_score"] == r2["risk_score"]
    assert r1["risk_tier"] == r2["risk_tier"]
    assert r1["triage_recommendation"] == r2["triage_recommendation"]
    assert r1["borderline_flag"] == r2["borderline_flag"]


def test_risk_score_is_reproducible(base_input):
    scores = [compute_risk_score(base_input) for _ in range(5)]
    assert len(set(scores)) == 1


# ---------------------------------------------------------------------------
# Risk tier boundary conditions
# ---------------------------------------------------------------------------

def test_tier_low_at_exact_boundary():
    cfg = get_config()
    # Score exactly at low_max should be Low
    # credit_score=850 → credit_risk=0; dti=0 → dti_risk=0; delinquencies=0 → delinq_risk=0
    # Craft input where score ≈ low_max exactly is hard; test the tier function directly
    assert assign_risk_tier(cfg.tier_low_max) == "Low"
    assert assign_risk_tier(cfg.tier_low_max - 0.01) == "Low"


def test_tier_moderate_just_above_low_boundary():
    cfg = get_config()
    assert assign_risk_tier(cfg.tier_low_max + 0.01) == "Moderate"


def test_tier_moderate_at_exact_boundary():
    cfg = get_config()
    assert assign_risk_tier(cfg.tier_moderate_max) == "Moderate"


def test_tier_high_just_above_moderate_boundary():
    cfg = get_config()
    assert assign_risk_tier(cfg.tier_moderate_max + 0.01) == "High"


def test_tier_low_score_zero():
    assert assign_risk_tier(0.0) == "Low"


def test_tier_high_score_100():
    assert assign_risk_tier(100.0) == "High"


# ---------------------------------------------------------------------------
# Borderline detection at margin edges
# ---------------------------------------------------------------------------

def test_borderline_exactly_at_escalation_threshold():
    cfg = get_config()
    # Score == escalation_threshold is inside the borderline zone
    assert detect_borderline(cfg.escalation_threshold) is True


def test_borderline_at_lower_edge():
    cfg = get_config()
    lower = cfg.escalation_threshold - cfg.borderline_margin
    assert detect_borderline(lower) is True


def test_borderline_at_upper_edge():
    cfg = get_config()
    upper = cfg.escalation_threshold + cfg.borderline_margin
    assert detect_borderline(upper) is True


def test_not_borderline_just_below_lower_edge():
    cfg = get_config()
    lower = cfg.escalation_threshold - cfg.borderline_margin
    assert detect_borderline(lower - 0.01) is False


def test_not_borderline_just_above_upper_edge():
    cfg = get_config()
    upper = cfg.escalation_threshold + cfg.borderline_margin
    assert detect_borderline(upper + 0.01) is False


def test_not_borderline_very_low_score():
    assert detect_borderline(0.0) is False


def test_not_borderline_very_high_score():
    assert detect_borderline(100.0) is False


# ---------------------------------------------------------------------------
# Weights sum validation
# ---------------------------------------------------------------------------

def test_weights_sum_to_one():
    cfg = get_config()
    total = (
        cfg.weight_credit_score
        + cfg.weight_dti_ratio
        + cfg.weight_delinquencies
        + cfg.weight_income_to_loan
    )
    assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, expected 1.0"


# ---------------------------------------------------------------------------
# Base recommendation
# ---------------------------------------------------------------------------

def test_recommendation_at_escalation_threshold():
    cfg = get_config()
    assert determine_base_recommendation(cfg.escalation_threshold) == "escalate_to_underwriting"


def test_recommendation_below_threshold_escalates():
    cfg = get_config()
    assert determine_base_recommendation(cfg.escalation_threshold - 1) == "escalate_to_underwriting"


def test_recommendation_above_threshold_declines():
    cfg = get_config()
    assert determine_base_recommendation(cfg.escalation_threshold + 0.01) == "recommend_decline"


# ---------------------------------------------------------------------------
# Risk score range
# ---------------------------------------------------------------------------

def test_risk_score_stays_in_valid_range():
    # Perfect applicant
    perfect = {
        "credit_score": 850,
        "monthly_income": 10000.0,
        "debt_to_income_ratio": 0.0,
        "recent_delinquencies": 0,
        "loan_amount_requested": 1000.0,
    }
    assert 0.0 <= compute_risk_score(perfect) <= 100.0


def test_worst_case_score_stays_in_range():
    worst = {
        "credit_score": 300,
        "monthly_income": 500.0,
        "debt_to_income_ratio": 1.0,
        "recent_delinquencies": 10,
        "loan_amount_requested": 100000.0,
    }
    assert 0.0 <= compute_risk_score(worst) <= 100.0


# ---------------------------------------------------------------------------
# Run scoring engine returns all required keys
# ---------------------------------------------------------------------------

def test_run_scoring_engine_returns_required_keys(base_input):
    result = run_scoring_engine(base_input)
    for key in ("risk_score", "risk_tier", "triage_recommendation", "borderline_flag"):
        assert key in result, f"Missing key: {key}"


def test_run_scoring_engine_risk_tier_valid_values(base_input):
    result = run_scoring_engine(base_input)
    assert result["risk_tier"] in ("Low", "Moderate", "High")


def test_run_scoring_engine_recommendation_valid_values(base_input):
    result = run_scoring_engine(base_input)
    assert result["triage_recommendation"] in (
        "escalate_to_underwriting", "recommend_decline"
    )
