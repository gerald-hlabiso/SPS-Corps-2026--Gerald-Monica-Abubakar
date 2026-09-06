# tools/scoring.py
# Mode 1 — Deterministic Scoring Engine
# All thresholds and weights are loaded from config/scoring_config.yaml via config_loader.
# No magic numbers — tune by editing the YAML file and restarting.

from config_loader import get_config


# ---------------------------------------------------------------------------
# Normalisation helpers (pure functions — no config dependency)
# ---------------------------------------------------------------------------

def _normalize_credit_score(credit_score: int) -> float:
    """Convert FICO (300–850) to risk contribution 0–100. Higher score = lower risk."""
    return ((850 - credit_score) / (850 - 300)) * 100


def _normalize_dti(dti_ratio: float) -> float:
    """Convert DTI (0.0–1.0) to risk contribution 0–100. Higher DTI = higher risk."""
    return dti_ratio * 100


def _normalize_delinquencies(recent_delinquencies: int) -> float:
    """Convert delinquency count to risk contribution 0–100. Capped at 5 = max risk."""
    return min(recent_delinquencies / 5, 1.0) * 100


def _normalize_income_to_loan(monthly_income: float, loan_amount_requested: float) -> float:
    """
    Compute loan-to-income risk contribution 0–100.
    Higher loan relative to annual income = higher risk. Cap at 5x annual income.
    """
    annual_income = monthly_income * 12
    if annual_income <= 0:
        return 100.0
    ratio = loan_amount_requested / annual_income
    return min(ratio / 5.0, 1.0) * 100


# ---------------------------------------------------------------------------
# Core scoring functions — each reads config fresh via get_config()
# ---------------------------------------------------------------------------

def compute_risk_score(validated_input: dict) -> float:
    """
    Compute weighted risk score (0–100) from validated input fields.
    Higher score = higher risk. Weights loaded from scoring_config.yaml.
    """
    cfg = get_config()

    credit_risk = _normalize_credit_score(validated_input["credit_score"])
    dti_risk = _normalize_dti(validated_input["debt_to_income_ratio"])
    delinquency_risk = _normalize_delinquencies(validated_input["recent_delinquencies"])
    income_loan_risk = _normalize_income_to_loan(
        validated_input["monthly_income"],
        validated_input["loan_amount_requested"],
    )

    risk_score = (
        cfg.weight_credit_score   * credit_risk
        + cfg.weight_dti_ratio    * dti_risk
        + cfg.weight_delinquencies * delinquency_risk
        + cfg.weight_income_to_loan * income_loan_risk
    )
    return round(risk_score, 2)


def assign_risk_tier(risk_score: float) -> str:
    """Assign Low / Moderate / High risk tier based on computed score."""
    cfg = get_config()
    if risk_score <= cfg.tier_low_max:
        return "Low"
    elif risk_score <= cfg.tier_moderate_max:
        return "Moderate"
    else:
        return "High"


def determine_base_recommendation(risk_score: float) -> str:
    """Return escalate_to_underwriting or recommend_decline based on escalation threshold."""
    cfg = get_config()
    if risk_score <= cfg.escalation_threshold:
        return "escalate_to_underwriting"
    return "recommend_decline"


def detect_borderline(risk_score: float) -> bool:
    """
    Return True if the score falls within the borderline margin around the
    escalation threshold.  Borderline cases are routed to Mode 2 (LLM reasoning).
    """
    cfg = get_config()
    lower = cfg.escalation_threshold - cfg.borderline_margin
    upper = cfg.escalation_threshold + cfg.borderline_margin
    return lower <= risk_score <= upper


def run_scoring_engine(validated_input: dict) -> dict:
    """
    Run the full Mode 1 scoring pipeline on validated input.

    Returns:
        Dict with keys: risk_score, risk_tier, triage_recommendation, borderline_flag
    """
    risk_score = compute_risk_score(validated_input)
    risk_tier = assign_risk_tier(risk_score)
    triage_recommendation = determine_base_recommendation(risk_score)
    borderline_flag = detect_borderline(risk_score)

    return {
        "risk_score": risk_score,
        "risk_tier": risk_tier,
        "triage_recommendation": triage_recommendation,
        "borderline_flag": borderline_flag,
    }
