# state.py
# Shared AgentState — single source of truth passed between all pipeline components.
# Every field is explicitly typed to prevent bugs at component handoff boundaries.
# State is fully independent per run — no persistence across loan application evaluations.

from typing import TypedDict, List, Optional
from config_loader import get_config


class AgentState(TypedDict):
    # -------------------------
    # Input
    # -------------------------
    application_input: dict          # Raw JSON input as received from CLI or API
    validated_input: dict            # Cleaned and normalised fields after Mode 1 validation

    # -------------------------
    # Mode 1 outputs (Deterministic Scoring Engine)
    # -------------------------
    risk_score: Optional[float]      # Computed numeric risk score
    risk_tier: Optional[str]         # Low / Moderate / High
    borderline_flag: bool            # True if application falls within borderline margin
    triage_recommendation: Optional[str]  # escalate_to_underwriting / recommend_decline

    # -------------------------
    # Orchestrator outputs (Policy Retrieval)
    # -------------------------
    policy_context: str              # Retrieved policy text; empty string if none found
    policy_retrieval_status: str     # "found" / "none_found"

    # -------------------------
    # Mode 2 outputs (LLM Reasoning Agent)
    # -------------------------
    decision_explanation: Optional[str]   # LLM-generated structured explanation
    policy_references: List[str]          # List of cited policy statements

    # -------------------------
    # Pipeline control fields
    # -------------------------
    llm_status: str                  # "success" / "retry" / "failed_after_retries" / "skipped"
    fallback_used: bool              # True if deterministic fallback was triggered
    retry_count: int                 # Current number of Mode 2 retry attempts
    max_retries: int                 # Maximum allowed retries (from config)

    # -------------------------
    # Human-in-the-Loop review
    # -------------------------
    pending_review: bool             # True if case requires human review before finalising
    review_reason: Optional[str]     # Why the case was flagged for review
    human_decision: Optional[str]    # "approve" / "override_escalate" / "override_decline"

    # -------------------------
    # Error tracking
    # -------------------------
    error_flag: bool                 # True if a pipeline error occurred
    error_stage: Optional[str]       # "mode_1" / "mode_2" / "mode_3" / None
    error_message: Optional[str]     # Human-readable error description

    # -------------------------
    # Output
    # -------------------------
    timestamp: str                   # ISO format execution timestamp
    final_output: Optional[dict]     # Terminal output artifact passed to CLI / API


def initial_state(application_input: dict) -> AgentState:
    """
    Returns a clean initial AgentState for a new loan application evaluation.
    Call this at the start of every pipeline run.
    """
    cfg = get_config()
    return AgentState(
        application_input=application_input,
        validated_input={},
        risk_score=None,
        risk_tier=None,
        borderline_flag=False,
        triage_recommendation=None,
        policy_context="",
        policy_retrieval_status="none_found",
        decision_explanation=None,
        policy_references=[],
        llm_status="skipped",
        fallback_used=False,
        retry_count=0,
        max_retries=cfg.max_retries,
        pending_review=False,
        review_reason=None,
        human_decision=None,
        error_flag=False,
        error_stage=None,
        error_message=None,
        timestamp="",
        final_output=None,
    )
