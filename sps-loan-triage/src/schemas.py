# schemas.py
# Pydantic schemas for structured input validation and LLM output enforcement.
# These schemas define the contract at every system boundary.

from pydantic import BaseModel, Field
from typing import List, Literal, Optional


# ---------------------------------------------------------------------------
# Input Schema — Loan Application
# ---------------------------------------------------------------------------

class LoanApplicationInput(BaseModel):
    """
    Structured input for a single unsecured personal consumer loan application.
    All five fields are required. Types and ranges are strictly enforced.
    """
    credit_score: int = Field(
        ge=300, le=850,
        description="Applicant credit score (FICO range: 300–850)"
    )
    monthly_income: float = Field(
        gt=0,
        description="Gross monthly income in USD"
    )
    debt_to_income_ratio: float = Field(
        ge=0.0, le=1.0,
        description="Debt-to-income ratio as a decimal (e.g., 0.43 = 43%)"
    )
    recent_delinquencies: int = Field(
        ge=0,
        description="Number of delinquencies in the past 24 months"
    )
    loan_amount_requested: float = Field(
        gt=0,
        description="Loan amount requested in USD"
    )


# ---------------------------------------------------------------------------
# Mode 2 Output Schema — LLM Reasoning Agent
# ---------------------------------------------------------------------------

class ReasoningAgentOutput(BaseModel):
    """
    Structured output from Mode 2 (Policy-Aware Reasoning Agent).
    Enforced via Ollama's native structured output (JSON schema injection via format param).
    """
    decision_explanation: str = Field(
        description=(
            "Concise, policy-aligned justification for the triage recommendation. "
            "Must reference specific input factors and policy clauses. "
            "Must not contradict or override the deterministic risk score or tier."
        )
    )
    policy_references: List[str] = Field(
        description=(
            "List of specific policy statements cited in the explanation. "
            "Must be drawn only from retrieved policy context. "
            "Return an empty list if no policy clauses were retrieved."
        )
    )
    confidence_note: Optional[str] = Field(
        default=None,
        description=(
            "Optional note on borderline uncertainty or edge case conditions "
            "observed during reasoning. Leave null if not applicable."
        )
    )


# ---------------------------------------------------------------------------
# Final Output Schema — Pipeline Output Artifact
# ---------------------------------------------------------------------------

class TriageOutput(BaseModel):
    """
    Final structured JSON output returned to the CLI and downstream systems.
    This is the terminal artifact produced by Mode 3 after validation.
    """
    risk_score: Optional[float] = Field(description="Computed numeric risk score")
    risk_tier: Optional[str] = Field(description="Low / Moderate / High")
    borderline_flag: bool = Field(description="True if application is near a decision threshold")
    triage_recommendation: Optional[str] = Field(
        description="escalate_to_underwriting or recommend_decline"
    )
    decision_explanation: Optional[str] = Field(
        description="LLM-generated explanation (null if LLM was skipped or failed)"
    )
    policy_references: List[str] = Field(
        default_factory=list,
        description="Policy statements cited in the explanation"
    )
    llm_status: str = Field(
        description="success / retry / failed_after_retries / skipped"
    )
    fallback_used: bool = Field(
        description="True if deterministic fallback was triggered due to LLM failure"
    )
    # Human-in-the-Loop fields
    pending_review: bool = Field(
        default=False,
        description="True if this case requires human review before finalising"
    )
    review_reason: Optional[str] = Field(
        default=None,
        description="Why the case was flagged for human review"
    )
    human_decision: Optional[str] = Field(
        default=None,
        description="Human reviewer decision: approve / override_escalate / override_decline"
    )
    # Error tracking
    error_flag: bool = Field(description="True if a pipeline error occurred")
    error_stage: Optional[str] = Field(description="Stage where error occurred, if any")
    error_message: Optional[str] = Field(description="Error description, if any")
    timestamp: str = Field(description="ISO format execution timestamp")


# ---------------------------------------------------------------------------
# Error Output Schema — Hard Failure
# ---------------------------------------------------------------------------

class ErrorOutput(BaseModel):
    """
    Minimal structured output returned when the pipeline encounters a hard failure.
    """
    error_flag: Literal[True] = True
    error_stage: str = Field(description="Pipeline stage where failure occurred")
    error_message: str = Field(description="Description of the failure")
    timestamp: str = Field(description="ISO format execution timestamp")
